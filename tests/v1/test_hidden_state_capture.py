# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.v1.hidden_state_capture import (
    HiddenStateCaptureBuffer,
    HiddenStateCapturePlan,
    capture_scheduled_hidden_states,
    drop_incompatible_aux_plans,
    hidden_state_capture_capability,
    validate_hidden_state_capture,
)


def test_hidden_capture_window_uses_teacher_forcing_positions():
    response = HiddenStateCapturePlan.from_window("req", 17, 1000, 2000, "collection")
    absolute = HiddenStateCapturePlan.from_window(
        "req", 17, 1016, 2016, "collection", coordinate="absolute"
    )
    assert (response.window_start_abs, response.window_end_abs) == (1016, 2016)
    assert response == absolute
    with pytest.raises(ValueError):
        HiddenStateCapturePlan.from_window("req", 17, 2000, 1000, "collection")


def test_window_logprobs_follow_accepted_hidden_rows_and_retry_deduplication():
    """Raw top-k labels must stay aligned when speculation rejects a suffix."""
    plan = HiddenStateCapturePlan.from_window(
        "req", 4, 0, 3, "collection", logprob_topk=2
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    logits = torch.tensor(
        [[1.0, 2.0, 4.0, 0.0], [2.0, 1.0, 0.0, 3.0], [4.0, 1.0, 0.0, 2.0]]
    )
    chunk = capture_scheduled_hidden_states(
        {"req": plan},
        ["req"],
        {"req": 6},
        {"req": 0},
        hidden,
        logits=logits,
        logits_indices=torch.tensor([3, 4, 5]),
    )["req"]
    expected_values, expected_ids = logits.log_softmax(-1).topk(2, dim=-1)
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk.accepted(3, 5))
    buffer.add(chunk.accepted(4, 6))
    result = buffer.finish()
    np.testing.assert_array_equal(result.hidden_positions, [3, 4, 5])
    torch.testing.assert_close(result.hidden_states, hidden[3:6])
    torch.testing.assert_close(result.logprobs, expected_values)
    assert torch.equal(result.logprob_token_ids, expected_ids)


def test_hidden_capture_selects_only_each_requests_window_rows():
    plans = {
        "a": HiddenStateCapturePlan.from_window("a", 4, 0, 2, "a"),
        "b": HiddenStateCapturePlan.from_window("b", 2, 1, 3, "b"),
    }
    hidden = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    chunks = capture_scheduled_hidden_states(
        plans,
        ["a", "b", "c"],
        {"a": 4, "b": 3, "c": 2},
        {"a": 0, "b": 2, "c": 8},
        hidden,
    )
    np.testing.assert_array_equal(chunks["a"].positions, [3])
    torch.testing.assert_close(chunks["a"].hidden_states, hidden[3:4])
    np.testing.assert_array_equal(chunks["b"].positions, [2, 3])
    torch.testing.assert_close(chunks["b"].hidden_states, hidden[4:6])
    assert "c" not in chunks

    after_window = capture_scheduled_hidden_states(
        plans, ["a"], {"a": 2}, {"a": 5}, hidden
    )
    assert after_window == {}


def test_hidden_capture_uses_prompt_logprob_batch_row_boundaries():
    plan = HiddenStateCapturePlan.from_window("b", 2, 1, 3, "collection")
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    chunks = capture_scheduled_hidden_states(
        {"b": plan},
        ["a", "b"],
        {"a": 3, "b": 3},
        {"a": 0, "b": 2},
        hidden,
        query_start_loc=np.array([0, 3, 6]),
    )
    np.testing.assert_array_equal(chunks["b"].positions, [2, 3])
    torch.testing.assert_close(chunks["b"].hidden_states, hidden[3:5])


def test_hidden_capture_rejects_mismatched_runner_row_boundaries():
    plan = HiddenStateCapturePlan.from_window("a", 1, 0, 1, "collection")
    with pytest.raises(ValueError, match="row boundaries"):
        capture_scheduled_hidden_states(
            {"a": plan},
            ["a"],
            {"a": 1},
            {"a": 0},
            torch.zeros(2, 1),
            query_start_loc=np.array([0, 2]),
        )


def test_hidden_capture_discards_rejected_speculative_rows_and_short_samples():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 10, 13, "collection", min_rows=2
    )
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    chunk = capture_scheduled_hidden_states(
        {"req": plan}, ["req"], {"req": 3}, {"req": 10}, hidden
    )["req"]

    buffer = HiddenStateCaptureBuffer(plan)
    result, reason = buffer.update(chunk, 10, 12, stopped=True)
    assert reason is None
    assert result is not None
    np.testing.assert_array_equal(result.hidden_positions, [10, 11])
    torch.testing.assert_close(result.hidden_states, hidden[:2])
    assert (result.hidden_position_start, result.hidden_position_end) == (10, 12)
    assert (result.hidden_window_start, result.hidden_window_end) == (10, 13)
    assert result.layer_ids == ()
    assert result.includes_final_layer

    early_eos = HiddenStateCaptureBuffer(plan)
    result, reason = early_eos.update(chunk, 10, 11, stopped=True)
    assert result is None
    assert reason == "insufficient_rows"


def test_hidden_capture_preserves_aux_final_layout_and_deduplicates_retries():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 0, 2, "collection", aux_layer_ids=(3,)
    )
    final = torch.tensor([[1.0], [2.0]])
    aux = torch.tensor([[10.0], [20.0]])
    chunk = capture_scheduled_hidden_states(
        {"req": plan}, ["req"], {"req": 2}, {"req": 0}, final, [aux]
    )["req"]
    torch.testing.assert_close(
        chunk.hidden_states, torch.tensor([[10.0, 1.0], [20.0, 2.0]])
    )
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk)
    buffer.add(chunk.accepted(1, 2))
    result = buffer.finish()
    assert result is not None
    assert result.hidden_layout == "aux_final"
    assert result.layer_ids == (3,)
    assert result.includes_final_layer
    np.testing.assert_array_equal(result.hidden_positions, [0, 1])
    torch.testing.assert_close(result.hidden_states, chunk.hidden_states)


def test_hidden_capture_dflash_layout_excludes_final_state():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 0, 1, "collection", aux_layer_ids=(3,), hidden_layout="dflash_aux"
    )
    chunk = capture_scheduled_hidden_states(
        {"req": plan},
        ["req"],
        {"req": 1},
        {"req": 0},
        torch.tensor([[1.0]]),
        [torch.tensor([[10.0]])],
    )["req"]
    torch.testing.assert_close(chunk.hidden_states, torch.tensor([[10.0]]))
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk)
    result = buffer.finish()
    assert result is not None
    assert result.layer_ids == (3,)
    assert not result.includes_final_layer


@pytest.mark.parametrize("non_blocking", [False, True])
def test_hidden_capture_keeps_bfloat16_on_cpu(non_blocking):
    plan = HiddenStateCapturePlan.from_window("req", 1, 0, 1, "collection")
    hidden = torch.tensor([[1.0]], dtype=torch.bfloat16)
    chunk = capture_scheduled_hidden_states(
        {"req": plan},
        ["req"],
        {"req": 1},
        {"req": 0},
        hidden,
        non_blocking=non_blocking,
    )["req"]
    assert chunk.hidden_states.dtype == torch.bfloat16
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk)
    result = buffer.finish()
    assert result is not None
    assert result.hidden_states.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layout", ["final", "aux_final", "dflash_aux"])
@pytest.mark.parametrize("logprob_topk", [0, 2])
def test_hidden_capture_async_rows_survive_graph_output_reuse(layout, logprob_topk):
    """The runner's output event makes copies readable despite graph replay reuse."""
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    main_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    source = torch.arange(12, dtype=torch.bfloat16, device="cuda").reshape(6, 2)
    main_stream.wait_stream(torch.cuda.current_stream())
    aux_ids = () if layout == "final" else (3,)
    plans = {
        req: HiddenStateCapturePlan.from_window(
            req,
            1,
            1,
            3,
            req,
            aux_layer_ids=aux_ids,
            hidden_layout=layout,
            logprob_topk=logprob_topk,
        )
        for req in ("a", "c")
    }
    with torch.cuda.stream(main_stream):
        hidden = torch.empty_like(source)
        aux = torch.empty_like(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=main_stream):
            hidden.copy_(source)
            aux.copy_(source + 100)
        snapshots = []
        for _ in range(2):
            graph.replay()
            chunks = capture_scheduled_hidden_states(
                plans,
                ["a", "b", "c"],
                {"a": 2, "b": 2, "c": 2},
                {"a": 1, "b": 1, "c": 1},
                hidden,
                [aux] if aux_ids else None,
                non_blocking=True,
                logits=hidden.float(),
                logits_indices=torch.arange(6, device="cuda"),
            )
            counts = torch.ones(2, dtype=torch.int32, device="cuda")
            sampled = SamplerOutput(
                torch.full((2, 1), 7, dtype=torch.int64, device="cuda"),
                None,
                None,
                counts,
                torch.zeros_like(counts),
            )
            snapshots.append(
                AsyncOutput(
                    ModelRunnerOutput(
                        req_ids=["a", "c"],
                        req_id_to_index={"a": 0, "c": 1},
                        hidden_state_capture=chunks,
                    ),
                    sampled,
                    counts,
                    main_stream,
                    copy_stream,
                    False,
                    None,
                )
            )
            source.add_(20)
    for step, pending in enumerate(snapshots):
        output = pending.get_output()
        assert output.sampled_token_ids == [[7], [7]]
        chunks = output.hidden_state_capture
        assert set(chunks) == {"a", "c"}
        for req, offset in (("a", 0), ("c", 4)):
            expected = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
            expected = expected[offset : offset + 2] + step * 20
            raw_logits = expected.float()
            if layout == "aux_final":
                expected = torch.cat((expected + 100, expected), dim=-1)
            elif layout == "dflash_aux":
                expected = expected + 100
            chunk = chunks[req]
            assert chunk.hidden_states.is_pinned()
            assert chunk.copied_bytes == (
                expected.numel() * expected.element_size() + 2 * logprob_topk * 12
            )
            np.testing.assert_array_equal(chunk.positions, [1, 2])
            torch.testing.assert_close(chunk.hidden_states, expected)
            if logprob_topk:
                values, ids = raw_logits.log_softmax(-1).topk(logprob_topk, dim=-1)
                assert chunk.logprobs.is_pinned()
                assert chunk.logprob_token_ids.is_pinned()
                torch.testing.assert_close(chunk.logprobs, values)
                assert torch.equal(chunk.logprob_token_ids, ids)


def test_hidden_capture_reports_sparse_positions_without_filling_gaps():
    plan = HiddenStateCapturePlan.from_window(
        "req", 1, 10, 13, "collection", min_rows=2
    )
    hidden = torch.tensor([[1.0], [2.0], [3.0]])
    chunk = capture_scheduled_hidden_states(
        {"req": plan}, ["req"], {"req": 3}, {"req": 10}, hidden
    )["req"]
    buffer = HiddenStateCaptureBuffer(plan)
    buffer.add(chunk.accepted(10, 11))
    buffer.add(chunk.accepted(12, 13))
    result = buffer.finish()
    assert result is not None
    np.testing.assert_array_equal(result.hidden_positions, [10, 12])
    torch.testing.assert_close(result.hidden_states, hidden[[0, 2]])


def test_hidden_capture_aux_capability_is_request_local():
    plan = HiddenStateCapturePlan.from_window(
        "a", 1, 0, 1, "collection", aux_layer_ids=(3,)
    )
    plans = {"a": plan, "b": plan}
    errors = drop_incompatible_aux_plans(plans, ["a"], (), None)
    assert errors == {"a": "aux_layers_unavailable"}
    assert plans == {"b": plan}


def test_hidden_capture_fails_closed_for_unmapped_runners():
    config = SimpleNamespace(
        device_config=SimpleNamespace(device_type="cuda"),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        model_config=SimpleNamespace(is_encoder_decoder=False, is_diffusion=False),
        speculative_config=None,
    )
    assert hidden_state_capture_capability(config) is None
    config.model_config.is_diffusion = True
    assert "diffusion" in hidden_state_capture_capability(config)
    config.model_config.is_diffusion = False
    config.scheduler_config.async_scheduling = True
    assert "async scheduling" in hidden_state_capture_capability(config)
    config.scheduler_config.async_scheduling = False
    config.device_config.device_type = "npu"
    assert "CUDA" in hidden_state_capture_capability(config)


@pytest.mark.parametrize(
    ("device_type", "is_diffusion", "reason"),
    [("npu", False, "only the CUDA GPU"), ("cuda", True, "diffusion")],
)
def test_hidden_capture_rejects_unsupported_config_before_admission(
    device_type, is_diffusion, reason
):
    from vllm.sampling_params import SamplingParams

    config = SimpleNamespace(
        device_config=SimpleNamespace(device_type=device_type),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        model_config=SimpleNamespace(
            is_encoder_decoder=False, is_diffusion=is_diffusion
        ),
        speculative_config=None,
    )
    request = SimpleNamespace(
        request_id="req",
        prompt_token_ids=[1, 2],
        prompt_embeds=None,
        params=SamplingParams(),
    )
    plan = HiddenStateCapturePlan.from_window("req", 2, 0, 1, "collection")
    with pytest.raises(VLLMValidationError, match=reason):
        validate_hidden_state_capture(plan, request, config)


@pytest.mark.parametrize("stop", ["stop", ["stop"]])
def test_hidden_capture_rejects_frontend_stop_strings(stop):
    """Frontend termination cannot return the scheduler's buffered partial window."""
    from vllm.sampling_params import SamplingParams

    request = SimpleNamespace(
        request_id="req",
        prompt_token_ids=[1, 2],
        prompt_embeds=None,
        params=SamplingParams(),
    )
    plan = HiddenStateCapturePlan.from_window("req", 2, 0, 4, "collection", min_rows=1)
    request.params = SamplingParams(stop=stop)
    with pytest.raises(VLLMValidationError, match="stop strings"):
        validate_hidden_state_capture(plan, request, None)


def test_hidden_capture_counts_replay_copies_without_new_tokens():
    """Actual D2H bytes include replayed rows even when no token is newly accepted."""
    from vllm.v1.hidden_state_capture import HiddenStateCaptureChunk

    plan = HiddenStateCapturePlan.from_window("req", 1, 0, 4, "audit")
    buffer = HiddenStateCaptureBuffer(plan)
    first = HiddenStateCaptureChunk(np.array([0, 1]), torch.zeros(2, 2), 16, 1.0)
    replay = HiddenStateCaptureChunk(np.array([0, 1]), torch.zeros(2, 2), 16, 1.0)
    final = HiddenStateCaptureChunk(np.array([2, 3]), torch.zeros(2, 2), 16, 1.0)
    assert buffer.update(first, 0, 2, stopped=False) == (None, None)
    assert buffer.update(replay, 2, 2, stopped=False) == (None, None)
    result, reason = buffer.update(final, 2, 4, stopped=False)
    assert reason is None
    assert result is not None
    np.testing.assert_array_equal(result.hidden_positions, [0, 1, 2, 3])
    assert result.copied_bytes == 48
    assert result.copy_ms == 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_window_logprobs_match_public_generation(vllm_runner):
    """Window supervision agrees with standard raw logprobs without changing tokens."""
    import os

    from vllm import SamplingParams

    model = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")
    with vllm_runner(
        model,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        logprobs_mode="raw_logprobs",
        gpu_memory_utilization=0.5,
    ) as runner:
        llm = runner.get_llm()
        params = SamplingParams(
            temperature=0, max_tokens=8, ignore_eos=True, logprobs=4
        )
        inputs = [{"prompt_token_ids": [100, 200, 300, 400]}]
        reference = llm.generate(inputs, params, use_tqdm=False)[0]
        plan = HiddenStateCapturePlan.from_window(
            "req", 4, 2, 6, "regression", min_rows=4, logprob_topk=4
        )
        actual = llm.generate(
            inputs, params, hidden_state_capture=plan, use_tqdm=False
        )[0]
        assert actual.outputs[0].token_ids == reference.outputs[0].token_ids
        result = actual.hidden_state_capture
        assert result is not None, actual.hidden_capture_skip_reason
        np.testing.assert_array_equal(result.hidden_positions, [5, 6, 7, 8])
        assert result.logprobs.shape == result.logprob_token_ids.shape == (4, 4)
        for row, values, ids in zip(
            actual.outputs[0].logprobs[2:6], result.logprobs, result.logprob_token_ids
        ):
            expected = sorted(row, key=lambda token: row[token].logprob, reverse=True)[
                :4
            ]
            assert set(ids.tolist()) == set(expected)
            torch.testing.assert_close(
                values,
                torch.tensor([row[token].logprob for token in ids.tolist()]),
                atol=2e-5,
                rtol=1e-5,
            )

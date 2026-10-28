# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-local, position-based capture of model output hidden states."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine import EngineCoreRequest


@dataclass(frozen=True)
class HiddenStateCapturePlan:
    request_id: str
    prompt_len: int
    window_start_abs: int
    window_end_abs: int
    collection_id: str
    min_rows: int = 1
    aux_layer_ids: tuple[int, ...] = ()
    hidden_layout: str = "final"
    logprob_topk: int = 0

    def __post_init__(self) -> None:
        if (
            self.prompt_len < 1
            or self.window_start_abs < self.prompt_len - 1
            or self.window_start_abs >= self.window_end_abs
            or self.min_rows < 1
            or self.min_rows > self.window_end_abs - self.window_start_abs
        ):
            raise ValueError("Invalid hidden-state capture plan")
        if self.hidden_layout not in ("final", "aux_final", "dflash_aux"):
            raise ValueError("Invalid hidden-state layout")
        if bool(self.aux_layer_ids) != (self.hidden_layout != "final"):
            raise ValueError("Hidden-state layout does not match auxiliary layers")
        if len(set(self.aux_layer_ids)) != len(self.aux_layer_ids):
            raise ValueError("Duplicate auxiliary hidden-state layers")
        if not isinstance(self.logprob_topk, int) or self.logprob_topk < 0:
            raise ValueError("logprob_topk must be a nonnegative integer")

    @classmethod
    def from_window(
        cls,
        request_id: str,
        prompt_len: int,
        start: int,
        end: int,
        collection_id: str,
        *,
        coordinate: Literal["response", "absolute"] = "response",
        min_rows: int = 1,
        aux_layer_ids: tuple[int, ...] = (),
        hidden_layout: str | None = None,
        logprob_topk: int = 0,
    ) -> HiddenStateCapturePlan:
        if coordinate == "response":
            offset = prompt_len - 1
        elif coordinate == "absolute":
            offset = 0
        else:
            raise ValueError(f"Unknown hidden-state coordinate: {coordinate}")
        return cls(
            request_id=request_id,
            prompt_len=prompt_len,
            window_start_abs=start + offset,
            window_end_abs=end + offset,
            collection_id=collection_id,
            min_rows=min_rows,
            aux_layer_ids=aux_layer_ids,
            hidden_layout=hidden_layout or ("aux_final" if aux_layer_ids else "final"),
            logprob_topk=logprob_topk,
        )


def hidden_state_capture_capability(vllm_config: VllmConfig) -> str | None:
    """Return a reason when row-to-position mapping is not supported."""
    if vllm_config.device_config.device_type != "cuda":
        return "only the CUDA GPU model runners have a capture adapter"
    if vllm_config.scheduler_config.async_scheduling:
        return "async scheduling can reorder in-flight hidden rows"
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        return "prefill context parallelism changes hidden-row ownership"
    if vllm_config.parallel_config.decode_context_parallel_size > 1:
        return "decode context parallelism changes hidden-row ownership"
    if vllm_config.model_config.is_encoder_decoder:
        return "encoder-decoder hidden positions need a separate mapping"
    if vllm_config.model_config.is_diffusion:
        return "diffusion hidden positions need a separate mapping"
    if (
        vllm_config.speculative_config is not None
        and vllm_config.speculative_config.enable_adaptive_verification
    ):
        return "adaptive verification changes GPU-side token row allocation"
    return None


def validate_hidden_state_capture(
    plan: HiddenStateCapturePlan,
    request: EngineCoreRequest,
    vllm_config: VllmConfig,
) -> None:
    """Validate a capture plan before assigning the internal request ID."""
    from vllm.sampling_params import SamplingParams
    from vllm.utils import length_from_prompt_token_ids_or_embeds

    params = request.params

    if not isinstance(params, SamplingParams) or params.n != 1:
        raise VLLMValidationError("Hidden-state capture requires a single completion")
    if params.stop:
        raise VLLMValidationError("Hidden-state capture does not support stop strings")
    if plan.logprob_topk:
        if plan.logprob_topk > vllm_config.model_config.get_vocab_size():
            raise VLLMValidationError("logprob_topk exceeds the model vocabulary")
        if vllm_config.parallel_config.enable_batch_sharded_sampling:
            raise VLLMValidationError(
                "Window logprobs do not support batch-sharded sampling"
            )
    prompt_len = length_from_prompt_token_ids_or_embeds(
        request.prompt_token_ids, request.prompt_embeds
    )
    if plan.prompt_len != prompt_len or plan.request_id != request.request_id:
        raise VLLMValidationError(
            "Hidden-state capture plan does not match the request"
        )
    if reason := hidden_state_capture_capability(vllm_config):
        raise VLLMValidationError(f"Hidden-state capture unavailable: {reason}")


@dataclass
class HiddenStateCaptureChunk:
    positions: np.ndarray
    hidden_states: torch.Tensor
    copied_bytes: int = 0
    copy_ms: float = 0.0
    logprobs: torch.Tensor | None = None
    logprob_token_ids: torch.Tensor | None = None

    def accepted(self, start: int, end: int) -> HiddenStateCaptureChunk:
        keep = (self.positions >= start) & (self.positions < end)
        return HiddenStateCaptureChunk(
            self.positions[keep],
            self.hidden_states[torch.from_numpy(keep)],
            self.copied_bytes,
            self.copy_ms,
            self.logprobs[torch.from_numpy(keep)]
            if self.logprobs is not None
            else None,
            self.logprob_token_ids[torch.from_numpy(keep)]
            if self.logprob_token_ids is not None
            else None,
        )


def drop_incompatible_aux_plans(
    plans: dict[str, HiddenStateCapturePlan],
    req_ids: list[str],
    available_layer_ids: tuple[int, ...],
    aux_hidden_states: list[torch.Tensor] | None,
) -> dict[str, str]:
    """Disable captures whose requested auxiliary layout cannot be produced."""
    errors: dict[str, str] = {}
    for req_id in req_ids:
        plan = plans.get(req_id)
        if plan is None or not plan.aux_layer_ids:
            continue
        if (
            aux_hidden_states is None
            or plan.aux_layer_ids != available_layer_ids
            or len(aux_hidden_states) != len(available_layer_ids)
        ):
            plans.pop(req_id)
            errors[req_id] = "aux_layers_unavailable"
    return errors


def capture_scheduled_hidden_states(
    plans: dict[str, HiddenStateCapturePlan],
    req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    num_computed_tokens: dict[str, int],
    hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor] | None = None,
    query_start_loc: np.ndarray | None = None,
    *,
    non_blocking: bool = False,
    logits: torch.Tensor | None = None,
    logits_indices: torch.Tensor | None = None,
) -> dict[str, HiddenStateCaptureChunk]:
    """Copy scheduled window rows, optionally deferring completion to the runner.

    With non_blocking=True, the caller must synchronize the current CUDA stream
    before reading or serializing returned CPU tensors. copy_ms then measures
    host submission time rather than completion time.
    """
    chunks: dict[str, HiddenStateCaptureChunk] = {}
    row_offset = 0
    for index, req_id in enumerate(req_ids):
        num_rows = num_scheduled_tokens[req_id]
        if query_start_loc is not None:
            row_offset = int(query_start_loc[index])
            if int(query_start_loc[index + 1]) - row_offset != num_rows:
                raise ValueError("Scheduled hidden-state row boundaries do not match")
        plan = plans.get(req_id)
        if plan is not None:
            start = num_computed_tokens[req_id]
            first = max(start, plan.window_start_abs)
            end = min(start + num_rows, plan.window_end_abs)
            if first < end:
                first_row = row_offset + first - start
                end_row = row_offset + end - start
                if plan.aux_layer_ids:
                    if aux_hidden_states is None or len(aux_hidden_states) != len(
                        plan.aux_layer_ids
                    ):
                        raise ValueError(
                            "Requested auxiliary hidden layers are unavailable"
                        )
                    parts = [state[first_row:end_row] for state in aux_hidden_states]
                    if plan.hidden_layout != "dflash_aux":
                        parts.append(hidden_states[first_row:end_row])
                    states = torch.cat(parts, dim=-1)
                else:
                    states = hidden_states[first_row:end_row]
                values = token_ids = None
                if plan.logprob_topk:
                    if logits is None or logits_indices is None:
                        raise ValueError(
                            "Window logprobs require raw logits and indices"
                        )
                    rows = torch.arange(first_row, end_row, device=logits.device)
                    indices = torch.searchsorted(logits_indices.contiguous(), rows)
                    indices = indices.clamp(max=logits_indices.numel() - 1)
                    torch._assert_async(
                        (logits_indices[indices] == rows).all(),
                        "Captured hidden rows have no corresponding logits",
                    )
                    scores = logits[indices].float()
                    values, token_ids = scores.topk(plan.logprob_topk, dim=-1)
                    values = values - scores.logsumexp(dim=-1, keepdim=True)
                copy_started = perf_counter()
                tensors = [states]
                if values is not None:
                    tensors.extend([values, token_ids])
                cpu_tensors = []
                for tensor in tensors:
                    if non_blocking and tensor.device.type == "cuda":
                        cpu_tensor = torch.empty_like(
                            tensor, device="cpu", pin_memory=True
                        )
                        cpu_tensor.copy_(tensor.detach(), non_blocking=True)
                    else:
                        cpu_tensor = tensor.detach().to("cpu")
                    cpu_tensors.append(cpu_tensor)
                chunks[req_id] = HiddenStateCaptureChunk(
                    positions=np.arange(first, end, dtype=np.int64),
                    hidden_states=cpu_tensors[0],
                    logprobs=cpu_tensors[1] if values is not None else None,
                    logprob_token_ids=cpu_tensors[2] if values is not None else None,
                    copied_bytes=(
                        sum(t.numel() * t.element_size() for t in tensors)
                        if states.device.type != "cpu"
                        else 0
                    ),
                    copy_ms=(
                        (perf_counter() - copy_started) * 1000
                        if states.device.type != "cpu"
                        else 0.0
                    ),
                )
        row_offset += num_rows
    return chunks


@dataclass
class HiddenStateCaptureResult:
    hidden_positions: np.ndarray
    hidden_states: torch.Tensor
    hidden_position_start: int
    hidden_position_end: int
    hidden_window_start: int
    hidden_window_end: int
    hidden_layout: str
    layer_ids: tuple[int, ...]
    includes_final_layer: bool
    request_id: str
    collection_id: str
    copied_bytes: int = 0
    copy_ms: float = 0.0
    logprobs: torch.Tensor | None = None
    logprob_token_ids: torch.Tensor | None = None


@dataclass
class HiddenStateCaptureBuffer:
    plan: HiddenStateCapturePlan
    chunks: list[HiddenStateCaptureChunk] = field(default_factory=list)
    copied_bytes: int = 0
    copy_ms: float = 0.0

    def add(self, chunk: HiddenStateCaptureChunk) -> None:
        self.copied_bytes += chunk.copied_bytes
        self.copy_ms += chunk.copy_ms
        if chunk.positions.size:
            self.chunks.append(chunk)

    def finish(self) -> HiddenStateCaptureResult | None:
        if not self.chunks:
            return None
        positions = np.concatenate([chunk.positions for chunk in self.chunks])
        states = torch.cat([chunk.hidden_states for chunk in self.chunks])
        positions, unique_indices = np.unique(positions, return_index=True)
        states = states[torch.from_numpy(unique_indices)]
        logprobs = token_ids = None
        if self.plan.logprob_topk:
            if any(
                chunk.logprobs is None or chunk.logprob_token_ids is None
                for chunk in self.chunks
            ):
                raise ValueError("Captured window has missing logprob rows")
            logprobs = torch.cat([c.logprobs for c in self.chunks])[
                torch.from_numpy(unique_indices)
            ]
            token_ids = torch.cat([c.logprob_token_ids for c in self.chunks])[
                torch.from_numpy(unique_indices)
            ]
        if len(positions) < self.plan.min_rows:
            return None
        return HiddenStateCaptureResult(
            hidden_positions=positions,
            hidden_states=states,
            hidden_position_start=int(positions.min()),
            hidden_position_end=int(positions.max()) + 1,
            hidden_window_start=self.plan.window_start_abs,
            hidden_window_end=self.plan.window_end_abs,
            hidden_layout=self.plan.hidden_layout,
            layer_ids=self.plan.aux_layer_ids,
            includes_final_layer=self.plan.hidden_layout != "dflash_aux",
            request_id=self.plan.request_id,
            collection_id=self.plan.collection_id,
            copied_bytes=self.copied_bytes,
            copy_ms=self.copy_ms,
            logprobs=logprobs,
            logprob_token_ids=token_ids,
        )

    def update(
        self,
        chunk: HiddenStateCaptureChunk | None,
        accepted_start: int,
        accepted_end: int,
        *,
        stopped: bool,
        error: str | None = None,
    ) -> tuple[HiddenStateCaptureResult | None, str | None]:
        """Settle returned rows using accepted output progress."""
        if error is not None:
            return None, error
        if chunk is not None:
            self.add(chunk.accepted(accepted_start, accepted_end))
        if not stopped and accepted_end < self.plan.window_end_abs:
            return None, None
        result = self.finish()
        if result is not None:
            return result, None
        reason = (
            "ended_before_window"
            if stopped and accepted_end <= self.plan.window_start_abs
            else "insufficient_rows"
        )
        return None, reason

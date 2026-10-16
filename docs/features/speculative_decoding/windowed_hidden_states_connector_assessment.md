# Connector window capture：实现评估与交付记录

## 结论与范围

已在 `vllm-ExampleHiddenStatesConnector` 的独立 main 工作区实现实验性代码；
没有修改 `verl-SpeCo` 或原 `vllm` 工作区。没有提交、推送或新建 PR。
开发基线为 `823eaf667d33125259d2d280b0ed00747fa92440`。

**目前不能认定为 #57736 的等价替代，也不能宣称 GPU 容量或吞吐改善。**
方案复用 Connector 生命周期，能在写入阶段过滤窗口外数据，并限制 window
导出的 pinned 内存；但 full-sequence block 分配、MRV1 依赖、真实 drafter
互斥和请求结束才发布，都是实际架构限制。GPU 运行与性能尚待验证。

## A. 实现文件和执行流程

| 文件 | 目的 |
| --- | --- |
| `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` | 请求窗口、block metadata、选择性导出、预算、取消清理和原子发布 |
| 同目录 `hidden_state_window.py` | 纯 CPU 区间、物理 block offset、layout 契约 |
| `vllm/model_executor/layers/hidden_state_cache.py` | 负 slot 和窗口外行不写入的 Triton store |
| `vllm/model_executor/models/extract_hidden_states.py` | 接入 masked store，按 layout 定义 cache 层数 |
| `vllm/v1/spec_decode/extract_hidden_states.py` | layout buffer，逐层复制以去掉额外全 batch stack |
| `vllm/v1/engine/input_processor.py` | frontend 校验窗口及不支持的 request 组合 |
| `vllm/v1/worker/gpu_model_runner.py` | 传递 normalized final；extractor 在上下文末端仍缓存已有行 |
| `vllm/config/speculative.py`、`vllm/config/kv_transfer.py` | layout 与 masked store 的编译 hash 隔离，默认 hash 保持兼容 |
| `tests/v1/kv_connector/unit/test_hidden_state_window.py` | 无 runtime 依赖的窗口契约测试 |
| `tests/v1/kv_connector/unit/test_hidden_states_connector.py` | completion、取消、copy fence、写入失败、预算和旧逻辑回归 |
| `tests/v1/kv_connector/extract_hidden_states_integration/test_window.py` | 本地 tiny Llama 数值、graph、BF16/FP16、分块与 cache reuse |
| `.buildkite/test_areas/misc.yaml` | 现有 extraction GPU job 的新增 source dependencies |
| `benchmarks/benchmark_connector_hidden_state_window.py` | baseline/full/window/PR 控制、worker 内存与运行/消费计时 |
| 本评估及 `windowed_hidden_states_connector.md`，原 extraction 文档链接 | 使用、限制与验证记录 |

1. frontend validation 校验后，`SamplingParams.extra_args.kv_transfer_params.hidden_state_window` 传递
   `start/end/min_rows`，无需新的 API/EngineCore request 字段。
2. InputProcessor 在送入 engine 前校验，既有 Connector 匹配查询保留防御检查；window 模式显式开启。
3. `build_connector_meta` 从 new/cached request 的 block IDs 计算窗口 offsets，
   增量更新 GPU 常驻 range table。恢复/抢占后的新 block table 重新计算。
4. CacheOnlyAttentionLayer 的 masked store 在 GPU 直接忽略窗口外行；
   图 replay 使用固定地址的 range table。不会把整个序列 D2H 后在 CPU 切片。
5. request_finished 根据 accepted output 长度裁剪窗口，处理 EOS/stop token/
   max_tokens；不足 min_rows 无 payload。response r 对应 P-1+r。
6. 只 gather 有效窗口，异步 D2H；大窗口使用有界 staging。保持 blocks 到
   copy event 完成；序列化可以稍后完成，消费者等待 flock。
7. 原子文件 rename 后释放文件锁；异常删除临时文件并写入 error sidecar，取消清理未使用的锁。提交失败在 copy stream 同步后结束发送，避免 blocks 泄漏。

核心新增 runner 逻辑只有两个接入点；InputProcessor 增加 frontend validation，避免无效窗口使 scheduler 抛异常。没有改 Scheduler、EngineCore、
OutputProcessor 或 KV allocator。两个 config 改动是为了避免编译缓存混用。
完整窗口会一直留在 hidden-state cache，直到请求结束，这不同于 RFC 的
window-end 立即发布和释放；后续 token 不再写入窗口外 hidden slots。

## B. 与 #57736 比较

比较依据：2026-10-16 读取的 [PR #57736](https://github.com/vllm-project/vllm/pull/57736)，
head `df01e5f1f19003fb803162364eb8438e39bf6bc1`，23 个 changed files；
实际读取了 capture module、MRV2 runner 和 scheduler 的远端 patch。
本地另一个 vllm checkout head `88cf9d3fa0` 较旧，不等同远端当前 head。
[RFC #56916](https://github.com/vllm-project/vllm/issues/56916) 明确 P-1+r
坐标和 partial-window 语义。

| 项目 | 本 Connector 原型 | #57736 当前 diff |
| --- | --- | --- |
| 文件范围 | 17 个文件（9 runtime、3 test、1 CI、1 benchmark、3 docs） | 23 个文件 |
| 核心侵入性 | MRV1 runner 两处、frontend validation、两个配置 hash；Connector 自管状态 | API/request/output、scheduler、engine、MRV1/MRV2 传递与完成管理 |
| GPU hidden cache | 按全序列 blocks 分配；没有 window-only 容量节省 | 直接从 target activations 选取，无专用全序列 hidden cache |
| D2H | 只复制 accepted window；rank0 导出 | 复制 scheduled window，scheduler 过滤 accepted；TP copy 策略依具体路径 |
| CPU/pinned | 小窗口异步；大窗口预算 staging + pageable final；host 同步增加延迟 | MRV2 fresh pinned + AsyncOutput fence，buffer 合并有额外成本 |
| 生命周期 | 请求结束导出、copy 完释放 blocks、flock 后可读 | window 完成即可生成 capture result 并撤销 plan |
| 真实 speculation | extractor 占唯一 drafter slot，明确拒绝 EAGLE3/DFlash/DSpark | 与 target verification 路径协作，scheduler 用 accepted output 定位 |
| Layer layout | engine-wide aux/final/组合；保留三维 file tensor | plan 含 layout/aux IDs；实际可用层须与 model outputs 一致 |
| 大窗口 | D2H/CPU 线性，pinned 有界，GPU cache 仍按序列增长 | D2H/CPU 线性；无 Connector full cache，但 pinned/window buffers 需评估 |
| Mooncake/File | 保留 path/flock 与现有 Connector 输出位置；可复用 CPU tensor 发布点 | 当前直接输出 tensor/接口序列化，需要另加 feature store adapter |
| verl 替换成本 | 非真实 drafter rollout 可少一次 replay；consumer 需理解 sparse positions | consumer 需接 capture result 与 store；更符合真实 speculative rollout |

PR 描述已报告其自身 GPU 性能数据；这些不是本次运行结果，也不能拿不同
模型/硬件的 PR 数字证明 Connector 方案好或坏。

### 分配限制与可量化资源模型

`HiddenStateCacheSpec` 注册到 `FullAttentionManager`；block 数由 sequence progress
决定，slot mapping 仍是 absolute token -> logical block -> physical block。
过滤 store 不影响 allocator。page 对齐和 hybrid group padding 还可能增加浪费。
range table 额外占 `num_physical_blocks * 2 * 8` 字节。

完整 sparse allocation 需要新的 cache spec/manager 定义 request 的 window
逻辑块，独立预算和 admission accounting、window-relative slot mapping、
抢占恢复/prefix-sharing 约束、worker block table 映射，以及独立于实际 drafter
的 extraction hook。仅改 Connector 或把 slot 设为 -1 不能做到这些。
如果主要目标是 GPU 容量和真实 drafter 共存，直接 target-row capture 后交给
Connector transport 可能比给 FullAttentionManager 增加稀疏能力更合适。

以下是公式估算，**不是测量**。H=4096，L=4，BF16/FP16，单行 32768 bytes，
export budget=64 MiB：

| Window rows | D2H hidden payload | 单窗口 live pinned 上界 | 大窗口 pageable final |
| --- | --- | --- | --- |
| 32 | 1 MiB | 1 MiB | 0 |
| 128 | 4 MiB | 4 MiB | 0 |
| 512 | 16 MiB | 16 MiB | 0 |
| 2048 | 64 MiB | 64 MiB | 0 |
| 8192 | 256 MiB | 64 MiB | 256 MiB |

多窗口 pinned 总预算按 worker 共享，writer backlog 通过等待施加背压；
legacy full requests 不受此预算限制。CPU final 大窗口加 staging 峰值约 W+budget，
不含 safetensors 库自身开销和其它 vLLM 内存。PyTorch pinned allocator reserved
不等于 live allocation，可能保留已释放 buffers。GPU cache 实际字节需读取
worker cache.numel()*element_size；proposer 仍有 batch buffer。

## C. 验证与复现

本机 Windows，借用原 vllm checkout 的 uv `.venv/Scripts/python.exe`：
PyTorch 2.14.1+cpu，CUDA=False，Triton 不可用。目标仓库未创建新的 CUDA 安装。

- 纯 CPU window/validation 契约：35 passed（最终完整重跑）。
- 真实 Connector suite：收集失败 `ModuleNotFoundError: fcntl`，没有执行测试体。
- GPU numerical、EOS/stop、graphs、D2H/cache reuse、GPU regression 和 benchmark：待验证。
- 真实 drafter rejection correctness：未实现，使用 capability error 明确拒绝。
- Linux 的 pre-commit 全套与真实模型 eval：待验证；静态检查结果见最终记录。

纯契约测试为何独立加载：它只有标准库依赖，允许 Windows 测区间和 layout；
不会导入整个 vLLM。真实 Connector 和 GPU 测试另外运行，不能用契约通过代替。

Linux GPU 先按 AGENTS.md 用 uv 建 `.venv`，安装 lint/test 依赖和可编辑 vLLM，
并安装 pre-commit hooks。随后：

```bash
VLLM_USE_V2_MODEL_RUNNER=0 uv run --no-sync --python .venv/bin/python \
  -m pytest tests/v1/kv_connector/unit/test_hidden_state_window.py \
  tests/v1/kv_connector/unit/test_hidden_states_connector.py -v
VLLM_USE_V2_MODEL_RUNNER=0 uv run --no-sync --python .venv/bin/python \
  -m pytest tests/v1/kv_connector/extract_hidden_states_integration -v
```

新增 GPU numerical 测试创建本地 tiny Llama checkpoint，不下载模型；HF 用同一
checkpoint 对 accepted sequence teacher-force，比较正确位置的 aux 和 normalized
final 值。混合请求包含 chunked prompt、超长窗口和窗口外结束。另有 GPU kernel
graph replay、负 slot、offset view、非对齐宽度、BF16/FP16 精确保真，128/2048/8192
行异步/分块导出并在 D2H 后覆写 cache 的测试。既有 predictable/full tests 保留。

benchmark 每个 mode 用独立进程/引擎，固定模型、输出长度、dtype、并发和图模式。
先 warmup 再测量，至少重复三轮并轮换模式顺序；不要将一次结果当结论。

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
for round in 1 2 3; do
  for concurrency in 1 8 32; do
    for width in 32 128 512 2048 8192; do
      for mode in baseline full window; do
        uv run --no-sync --python .venv/bin/python \
          benchmarks/benchmark_connector_hidden_state_window.py \
          --model /path/to/model --mode "$mode" --window-size "$width" \
          --concurrency "$concurrency" --requests 64 --output-len 8256 \
          --report "results/$round-$mode-$width-$concurrency.json"
      done
    done
  done
done
```

baseline 不启用 extraction，full 使用 legacy full export，window 使用 masked
store。默认 auxiliary layout 保持原 Example 数据类型。要测未经修改的原 Example，
把 benchmark 脚本带到基线 commit 的独立 checkout，运行 full（不要切换/重置当前
开发目录）；新增的默认 auxiliary 字段会被旧 extraction 忽略。

PR 控制在已有 PR checkout 中用同一 benchmark 脚本的绝对路径运行
`--mode pr57736 --layout final`。当前脚本的 PR control 只测 final layout。
Connector 使用 `--layout final` 可以按相同 normalized-final 数据尺寸配对，但这时
full 是 legacy writer + 新 final extraction 扩展，不能称为原 Example final 功能。
aux-only 控制与 final 不同 layer count 时不得直接比较字节/吞吐，也不能据此
证明数值等价。真实 drafter 场景应使用 PR 自带的专用测试和 benchmark。

JSON 分开记录：generation throughput/TTFT/按事件 token 数均摊的 ITL；文件
等待+读取+清理；worker GPU allocated/reserved peak；hidden-cache actual bytes；
10ms 采样 worker RSS peak；可用时的 PyTorch host allocator stats；window live
pinned accounting 上界、copied bytes、host copy wait、serialization time。
full 的 copied bytes 可从实际返回 hidden tensor 的字节数推导（不是 profiler
测量）；额外 token/position metadata、GPU H2D 和文件字节单独考虑。
host copy wait 不是 GPU copy execution latency。需要精确带宽和 overlap 时另用
Nsight/torch profiler，并把 profiling 与吞吐测量分开。PR 原生 tensor 不写文件，
不能将 file export 与 PR tensor return 的端到端耗时当作相同工作。

## D. verl-SpeCo 集成建议与 Mooncake 生命周期

实际只读参考中找到 vllm_runtime、vllm_feature_client、target_feature_replay 和
integration/transferqueue_bridge；附件指定的 producer/hidden_states_store.py、
mooncake_hidden_states_connector.py 和 integration 同名文件均不存在。
没有据此假定本地已有可直接复用的 Mooncake 类。

- `vllm_runtime.py`：为非真实 drafter 的 feature-serving engine 配置 window flag、
  layout 和导出预算，显式禁用 prefix/async，并选择 MRV1。
- `vllm_feature_client.py`：generate 请求加 `kv_transfer_params.hidden_state_window`；
  读取现有 hidden_states_path，等待 flock。无 path 时处理 skip reason。
  按 hidden_positions/response_positions 对齐，而不是强制全 prompt token_ids 匹配。
- `target_feature_replay.py`：window payload 不再是完整 replay prefix；读取
  target_token_ids 和 sparse position metadata，保留 aux 顺序；final layout 不能
  再做一次 final norm。DFlash 选 auxiliary，EAGLE3/DSpark 根据训练需要选组合。
- File -> Mooncake：复用 `_write_tensors` 的 CPU-ready 发布时机或抽取 store adapter。
  数据必须在 copy event 完成后才 publish handle。GPU block 可以先释放，CPU
  feature backing buffer/segment 的 lifetime 应由 consumer leases/ACK 决定。
- Producer 发布后不能立即 close segment；所有 Consumer ranks 读取/ACK 后由
  owner close。支持多读者引用计数或 task teardown 回收；取消/超时采用有界 lease
  expiry。TransferQueue 现有代码特意延迟到 task teardown 回收，因为 SP/TP 多 rank
  会读取同一 sample；不能改成 first-reader-delete。采集完成与 feature 消费完成
  是两个不同状态，不能以 KV finished_sending 代替 consumer ACK。

本次没有实现 Mooncake feature transport 或 lease 协议，也没有把 Mooncake KV
transfer 当 feature transport。真实 speculative rollout 暂时仍需原 replay 路径。

## Checklist 评估

按 pr-checklist 检查了设计、测试、性能与兼容性。它要求新 GPU feature 优先 MRV2，
并要求 end-to-end 验证；本原型依附现有 MRV1 extractor，因此不是可直接提交的
生产替代。runtime diff 超过 250 行，主要来自 window metadata、masked store 和
导出背压，维护成本不可忽略。大窗口 completion-time synchronize 是明确 tradeoff，
不可声称 steady-state 全异步。需 Linux GPU 回归、模型 eval 和真实并发性能证据
后才能决定是否继续以此架构替换 #57736。没有新建 PR，因此不重复提交现有工作；
AI 协助用于实现和验证准备，人类仍需逐行审阅并执行目标硬件验证。

Windows 复现纯契约命令（`$runtime` 指现有 uv venv Python）：

```powershell
$runtime = 'D:/workspace/windowed-hidden-state-collection/vllm/.venv/Scripts/python.exe'
uv run --no-project --python $runtime python -m pytest --noconftest -o addopts= tests/v1/kv_connector/unit/test_hidden_state_window.py -q
```

`--noconftest` 仅用于不依赖 runtime 的纯契约文件。完整 Connector/GPU suite
应保留正常 fixtures，并在 Linux 环境运行。测试没有通过伪造 fcntl/CUDA 模块
绕过平台限制。

最终静态验证记录（2026-10-16）：

- `ruff check` / `ruff format`：所有 13 个改动 Python 文件通过。
- `compileall -q`：所有改动 Python 文件通过。
- mypy 1.20.2，`--platform linux --python-version 3.10`：8 个 runtime 文件和
  3 个测试文件通过（model module 与 benchmark 按仓库 hook 的规则不在此检查组）。
- pre-commit hooks 已安装。完整首次运行暴露的实际类型/API 问题已修复。
  UTF-8 环境复跑 Ruff、typos、Markdown、SPDX、forbidden imports、accelerator API、
  config validation 等可用 hooks 通过。
- 复跑跳过 `mypy-3.10`（以等价 Linux-target 命令另验）、`check-filenames`、
  `update-dockerfile-graph`、`suggestion`（依赖不可用的 Bash）。
  不能将这次复跑称为全量 pre-commit 通过。
- `git diff --check` 通过。benchmark `--help` 是 CLI 入口检查，不是 GPU 运行验证。

未执行任何模型性能或准确率评估；GPU teacher-forcing suite 和 benchmark
命令是下一轮硬件验证入口。当前没有生产替代已完成或性能胜出的结论。

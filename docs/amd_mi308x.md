# AMD MI308X（gfx942）平台差异手册 —— 插桩/Profiling 实测结论固化

适用卡：**AMD MI308X**（gfx942, 80 CU, HBM 实测流读峰值 **3684 GB/s**——实测值非 datasheet,
多 agent 共用 8 卡机, 跨机 4× RDMA NIC (vsolar_rni, 300 Gb/s/口 raw=37.5 GB/s), 同机 XGMI）。
本文把证据分成三类并逐条标注: **measured** = MI308X 实卡执行输出; **compiled** =
本机 ROCm 工具链/代码构建与离线自测通过, 但没有在 MI308X 上运行该路径; **unverified**
= 尚无本机证据。2026-11 的 same-run `rocprof-sys-run --rocm-events` GEMM 路线已在
MI308X 上 **measured**: kernel dispatch slice 与 raw ROCm PM counter 位于同一个 Perfetto
proto。该 measured 结论只覆盖 MI308X GEMM 路线; 其他 operator、语言路线、collector
fallback 或离线 CLI 自测都必须单独取证, 不能继承这个状态。与 NV 侧 (nsys/gh100) 的
差异是本质性的, 不要把 NV 侧流程平移过来。

## 1. 设备时间戳：s_memrealtime, 不是 clock64

- **用** `__builtin_amdgcn_s_memrealtime()`（gfx942 全 GPU 同基线, ns 级, 跨 CU 直接可比）。
- **禁** `clock64()`：AMD 上也是 **per-CU 周期计数**, 跨 CU 相位差实测上百 ms
  （mega EP16 16-rank 相位对齐时实锤）——与 NV 侧 per-SM 的问题同构但量级更坏, 没有任何
  校准能救, 跨 CU/跨 rank 对齐一律走 s_memrealtime。

### 1.1 计时模式与观测窗口

- `FI_TIMER=0`：每条记录读取 `s_memrealtime`。
- `FI_TIMER=1`：语义区间内的记录读取 `s_memtime`；每个物理 wave 的 OPEN/CLOSE 各写一对相邻的 `s_memrealtime` + `s_memtime` 锚点，再按锚点把本地周期换算到全卡时基。两条锚点记录必须计入 slot/cap 预算。
- `observation window` 包含 OPEN/CLOSE 和计时锚点；`semantic envelope` 只覆盖首个到最后一个语义事件。两者分别写入 trace/report，不能互相替代。
- `clock64()` 是 CU 本地周期计数器，不能直接作为跨 CU 或跨 rank 的时间轴。

## 2. Collector 选择：same-run `rocprof-sys-run --rocm-events` 为主

MI308X 的主路线是在启动 workload 的同一进程里执行：

```bash
rocprof-sys-run --rocm-events ... -- <app>
```

raw ROCm PM counter 会作为 `GPU <COUNTER> [0]` counter track 与 kernel dispatch slice
一起写入同一个 Perfetto proto。用 `weaveprof/pipeline/capture/build_rocprof_sys_sw_hw.py` 按 dispatch
start timestamp 匹配 counter sample: 每个 dispatch 只使用同一 run、同一 dispatch 的
timestamp 邻近样本 (`same_run=true`, `same_dispatch=true`), 容差必须显式写入对齐元数据。
不得把另一个 run 的 counter、只用 wall-clock 邻近猜出来的样本, 或别的 kernel 的 aggregate
拼到当前 dispatch 上。

rocprof-sys 产物路径带 timestamp/PID 后缀; manifest 记录发现的真实路径, 不伪造固定文件名。
若 `rocprof-sys` 主路线不可用或不兼容, 才允许**明确降级**到 rocprofv3, 并把
`collector.fallback.from/to/trigger/reason`、候选探测与版本探测写入 manifest。
rocprofv3 `MfmaUtil` 只是可选的独立 calibration/fallback, 不是主来源; 使用时必须保留
`rocprofv3:MfmaUtil` provenance, 直接按 percent 读取, 不再乘 100, 且不得与 primary
same-run series 混成同一来源。

manifest 的 `collector` 是实际执行事实, 至少记录: 实际 family/path/version、
`version_probe.argv/stdout/stderr/exit_code`、`capability_probe`、完整 `argv`/`argv_shell`、
退出码、fallback 原因、以及 `capabilities.sampling`。其中
`sampling.periodic_pm_sampling=false`、`equivalent_to_nsys_200khz=false` 和
`counter_granularity=per-dispatch` 必须如实保留。可选 amd-smi polling 仍只是约 2–5 Hz
的粗粒度调试上下文, 不得写成主 raw PM 路线, 也不得展示成 compute/memory utilization。

### 2.1 `FI_LEVEL`、`FI_SAMPLE` 与 fast/full 证据

`FI_LEVEL` 只控制 recorder 记录层级，`FI_SAMPLE` 独立控制 K block 事件密度；两者不是 CLI `fast`/`deliver` 命令的别名，也不能互相替代：

- `FI_LEVEL=0`：off，不记录。
- `FI_LEVEL=1`：entry-only，只记录 OPEN/CLOSE（`FI_TIMER=1` 时再加计时锚点），不记录 K block 语义事件。它不是 full audit，不能用于 full-audit 证据。
- `FI_LEVEL=2`：full-events，记录完整事件；K block 密度由 `FI_SAMPLE` 决定。
- `FI_SAMPLE=0`：full audit，覆盖所有 K block。
- `FI_SAMPLE=1`：fast sampled feedback，每个物理 wave 只采一个代表性 K block，并保留完整 wave/GEMM envelope。

fast/low-overhead 必须写成 `FI_LEVEL=2 FI_SAMPLE=1`。本机 2048³ FP16 Triton 参考为 14 records/wave（含两条 timer 锚点），稳态比 0.975x，通过 10% gate；它仍是开发反馈数据，`delivery_evidence=false`。

full audit 必须写成 `FI_LEVEL=2 FI_SAMPLE=0`。历史 full-K 参考为 390 records/wave，稳态比 1.703x；该记录布局早于新增的两条 GEMM envelope 边界。修正后的 full-audit 布局在 64 个 K block、带两条 timer 锚点时是 392 records/wave。10% gate 只使用稳态 kernel 时间，因此 full audit 的 1.703x 明确未通过；正确性、overflow 和快采样结果通过都不能抵消这个性能失败。

正确性证据与采样密度无关：两种模式都必须用独立参考全量比较每个输出的每个元素，`FI_SAMPLE` 只减少事件记录，不允许减少或抽样输出检查。两种模式还必须分别保留 record 数、稳态比和 SASS/ISA 变化；不得用 `FI_LEVEL=1`、快采样结果或 `run.py fast` 认证 full audit。

### 2.2 Integration anchor 与绝对时钟

同 run 内，raw ROCm PM counter 与 kernel dispatch slice 按同一 dispatch 的 start timestamp 和显式容差配对。合并到 Perfetto 时，硬件 aggregate 的显示起点可以放在软件 observation start，并记录 `source_clock_offset_ns`。该放置必须标记为：

- `start_placement_is_integration_anchor=true`
- `start_placement_measured=false`
- `absolute_clock_equality_claimed=false`

该 offset 只用于把同一 dispatch 的软硬件证据放进同一显示窗口，不证明两个时钟绝对相等。对齐质量使用 `duration_ratio_hw_over_sw` 与声明容差判断；不得由显示 offset 反推绝对时钟校准。

## 3. ROCm PM counters：persistent kernel 禁用

grid >= 80 CU 常驻 + 自旋等待的 persistent kernel（mega 类）**禁止挂 rocprof PMC**——
会挂死或采到无效值 (**measured**, 本机有 hang + GPU reset 风暴前科)。该保护在 collector
选择之后、任何 GPU 动作之前执行, 对 same-run raw ROCm events 与 rocprofv3 fallback
一致 fail-closed: 请求 raw PM counters 时契约必须显式 `platform.persistent=false`;
true、字段缺失、契约缺失都拒绝。这类核的唯一时间轴手段是自制相位戳（见 §6）。
counter aggregate 只用于 per-dispatch 的普通 kernel。timeline-only 可以采 persistent
kernel, 因为 trace 只包 dispatch/API 事件边界, 不进 CU。

## 4. 无 200 kHz PM 采样等价物

NV 侧 nsys 的 200 kHz hardware bundle 时间轴采样在 ROCm 侧**没有等价物**。MI308X 主路线
提供的是同一个 Perfetto proto 内的 per-dispatch raw ROCm PM counter 与 dispatch slice;
rocprofv3 fallback 同样只有 per-dispatch / per-kernel 的 aggregate。两者都不是周期 GPU
PM 采样。amd-smi 的可选约 2–5 Hz polling 只作粗粒度调试上下文, 不进入 canonical
compute/memory utilization。要时间轴分辨率靠 §6 的相位戳+CDF 方法, 不要在任一 collector
里找 200 kHz 开关。manifest 必须把 nonperiodic、per-dispatch、同 run 配对和 fallback
语义如实写进 `sampling` / `capture_contract`, 禁止把 trace 密度、dispatch 数量、amd-smi
轮询率或 counter 数量换算成“等效 200 kHz”。

## 5. 互联计数：跨机在 NIC 侧, 不在 GPU PMC

与 NV 的本质差异：NVLink 计数器在 GPU PMC 内（nsys 直接采 NVLink TX/RX User Data）；
MI308X **跨机流量走 RDMA NIC（ibv/UNICM）, 字节计数完全不在 GPU PMC**。

- `communication_util` 只有在同一 run、同一 dispatch 上采到 XGMI、RDMA 或 fabric
  counter 时才有 canonical 数据。没有这种 same-dispatch counter 时必须写成
  `unavailable` 并保留原因；**绝不补零**, 也不要用 `TCC_EA0_*_GMI_CREDIT_STALL`
  stall 周期、`EA 全向请求 − DRAM 请求` 的差值、amd-smi NIC 或 GPU busy 值硬凑成
  communication utilization。
- 跨机带宽用 **NIC 侧旁路**：ibstat / amd-smi 的 NIC 项 / 或 proxy 层自打点
  （per-(rank,peer) 的 `disp MB/ms=GB/s | ret MB/ms=GB/s` 行, 生产实例见 §6 引用）。
  这些是独立的 NIC/RDMA 证据; 如果没有与 GPU dispatch 做同一 run 的 timestamp pairing,
  就只能标成外部旁路, 不能冒充 same-dispatch GPU counter。
- 同机 XGMI：本机 rocprofv3 build 不暴露 XGMI 带宽计数器, 只有 GMI credit-stall
  拥塞代理。该代理只说明“等过”, 不是字节, 也不是 `communication_util`。

## 6. 相位戳 + CDF 方法（生产实例）

persistent kernel 的插桩总图 = **prog[] 固定槽位 + rdclock()（s_memrealtime）打点 +
host 侧打印相位行 + per-mb CDF 曲线**。生产实例（含打点槽位编号表、env 门控纪律、
读数纪律）见：
`/workspace/weave_new1/weave_v1/rebuttal/amd_mi308x/megamoe/mn_v2/notes/instrumentation_map.md`
——mega M1-M6 全系 EP16 迭代（drain/RRS/RQT/BB 相位分解、w01stat、mb-done CDF）都靠这套。

## 7. 占用检查与设备选择（多 agent 共用机纪律）

- **amd-smi process**：8 张卡列出**同一个 pid** = host 侧系统进程（容器内不可见, 不持
  用户显存）, 是正常现象不算占用；真正的用户占用会显示不同 pid + 显存列。跑 GPU 前
  还要先查 `/tmp/*_RUNNING` flag 文件（多 agent 互斥纪律）, 双重确认后才动卡。
- **HIP_VISIBLE_DEVICES**：每进程绑一卡, 容器内从 0 重编号；跨机双机档各 rank
  `HIP_VISIBLE_DEVICES=<localrank>`。
- 僵尸进程（Z 态挂 PID1）不持 GPU/端口资源, 见到不用处理, 但别误读成活占用。

## 8. Same-run raw counter 定义与证据状态（MI308X GEMM route, **measured**）

主路线的 track 名形如 `GPU SQ_VALU_MFMA_BUSY_CYCLES [0]`; 当前 GEMM 计算层使用
`SQ_VALU_MFMA_BUSY_CYCLES` 与 `GRBM_GUI_ACTIVE`。Matrix Core utilization 的公式必须保持为：

```text
100 * SQ_VALU_MFMA_BUSY_CYCLES /
    (GRBM_GUI_ACTIVE * (CU_NUM * SIMD_PER_CU / XCC_NUM))
```

MI308X 默认 `CU_NUM=80`、`SIMD_PER_CU=4`、`XCC_NUM=4`, 因此分母使用
`80*4/4 = 80` SIMD lanes per XCC。`compute_util` 与 auxiliary `tensor_util` 使用同一个
derived value; stable semantic IDs 不变, 计算轨道可标 `compute_util [matrix_core]` /
`tensor_util [matrix_core]`。该值是 per-dispatch aggregate, 不能声称 kernel 内时间分辨率。

其他 stable track 的口径：

- `simd_util` 单独来自 `SIMD_UTILIZATION*100`, 标 `simd_util [SIMD/CU]`; 绝不冒充 Matrix Core。
- `sm_active` 直接来自 `GPU_UTIL`, 标 `sm_active [GPU active]`; 它也不是 tensor/compute utilization。
- HBM `memory_util` 是**显式 estimate**: `(TCC_EA0_RDREQ_DRAM + TCC_EA0_WRREQ_DRAM) * 64 B /
  dispatch_duration / 3684 GB/s`, 并保留 read/write 方向曲线。3684 GB/s 是 MI308X 实测峰值,
  64 B/request 是写入 trace metadata 的显式假设; 展示时必须带 `[HBM request estimate]` 语义。
- `communication_util` 没有 same-dispatch XGMI/RDMA/fabric counter 时只能是 `unavailable` +
  reason; 不得零填充, 不得用 proxy 或 amd-smi 代替。
- 可选 amd-smi `gpu_busy` / `memory_busy` / `gpu_memory_used_mib` 只是粗粒度 debug
  overlay (`canonical_mapping=null`, `auxiliary_only=true`), 不能展示成 compute 或 memory utilization。

本节的 measured 状态只覆盖 MI308X GEMM 路线。rocprofv3 `--list-metrics` 仍可用于独立
校准或 fallback 候选核对; `MfmaUtil` 是 optional independent calibration/fallback,
不是 primary source。旧 `configs/metrics/rocprof_gfx942_moe.json` 的 `_sum` 名称属于
rocprofv3 explicit-PMC 词汇, 不能据此改写 same-run rocprof-sys raw track 名。

### 8.1 MI308X 出口限规 SKU 算力口径辨析（为何不像 H20 那样乘常数因子，**measured**）

需要严格区分 **NVIDIA H20 的硬件宽度缩减（需 `×4` 修正）** 与 **AMD MI308X 的动态限频机制（保持 `×1.0`，严禁乘 `606.2 / 398.7 = 1.52×`）**：

1. **H20 (`sm_90a`) 为何必须 `×4`**：`nsys` 的 `Tensor Active [Throughput %]` 以完整 GH100 核心的每周期 Tensor 吞吐宽度为分母，而 H20 每 SM 的 Tensor 吞吐被硬件裁剪为满血 GH100 的 `1/4`，导致 100% 跑满的 H20 GEMM 原始读数封顶在 `25%`，因此必须乘以 `tensor_active_scale = 4.0`。
2. **MI308X (`gfx942`, 80 CUs) 为何严禁乘 `1.52×`**：
   - `rocminfo` 报告 `Max Clock Freq = 1850 MHz`，按未限频 CDNA3 理论公式计算为 `80 CU × 4 SIMD × 1024 FP8 FLOPs/clk × 1.85 GHz = 606.208 TFLOPS`；但出口合规版 MI308X 的官方物理稠密 FP8 峰值上限为 **`398.7 TFLOPS`**（BF16 为 **`199.3 TFLOPS`**）。
   - 真机 80 µs 硬件 PMC 实测表明：MI308X 并非每周期屏蔽部分 MFMA 槽位，而是**在密集 MFMA 负载下由固件将实际着色器运行主频动态压制在 `1.373–1.380 GHz`**（由 `(GRBM_GUI_ACTIVE / 4) / dt_ns` 直接测得：`327,680 FLOPs/clk × 1.378 GHz = 451.5 TFLOPS` 原始管线时钟上限）。
   - 因为 `100 * SQ_VALU_MFMA_BUSY_CYCLES / (80 * GRBM_GUI_ACTIVE)` 的分母 **`GRBM_GUI_ACTIVE` 本身统计的就是降频后（`~1.38 GHz`）真实流逝的活跃着色器周期数**，频率缩减已经在分母中被自然消去！在 MI308X 真机上运行 `8192×4096×2048` 稠密 FP8 GEMM（实测 **`376.65 TFLOPS`**，已达 `398.7 TFLOPS` SKU 峰值的 **`94.5%`**）时，该公式原始读数直接达到 **`max = 82.90%`、`p95 = 76.29%`（中位数 `72.14%`）**。若错误乘上 `606.2 / 398.7 = 1.5205×`，读数将溢出至 **`126.1% > 100%`**。
3. **图例与全双工 xGMI 规范**：
   - Panel (A) 计算曲线应标注为 **`FP8 Matrix Core Pipe Active % (398.7T MI308X SKU)`**（表示当前 `~1.38 GHz` 真实运行主频下的 Matrix Core 管线周期占空比），绝不可误标为 `% of 606.2T`。
   - 机内全双工 xGMI 硬件计数器（`communication_rx_util` 入向与 `communication_tx_util` 出向，单向理论带宽 `7 × 41.03 = 287.2 GB/s`）必须在 Panel (A) 中拆分为 **`Intra-Node xGMI RX (In)`** 与 **`Intra-Node xGMI TX (Out)`** 两条独立曲线，严禁做 `max()` 合并或相加。

## 9. Native Perfetto 导出与 Trace Processor 验收（capture -> validate）

合并后的 Chrome JSON `.perfetto.json.gz` 只是中间调试格式, 用来检查合并结果或离线修复。
最终可打开、可交付的文件必须是 Perfetto native TrackEvent protobuf
`.perfetto-trace` 或 `.perfetto-trace.gz`。Chrome JSON 改名、只做 JSON 解析、只列压缩包
内容, 或发送包着 trace 的 `tar.gz`, 都不算 native trace 交付证据。

full audit 的 native export 使用这个 exact 命令:

```bash
python3 -B weaveprof/pipeline/capture/export_perfetto_native.py --trace <merged_chrome_json.gz> --out <name__card-mi308x__...__fi-l2-s0.perfetto-trace.gz>
```

fast sampled development capture 使用 `FI_LEVEL=2 FI_SAMPLE=1` 和 `fi-l2-s1` token;
它不是 full audit, 不得改名为 `fi-l2-s0`。

预览可以用 `--preview-k-blocks 0,31,63`; 该选项保留 store 和 GEMM envelope, 只缩减 K block:

```bash
python3 -B weaveprof/pipeline/capture/export_perfetto_native.py --trace <merged_chrome_json.gz> --out <name__card-mi308x__...__fi-l2-s0-preview.perfetto-trace.gz> --preview-k-blocks 0,31,63
```

preview 只用于检查, 不是交付件。full audit 导出必须不带 `--preview-k-blocks`, 输出完整的
`<name__card-mi308x__...__fi-l2-s0.perfetto-trace.gz>`。`.perfetto-trace` 非压缩形式同样有效;
文件名 token 是 `fi-l<FI_LEVEL>-s<FI_SAMPLE>` 的显式配置。capture manifest 必须同时记录实际
`FI_LEVEL` 和 `FI_SAMPLE`, 文件名不能替代 manifest。`FI_LEVEL=1` 是 entry-only, 不是 full audit。

导出前读取 merged Chrome JSON 的 `otherData.meta.card`; 先去空白并转小写, 结果必须精确等于
MI308X 的 `mi308x`, 输出文件名必须含 `card-` 加这个值得到的 `card-mi308x`。不得从 GPU
index、marketing name 或架构名反推; meta 缺失、指向别的卡, 或文件名 token 与 meta 不一致时
fail-closed。exporter 不会替调用者选择或改写这个文件名。
命令 stdout 是 native export report, 必须和 trace 一起保留, 用于核对 `semantic_counts`、
`physical_waves`、`wave_tracks`、`hardware_tracks` 和 `hardware`。H20/B300 使用同名规则及
`card-h20` / `card-b300`, B300 的 `sm_100a` 与 `sm_103a` 仍分别保留。

native trace 的 track 契约:

- 每个物理 wave 一条独立 track, 归在对应 SM/CU 下。wave track 上的语义 slice 名只能是
  `load`、`mma`、`store`; software dispatch track 上有一个 `gemm` envelope。不要改成
  source label、alias 或自造阶段名。
- `compute_util [matrix_core]`、`tensor_util [matrix_core]`、`simd_util [SIMD/CU]`、
  `memory_util [HBM request estimate]`、`memory_read_util [HBM request estimate]`、
  `memory_write_util [HBM request estimate]` 和 `sm_active [GPU active]` 都是
  per-dispatch aggregate 平带: 同一 dispatch 的起止点使用同一个 aggregate value。零值边界
  只是 display break; 不得插值、细分或声称有 kernel 内硬件分辨率。
- 没有 same-dispatch XGMI/RDMA/fabric counter 时, `communication_util [interconnect]`
  track 显示 instant marker `unavailable` 和 reason; 不画零值, 不用 proxy 补线。
- `software_load_active_waves`、`software_mma_active_waves`、
  `software_store_active_waves` 是 1 us bin 的区间派生占用, 不是硬件曲线; 标签里的
  `[derived from intervals]` 必须保留。

`validate` 必须用 `trace_processor_shell` 实际导入最终 native trace。文件扩展名正确、JSON
可解析、exporter 返回 0 或 `tar -tzf` 能列出文件都不算通过。逐个执行下面的 SELECT,
要求每次 `trace_processor_shell` 退出码为 0、`slices > 0`、`import_errors = 0`, 且
查询得到的 `wave_tracks` 同时等于 export report 的 `physical_waves` 与 `wave_tracks`:

```sql
SELECT
  (SELECT COUNT(*) FROM slice) AS slices,
  (SELECT COUNT(*) FROM stats
   WHERE value > 0
     AND (severity = 'error'
          OR name IN ('json_parser_failure',
                      'json_tokenizer_failure',
                      'slice_spill_overlapping_complete_event',
                      'slice_drop_overlapping_complete_event',
                      'trace_sorter_negative_timestamp_dropped'))) AS import_errors,
  (SELECT COUNT(*) FROM track
   WHERE name GLOB 'wave *' OR name GLOB 'cta * / warp *') AS wave_tracks;

SELECT name, COUNT(*) AS slices
FROM slice
WHERE name IN ('load', 'mma', 'store', 'gemm')
GROUP BY name
ORDER BY name;

SELECT t.name,
       (SELECT COUNT(*) FROM counter AS c WHERE c.track_id = t.id) AS counter_points,
       (SELECT GROUP_CONCAT(DISTINCT s.name) FROM slice AS s WHERE s.track_id = t.id) AS slice_markers
FROM track AS t
WHERE t.name LIKE '%util%' OR t.name LIKE 'sm_active%'
   OR t.name LIKE 'software_%_active_waves%'
ORDER BY t.name;
```

验收还要求: wave track 上的 slice 名只出现 `load`/`mma`/`store`; 全部 `load`、`mma`、
`store` 数量与 report 的 `semantic_counts` 对账; native trace 恰好有一个 `gemm`
envelope。report `hardware` 中每个实测 track 都必须存在且有 counter point, 这些
utilization track 的数量必须等于 report 的 `hardware_tracks`;
`communication_util [interconnect]` 缺失证据时必须由上面的 `slice_markers` 显示
`unavailable`。software active-wave track 只能按区间派生占用验收, 不能混入硬件
utilization 台账。preview 的 slice 数量是有意缩减的, 不能当作完整交付验收。

该导出属于现有 `scan -> instrument -> capture -> validate` 中的 capture 后处理和 validate
验收, 不增加 workflow 阶段; 对 MI308X 的 HIP C++ 与 Triton 使用同一套导出和验收入口,
不拆 HIP 专用流程, 也不引入 MLIR。H20、B300 与 MI308X 继续按各卡路由执行同一工作流,
B300 的 `sm_100a`/`sm_103a` 仍分别取证。

## 10. Canonical 文件名与 unavailable 语义

merged Chrome JSON 调试输入可以保留
`trace__<model>_<stage>__card-mi308x__bs<B>_sq<S>[_kv<K>]__<dtype>__grid<G>_warp<W>_ph<P>__<bundle>.perfetto.json.gz`;
最终 native trace 必须带匹配 capture manifest 的显式配置 token:
full audit 为
`trace__<model>_<stage>__card-mi308x__bs<B>_sq<S>[_kv<K>]__<dtype>__grid<G>_warp<W>_ph<P>__fi-l2-s0.perfetto-trace`,
fast sampled development feedback 为同名 `__fi-l2-s1.perfetto-trace`, 两者均可使用 `.gz`。
receipt 名是 `<trace>.delivery_receipt.json`, 同样继承 `card-mi308x`。
不要把 card token 删成隐含默认值, 也不要替换成 GPU index、marketing name 或别的 card。
任何缺少证据的 stable curve 都写 `unavailable` + reason; 只有真实采到的 0 才是 0,
缺失证据绝不是 0。

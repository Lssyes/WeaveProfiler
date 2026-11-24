# WeaveProfiler

简体中文 | [English](README_EN.md)

## 🚀 为什么是 WeaveProfiler

> **把 GPU kernel 内部每个 Warp / Wave 的微秒级执行时间线，和同一次运行的硬件利用率曲线画在同一张图里——面向写高性能 GPU kernel 的工程师与 AI coding agent 的低开销剖析工具。**

WeaveProfiler 是一款低开销 GPU kernel 内部时间线剖析与可复现验证工具，面向正在打磨高性能算子（如 MoE Megakernel、FlashAttention、FlashMLA、Persistent GEMM 等）的工程师，以及自动迭代优化 kernel 的 AI coding agent。

NVIDIA `nsys`、AMD `rocprof-sys` 这类厂商工具主要反映整张卡的总体利用率，利用率突然掉下去时，你看不到到底是哪颗 SM/CU、哪个 Warp 在干等；而直接在源码里手工插桩计时，又很容易引发寄存器溢出或打乱编译器的指令调度。WeaveProfiler 会把代码复制到隔离的临时影子副本中插桩，把每个 Warp 的阶段时间线和同一次运行的硬件利用率曲线对齐画进一张 `overview.png`，再用四道硬性检查把关——只要任意一项不通过，就拒绝给出通过结论，确保每一轮性能数据真实可信。

![NVIDIA H20 Triton num_stages=5 GEMM 四面板软硬对齐图](docs/images/case_h20_triton_s5_4panel.png)

> 图：NVIDIA H20（`sm_90a`）上 `4096×4096×2048` Triton `num_stages=5` GEMM 的四面板实测图。无需手改 Triton 核函数的 K 维主循环代码，工具自动从生成的 PTX 中追踪异步数据搬运（`cp.async`，最大在途深度 `8`）与矩阵计算（`wgmma`，最大在途深度 `2`）两条异步流水线（实测 `122.42 TFLOPS` = `93.7%` cuBLAS，稳态开销 `+5.46%`，输出误差 `max_diff=0.0`，`200 kHz nsys`），清晰呈现出前 `13` 个完整波次以及末尾只剩约 `10` 颗 SM 仍在收尾、导致利用率明显下降的过程。

---

## ✨ 亮点

- 🧵 **SM/CU 与 Warp/Wave 级核内时间线**：每个 Warp/Wave 写自己独立的内存区域（仅 `lane 0` 顺序写入 16 字节记录，不用 `atomicAdd`），尽量不拖慢核函数、不引入线程争抢。
- 📈 **核内时间线与硬件曲线同图对照**：把核内软件时间线和同一次运行的硬件利用率曲线（NVIDIA 为 `200 kHz nsys`，AMD 为按 dispatch 采集的 `rocprof-sys`）合画在一张 `overview.png` 里，打开就能看，默认不生成庞大的 trace 文件（需要时可按需导出 `.perfetto.json.gz`）。
- 🌊 **异步流水线深度与等待定位**：同时记录 `TMA`/`cp.async` 数据搬运与 `WGMMA` 计算的在途请求深度，标出每次发起位置，以及没能被计算掩盖的等待时间（`Exposed Wait`）。
- 🛡️ **隔离影子副本编译，不改生产源码**：所有插桩、编译和运行都在临时影子副本里进行，不用给 kernel 函数加额外参数，完全不改动你的生产源码。
- ✅ **四道硬性检查 + 精简报告 + 可校验封存**：四道硬性检查（编译溢出、输出一致、事件完整、`±10%` 开销）中任意一项不通过就不出结论；通过后输出完整诊断报告 `feedback.json`、面向 AI agent 的精简摘要（`< 2 KB`）以及带 HMAC-SHA256 校验的封存记录 `seal.json`。
- 🤖 **多平台多语言与 AI Coding Agent 集成**：支持 `CUDA C++`、`HIP C++`、`Triton`、`CuTeDSL`，覆盖 H20 / B300 / `sm_86` / MI308X 等平台，并通过 `Skill + MCP + Hook` 接入 Claude Code、Codex CLI、Qwen Code 与 Antigravity。

---

## 🔍 它能回答的问题

- **哪个 Warp / Wave 在等，等在哪里？** 把同一颗 SM/CU 内负责数据搬运、计算、调度的各物理 Warp/Wave 拆成独立泳道，并用亮黄色斜线标出没被计算掩盖的等待时间（`Exposed Wait`）及占比，直接定位阻塞源。
- **`TMA`/`cp.async` 和 `WGMMA` 流水线跑满了吗？** 用深浅不同的色块显示同时在途的异步请求数，并用 `▼` 标出每次发起位置，能直观看出流水线怎么爬满、稳定在几级深度以及结尾怎么逐步排空。
- **末尾利用率下降是少数 SM 拖尾，还是各 SM 负载不均？** 全卡所有 SM/CU 纵向排开，一眼分清是最后几颗拖尾的 SM 还没收尾，还是不同 SM 之间的负载本身就不均衡。
- **硬件 Tensor Core 或显存带宽掉下去时，代码在执行哪一段？** 硬件曲线与核内阶段共享同一条微秒时间轴，能直接区分核间空等、冷启动阶段显存带宽爬坡，还是桌面系统上的 WDDM 抢占中断。
- **还有多少优化空间，是否已经撞上硬件上限？** `causal.py` 对照该卡实测的算力和带宽上限，估算每个阶段还有多少优化空间，避免在已经撞上硬件瓶颈的地方白费功夫。

---

## 🖼️ 怎么读四面板图

绘图模块 `viz.py`（`render_swhw_overview_png`）按同一条微秒时间轴画出最多 4 个对齐面板。**如果缺少同一次运行采集到的真实硬件曲线，默认拒绝生成图片**：

- **(A) 硬件利用率曲线**：显示同一次采集到的整卡指标（纵轴 `0–100%`；NVIDIA 为 `200 kHz nsys`，AMD 为 `rocprof-sys`），包括计算单元、`Active SMs`、`SM Issue`、**`HBM/DRAM` 读+写总带宽**，以及分别独立绘制的全双工 `NVLink TX` 与 `NVLink RX`（不把收发相加；缺少对应多机 RDMA 计数器时明确标为 `unavailable`，不会用 `0` 填充）。
- **(B) 全卡物理 SM / CU 执行总览**：把整张卡的所有物理 SM/CU 纵向排开（以 `2400 px` 宽度渲染），叠画主执行阶段与后台异步 `MEM` 活动，直观看出波次边界、末尾波次里少数掉队的核心以及耗时异常长的核心。
- **(C) 代表性 SM / CU 的逐 Warp / Wave 时间线**：自动选取 CTA、角色与事件最丰富的 SM/CU（支持 `--sm` 手动指定），每条泳道上部画主执行阶段和亮黄色斜线标注的 `Exposed Wait`，下方按需展开绿色的 `MEM`（搬运）和蓝色的 `MMA`（计算）在途深度色块及 `▼` 发起位置，右侧汇总事件数、平均并发度 `d` 与等待占比。
- **(D) 自动局部放大单个线程块（CTA）**：当相邻事件在图中挤得太密（间距 `< 28 px`）时自动展开，优先放大等待占比最高的 CTA（整体几乎无等待时则取稳态耗时居中的 CTA）；若循环超过 `48` 次迭代，则截取等待最重的一段周围的 `32` 次迭代，并在 (A)–(C) 面板用洋红虚线框同步标出放大区间。

> 💡 **H20 读图提示**：在 H20 上，`nsys` 的 `Tensor Active` 按完整 GH100 芯片归一，而 H20 每颗 SM 算力只有其 `1/4`，因此面板 (A) 满载读数大约在 `25%`，乘以 `4` 才是相对 H20 峰值的真实利用率（`causal.py` 的分析结果已自动乘以 `4`）。

---

## 📊 实测案例

### 8×H20 生产级 DeepGEMM MegaMoE 优化（`1.47×` 加速）

在 8×H20 NVLink 节点上剖析 Qwen3.5-397B DeepGEMM MegaMoE（`EP=8, S=1024, E=512`，`200 kHz nsys`，详述见 [h20_megamoe_optimization.md](docs/case_studies/h20_megamoe_optimization.md)）：对齐时间线定位出 ① `GEMM1`/`GEMM2` 拆成两个 kernel 导致约 `65 µs` 核间空等（`t ≈ 280–345 µs`），以及 ② `GEMM2` 中同步 `Scatter` 阻塞计算 warpgroup 而 `Disp` 线程束闲置；融合为动态队列单个持久化 kernel 并将 `Scatter` 异步交给 `Disp` 后取得 **`1.47×`**（耗时降低 `32.1%`）加速，优化后活跃 SM 均值达 `94.0%`：

| 版本 | 端到端耗时 | FP8 Tensor Core（均值 / 峰值） | HBM 带宽（均值 / 峰值） | NVLink 通信（均值 / 峰值） |
| --- | ---: | ---: | ---: | ---: |
| 原始分核版本（图 a） | `613.2 µs` | `27.9% / 37.1%` | `36.4% / 47.6%` | `8.1% / 78.0%` |
| 持久化融合 + 异步 Scatter（图 b） | **`416.4 µs`（`1.47×`）** | **`41.8% / 62.6%`** | **`55.7% / 81.5%`** | `8.8% / 50.4%` |

![8×H20 DeepGEMM MegaMoE 优化前后对比图](docs/images/case_h20_megamoe_1p47x.png)

> 图：8×H20（`78` 颗 SM，`200 kHz nsys`）DeepGEMM MegaMoE 优化前后对比，直观展示消除约 `65 µs` 核间空等并将 `Scatter` 异步化后的加速收益。

### H20 生产级算子剖析（不改 kernel 函数签名）：FlashInfer 0.6.8 Hopper FA3

在 H20（`sm_90a`）上对生产级 CUDA C++ 算子 **FlashInfer 0.6.8 Hopper FA3 Prefill**（`B=5, H=16, S=4096, D=128`）进行不改函数签名的实测剖析（`200 kHz nsys`）：

- **全覆盖与极低开销**：覆盖全部 **`12/12`** 物理 Warp，稳态相对开销仅 **`+0.15%`**，输出完全一致（**`max_diff=0.0`**）。
- **核内分工与重叠**：Panel (C) 清晰区分负责数据搬运的 `W00 producer_tma`、空闲的 `W01–W03 producer_idle` 与两个计算线程束组 `W04–W11 math_wg1/2`；Panel (D)（`CTA 26`，`1.6–150.6 µs`）把每个计算 warp 拆成主任务和异步矩阵计算两行展示，直接呈现 `Softmax/Wait` 与异步 `WGMMA QK/PV`（深度 `d=1/2`）重叠执行以及 tile 间的 `O Store` 写回。

![H20 FlashInfer 0.6.8 Hopper FA3 四面板对齐图](docs/images/case_h20_flashinfer_fa3.png)

> 图：H20（`sm_90a`）FlashInfer 0.6.8 Hopper FA3 Prefill（`B=5, H=16, S=4096, D=128`）实测四面板图，以 `+0.15%` 开销与 `max_diff=0.0` 完整呈现 `12/12` 物理 Warp 分工与异步计算重叠。

### 更多已验证案例（链接）

- **FlashInfer 0.6.8 Hopper FlashMLA（H20 `sm_90a`）** — [case_h20_flashmla.png](docs/images/case_h20_flashmla.png)：`DeepSeek-V3` 规模（`B=64, H=128, KV=2048`），覆盖 `8/8` 个 Warp，稳态开销 **`+9.96%`**、`0 B` 寄存器溢出、`max_diff=0.0`、`200 kHz nsys`；Panel (D) 自动放大满载双波次的 `SM 2`（`CTA 22`，`140.9–248.8 µs`），同时展示 `WG0` 的异步 `cp.async` 搬运（深度 `d=1/2`）、等待与 `PV1` 计算，以及 `WG1` 的 `Softmax / WGMMA` 阶段。
- **vLLM FLA FlashKDA（H20 `sm_90a`）** — [case_h20_flashkda.png](docs/images/case_h20_flashkda.png)：`Triton chunk_delta_h`（`B=1, T=4096, H=64, D=128`），无需手改 Triton K 维主循环代码、自动从 PTX 追踪 `cp.async`（最大深度 `5`）与 `wgmma` 两条异步流水线，覆盖 `4/4` 个 Warp，开销 **`+8.14%`**、`0 B` 溢出、`max_diff=0.0`；Panel (D) `CTA 14`（`216.0–232.5 µs`）直接暴露约 **`65%`**（`64.2–65.6%`）未被掩盖的等待时间。
- **RTX 3050 `sm_86` WDDM 抢占与冷启动排障** — [sm86_wddm_preemption.md](docs/case_studies/sm86_wddm_preemption.md) / [case_sm86_wddm_preemption.png](docs/images/case_sm86_wddm_preemption.png)：`8192³` FP16 Triton GEMM（完整 kernel 耗时 `86–101 ms`）在 `16,384` 条泳道上采集 `25.46M` 条原始记录且 `0` 溢出（软硬件窗口时长偏差 `1.07%`），对齐图清晰呈现 `0–25 ms` 的 GDDR6 冷启动带宽爬坡（均值 `55.9%`、峰值 `98.0%`）与 `t ≈ 30–33 ms` 处桌面 WDDM 显示抢占造成的利用率骤降归零。
- **AMD MI308X `gfx942` 采样开销检查与带宽标定** — [amd_mi308x.md](docs/amd_mi308x.md) / [README_card-mi308x.md](examples/triton/card-mi308x/README_card-mi308x.md)：实测 HBM 流式读峰值 `3684 GB/s`；在 `2048³` FP16 Triton GEMM 上，快速采样模式（`FI_LEVEL=2 FI_SAMPLE=1`）为 `14` 条记录/wave、耗时倍率 `0.975×`（通过 10% 检查，标明仅作快速调试、不作正式完整验证证据），而未做采样的全量打点对照（`FI_LEVEL=2 FI_SAMPLE=0`）为 `390` 条记录/wave、耗时倍率 **`1.703×`**，被 10% 开销检查直接拦截。

---

## 🧩 支持范围

| 卡型 (`card`) | 架构 (`arch`) | 并行粒度 | 支持语言 | 硬件采样与实测状态 |
| --- | --- | --- | --- | --- |
| **`h20`** (NVIDIA H20) | `sm_90` / `sm_90a` | `78` SMs, warp32 | `cuda_cpp`, `triton`, `cutedsl` | `200 kHz nsys`（`--capture-range=cudaProfilerApi`）；CUDA C++ / Triton / CuTeDSL 已有实测配置（注：CuTeDSL 的 `cute.gemm` `wgmma` 路径暂未实测）；因果分析会自动对 `Tensor Active` 做 `×4` 校正 |
| **`b300`** (NVIDIA B300) | `sm_100a` / `sm_103a` | 如 `sm_103a` `148` SMs, warp32 | `cuda_cpp`, `triton`, `cutedsl` | `200 kHz nsys`；`sm_103a` 的 CUDA C++ / Triton 已实测，`sm_100a` 仅验证编译，CuTeDSL 处于实验阶段（参考 [L20D 峰值校准说明](docs/l20d_achievable_peak_rebase.md)） |
| **`sm86`** (Ampere) | `sm_86` | 如实测 RTX 3050 `16` SMs, warp32 | `cuda_cpp`, `triton`, `cutedsl` | `200 kHz nsys`（WSL2 缺硬件计数器时回退无硬件曲线模式）；基础卡级配置已在 RTX 3050 实测（暂未单列独立算子模板）；`triton`/`cutedsl` 默认启用 `WP_TIMER=1`（`%clock64` 约 `27 ns` 步长） |
| **`mi308x`** (AMD MI308X) | `gfx942` | `80` CUs (`4` XCC), wave64 | `hip_cpp`, `triton`（无 CuTeDSL） | 主用 `rocprof-sys`（`rocprofv3` 作备选），按 dispatch 聚合、无 `200 kHz` 周期采样；GEMM 路线已实测；`grid >= 80 CU` 的持久核禁止采集 PMC |

- **AI 编程代理集成**（`install-agent --agent`）：原生支持 **Claude Code**（`claude`）、**OpenAI Codex CLI**（`codex`）、**Qwen Code**（`qwen`）与 **Antigravity**（`agy` / `antigravity`）。

---

## 📦 安装

`pyproject.toml` 当前仅声明了 `requires-python = ">=3.10"`，**未在 `dependencies` 中列出运行时依赖包**，请按使用场景安装对应组件：

- **Python 版本**：`>= 3.10`（源码使用 `X | Y` 联合类型语法）。
- **Python 依赖**：
  - **绘图必需**：`numpy`、`matplotlib`（生成 `overview.png` 必需；核心编解码、检查与因果分析模块仅用标准库）；
  - **按需可选**：`msgspec`（加速 JSON 读写）、`perfetto`（仅原生 Perfetto protobuf 导出与 `TraceProcessor` 校验需要）、`torch` / `triton` / `cutlass`（对应前端核函数需要）。
- **系统工具链**：
  - NVIDIA：CUDA Toolkit（`nvcc`、`cuobjdump`）+ Nsight Systems（`nsys`）；
  - AMD：ROCm（`hipcc`）+ `rocprof-sys`（及备选 `rocprofv3`）；
  - 可选验收/沙箱：`trace_processor_shell`（仅 `run.py deliver` 的完整 Perfetto 导出验收需要）、Linux `bwrap`（`guard-exec` 可选只读挂载）。

```bash
pip install -e . --no-build-isolation   # 或不安装，直接通过 python3 -B run.py / python3 -B -m weaveprof 运行
pip install numpy matplotlib            # 渲染 overview.png 必需
python3 -B -m weaveprof doctor          # 检查 Python、GPU 互斥锁目录、GPU 与 profiler 就绪状态
```

> ⚠️ **GPU 计数器权限提示**：NVIDIA 遇到 `ERR_NVGPUCTRPERM`（`Insufficient Privilege`）权限报错时会直接中止，而不会悄悄丢掉硬件曲线——需在 Linux 配置 `NVreg_RestrictProfilingToAdminUsers=0` / `CAP_SYS_ADMIN`（或使用 `sudo` 授权执行 `nsys`），在 Windows NVIDIA 控制面板允许所有用户访问 GPU 性能计数器。未取得硬件曲线时，`viz.py` 默认拒绝生成只有软件时间线的 `overview.png`。

---

## ⚡ 快速开始

### 方式 A：接入 AI 编程代理（`Skill + MCP + Hook`）

```bash
python3 -B run.py install-agent \
  --workspace <your_repo> \
  --protect <kernel_file> \
  --agent claude            # 可选：all | both | claude | codex | qwen | agy | antigravity
```

安装后受保护文件会被锁定并备份到工作区之外的独立目录；AI 代理通过 MCP 工具 **`weaveprof_megakernel_audit`** 在临时影子副本中完成编译、测速、四道检查与 `overview.png` 渲染，生成通过的 `seal.json` 后才会解锁后续优化编辑。

### 方式 B：Python API 直接运行 Megakernel 剖析

> 注：`run.py` 本身**不提供** `megakernel-audit` 子命令；如需从命令行调用，可使用 `python3 -B weaveprof/aka_adapter.py megakernel-audit`（从标准输入读取 JSON）。

```python
from weaveprof.megakernel import run_megakernel_audit

report = run_megakernel_audit(
    workspace="./my_kernel_repo",
    entry_source="kernel.cu",                 # 相对 workspace 的路径
    inst_files={"kernel.cu": open("kernel_inst.cu").read()},  # 或传 inst_dir=...
    build_cmd="make -j8",
    bench_cmd="python3 bench_mk.py",
    capture_nsys=True,
    card="h20",
    out_dir="../wp_audit_out",  # 必须位于 workspace 之外
)
```

**运行前需要准备三部分输入**：

1. **插桩副本**（`inst_files` 源码映射或 `inst_dir` 目录）：只覆盖临时影子副本里的插桩版本（`L_inst`），原版会自动以 `WP_LEVEL=0` 关闭打点编译作基线，生产源码保持不动；
2. **`names.json`**：把事件 ID、语义分类（须取自 `configs/semantic_ids.json`）与线程角色编号映射成可读名称；
3. **基准脚本 `bench_cmd`**：
   - 执行 **ABBA 交替计时**（按 `Base-Inst-Inst-Base` 的顺序交替运行原版与插桩版，以抵消 GPU 升温、降频带来的单向漂移）；
   - 将两份输出比对文件、二进制记录文件（`capture.wp02.bin` 或 `capture.fi02.bin`）与 `runtime.json` 写入 `$WEAVEPROF_OUT_DIR`；
   - 四道检查都通过后，启用 `capture_nsys=True` 时会带着环境变量 **`WEAVEPROF_NSYS_CAPTURE=1`** 再运行一次 `bench_cmd`；此时脚本应跳过 ABBA 多轮循环，仅执行一次被 `cudaProfilerStart()` / `cudaProfilerStop()` 包裹的 kernel 启动，并将同一次运行产生的二进制记录文件写入 `$WEAVEPROF_OUT_DIR`。

最小 `names.json` 与 `runtime.json` 示例（按 `megakernel.py` 约定）：

```json
{
  "events": {"1": "kv_topk_load", "3": "sparse_mla_mma", "5": "epilogue"},
  "semantics": {"1": "load", "3": "mma", "5": "epilogue"},
  "roles": {"0": "persistent_cta"}
}
```

```json
{
  "kernel_name": "sparse_mla_decode",
  "warmup": 3,
  "timing_protocol": "ABBA",
  "baseline_ns": [374000.0, 374400.0, 373900.0],
  "instrumented_ns": [381106.0, 381513.6, 381004.1],
  "output_files": [["baseline.bin", "instrumented.bin"]],
  "l0_stats": {"regs": 72, "spill_stores": 16, "spill_loads": 16, "stack_bytes": 16, "stl": 2, "ldl": 2},
  "inst_stats": {"regs": 72, "spill_stores": 16, "spill_loads": 16, "stack_bytes": 16, "stl": 3, "ldl": 3}
}
```

**产物清单（输出至 `out_dir`）**：

- `overview.png`：四面板软硬对齐图（默认生成；缺少硬件曲线时拒绝出图）。
- `feedback.json`：完整检查细节与因果诊断报告（API 返回给 AI 代理的摘要压至 `< 2 KB`）。
- `verify.json` / `seal.json`：四道检查判定摘要与 HMAC-SHA256 封存记录。
- `nsys_hw.json` / `trace.perfetto.json.gz`：启用 `capture_nsys` 时产出硬件摘要；体积较大的 `.perfetto.json.gz` 文件默认不输出（需显式 `write_perfetto_gz=True`）。

---

## 🛡️ 四道检查

| 检查项 | 检查什么 | 输入来自哪里 |
| --- | --- | --- |
| **`assembly`** | 比对局部访存与溢出增量，要求 `delta_spill_stores <= 0 B`、`delta_spill_loads <= 0 B`、`delta_stack_bytes <= 0 B`、`delta_stl <= 4`、`delta_ldl <= 4`（**即使编译器报告的寄存器数没涨，也能抓出隐性溢出**） | `bench_cmd` 写入的 `runtime.json` 中 `l0_stats` / `inst_stats` |
| **`output`** | 检验基线与插桩版输出是否指向两份不同的非空文件，且 **SHA-256 逐字节一致** | `output_files` 指向的两份磁盘文件（默认未强制要求磁盘文件比对时，回退读取 `runtime.json` 中的哈希字段或 `output_match`） |
| **`events`** | 检查 `overflow == 0`、起止配对错误 `== 0`、`spans > 0`；**拦住空占位探针和只包住整个大循环的粗粒度打点**，且默认要求覆盖全部物理 Warp | 解码后的 `capture.*.bin` + `names.json`，辅以 `runtime.json` 可选字段（`sass_body_insns`、`expected_warps_per_cta`） |
| **`overhead`** | 要求 `warmup >= 1`、`>= 3` 组正数成对样本、中位数相对变化 **`\|Δt\| <= 10%`**（**插桩后莫名变快超过 `10%` 同样会失败，防止打点改变了编译器优化**） | `bench_cmd` 写入的 `runtime.json` 中 `baseline_ns` / `instrumented_ns` / `warmup` 等 |

> ℹ️ **Megakernel 路径的数据边界**：在 `run_megakernel_audit` 中，`assembly`、`overhead`（以及默认未强制要求磁盘文件比对时的 `output` 回退项）依赖 `bench_cmd` 生成的 `runtime.json`；`megakernel.py` 本身不直接调用 `cuobjdump` 做指令级比对。如果需要工具独立反汇编校验并强制检查输出文件，请改用 `wp.profile` / `wp.verify` 或分阶段命令 `validate`。

---

## 🔧 设备端记录器速查

以下开关与宏定义严格对应 `recorders/cuda_cpp/wp_lite.cuh`（同时定义 `WP_*` 与不相等的旧 `FI_*` 会触发编译 `#error`）：

| 编译开关 | 可选值（CUDA 默认） | 行为说明 |
| --- | --- | --- |
| **`WP_LEVEL`** | `0 \| 1 \| 2`（`0`） | `0` 为关闭打点（编译为空操作，用于基线计时）；`1` 只记录 `OPEN`/`CLOSE`、作用域与角色切换；`2` 记录全部事件与阶段边界切换 |
| **`WP_TIMER`** | `0 \| 1`（`0`） | `0` 读取全局时钟 `%globaltimer`；`1` 主体改用 `%clock64`，并在 `OPEN`/`CLOSE` 各占 `2` 槽（共多占 `2` 槽）写入双时钟锚点 |
| **`WP_GRAN`** | `0 \| 1`（`0`） | `0` 为 Warp/Wave 粒度（仅 `lane 0` 写入）；`1` 为 CTA/SM 粒度（仅 `thread 0` 写入） |
| **`WP_CTX_REG`** | 宏开关（默认关） | 单 32-bit 寄存器上下文（`0 B` SMEM），上限 **`65,535` 槽位** / **`32,767` 条/泳道**；支持 `WP_CTX_REG_NOINLINE` 把写入辅助函数标为 `__noinline__`，避免内联撑大主循环 |
| **`WP_CTX_SMEM`** | 宏开关（默认关） | **主循环不占寄存器，改用静态共享内存**，每槽位占用 **`16 B`**（32 warp 默认 `512 B/CTA`，`WP_GRAN=1` 为 `16 B/CTA`） |

**何时切换上下文模式**：默认优先从 `-DWP_CTX_REG` 开始；遇到以下任一情况立即切换或调整打点位置：

- **每线程可用寄存器余量 <= 2 或出现寄存器溢出/栈帧/STL 增量**（即使报告的 `regs` 计数没变！）：剩余共享内存 `>= 16 B` 时切到 **`-DWP_CTX_SMEM`**（SMEM 不足 32 槽可叠加 `-DWP_GRAN=1` 降至 `16 B/CTA`）。
- **寄存器与共享内存同时耗尽（剩余 SMEM `< 16 B`）**：回退到 **`-DWP_CTX_REG_NOINLINE`**（1 个寄存器、`0 B` SMEM，把写入辅助函数隔离成 `__noinline__`）。
- **编译器循环展开（loop unrolling）或指令调度被破坏**：停止在内层热循环打点，将探针**外提到外层阶段边界**。

**6 个核心宏速查**：

| 宏定义 | 用途说明（1 句话速览） |
| --- | --- |
| **`WP_OPEN(role)`** | 在核函数入口由 `lane 0`/`thread 0` 初始化槽位上下文，并写入带物理 SM ID 的 `OPEN` 起始记录（`WP_TIMER=1` 时同步写起始锚点）。 |
| **`WP_CLOSE()`** | 在核函数退出路径写入 `CLOSE` 结束记录（与结束锚点），并在缓冲区写满时设置溢出标志位。 |
| **`WP_B(id, aux)`** | 在默认事件组（group 0）写入事件 `id` 的开始（Begin）16 字节记录，附带自定义 `aux` 上下文参数。 |
| **`WP_E(id, aux)`** | 在默认事件组（group 0）写入事件 `id` 的结束（End）16 字节记录，与对应 `WP_B` 配对闭合。 |
| **`WP_TRANS_PRED(pred, g_end, end_id, g_beg, beg_id, aux)`** | 仅读取一次计时器，以相同时间戳连续写入上一阶段结束与下一阶段开始两条记录。 |
| **`WP_KERNEL_SCOPE(role)`** | 常规模式下为 RAII 作用域宏，**但在 `-DWP_CTX_SMEM` 下会直接展开为非 RAII 的入口宏（底层展开为 `FI_OPEN(role)`），退出前必须显式调用 `WP_CLOSE()`**。 |

---

## 🧭 其他入口

- **分阶段 CLI 流水线**：`python3 -B run.py scan → instrument → capture → validate → deliver`（另含只做检查、不签发验收回执的 `check`）将源码语义扫描、按源码哈希校验生成并应用插桩修改、硬件采集、完整规则检查以及基于 `trace_processor_shell` 的最终 trace 导出验收（`deliver`）拆成独立阶段。
- **内置 H20 GEMM 探针（`gemm-probe`）**：`python3 -B run.py gemm-probe --card h20 --arch sm_90` 一键编译并运行内置 H20 `sm_90` GEMM 软硬件对照探针（支持 `--profile full|quick`，结果固定标明仅供诊断参考、不作正式验收证据）。
- **快速构建与验证（`fast`）**：`python3 -B run.py fast --job <fast_job.json>` 先哈希缓存并并行编译关闭打点与全量打点两个版本，再跑快速验证（同样标明不作正式验收证据，不替代完整验证流程）。
- **Python 闭环接口与模块 CLI（`wp.profile` / `wp.verify` / `-m weaveprof`）**：`wp.profile` 负责自动定位打点位置（`identify`）、独立比对汇编指令并在开销或调度受影响时多轮外提探针（配合 `wp.locate_or_remap_0token` 随代码修改免 LLM 重新映射位置）；`wp.verify` 则针对已有的插桩配置（`probe.json`）做单轮确定性验证；`python3 -B -m weaveprof` 另暴露 `identify`、`locate`、`verify`、`validate`、`profile`、`seal-check`、`doctor`、`causal`、`render` 共 9 个子命令。
- **Coding Agent 防护工具链（`install-agent` / `guard_hook.py` / `mcp-server` / `guard-exec`）**：`install-agent` 写入代理专属 MCP 与生命周期 Hook，由 `guard_hook.py` 在工具调用前后及退出阶段拦截未经校验封存的改动、源码中遗留的打点代码与不合规 `.gz` 文件，配合 `guard-exec` 在代理运行期间持续守护受保护文件（并可选 Linux `bwrap` 只读挂载）：

```bash
python3 -B run.py guard-exec --workspace <your_repo> -- codex exec "定位核内等待瓶颈与末尾 SM 掉速原因"
```

---

## 📌 使用须知

- **Megakernel 路径需要自行准备输入**：`run_megakernel_audit` 不会自动改写大型多文件算子源码，需由调用方提供临时影子副本中的插桩代码（`inst_files` / `inst_dir`）、`names.json` 以及实现 ABBA 计时、产出比对文件、`runtime.json`、`capture.*.bin` 并响应 `WEAVEPROF_NSYS_CAPTURE=1` 的 `bench_cmd`。
- **Megakernel 路径的数据来源边界**：该路径的 `assembly`、`overhead` 与可选 `output` 回退项直接读取 `runtime.json`；需要由工具独立驱动 `cuobjdump` 做全量指令序列比对时，请改用 `wp.profile` / `wp.verify` 或分阶段命令 `validate`。
- **软硬件时间对齐策略差异**：`megakernel` 路径下软件时间线与 `nsys` 硬件曲线各自以核函数起点为零点叠放，`±5%` 跨度与 `≥ 90%` 采样密度检查只给提醒；如果需要严格的跨时钟偏移计算与尾部超差拦截，请使用 `weaveprof/pipeline/validate/tools/fi_align_nsys.py`。
- **H20 面板 (A) 与因果校准差异**：当前 `overview.png` 的面板 (A) 直接画 `nsys` 原始读数（H20 满载约 `25%`），而 `causal.py` 的分析结果已自动乘以 `4`。
- **HMAC `seal.json` 的本地权限边界**：`seal.json` 能校验产物在生成后是否被改动过，但签名密钥 `.supervisor.key` 存在当前用户自己的本地目录（`0o600`）下，因此防不住同一操作系统用户绕过 Agent 会话直接操作本地文件。
- **GPU 计数器权限与架构限制**：NVIDIA 遇到 `ERR_NVGPUCTRPERM` 需先开启计数器访问权限，缺失硬件曲线时默认拒绝生成纯软件 PNG；AMD MI308X `rocprof-sys` 仅提供按 dispatch 聚合计数器（无 `200 kHz` 周期采样），且 `grid >= 80 CU` 持久核禁止采集 ROCm PMC。
- **Python 运行依赖需按需安装**：`pyproject.toml` 尚未声明 `dependencies`，请按需手动安装 `numpy`、`matplotlib` 等运行包。
- **离线测试范围与跳过条件**：离线套件入口为 [tests/run_all.sh](tests/run_all.sh)（包含 `40` 项单元测试与 `__pycache__` 检查，要求 Python `>= 3.10` 及 GNU `find`/`date`，缺少 `numpy`/`torch`/`triton`/`hipcc` 时对应条目会按条件跳过或只做静态检查）；真卡 GPU 测试需在对应硬件环境单独执行。

---

## 🗂️ 仓库结构 / 📚 文档索引

```text
WeaveProfiler/
├── run.py                  # 统一 CLI（18 个子命令：scan/instrument/capture/validate/deliver/install-agent 等）
├── SKILL.md                # AI 编程代理使用规范与全参数参考
├── plugin.json             # 代理插件声明与 6 个插件工具定义
├── pyproject.toml          # 包元数据与 `weaveprof` 入口定义
├── weaveprof/              # 核心引擎（megakernel.py, harness.py, causal.py, viz.py, api.py, backend.py, mcp_server.py, guard_hook.py, pipeline/）
├── recorders/              # 跨语言设备端记录器（wp_lite.cuh, wp_lite_hip.h, wp_triton*.py, wp_cute.py）
├── configs/                # 卡型规格（cards/）、语言模式（languages/）、指标归一（metrics/）、语义表与配方（recipes/）
├── examples/               # 双角色 demo、MI308X Triton 示例、多语言参考核与主机脚手架
├── docs/                   # 设计文档、硬件标定记录、案例分析与四面板示例图（images/）
└── tests/                  # 离线测试套件（run_all.sh）与硬件回归用例
```

- **架构与指南**
  - [系统设计与原型记录 (`docs/design.md`)](docs/design.md) — 记录器架构、SM90 原型数据与对齐规范说明
  - [代理技能全手册 (`SKILL.md`)](SKILL.md) — 完整工作流规则、参数配置与排障手册
  - [离线测试入口 (`tests/run_all.sh`)](tests/run_all.sh) — `40` 项离线单元测试与无 `__pycache__` 残留检查脚本
- **硬件标定与采样专题**
  - [AMD MI308X `gfx942` 实测指南 (`docs/amd_mi308x.md`)](docs/amd_mi308x.md) — CDNA3 时钟、`rocprof-sys` 聚合指标与持久核约束
  - [`nsys --capture-range` 采样窗口专题 (`docs/nsys_pm_sampling.md`)](docs/nsys_pm_sampling.md) — 解决 `200 kHz` 缓冲区刷盘丢窗问题
  - [L20D `sm_103` 可达峰值重定标 (`docs/l20d_achievable_peak_rebase.md`)](docs/l20d_achievable_peak_rebase.md) — B300 衍生 SKU 实测带宽与算力基准
  - [慢路径与异常排查 (`docs/slow-path.md`)](docs/slow-path.md) — 慢路径诊断与边界场景记录
- **案例分析与示例**
  - [8×H20 DeepGEMM MegaMoE 优化 (`docs/case_studies/h20_megamoe_optimization.md`)](docs/case_studies/h20_megamoe_optimization.md) — `1.47×` 加速诊断与改造细节
  - [RTX 3050 `sm_86` WDDM 抢占分析 (`docs/case_studies/sm86_wddm_preemption.md`)](docs/case_studies/sm86_wddm_preemption.md) — 冷启动带宽爬坡与桌面抢占排障
  - [MI308X Triton 示例说明 (`examples/triton/card-mi308x/README_card-mi308x.md`)](examples/triton/card-mi308x/README_card-mi308x.md) — MI308X `2048³` FP16 GEMM 采样对照实操

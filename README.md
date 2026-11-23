# WeaveProfiler

简体中文 | [English](README_EN.md)

> **让 AI 看清 GPU 核内每一微秒在跑什么——面向 RSI（Recursive Self-Improvement）与 Megakernel 开发的自动核内 Profiler。**

WeaveProfiler 把**自动低开销核内插桩**和**同次 200 kHz 硬件利用率曲线**对齐到同一张**四联 PNG 图**里，以 **`Skill + MCP + Hook`** 形态接入 Claude Code、Qwen Code、OpenAI Codex CLI、Antigravity 等 AI 编程代理。支持 CUDA C++、HIP C++、CuTeDSL 和 Triton，在隔离临时副本中完成插桩与验证，不改生产源码、不改函数签名。

---

## 1. 为什么 RSI 需要 WeaveProfiler

在 **RSI（Recursive Self-Improvement，递归自我改进）** 流程里，AI 代理需要持续“写 kernel → 定位瓶颈 → 改写微架构 → 复测验证”，自主演化高性能 **Megakernel**（MegaMoE、FlashAttention、Persistent GEMM 等）。只看一个总耗时，AI 只能盲猜；要让 RSI 真正闭环，必须有既能看清核内代码语义、又能全自动运行的 profiler。

以往的方案总在两边二选一：

- **厂商自动工具（`nsys` / `rocprof-sys` 等）**：自动好用，但只能看到全卡累加的硬件利用率曲线，不知道掉谷那一微秒卡在哪个 SM、哪个 Warp、哪段异步搬运或等待。（`ncu` 靠多次重放算整核平均，没有时间轴，持久核与多卡核基本跑不动。）
- **手动核内插桩**：能对应到具体代码阶段，但写起来复杂易错，稍不注意就触发寄存器 spill 或打乱调度把耗时测歪，异步流水线（`cp.async` / TMA / `wgmma` / `tcgen05`）更难配对，无法放进自动优化循环。

**WeaveProfiler 把两者合二为一：**

1. **自动核内插桩 + 同次硬件曲线对齐**：在隔离副本里注入低开销探针（热路径零原子、不改签名；对 NVIDIA Triton 还支持 PTX 级自动插桩，零源码改动剥离 `cp.async` / `wgmma` 在途深度与暴露等待），与同一次运行的 200 kHz 硬件利用率曲线刚性对齐，直出一张**四联 PNG 图**。
2. **四项确定性检查保真**：反汇编零 spill、输出逐字节一致、事件完整、稳态开销 `≤ 10%`；任一项不过自动外提粗化重测，避免拿测歪的数据误导优化。
3. **`Skill + MCP + Hook` 融入 RSI 框架**：直接复用当前宿主 AI 代理读代码、定探针、看图定位瓶颈，本地 MCP 负责隔离实测与跨轮对比，Hook 守住“先测后改、不污染原码、必交 PNG”——方便接入主流 RSI 框架，是开发复杂 Megakernel 的利器。

---

## 2. 四联对齐图读法

默认直接输出静态 PNG（`overview.png`），不落盘大体积 `.perfetto.json.gz`。前 3 栏共享同一微秒（`µs`）横轴，当全核事件过密（相邻间距 `< 12 px`）时自动追加第 4 栏：

- **(A) 硬件利用率曲线**：同次采集的整卡指标（0–100%，NVIDIA 为 200 kHz `nsys`，AMD 为 `rocprof-sys`），含计算流水线（Tensor / MFMA）、HBM 读+写总带宽、独立绘制的全双工互联（`NVLink TX` 与 `NVLink RX` 绝不合并；AMD 为 `xGMI`）、可选跨机 RDMA 以及活跃 SM/CU 占比。
- **(B) 全卡物理 SM / CU 瀑布**：纵轴列出全部物理核心，直观暴露波次台阶、尾波拖尾与负载倾斜。
- **(C) 单个 SM / CU 的全物理 Warp / Wave 时间线**：每个 Warp 分为「**上层主执行轨（MMA / 亮黄斜纹 `Exposed Wait` / Store）+ 下层按需 0/1/2 条极薄异步引擎副轨（Mem 绿带 + 洋红发射针 `▼`；MMA 蓝带 + 深紫发射针 `▼`）**」，用浅/中/深三档热力带压缩不同 `num_stages` 并发深度。
- **(D) 单 CTA 微距放大（Zoom-In）**：全核过密时自动触发，确定性选取 `Exposed Wait` 占比最高的完整 CTA（无等待时取稳态中位数 CTA；超 48 拍截最重气泡周围 32 拍，支持 `--zoom-cta` / `--zoom-window` 手动指定），逐拍展示发射针、在途深度爬坡/排空与等待气泡；(A)–(C) 栏同步用洋红虚线框标出放大区间。

---

## 3. 工作原理与核心设计

```text
受保护源码 (只读) ──► 宿主 AI 代理 (按 Skill 读源码，决策各阶段探针 / 隔离副本改动)
                            │
                            ▼  (调用本地 MCP 验证工具)
                 本地验证器 (隔离副本插桩 → 编译基线/插桩 → 实卡运行 → 四项检查)
                   ├── 任一项失败 ──► 返回失败诊断与热点摘要给代理，外提/粗化后重测
                   └── 四项全通过 ──► 对齐同次硬件曲线 → 渲染四联 PNG → 签发封签并解锁优化
```

- **直接复用宿主 AI 代理**：默认 `Skill + MCP + Hook` 模式下，由当前运行的 AI 代理读源码、决策插桩位置并解读四联图，无需单独再配一套 LLM API Key。（若脱离宿主代理走无头脚本 `wp.identify` / `weaveprof_profile`，可配置兼容 OpenAI 的模型端点，或直接手写 `probe.json` 离线调用本地验证器。）
- **低扰动核内记录器**：
  - 按 `CTA × 物理 Warp/Wave` 预切私有缓冲，每 Warp/Wave 仅 `lane 0` 顺序写 16 字节流式记录（NVIDIA `st.global.cs.v4.u32`、AMD `global_store_dwordx4 ... nt`），**热路径零原子**。
  - CUDA C++ 默认单 32-bit 寄存器上下文（`0 B` 共享内存，上限 `32,767` 条/泳道），若触发 spill 自动回退静态共享内存或非内联函数；HIP C++ 把游标缓存在每条 Wave 的固定槽位中，`__device__` 辅助函数无需改参数即可续写。
  - 支持全卡墙钟（`%globaltimer` / `s_memrealtime`）与双锚点周期计数（`%clock64` / `s_memtime`），支持相邻边界合并。
- **不改函数签名接入**：CUDA/HIP 走编译单元内部设备符号绑定（`cudaMemcpyToSymbol` / `hipMemcpyToSymbol`）；Triton/CuTeDSL 走同名透明代理（自动去除插桩副本 `@triton.autotune` 并固定基线最优配置）；NVIDIA Triton 额外提供 PTX 级自动异步插桩（`triton_ptx_async.py`）。
- **四项阻塞检查**：① **反汇编与资源**（核心指令保序/多重集保真，新增寄存器 spill `≤ 0 B`，新增 SMEM `≤ 0 B`）；② **计算输出**（插桩与基线输出逐字节一致）；③ **事件完整与语义覆盖**（起止配对无溢出、默认覆盖 CTA 全部物理 Warp/Wave、拒绝粗块吞并）；④ **稳态开销**（热身后中位数变化 `|Δt| ≤ 10%`，正负超限均拒绝）。
- **工作区外防误操作护栏**：受保护源码备份、HMAC 封签与会话密钥存放在 `~/.cache/weaveprof/sessions/`；生命周期钩子（`PreToolUse` / `PostToolUse` / `Stop`）保证未过基线不可改源码、生产源码不残留探针、交付必含实测 PNG。

---

## 4. 实测案例（NVIDIA H20，200 kHz `nsys`）

### 4.1 8×H20 生产级 DeepGEMM MegaMoE 优化（`1.47×` 加速）

在 8×H20 NVLink 节点上剖析 **Qwen3.5-397B DeepGEMM MegaMoE**（`EP=8, S=1024, E=512`），对齐时间线定位出两处瓶颈：① `GEMM1`/`GEMM2` 拆核与静态专家分配造成约 `65 µs` 核间空等；② `GEMM2` 中同步 `Scatter` 阻塞计算组而 `Disp` 线程束闲置。融合为单核动态队列持久化算子并把 `Scatter` 异步卸载到 `Disp` 后取得 `1.47×` 加速（完整过程见 [h20_megamoe_optimization.md](docs/case_studies/h20_megamoe_optimization.md)）：

![8×H20 DeepGEMM MegaMoE 优化前后软硬对齐图对比](docs/images/case_h20_megamoe_1p47x.png)

| 版本 | 端到端耗时 | FP8 Tensor Core（均值 / 峰值） | HBM 带宽（均值 / 峰值） |
| --- | ---: | ---: | ---: |
| 原始分核版本（图 a） | `613.2 µs` | `27.9%` / `37.1%` | `36.4%` / `47.6%` |
| 优化后持久化融合版本（图 b） | **`416.4 µs`（`1.47×`）** | `41.8%` / `62.6%` | `55.7%` / `81.5%` |

### 4.2 开源算子库零签名修改剖析

不改动原始开源库函数签名与调用链，在 H20 上完成全自动闭环剖析（以下三张为历史实测软硬对齐图，单核事件较稀疏未触发 (D) 栏）：

| 开源算子 | 语言与规模（H20） | 物理 Warp 覆盖 | 稳态相对开销与结果 |
| --- | --- | :---: | --- |
| **FlashInfer 0.6.8 Hopper FA3**（跨 3 头文件） | CUDA C++，`B=5, H=16, S=512/4096, D=128` | 12 / 12 | `-1.62%`（调度微扰，在 `±10%` 容差内），软硬边界差 `0.33%` |
| **FlashInfer 0.6.8 Hopper FlashMLA**（`mla_hopper.cuh`） | CUDA C++，`B=64, H=128, KV=512/4096` | 8 / 8 | 第 1 轮 `+14.55%` 被拦截 → 自动外提粗化后 `+6.11%`，`0 B` spill |
| **vLLM FLA FlashKDA**（`chunk_delta_h.py`） | Triton 3.6，`B=1, T=512/8192, H=64, D=128` | 4 / 4 | 固定基线最优调优配置，自动合并边界后 `+6.74%`，`0 B` spill |

![FlashInfer 0.6.8 Hopper FA3 H20 软硬对齐图](docs/images/case_h20_flashinfer_fa3.png)
![FlashInfer 0.6.8 Hopper FlashMLA H20 软硬对齐图](docs/images/case_h20_flashmla.png)
![vLLM FLA FlashKDA H20 软硬对齐图](docs/images/case_h20_flashkda.png)

---

## 5. 快速上手与支持矩阵

| 维度 | 支持范围 |
| --- | --- |
| GPU 硬件 | NVIDIA H20（`sm_90` / `sm_90a`）、B300（`sm_100a` / `sm_103a`）、Ampere（`sm_86`）、AMD MI308X（`gfx942`） |
| 算子语言 | CUDA C++、HIP C++、CuTeDSL（NVIDIA）、Triton（NVIDIA PTX 与 AMD CDNA3） |
| AI 代理 | 原生集成 Claude Code、Qwen Code（`qwen`）、OpenAI Codex CLI、Antigravity（`agy`） |
| 依赖 | Python 3.10+、`nvcc`/`nsys` 或 `hipcc`/`rocprof-sys`、`matplotlib`；默认复用宿主代理，无需额外 API Key |

两步接入你的算子仓库（安装后 `--protect` 文件进入锁定，首次基线剖析通过后自动解锁优化）：

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git
python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_repo \
  --protect kernel.cu \
  --card h20   # 可选：h20 | b300 | mi308x | sm86
```

无交互批处理（Linux 下可选 `bubblewrap` 只读隔离挂载）与本地离线自检：

```bash
python3 -B WeaveProfiler/run.py guard-exec --workspace /path/to/your_kernel_repo \
  -- codex exec "剖析该核函数的瓶颈与跨 SM 负载不均"

# 无需 GPU 的离线验证
python3 -B WeaveProfiler/tests/package_structure.py
python3 -B -m unittest WeaveProfiler/tests/test_wp_megakernel.py WeaveProfiler/tests/test_mcp_hook_installer.py
```

**常用 MCP 工具与边界提示**：核心 MCP 工具包括 `weaveprof_megakernel_audit`（主入口：隔离实测 + 四项检查 + 四联图 + 封签）、`weaveprof_status`、`weaveprof_analyze`、`weaveprof_compare_rounds`、`weaveprof_verify_seal`。已知边界：CUDA 单寄存器模式每泳道上限 `32,767` 条记录；AMD `rocprof-sys` PM 计数器为按 dispatch 聚合粒度；更详细的 CLI、记录器编译开关与平台细则见 [SKILL.md](SKILL.md)。

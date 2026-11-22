# WeaveProfiler (`wp`)
简体中文 | [English](README_EN.md)

WeaveProfiler 是一套面向 GPU 核内（intra-kernel）低开销自动插桩与软硬时间线对齐工具，帮助开发者和 AI Coding Agent（Claude Code、Codex CLI 等）看清复杂 GPU kernel 内部每个 SM、每个 Warp 在每一微秒都在干什么，并直接与 `nsys` / `rocprof` 硬件利用率曲线对齐。

全程**不改动你的原始源码、不改动函数签名和上层调用代码**，本地开箱即用。

---

## 一、快速上手

### 支持范围

| 维度 | 支持内容 |
| --- | --- |
| 硬件 | NVIDIA H20（`sm_90a`）、NVIDIA B300（`sm_100a` / `sm_103a`）、AMD Instinct MI308X（`gfx942`） |
| 语言 | CUDA C++、HIP C++、CuTeDSL、Triton（NVIDIA 与 AMD） |
| Agent | 原生支持 Claude Code 与 OpenAI Codex CLI；其他命令行 Agent 可用 `guard-exec` 包装 |

### 两步安装

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git
python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_repo \
  --protect kernel.cu \
  --card h20
```

执行完上面这条命令后，工具会自动在你的算子目录里配好 MCP、Hook 和 Skill 说明。之后直接进入目录运行 `claude` 或 `codex` 即可。

如果你需要无交互批处理运行，可以用包装命令：

```bash
python3 -B WeaveProfiler/run.py guard-exec \
  --workspace /path/to/your_kernel_repo \
  -- codex exec "分析这个 kernel 的瓶颈和跨 SM 负载不均"
```

---

## 二、真实案例与 H20 实卡效果

### 2.1 论文核心案例：8×H20 生产级 DeepGEMM MegaMoE 诊断与优化（1.47× 加速）

在 8×NVIDIA H20（`sm_90a`，78 SM，NVLink 全互联）上剖析生产级 **Qwen3.5-397B DeepGEMM MegaMoE**（`EP=8, S=1024, E=512`），WeaveProfiler 的软硬同轴时间线直接定位出两个核心瓶颈，指导重构实现 **`1.47×` 端到端加速**（详细过程见 [docs/case_studies/h20_megamoe_optimization.md](docs/case_studies/h20_megamoe_optimization.md)）：

![8×H20 DeepGEMM MegaMoE 优化前后软硬同轴时间线对比](docs/images/case_h20_megamoe_1p47x.png)

| 版本 | 端到端耗时 | FP8 Tensor Core（均值 / 峰值） | HBM 带宽（均值 / 峰值） | 定位出的瓶颈与改动 |
| --- | ---: | ---: | ---: | --- |
| 原始分核 DeepGEMM（图 a） | `613.2 µs` | `27.9%` / `37.1%` | `36.4%` / `47.6%` | ① GEMM1 与 GEMM2 之间有约 65 µs 的核间空等；② GEMM2 里的同步 `Scatter` 卡住了计算 Warp |
| **优化后融合 Megakernel（图 b）** | **`416.4 µs`（`1.47×`）** | **`41.8%` / `62.6%`** | **`55.7%` / `81.5%`** | ① 融合成单核持久 kernel + 动态领任务（94.0% 活跃 SM）；② 异步 `Scatter` 挪给空闲的 `Disp` Warp |

### 2.2 开源算子库零改代码实测（直接本地输出软硬对齐图）

测试环境：NVIDIA H20（`sm_90a`，78 SM），输入为完全未改动的开源算子库源码，全程自动完成“识别打点位置 → 临时目录插桩编译 → 实卡测速与四项检查 → 必要时自动调整重测 → 同次 200.3 kHz `nsys` 对齐并直接出图”。

| 开源算子库与核函数 | 语言与测试规模（H20） | 物理 Warp 覆盖 | 实测结果与稳态开销 |
| --- | --- | :---: | --- |
| **FlashInfer 0.6.8 Hopper FA3**（`prefill_sm90.cuh` 等 3 个头文件） | CUDA C++，`B=5, H=16, S=512, D=128`（长序 `S=4096`） | **12 / 12**（`W00..W11`） | 0 签名改动；长序稳态开销 **`-1.62%`**，软硬边界差 `0.33%` |
| **FlashInfer 0.6.8 Hopper FlashMLA**（`mla_hopper.cuh`，DeepSeek-V3 页式 MLA） | CUDA C++，`B=64, H=128, KV=512`（长序 `KV=4096`） | **8 / 8**（`W00..W07`） | 第 1 轮过细打点（`+14.55%`）被自动拦下 → 第 2 轮自动收敛通过：**`+8.25%`**（`KV=512`）/ **`+6.11%`**（`KV=4096`），0 B 溢出 |
| **vLLM FLA FlashKDA**（`chunk_delta_h.py`，线性注意力递推） | Triton 3.6，`B=1, T=512, H=64, D=128`（长序 `T=8192`） | **4 / 4**（`W00..W03`） | 锁定原 `@autotune` 最优配置；自动外提与合并后通过：**`+6.74%`**（`T=512`）/ **`+8.14%`**（`T=8192`），0 B 溢出 |
| **DeepGEMM**（`8×H20` 生产级 MegaMoE，Qwen3.5-397B） | CUDA C++，`EP=8, S=1024, E=512` | 全 Warp 角色 | 端到端 **`613.2 → 416.4 µs`（`1.47×`）** |

以下三张图均为 WeaveProfiler 在 H20 上直接本地渲染输出的原始三栏软硬对齐图（`overview.png`）：

- **FlashInfer 0.6.8 Hopper FA3 Prefill**（CUDA C++ 跨 3 个头文件，`B=5, H=16, S=512, D=128`，全 12 个物理 Warp：`W00` TMA 预取、`W01..W03` 空闲、`W04..W11` 双计算组）：

![FlashInfer 0.6.8 Hopper FA3 H20 软硬对齐图](docs/images/case_h20_flashinfer_fa3.png)

- **FlashInfer 0.6.8 Hopper FlashMLA**（CUDA C++ `mla_hopper.cuh`，`B=64, H=128, KV=512`，全 8 个物理 Warp，清晰呈现 128 个块在 78 个 SM 上的双波次台阶）：

![FlashInfer 0.6.8 Hopper FlashMLA H20 软硬对齐图](docs/images/case_h20_flashmla.png)

- **vLLM FLA FlashKDA**（Triton 3.6 `chunk_delta_h.py`，`B=1, T=512, H=64, D=128`，全 4 个物理 Warp，清晰分解加载、8 轮分块矩阵乘与状态衰减更新、以及写回）：

![vLLM FLA FlashKDA H20 软硬对齐图](docs/images/case_h20_flashkda.png)

---

## 三、核心设计

WeaveProfiler 围绕**两个核心设计点**和**一层工程实现**构建：

### 3.1 设计点一：怎么保证自动插桩不出错（大模型看懂代码 + 固定程序插桩验收）

- **大模型只负责看代码，固定程序负责插计时语句**：大模型不会直接改你的源码，它只读源码并对照跨语言示例库，输出一份 JSON 清单（`probe.json`，记录每个 Warp 角色的判断条件以及各阶段的起止行号）。真正往代码里插计时调用的工作，全部由固定的 Python 程序在独立临时目录复制出来的源码副本里完成，保证语法稳定、结果可复现，且绝不碰原文件。
- **防篡改与防作弊机制（MCP 服务 + HMAC 签名 + Hook + 只读隔离）**：
  - **独立 MCP 服务与防篡改签名**：插桩、编译、实测和出图运行在独立的 MCP 服务进程中。只有全部检查通过时，服务才会用对 Agent 不可见的私钥，对**原始源码哈希、`probe.json` 哈希、实测时间线文件哈希与四项检查结果**联合计算 HMAC-SHA256 签名并生成 `seal.json`；后续流程和结束任务时都会重新核验哈希与签名，防止擅自篡改结果。
  - **Hook 防护（`PreToolUse / PostToolUse / Stop`）**：自动拦住 Agent 绕过 `probe.json` 直接往生产源码里硬塞计时代码、或篡改验收脚本与签名文件；任务结束前必须确认当前源码有合法通过记录且不含任何残留计时语句。
  - **只读权限隔离**：验证脚本、编译器配置与签名私钥在操作系统层面与工作区隔离（只读挂载），避免被常规命令改写。
- **四项自动验收，不通过就自动重来**：
  1. **反汇编检查**：排除打点指令本身和寄存器编号差异后，核对核心计算与访存指令的主要顺序和主循环展开结构没变，且没有新增寄存器溢出；
  2. **计算结果检查**：核对插桩后的数值输出与原算子在误差范围内一致；
  3. **事件完整性检查**：核对每个阶段的起止打点成对闭合（`t_end >= t_begin`）、缓冲区无丢包、拦截空壳打点与粗块折叠，且覆盖 CTA 内全部物理 Warp；
  4. **运行开销检查**：与原算子热态交替测速，要求端到端稳态耗时增量不超过上限（默认 `10%`）。
  - 只要任意一项不通过，验证程序就拒绝交付，把具体报错（`feedback.json`）**连同这一轮采到的时间线文件一起**返回给大模型；大模型对照源码和时间线自动修改 `probe.json`（比如发现高频内层循环打点导致开销超标，就自动提到外层循环边界），直到四项全部通过。

### 3.2 设计点二：怎么保证测量准确（低扰动片内记录）

- **每 Warp 独立缓冲 + 绕过缓存写入**：按 `CTA × 物理 Warp` 预先分好各自独立的显存切片，由每个 Warp 的 `lane 0` 顺序追加 16 字节事件记录，不需要原子锁；写入使用绕过缓存的流式全局写指令（如 `st.global.cs.v4.u32`），避免挤掉原算子的数据缓存。
- **常量内存放配置 + 单寄存器存进度**：把缓冲区基址和容量放在只读常量内存里，热循环里只用 **1 个 32 位寄存器**打包保存“槽号、当前写位置、溢出标记”，且占 **0 B 共享内存**；如果遇到原算子寄存器已经彻底用满、多用 1 个寄存器都会溢出的极端情况，就自动改用 16 字节共享内存存写位置，并调用禁止内联的独立记录函数（`__noinline__`），让主循环完全不额外占用寄存器。
- **同步时钟 / 双锚点计时 + 相邻阶段合并**：支持直接读全卡同步纳秒时钟（`%globaltimer` / `s_memrealtime`），也支持只在 Warp 起止处记同步时钟锚点、循环内改读更轻量的周期计数器（`%clock64` / `s_memtime`）离线换算；上一阶段结束紧挨下一阶段开始时，自动合并成读一次时钟、连续写入前后两条记录，减少一半边界打点开销。

### 3.3 工程实现：不改函数签名接入与本地直接出图

- **不改动函数签名和调用栈**：
  - **CUDA C++（`recorders/cuda_cpp/wp_lite.cuh`）与 HIP C++（`recorders/hip_cpp/wp_lite_hip.h`）**：`__global__` 入口不新增任何参数，启动前通过符号拷贝（`MemcpyToSymbol`）或 `.cubin` 动态查找符号（`cuLibraryGetGlobal`）传入缓冲区地址；跨多个头文件的 `__device__` 子函数之间通过每物理 Warp 独立的进度槽交接写位置。
  - **Triton（`recorders/triton/wp_triton.py`、`wp_triton_amd.py`）与 CuTeDSL（`recorders/cutedsl/wp_cute.py`）**：在外层套一层同名透明代理，自动解开 `@heuristics` / `@autotune` 装饰器并固定原算子选出的最优配置，只在最底层启动核函数时隐式传入缓冲指针，外部调用代码一行不用改。
- **本地直接渲染三栏软硬对齐图（`overview.png`）**：完成同次软硬件时钟对齐后，直接通过 `weaveprof/viz.py` 在本地生成一张共享微秒时间轴的三栏 PNG 图（同时按需保留 Perfetto 轨迹与结构化摘要）：
  - **上栏 (A)**：同次 `nsys` / `rocprof` 采样的硬件利用率曲线（Tensor/Matrix Core、HBM 带宽、NVLink 带宽、活跃 SM 占比）；
  - **中栏 (B)**：全卡所有物理 SM（`SM0..SM{N-1}`）的逐分块多阶段执行瀑布图；
  - **下栏 (C)**：代表性 SM 内全部物理 Warp 角色的细粒度时间线。

---

## 四、MCP 工具与 Hook 规则

### MCP 工具（`weaveprof/mcp_server.py`）

安装后会自动向 Claude Code 和 Codex CLI 注册 7 个 MCP 工具：

| 工具 | 作用 |
| --- | --- |
| `weaveprof_status` | 查看当前受保护的源码文件、基线与实时哈希、已通过验证的记录数及当前最优耗时。 |
| `weaveprof_megakernel_audit` | 在临时目录复制源码编译插桩、交替测速、执行四项检查、分析跨 SM 负载倾斜、生成对齐图并签发 `seal.json`。 |
| `weaveprof_profile` | 从 0 自动识别打点位置并跑完闭环验证（生成 `probe.json` → 临时目录插桩实测 → 不通过自动修正 → 签发 `seal.json`）。 |
| `weaveprof_plan_context` | 根据源码与寄存器 / 共享内存余量，推荐片内记录器的上下文存放模式与编译选项。 |
| `weaveprof_analyze` | 对已对齐的软硬轨迹做瓶颈归因，输出算力 / 带宽摘要、跨 SM 偏斜与优化建议。 |
| `weaveprof_compare_rounds` | 对比上一轮与当前轮的剖析结果，判断优化是否有效、瓶颈是否消除以及是否需要回退。 |
| `weaveprof_verify_seal` | 用独立会话密钥核验 `seal.json` 签名与产物哈希。 |

### Hook 自动防护（`weaveprof/guard_hook.py`）

会话密钥和原始源码备份存放在工作区外的 `~/.cache/weaveprof/sessions/<workspace_hash>/`：

- **调用工具前（`PreToolUse`）**：禁止读取签名密钥与校验脚本；禁止在获得基线通过签名之前修改受保护的算子源码；即使解锁编辑后，也禁止直接往生产源码里写入 `wp_lite.cuh`、`WP_OPEN`、`WP_B(` 等插桩代码；禁止修改或删除签名文件与 Hook 配置。
- **调用工具后（`PostToolUse`）**：实时检查受保护源码哈希；若未获基线签名就被命令偷偷修改，自动从外部备份还原并报错；若源码中混入插桩代码立即拦截。
- **结束任务前（`Stop` / `SubagentStop`）**：确认当前受保护源码与某个已签名通过的版本字节一致，且无插桩代码残留。

---

## 五、目录结构与本地测试

```text
WeaveProfiler/
├── README.md              # 简体中文文档
├── README_EN.md           # English documentation
├── SKILL.md               # Agent Skill 入口说明（Claude Code / Codex CLI / Antigravity）
├── pyproject.toml         # Python 包配置与 weaveprof CLI 入口
├── plugin.json            # Agent 插件与工具清单
├── run.py                 # 统一命令行入口
├── recorders/             # 各语言核内记录器（wp_lite.cuh / wp_lite_hip.h / wp_cute.py / wp_triton.py / wp_triton_amd.py）
├── weaveprof/             # 核心后端、MCP 服务、Hook 防护、自动插桩、对齐与绘图模块
├── configs/               # H20/B300/MI308X 硬件配置、硬件指标定义、跨语言示例库与配方
├── docs/                  # 真实案例报告与各平台硬件采集说明
├── examples/              # 各语言最小插桩参考示例
└── tests/                 # 离线单元、结构契约与安全对抗测试
```

### 离线测试（无需 GPU）

```bash
python3 -B tests/package_structure.py
python3 -B -m unittest tests/test_mcp_hook_installer.py tests/test_wp_megakernel.py
bash tests/run_all.sh
```

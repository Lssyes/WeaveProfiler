# WeaveProfiler (`wp`)

面向 GPU 核内（intra-kernel）低扰动插桩、软硬时间线对齐与 Agent 自动化闭环优化的基础设施，专攻 **Sparse Attention / GDN / MoE 等持久 Megakernel**。

- **硬件支持**：NVIDIA H20（`sm_90a`）、NVIDIA B300（`sm_100a` / `sm_103a`）、AMD Instinct MI308X（`gfx942`）
- **前端语言**：CUDA C++ / HIP C++ / CuTeDSL / Triton
- **Agent 闭环**：开箱即用的 **Skill + MCP Server + 统一强门禁 Hook + 一键部署器**，原生适配 **Claude Code** 与 **OpenAI Codex CLI**；`guard-exec`（Linux 自动启用 `bwrap` 只读保护）支持任意 CLI Agent。

---

## 1. 新用户 2 行快速上手

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git
python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_repo \
  --protect kernel.cu \
  --card h20
```

部署完成后，进入你的算子目录直接运行 `claude` 或 `codex`（MCP、Hook、Skill 均已自动接线）；或用无交互批量监工运行：

```bash
# 交互式运行（自动启用 MCP 工具与硬门禁 Hook）
cd /path/to/your_kernel_repo && claude
cd /path/to/your_kernel_repo && codex

# 无交互批量监工（Linux 自动以 bwrap 只读保护 WeaveProfiler 与配置/密钥）
python3 -B WeaveProfiler/run.py guard-exec \
  --workspace /path/to/your_kernel_repo \
  -- codex exec "使用 weaveprof_megakernel_audit 剖析基线并消除跨 SM 慢尾偏斜"
```

### `install-agent` 自动生成的四层防护

| 防护层 | Claude Code | OpenAI Codex CLI | 核心保障 |
| :--- | :--- | :--- | :--- |
| **MCP Server** | `.mcp.json` + `.claude/settings.local.json` | `.codex/config.toml` | 零依赖 stdio，暴露 7 个 `weaveprof_*` 工具（状态、Megakernel 审计、闭环剖析、上下文规划、因果归因、回合对比、验签） |
| **强门禁 Hook** | `.claude/settings.json` | `.codex/hooks.json` + `hooks = true` | `PreToolUse` / `PostToolUse` / `Stop` 拦截未封签修改、按基线字节自动回滚 `Bash` 静默篡改、禁绝生产代码手工插桩、强制修改后重封才允许结束 |
| **Skill & 指南** | `.claude/skills/weaveprofiler/SKILL.md` + `CLAUDE.md` | `.agents/skills/weaveprofiler/SKILL.md` + `AGENTS.md` | 自动注入硬件不变量契约与证据驱动优化闭环规范 |
| **工作区外密钥与备份** | `~/.cache/weaveprof/sessions/<id>/.supervisor.key` | 同左（`0600` 权限） | HMAC 防伪密钥与基线字节备份位于工作区之外，Hook 阻断读取与伪造 `seal.json` |

---

## 2. 核心能力

### 2.1 一等 `wp` 零签名改动录制器（`recorders/`）
- **CUDA C++**（`recorders/cuda_cpp/wp_lite.cuh`）：
  - 上下文模式：`-DWP_CTX_1R`（1 个 32-bit 寄存器 + `0 B` SMEM）、`-DWP_CTX_1R_NOINLINE`、`-DWP_SMEM_ZREG`（`16 B` static `__shared__`，循环体 `0` 寄存器）；
  - 多文件 `.so` Megakernel（如 `DSA-learn/mk`）：加 `-DWP_LEVEL=2 -DWP_CTX_1R -DWP_DEFINE_HOST_API`，Python 侧用 `weaveprof.WpSoController(so_path)` 零样板驱动；
  - 运行时 `.cubin` JIT（如 DeepGEMM / `weave` `cudaLibrary_t`）：调用 `wp_attach_library(lib, &buf)`，保持 `STO_ENTRY == 1` 且零 `__global__` 参数改动。
- **CuTeDSL / Triton / HIP C++**：`recorders/cutedsl/wp_cute.py`、`recorders/triton/wp_triton.py`、`recorders/triton/wp_triton_amd.py`、`recorders/hip_cpp/wp_lite_hip.h`。
- **高速解码器**：`weaveprof/pipeline/validate/tools/wp_decode.py` 同时兼容 `WPLITE02` 与 `FILITE02` 格式。

### 2.2 极速 Megakernel 审计引擎（`weaveprof/megakernel.py`）
- **影子副本编译与运行**：所有插桩只在影子副本中进行，生产源码全程零改动；支持内容寻址 `L0` 编译缓存与并行 `L0` / `L_inst` 构建；
- **四道硬保真门禁 + HMAC 防伪封签（`seal.json`）**：
  1. 编译器 spill / `STL` / `LDL` 增量门禁（防寄存器数不变但悄悄 local-memory spill）；
  2. 逐字节 / 容差输出一致性门禁；
  3. 64 ns 空壳探针与溢出 / 配对完整性门禁；
  4. 对称 ABBA 热态开销扰动门禁（`|overhead| <= 10%`，同时拦截正向膨胀与负向编译器循环展开扰动）；
- **毫秒级跨 SM 偏斜诊断**：`analyze_megakernel_skew` 自动切分慢尾 SM 与早停 SM、识别 `grid_sync` 屏障饥饿；
- **200 kHz `nsys` SQLite 直接抽取**：`extract_nsys_sqlite_fast` 按 `typeId` 分段 O(log N) 二分定位，无需全表扫描。

### 2.3 D3 软硬因果归因与回合对比（`weaveprof/causal.py`）
- 结合实测 Roofline 峰速标定、软硬时间线对齐、反事实加速上界排序与回合 Diff（`weaveprof_compare_rounds`），给出明确的下一轮优化指令与回退判定。

---

## 3. 目录结构（11 个顶层条目）

```text
WeaveProfiler/
├── README.md              # 简体中文主文档（本文件）
├── SKILL.md               # Agent Skill 规范入口（Claude Code / Codex CLI / Antigravity）
├── pyproject.toml         # Python 包配置与 weaveprof CLI 入口
├── plugin.json            # Agent 插件与 MCP 工具声明
├── run.py                 # 统一 CLI 入口（install-agent / guard-exec / mcp-server / megakernel-audit / profile / ...）
│
├── recorders/             # 【核内插桩前端】零 __global__ 签名改动录制器（直接 #include 或 import）
│   ├── cuda_cpp/          #   wp_lite.cuh（CUDA C++ / .so / .cubin JIT，兼容 fi_lite.cuh）
│   ├── hip_cpp/           #   wp_lite_hip.h（AMD HIP C++）
│   ├── cutedsl/           #   wp_cute.py（CUTLASS CuTeDSL）
│   └── triton/            #   wp_triton.py / wp_triton_amd.py（NVIDIA & AMD Triton）
│
├── weaveprof/             # 【核心 Python 引擎、Agent 基础设施与内部流水线】
│   ├── mcp_server.py      #   零依赖 stdio MCP Server（暴露 7 个 weaveprof_* 工具）
│   ├── guard_hook.py      #   Claude Code & Codex CLI 统一强门禁 Hook（PreToolUse / PostToolUse / Stop）
│   ├── agent_installer.py #   一键工作区部署器（install-agent / guard-exec + Linux bwrap）
│   ├── megakernel.py      #   极速 Megakernel 审计、跨 SM 偏斜诊断、nsys SQLite O(log N) 抽取
│   ├── causal.py          #   D3 软硬因果归因、Roofline 标定、反事实加速上界
│   ├── harness.py         #   四道保真门禁、ABBA 协议与 HMAC 防伪封签（EpisodeSealer）
│   ├── api.py / cli.py / backend.py / contract.py / workspace.py / replay.py / aka_adapter.py
│   └── pipeline/          #   内部四阶段自动插桩与采集流水线（由 run.py / weaveprof 内部调度）
│       ├── core/          #     路由注册表、语义表、快速路径与测量执行器
│       ├── scan/          #     源码静态扫描、算子与门控探测、探针放置规划
│       ├── instrument/    #     自动插桩规则引擎与源码补丁生成
│       ├── capture/       #     nsys / rocprof 硬件采集与 Perfetto 软硬融合
│       └── validate/      #     WPLITE02 解码器（tools/wp_decode.py）、交付门禁与 MANIFEST 校验
│
├── configs/               # 【硬件卡档与分析配置】
│   ├── cards/             #   H20 (sm_90)、B300 (sm_100a/sm_103a)、MI308X (gfx942) 实测峰速与架构参数
│   ├── metrics/           #   nsys / rocprof 硬件计数器与采样配置
│   ├── languages/         #   CUDA / HIP / CuTeDSL / Triton 静态匹配规则
│   ├── operators/         #   Attention / GEMM / MoE / GDN 算子语义库
│   ├── recipes/           #   卡型 × 语言 默认插桩与采集配方
│   ├── templates/         #   GEMM / Megakernel 探针模板
│   ├── validation/        #   校验规则与 JSON Schema
│   └── semantic_ids.json  #   统一语义事件 ID 表
│
├── docs/                  # 【架构与平台文档】设计原理、慢路径说明、H20/B300/MI308X 硬件采样指南
├── examples/              # 【参考示例】各语言最小插桩示例（CUDA C++ / HIP / CuTeDSL / Triton / Host）
└── tests/                 # 【测试套件】离线单元、结构契约与对抗安全测试
```

---

## 4. NVIDIA H20（`sm_90a`，78 SMs）实卡验证数据

### 场景 1：GLM-5.3 SparseMLA Decode（CuTeDSL，78 SMs × 12 warps）

| 指标 | 实测结果 |
| :--- | :--- |
| `install-agent` 一键部署 | **320 ms** |
| `WPLITE02` 206k 记录读取 | **101.9 ms** |
| 200 kHz `nsys` SQLite 抽取 | **24.2 ms**（148 点，`sm_active = 99.4%`） |
| MCP 完整审计往返 | **4.31 s** |
| 输出一致性 | **14,303,232 B**，**0 mismatch** |
| 热态开销 / 核函数耗时 | **+4.996%**（`747.3 us`） |
| 64 ns 空壳探针拦截 | **已拒收** |

### 场景 2：多文件 CUDA `.so` Sparse TopK MQA 闭环优化

| 指标 | 实测结果 |
| :--- | :--- |
| 冷启 / 热启（`L0` 缓存命中） | **7.79 s** / **7.00 s** |
| V0 基线 | **689.0 us**（开销 **+1.55%**，`32 -> 32` 寄存器，`0 B` spill） |
| V0 跨 SM 偏斜诊断 | `50` 慢尾 SM vs `28` 早停 SM，`spread = 518.8 us` |
| V1 动态 Tile 队列优化后 | **495.0 us**（**1.392x 真实加速**，`spread` 压至 **83.3 us**，开销仅 **+0.064%**） |
| 强门禁 Hook 对抗验证 | 未封签 Edit / Bash 篡改自动回滚 / 手工插桩 / 未重封 Stop 全部拦截生效 |

### 场景 3：运行时 `.cubin` `cudaLibrary_t` JIT（`weave` 风格）

| 指标 | 实测结果 |
| :--- | :--- |
| 入口符号与签名约束 | **`STO_ENTRY == 1`**，零 `__global__` 参数改动 |
| 编译 + 加载 + 运行 + 解码 | **2.83 s** |
| 数值精度与完整性 | `max_ref_abs_err = 8e-6`；`15,840` 记录 / **`0` overflow** / **`0` 解码错误** |

---

## 5. 常用命令与离线验证

统一入口：`python3 -B WeaveProfiler/run.py <子命令>`

| 子命令 | 用途 |
| :--- | :--- |
| `install-agent` | 一键部署工作区四层防护（`--workspace` / `--protect` / `--card` / `--bench`） |
| `guard-exec` | 无交互批量监工运行任意 CLI Agent（Linux 自动启用 `bwrap` 只读保护） |
| `agent-status` | 查看当前工作区受保护文件 SHA、封签状态与门禁解锁情况 |
| `mcp-server` | 启动零依赖 stdio MCP Server（7 个 `weaveprof_*` 工具） |
| `megakernel-audit` | 执行影子副本 Megakernel 审计（四道门禁 + ABBA + 偏斜诊断 + HMAC 封签） |

在 `WeaveProfiler/` 根目录执行离线回归验证：

```bash
python3 -B tests/package_structure.py
python3 -B -m unittest tests/test_mcp_hook_installer.py tests/test_wp_megakernel.py
bash tests/run_all.sh
```

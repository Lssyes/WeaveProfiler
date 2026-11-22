# WeaveProfiler (`wp`)
[简体中文](README.md) | English

WeaveProfiler is a low-overhead intra-kernel GPU profiler and software–hardware timeline aligner built for both human engineers and AI coding agents (Claude Code, Codex CLI, Qwen Code, Antigravity, etc.). It reveals what every physical SM and every physical warp is doing microsecond by microsecond, aligned directly with same-run `nsys` / `rocprof` hardware utilization curves.

It runs directly on your local GPU machine **without modifying your original kernel source, function signatures, or caller code**.

---

## 1. Quickstart

### Supported Platforms & Frontends

| Dimension | Support |
| --- | --- |
| Hardware | NVIDIA H20 (`sm_90a`), NVIDIA B300 (`sm_100a` / `sm_103a`), AMD Instinct MI308X (`gfx942`) |
| Languages | CUDA C++, HIP C++, CuTeDSL, Triton (NVIDIA and AMD) |
| Agents | Native Claude Code, OpenAI Codex CLI, Qwen Code (`qwen`), and Antigravity (`agy`); any CLI agent via `guard-exec` |

### 2-Step Local Setup

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git
python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_repo \
  --protect kernel.cu \
  --card h20
```

This wires MCP, lifecycle Hooks, and the Skill for all 4 agents in one command (or pass `--agent claude|codex|qwen|agy`):
- **Claude Code**: `.mcp.json` + `.claude/settings*.json` + `CLAUDE.md`
- **Codex CLI**: `.codex/config.toml` + `.codex/hooks.json` + `AGENTS.md`
- **Qwen Code (`qwen`)**: `.qwen/settings.json` (MCP + native JSON-protocol `PreToolUse / PostToolUse / Stop / SubagentStop` hooks) + `.qwen/skills/weaveprofiler/SKILL.md` + `QWEN.md` (approve project hooks on first launch or trust the workspace so `/hooks` shows `active`)
- **Antigravity (`agy`)**: native workspace plugin at `.agents/plugins/weaveprofiler/` (`plugin.json`, `mcp_config.json`, `hooks.json` with `PreToolUse / PostToolUse / PostInvocation / Stop`, `rules/AGENTS.md`, `skills/weaveprofiler/SKILL.md`) + `GEMINI.md`

Then simply enter the directory and run `claude`, `codex`, `qwen`, or `agy`.

For non-interactive CLI runs:

```bash
python3 -B WeaveProfiler/run.py guard-exec \
  --workspace /path/to/your_kernel_repo \
  -- codex exec "Profile this kernel and diagnose cross-SM load imbalance"
```

---

## 2. Real-World Case Studies & H20 Silicon Results

### 2.1 Paper Case Study: 8×H20 Production DeepGEMM MegaMoE Optimization (`1.47×` Speedup)

On an 8×NVIDIA H20 system (`sm_90a`, 78 SMs, full-mesh NVLink) running production **Qwen3.5-397B DeepGEMM MegaMoE** (`EP=8, S=1024, E=512`), WeaveProfiler's aligned SW+HW timeline directly pinpointed two root bottlenecks and guided refactoring to a **`1.47×` end-to-end speedup** (full walkthrough in [docs/case_studies/h20_megamoe_optimization.md](docs/case_studies/h20_megamoe_optimization.md)):

![8×H20 DeepGEMM MegaMoE Before/After Co-Axial Timeline](docs/images/case_h20_megamoe_1p47x.png)

| Variant | Latency | FP8 Tensor Core (Avg / Peak) | HBM BW (Avg / Peak) | Root Cause & Optimization |
| --- | ---: | ---: | ---: | --- |
| Original split DeepGEMM (a) | `613.2 µs` | `27.9%` / `37.1%` | `36.4%` / `47.6%` | ① ~65 µs idle gap between GEMM1 & GEMM2 launches; ② Synchronous `Scatter` blocks GEMM2 compute warpgroups |
| **Optimized fused Megakernel (b)** | **`416.4 µs` (`1.47×`)** | **`41.8%` / `62.6%`** | **`55.7%` / `81.5%`** | ① Single-kernel fusion + dynamic expert queue (`94.0%` active SMs); ② Async `Scatter` offloaded to idle `Disp` warps |

### 2.2 Zero-Signature Profiling of Open-Source Libraries (Direct Local 3-Panel PNGs)

Measured on NVIDIA H20 (`sm_90a`, 78 SMs) starting from completely unmodified open-source library code (stage identification -> deterministic injection in an isolated temp copy -> compile & benchmark -> four checks -> automatic refinement on failure -> same-run 200.3 kHz `nsys` aligned PNG):

| Open-Source Library & Kernel | Frontend & Shape (H20) | Warp Coverage | Closed-Loop Result & Overhead |
| --- | --- | :---: | --- |
| **FlashInfer 0.6.8 Hopper FA3** (`prefill_sm90.cuh` + 2 mainloop headers) | CUDA C++, `B=5, H=16, S=512, D=128` (also `S=4096`) | **12 / 12** (`W00..W11`) | Zero signature edits; `-1.62%` steady-state overhead (`S=4096`), `0.33%` SW–HW boundary delta |
| **FlashInfer 0.6.8 Hopper FlashMLA** (`mla_hopper.cuh`, DeepSeek-V3 paged MLA) | CUDA C++, `B=64, H=128, KV=512` (also `KV=4096`) | **8 / 8** (`W00..W07`) | Round 1 over-fine probes (`+14.55%`) auto-rejected -> Round 2 auto-pruned PASS: **`+8.25%`** (`KV=512`) / **`+6.11%`** (`KV=4096`), 0 B spill |
| **vLLM FLA FlashKDA** (`chunk_delta_h.py`, linear attention recurrence) | Triton 3.6, `B=1, T=512, H=64, D=128` (also `T=8192`) | **4 / 4** (`W00..W03`) | Pins baseline winning `@autotune` config; auto-hoist + boundary fusion PASS: **`+6.74%`** (`T=512`) / **`+8.14%`** (`T=8192`), 0 B spill |
| **DeepGEMM** (`8×H20` production MegaMoE, Qwen3.5-397B) | CUDA C++, `EP=8, S=1024, E=512` | All warp roles | End-to-end **`613.2 → 416.4 µs` (`1.47×`)** |

All three figures below are raw 3-panel PNGs (`overview.png`) rendered directly by WeaveProfiler on H20:

- **FlashInfer 0.6.8 Hopper FA3 Prefill** (CUDA C++ across 3 headers, `B=5, H=16, S=512, D=128`, all 12 physical warps):

![FlashInfer 0.6.8 Hopper FA3 H20 Overview](docs/images/case_h20_flashinfer_fa3.png)

- **FlashInfer 0.6.8 Hopper FlashMLA** (CUDA C++ `mla_hopper.cuh`, `B=64, H=128, KV=512`, all 8 physical warps, showing the `128 = 78 + 50` two-wave step):

![FlashInfer 0.6.8 Hopper FlashMLA H20 Overview](docs/images/case_h20_flashmla.png)

- **vLLM FLA FlashKDA** (Triton 3.6 `chunk_delta_h.py`, `B=1, T=512, H=64, D=128`, all 4 physical warps, breaking down load, 8 recurrence chunks, and store):

![vLLM FLA FlashKDA H20 Overview](docs/images/case_h20_flashkda.png)

---

## 3. Core Engineering Design

WeaveProfiler is built around **two core design pillars** and **one engineering implementation layer**:

### 3.1 Design Point 1: Ensuring Correctness (LLM Understanding + Deterministic Injection & Verification)

- **The LLM only reads code; deterministic Python inserts the probes**: The LLM never edits your kernel source directly. It reads the source against cross-language exemplars and outputs a structured `probe.json` (warp-role predicates and stage start/end line numbers). Fixed Python code copies your source into an isolated temp directory and deterministically injects the timing calls, keeping results reproducible and leaving original source files untouched.
- **Tamper-resistant execution (MCP Server + HMAC Seal + Hooks + Isolation)**:
  - **MCP Server & HMAC-SHA256 Seal**: Instrumentation, compilation, benchmarking, and rendering run inside a separate MCP server process; the MCP surface rejects forged `runtime_metrics` or pre-captured `dump_path` inputs. Only when all four checks pass does it compute an HMAC-SHA256 signature (`seal.json`) over **original source hash + artifact hashes + check results** using a session key stored outside the workspace (`0600`) and stripped from all build/benchmark child processes. Immutable session policy fields in `session.json` are likewise protected by `policy_hmac`, and verification fails closed.
  - **Lifecycle Hooks (`PreToolUse / PostToolUse / PostToolUseFailure / Stop / SubagentStop`)**: Automatically block editing protected production files before baseline sealing, inserting probe code, or modifying/deleting seals or hook configs (restoring exact backups on out-of-band shell mutations).
  - **Read-Only Isolation**: On Linux, `run.py guard-exec` mounts verification scripts, hook/MCP configs, and session keys read-only via `bwrap` when available.
- **Four Blocking Checks & Closed-Loop Refinement**:
  1. *Disassembly & resource check*: Compares compiled SASS/ISA to verify core compute/memory instruction multiset and structure are preserved (tolerating scheduling-only `ptxas -O3` instruction interleaving while rejecting dropped ops or unrolling collapse) with zero added register spills, stack frames, or `STL`/`LDL` explosions.
  2. *Output check*: Verifies **bitwise identical** outputs against the uninstrumented baseline.
  3. *Event completeness & semantic coverage check*: Verifies every begin/end pair is closed (`t_end >= t_begin`), zero drops/overflow, no hollow/collapsed stages or nested outer macro-wrappers, full per-CTA physical warp coverage, and operator-specific core stages (e.g., `load`, `mma`, `softmax`, `epilogue`/`store` on attention kernels).
  4. *Runtime overhead check*: Interleaved ABBA warm timing against the original kernel stays within the configured symmetric bound (default `±10%`).
  - Whenever any check fails, the verifier rejects delivery and returns `feedback.json` **together with failed-run timeline hotspots and semantic warnings** to the LLM, while `run_rsi_placement_search` enables parallel CPU compile pre-screening and persistent prior-guided convergence.

### 3.2 Design Point 2: Ensuring Measurement Accuracy (Low-Perturbation In-Kernel Recording)

- **Per-warp private buffers & cache-bypassing stores**: Each `(CTA, physical warp)` gets its own sequential memory slice written by `lane 0` in 16-byte records with **zero atomics on the hot path** (an atomic flag is used only on the cold overflow path), using cache-bypassing streaming stores (`st.global.cs.v4.u32`) so kernel data caches are not polluted.
- **Constant-memory descriptor & single-register cursor (CUDA C++)**: The CUDA recorder stores buffer base address and capacity in read-only constant memory while `-DWP_CTX_REG` packs `(slot, cursor, overflow)` into **a single 32-bit register** (`0 B` shared memory). On register-saturated kernels where even 1 extra register triggers spilling, the selector switches to `-DWP_CTX_SMEM` (16 B static shared memory + `__noinline__` helper, zero extra live registers) or `-DWP_CTX_REG_NOINLINE`; HIP C++ uses the standard register context.
- **Dual-anchor timing & adjacent boundary fusion**: Supports chip-wide synchronized clocks (`%globaltimer` / `s_memrealtime`) or open/close synchronized anchors plus lightweight cycle counters (`%clock64` / `s_memtime`) converted offline. Adjacent end+begin stage boundaries automatically merge into a single clock read + paired write.

### 3.3 Implementation: Zero Signature Changes & Direct Local 3-Panel PNGs

- **Zero function-signature or call-stack changes**:
  - **CUDA C++ (`recorders/cuda_cpp/wp_lite.cuh`) & HIP C++ (`recorders/hip_cpp/wp_lite_hip.h`)**: `__global__` parameter lists stay untouched; the host binds the buffer address via `cudaMemcpyToSymbol` / `hipMemcpyToSymbol` or dynamic CUDA `.cubin` symbol lookup (`cudaLibraryGetGlobal`), while nested `__device__` helpers across headers hand off the write position via per-warp progress slots.
  - **Triton (`recorders/triton/wp_triton.py`, `wp_triton_amd.py`) & CuTeDSL (`recorders/cutedsl/wp_cute.py`)**: A transparent same-name wrapper proxy passes the buffer pointer only at the lowest kernel launch site with zero caller edits (the Triton proxy also unwraps `@heuristics` / `@autotune` and pins the baseline winning configuration).
- **Direct local 3-panel SW+HW overview PNG (`overview.png`)**: After aligning software events with same-run `nsys` / `rocprof` hardware metrics on one shared microsecond axis, `weaveprof/viz.py` directly renders a compact 3-panel PNG locally (while optionally keeping Perfetto traces):
  - **Panel (A)**: Same-run hardware utilization curves (Tensor/Matrix Core, HBM BW, NVLink BW, Active SMs).
  - **Panel (B)**: All physical SMs (`SM0..SM{N-1}`) multi-stage execution waterfall.
  - **Panel (C)**: Representative SM per-physical-warp detailed timeline.

---

## 4. MCP Tools & Hook Rules

### MCP Tools (`weaveprof/mcp_server.py`)

`install-agent` registers 7 MCP tools for Claude Code, Codex CLI, Qwen Code, and Antigravity:

| Tool | Purpose |
| --- | --- |
| `weaveprof_status` | Show protected files, baseline/live SHA-256 hashes, valid verified seals, and current best latency. |
| `weaveprof_megakernel_audit` | Instrument in an isolated temp copy, benchmark, run all 4 checks, analyze cross-SM skew, render `overview.png`, and issue `seal.json`. |
| `weaveprof_profile` | End-to-end closed-loop profiling from scratch (`probe.json` -> temp-copy inject & verify -> auto-refine on failure -> `seal.json`). |
| `weaveprof_plan_context` | Recommend recorder state placement and compiler flags based on register/shared-memory headroom. |
| `weaveprof_analyze` | Run bottleneck attribution on an aligned trace, returning compute/bandwidth summaries, cross-SM skew, and ranked optimization ideas. |
| `weaveprof_compare_rounds` | Compare previous and current profiling rounds to verify whether bottlenecks were eliminated or regressed. |
| `weaveprof_verify_seal` | Verify the HMAC signature and bound artifact hashes of a `seal.json`. |

### Hook Protection (`weaveprof/guard_hook.py`)

Session keys and clean source backups live outside the repo under `~/.cache/weaveprof/sessions/<workspace_hash>/`:

- **`PreToolUse`**: Blocks reading keys/verifiers, blocks editing protected source files before a verified baseline seal exists, blocks writing probe code (`wp_lite.cuh`, `WP_OPEN`, `WP_B(`) into production files, and blocks altering/deleting seals or hook configs (including directory-level deletes/moves).
- **`PostToolUse` / `PostToolUseFailure`**: Checks live SHA-256 of protected files and installed hook/MCP configs; restores exact backups and blocks on unauthorized mutations or inserted probe macros.
- **`Stop` / `SubagentStop`**: `Stop` requires live protected source bytes to match a validly signed `PASS` version with zero probe code remaining; `SubagentStop` checks for unauthorized mutations, config tampering, and leftover probe code.

---

## 5. Repository Layout & Offline Tests

```text
WeaveProfiler/
├── README.md              # Simplified Chinese documentation
├── README_EN.md           # English documentation
├── SKILL.md               # Agent skill entry (Claude Code / Codex CLI / Qwen Code / Antigravity)
├── pyproject.toml         # Python package config and weaveprof CLI entry
├── plugin.json            # Agent plugin manifest
├── run.py                 # Unified CLI entry point
├── recorders/             # In-kernel recorders (wp_lite.cuh / wp_lite_hip.h / wp_cute.py / wp_triton.py / wp_triton_amd.py)
├── weaveprof/             # Core backend, MCP server, hooks, instrumentation, alignment, and plotting
├── configs/               # H20/B300/MI308X configs, hardware metrics, cross-language exemplars, and recipes
├── docs/                  # Case studies and platform profiling guides
├── examples/              # Minimal instrumentation examples per language
└── tests/                 # Offline unit, contract, and adversarial security tests
```

### Offline Verification (No GPU Required)

```bash
python3 -B tests/package_structure.py
python3 -B -m unittest tests/test_mcp_hook_installer.py tests/test_wp_megakernel.py
bash tests/run_all.sh
```

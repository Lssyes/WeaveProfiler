# WeaveProfiler (wp)
[简体中文](README.md) | English

## Overview & Quickstart

WeaveProfiler (`wp`) is a toolkit for low-overhead intra-kernel instrumentation on GPUs, software/hardware timeline alignment, and closed-loop operator optimization together with AI coding agents. It targets complex persistent kernels (Sparse Attention, GDN, MoE) as well as common CUDA/HIP/CuTeDSL/Triton operators.

Supported hardware: NVIDIA H20 (`sm_90a`), NVIDIA B300 (`sm_100a` / `sm_103a`), AMD Instinct MI308X (`gfx942`).

Supported frontends: CUDA C++, HIP C++, CuTeDSL, Triton (NVIDIA and AMD).

Supported agents: Claude Code and OpenAI Codex CLI natively; any command-line agent through `guard-exec` wrapping.

Quickstart (2 steps):

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git
python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_repo \
  --protect kernel.cu \
  --card h20
```

After entering the operator directory, run `claude` or `codex` directly — MCP, hooks and the skill are already configured. Alternatively, use the non-interactive wrapper:

```bash
python3 -B WeaveProfiler/run.py guard-exec \
  --workspace /path/to/your_kernel_repo \
  -- codex exec "Profile the baseline kernel and diagnose cross-SM load imbalance"
```

## Measured From-Zero-to-Instrumented Wall-Clock Time (H20)

The input is fully uninstrumented source. Starting from zero, the flow covers: locate instrumentation points -> rule-based injection generating shadow code -> compile -> on-device ABBA timing with byte-level output comparison -> four fidelity gates -> automatic loop re-hoisting and re-measurement if needed -> issue `seal.json` on pass.

**Triton 3.6 Sparse/Flash Attention (0 lines changed in external callers):**
- Pure deterministic rule path (0 LLM calls): first round identification 2 ms + injection 25.6 ms + JIT/run/decode 1.93 s = 1.96 s (first-round probe in the inner loop raised overhead by 19.9%, did not pass gates) -> automatic loop re-hoist 42.4 ms + JIT/run/decode 1.28 s -> **total 3.29 s to PASS** (overhead +4.95%, 0-byte diff, 0 overflow, 3,072 spans).
- LLM semantic identification path (`deepseek-v4.1-flash-1017`, `effort=low`): first-round identification 5.09 s + injection 40.0 ms + run 2.00 s + second-round automatic re-hoist 1.33 s -> **total about 8.5 s**.

**CUDA C++ multi-file `.so` Sparse TopK MQA (0 `__global__` signature changes, 0 host changes):**
- `deepseek-v4.1-flash-1017` (`effort=low`) + deterministic expansion + automatic loop re-hoist: first-round identification 10.01 s + expansion 21.8 ms + parallel `L0`/`L_inst` compilation and on-device run 8.71 s = 18.75 s (first round hit +20.1% inner-loop overhead and overflow gates) -> second round, 0 LLM, automatic re-hoist 30.4 ms + `L0` cache-hit incremental compile/run/seal 7.75 s -> **total 26.53 s to PASS seal** (overhead +5.70%, 0 spill, 0-byte diff, 0 overflow, `seal_verified=true`).
- Note: if first-round identification were set to a slow-thinking depth (for example `qwen4-max` with `reasoning_effort=max`), a single identification alone takes about 189 s. Therefore `identify_source` defaults to `low` for the first round (5–12 s) and only raises to `max` on failure-repair rounds.

**`ziyuhuang123/weave` 348-line `sm90_fp8_mqa_logits.cuh` (`CUTLASS_GLOBAL __launch_bounds__`):**
- From-zero identification plus deterministic rule expansion: `offline` heuristic 25 ms; `deepseek-v4.1-flash-1017` (`low`) 11.72 s (identified `load/compute/wait/mma/store` 5 stages, `zero_sig_change=true`); `.cubin` `cudaLibrary_t` compile-load-run-decode 2.83 s (`STO_ENTRY == 1`).

**Subsequent optimization iterations (second round onward, `L0` compile cache hit):**
- Triton approximately **1.3–2.0 s**; CUDA C++ `.so` approximately **7.0–7.8 s**.

## Core Engineering Design

### Register compression and low-perturbation in-kernel recording

`WP_CTX_1R` compresses the slot number (`slot`, high 16 bits), write cursor (`cur`, low 15 bits) and an overflow-disable flag (sign bit) into a single 32-bit register. The recording buffer metadata is placed in `__constant__ wp_fi_c` (constant cache, 0 B SMEM, no per-thread general-purpose register). `-DWP_CTX_1R_NOINLINE` moves the cold path into a `__noinline__` helper, reducing main-loop register pressure.

`WP_SMEM_ZREG` is intended for kernels that use `setmaxnreg` or are extremely register-constrained. It places the context in 16 B of static `__shared__` memory and pairs with a `__noinline__` writer function, so 0 additional registers stay live in the loop body.

For memory access and timing: 16-byte streaming store instructions `st.global.cs.v4.u32` avoid polluting L2 business data; support for `%globaltimer` (`-DWP_TIMER=0`) and for `%clock64` local clock with OPEN/CLOSE dual-anchor calibration (`-DWP_TIMER=1`); support for PTX predicate recording without control-flow divergence and merge of adjacent stage timestamps (`WP_TRANS_PRED`).

### Four-language support with zero call-stack changes

**CUDA C++** (`recorders/cuda_cpp/wp_lite.cuh`) and **HIP C++** (`recorders/hip_cpp/wp_lite_hip.h`):
- `__global__` entry points acquire no new parameters; they bind via `__constant__ wp_fi_c` / `__device__ wp_g` and host-side `wp_attach`.
- `__device__` sub-function call chains do not change signatures; `WP_KERNEL_SCOPE` + `WP_DEVICE_SCOPE` + `WP_SYNC_CURSOR_OUT/IN` synchronize the cursor through `wp_g.cur[slot]`, so the context does not need to be passed through every level.
- Multi-file `.so` supports `-DWP_DEFINE_HOST_API` to export `extern "C" wp_init/wp_reset/wp_dump/wp_free`; the Python side controls it through `weaveprof.WpSoController`.
- Runtime `.cubin` JIT (for example DeepGEMM / `weave`'s `cudaLibrary_t`) is bound via `wp_attach_library(lib, &buf)`, keeping `STO_ENTRY == 1` and adding no new `__global__` symbols.

**Triton (NVIDIA / AMD)** (`recorders/triton/wp_triton.py`, `wp_triton_amd.py`) and **CuTeDSL** (`recorders/cutedsl/wp_cute.py`):
- The rule engine replaces the original kernel name with a same-name transparent proxy (`_FiTritonProxy` / `_FiCuteProxy` / `_FiCuteDescriptor`, supporting module functions and class methods); external Python call sites `kernel[grid](...)` or `obj.method(...)` require 0 line changes.
- At runtime the recording buffer is injected through `wp_attach(buf)`; the Triton proxy preserves the uninstrumented `L0` `@triton.autotune` and directly reuses its winning configuration.

### Shadow copies and sandbox isolation

**Independent shadow copies**: all instrumentation, compilation and runs happen inside independent `L0` (uninstrumented) and `L_inst` (instrumented) copies outside the workspace; production source is never touched. The production source SHA-256 is re-checked before sealing.

**Content-addressed `L0` cache and parallel builds**: `L0` build artifacts are cached by source-tree hash and compiled in parallel with `L_inst`; re-measurement skips duplicate baseline builds.

**Linux `bwrap` read-only mounts and a 100 ms watchdog thread**: on Linux, `guard-exec` uses `bubblewrap` to mount the `WeaveProfiler` directory, `.claude/`, `.codex/`, `.agents/skills/weaveprofiler/`, `.mcp.json` and `.supervisor.key` read-only, and runs a 100 ms background watchdog thread that detects and rolls back unsigned tampering.

### Four fidelity gates and HMAC anti-forgery seal

- **Compiler resource and spill check**: besides register count, parse `nvcc`/`cuobjdump` `spill_stores`, `spill_loads`, `stack_bytes` and SASS `STL`/`LDL` deltas, blocking cases where the register count is unchanged but the compiler silently spills to local memory.
- **Output consistency check**: byte-level (or tolerance-specified) comparison of `L0` and `L_inst` output tensors.
- **Probe integrity and hollow-probe rejection**: requires 0 overflow, 0 pairing errors, and rejects hollow probes whose duration is close to the recorder's own latency (for example 64 ns).
- **Symmetric ABBA hot-state overhead gate**: requires alternating hot-state timing (ABBA) with `|overhead| <= 10%`, blocking both positive slowdown and negative distortion caused by instrumentation disrupting the compiler's loop unrolling or scheduling (`NEGATIVE_OVERHEAD_PERTURBATION`).
- **HMAC seal**: `seal.json` is issued using `~/.cache/weaveprof/sessions/<id>/.supervisor.key` outside the workspace, binding source, configuration, verification results and trace summary.

### Timeline visualization and optimization comparison

- **Remote 3-panel SW+HW overview PNG (`overview.png`)**: `run_megakernel_audit` and `weaveprof_megakernel_audit` automatically render a single-run 3-panel overview PNG directly on the GPU host via `weaveprof/viz.py` (~1.4 s, ~250–350 KB) on one locked microsecond axis, and do not emit `.perfetto.json.gz` by default (opt in with `write_perfetto_gz=True`):
  - **Panel (A)**: Same-run 200 kHz `nsys` hardware utilization curves (`Tensor Active`, `DRAM BW`, `SM Issue`, `Active SMs`) with boxed peak/mean KPI chips.
  - **Panel (B)**: All physical SMs (`SM0..SM{N-1}`) dense per-tile stage waterfall showing cross-SM tail skew.
  - **Panel (C)**: Representative SM (`SM 0`) per-warp-role (or concurrent CTA slot) timeline.
- **Compact `<2 KB` causal JSON & cross-SM imbalance**: Alongside `overview.png`, the audit returns `<2 KB` JSON (`stage_summary`, quartile hardware trajectories `hw_quartiles`, `megakernel_skew`, and ranked `causal_opportunities`) readable by both multimodal and text-only coding agents.
- **Before/after comparison**: Using the current card's measured compute throughput and bandwidth ceiling (H20 / B300 / MI308X), compare pre-change and post-change duration and stage-share changes, and decide whether the change is effective or should be reverted.

## MCP Server Reference

`weaveprof/mcp_server.py` is a zero-external-dependency stdio JSON-RPC 2.0 server (compatible with newline-delimited and `Content-Length` framing; at startup, fd 1 is redirected to stderr to prevent stray output from breaking the protocol; output directories and build/measure commands are validated internally for path and command legality).

It exposes 7 tools:

1. `weaveprof_status` — shows the current workspace's protected-file list, baseline SHA-256, live SHA-256, count of valid PASS seals, whether the Edit and Stop gates are unlocked, and the current best duration.
2. `weaveprof_megakernel_audit` — runs (or works from an already-collected dump) a single-file or multi-file kernel audit inside an independent shadow copy: compiles `L0`/`L_inst` in parallel (with `L0` cache), runs ABBA timing, passes the four fidelity gates, emits cross-SM load analysis and bottleneck hints, generates a Perfetto trace and an HMAC `seal.json`, and automatically registers valid seals to the current session.
3. `weaveprof_profile` — runs the from-zero automatic identification and closed-loop instrumentation profile (produce `probe.json` -> rule-inject into a shadow copy -> ABBA timing and gate validation -> automatic loop re-hoist on failure -> issue and register `seal.json`).
4. `weaveprof_plan_context` — given kernel source, architecture and register/shared-memory headroom, plans a low-overhead context mode (`WP_CTX_1R` vs `WP_SMEM_ZREG`) and compile-parameter recommendations.
5. `weaveprof_analyze` — attributes bottlenecks on an already-aligned software/hardware trace JSON and outputs a compute/bandwidth utilization summary, cross-SM skew statistics and yield-ordered optimization suggestions.
6. `weaveprof_compare_rounds` — compares the previous round's and current round's analysis report, identifies eliminated / residual / newly introduced bottlenecks, and gives continue-optimizing or revert recommendations.
7. `weaveprof_verify_seal` — verifies the HMAC of a specified `seal.json` using the out-of-workspace session key (HMAC + bound-artifact digest + overhead-limit check); on pass, registers it to the session.

## Hook Guard Reference

`install-agent` writes the hooks to both Claude Code (`.claude/settings.json`) and OpenAI Codex CLI (`.codex/hooks.json` plus `.codex/config.toml` with `hooks = true`). They follow a unified protocol: read JSON from stdin; allow with exit `0`; block with a clear reason on stderr and exit `2`.

All session state and keys live outside the workspace under `~/.cache/weaveprof/sessions/<workspace_hash>/` (containing `.supervisor.key`, `session.json`, and `backups/` with raw-byte baselines).

Three trigger phases:

**PreToolUse (before the tool call):**
- Block reads of or access to `.supervisor.key`, the session store directory, `guard_hook.py`, `mcp_server.py` and related environment variables.
- Block any command or argument that sets `WP_SAMPLE` / `FI_SAMPLE > 1`, block re-running `install-agent` / `guard-exec` to reset the baseline, and block modifications or deletions of `seal.json` / `verify.json` / `episode_ledger.json` and `.claude/` / `.codex/` / `.mcp.json` / `~/.claude/` / `~/.codex/` (this also covers `bench_cmd`, `build_cmd`, `out_dir` and similar fields passed through MCP tools).
- Before any HMAC-verified baseline `PASS` seal exists, block `Edit` / `Write` / `apply_patch` / `Bash` from modifying any protected operator file.
- Even after edit is unlocked, block insertion of `wp_lite.cuh`, `WP_OPEN`, `WP_B(`, `FI_OPEN` and similar probe code into production operator source; instrumentation may only occur in shadow copies.

**PostToolUse (after the tool call):**
- Verify the protected files' live SHA-256. If they were silently modified via `Bash` or scripts before a valid `PASS` seal exists, immediately restore original bytes from the out-of-workspace `backups/` and block with an error.
- Scan protected files for newly added probe macros and block on detection.

**Stop / SubagentStop (before ending a turn):**
- Require at least one HMAC-verified `PASS` seal in the session; the current bytes of each protected file must match some sealed `PASS` version (after code changes, the audit must be run again and a new seal issued, or the file must be reverted to a sealed version); files must not contain probe macros.
- A single-turn anti-infinite-loop counter is included: after three consecutive blocks in one stop sequence, the fourth is allowed and the counter is immediately zeroed, taking effect again on the next turn; the gate is not permanently disabled.

## Repository Layout

```text
WeaveProfiler/
├── README.md              # Simplified Chinese documentation
├── README_EN.md           # English documentation
├── SKILL.md               # Agent skill entry (Claude Code / Codex CLI / Antigravity)
├── pyproject.toml         # Python package config and weaveprof CLI entry
├── plugin.json            # Agent plugin and tool manifest
├── run.py                 # Unified command-line entry
├── recorders/             # In-kernel recorder frontends (cuda_cpp / hip_cpp / cutedsl / triton)
├── weaveprof/             # Core Python package (mcp_server / guard_hook / agent_installer / megakernel / causal / harness / api / cli + pipeline/{core,scan,instrument,capture,validate}/)
├── configs/               # Hardware tiers (H20/B300/MI308X), hardware metrics, language matching patterns, operator semantics, recipes and schemas
├── docs/                  # Design notes and per-platform hardware sampling guides
├── examples/              # Minimal instrumentation reference examples for each language
└── tests/                 # Offline unit tests, structural contract tests and adversarial security test suite
```

## Real-World Case Studies & H20 Silicon Verification

### 1. Paper Case Study: 8×H20 Production DeepGEMM MegaMoE Optimization (`1.47×` Speedup)

On an $8\times$NVIDIA H20 node (`sm_90a`, 78 SMs, full-mesh NVLink) running production **Qwen3.5-397B DeepGEMM MegaMoE** (`EP=8`, `S=1024`), WeaveProfiler's co-axial SW+HW timeline exposed two root bottlenecks and guided refactoring to a **`1.47×` end-to-end speedup** (`613.2 µs → 416.4 µs`, full walkthrough in [docs/case_studies/h20_megamoe_optimization.md](docs/case_studies/h20_megamoe_optimization.md)):

![8×H20 DeepGEMM MegaMoE Before/After Co-Axial Timeline](docs/images/case_h20_megamoe_1p47x.png)

| Variant | Latency | FP8 Tensor Core (Avg / Peak) | HBM BW (Avg / Peak) | Diagnosis & Optimization |
| --- | ---: | ---: | ---: | --- |
| Original split DeepGEMM (a) | `613.2 µs` | `27.9%` / `37.1%` | `36.4%` / `47.6%` | ① ~65 µs inter-kernel barrier bubble between GEMM1 & GEMM2; ② Synchronous `Scatter` blocks GEMM2 compute warpgroups |
| **Optimized fused Megakernel (b)** | **`416.4 µs` (`1.47×`)** | **`41.8%` / `62.6%`** | **`55.7%` / `81.5%`** | ① Single-kernel fusion + dynamic expert queue (`94.0%` active SMs); ② Async `Scatter` offloaded onto idle `Disp` warps |

### 2. Zero-Signature Instrumentation of Production Open-Source Libraries (FlashInfer Hopper FA3)

Without modifying any FlashInfer 0.6.8 Python wrappers, TVM-FFI bindings, or kernel signatures, WeaveProfiler instruments across three C++ template headers (`prefill_sm90.cuh`, `mainloop.cuh`, `mainloop_mma.cuh`), captures all 12 physical warps (`W00..W11`), and directly renders a same-run 200.3 kHz `nsys` 3-panel PNG:

![FlashInfer 0.6.8 Hopper FA3 H20 3-Panel Overview](docs/images/case_h20_flashinfer_fa3.png)

### 3. Additional H20 Silicon Verification Highlights

- **GLM-5.3 SparseMLA Decode** (CuTeDSL, 78 SMs × 12 warps, [full 78-SM figure](docs/images/case_h20_glm53_sparse_mla.png)): hot-state overhead **`+4.99%`** (`768.0 µs`, 6 hollow 64 ns stubs removed), 0-byte diff across 14.3 MB output, `1.15 µs` `nsys` alignment error — exposing 28/78 SMs idling `185.7 µs` (49.3%) at `grid_sync(5)`.
- **Qwen3.8-Next GDN Prefill SerialMK** (CUDA C++, 78 SMs × 16 warps, [full 78-SM figure](docs/images/case_h20_qwen38_gdn_prefill.png)): **`+0.026%`** end-to-end overhead (`223.29 → 223.35 ms`) at 128 regs / 232 KB SMEM, exposing a `4.0 ms` 28-SM straggler barrier wait in `P5 GDN`.
- **CUDA `.so` Sparse TopK MQA optimization loop**: V0 `689.0 µs` (overhead `+1.55%`, 32→32 registers, 0 spill, 50 vs 28 SM spread `518.8 µs`) → V1 dynamic tile queue `495.0 µs` (`1.392×` speedup, spread `83.3 µs`, overhead `+0.064%`).
- **Runtime `.cubin` `cudaLibrary_t` JIT (`weave`-style)**: `STO_ENTRY == 1`, zero `__global__` signature changes, compile+load+run+decode `2.83 s`, `max_ref_abs_err = 8e-6`, 15,840 records / 0 overflow / 0 decode errors.

## CLI & Offline Tests

```bash
python3 -B tests/package_structure.py
python3 -B -m unittest tests/test_mcp_hook_installer.py tests/test_wp_megakernel.py
bash tests/run_all.sh
```

# WeaveProfiler

[简体中文](README.md) | English

> **Give AI microsecond X-ray vision inside GPU kernels — the automated SW+HW intra-kernel diagnostic engine built for RSI (Recursive Self-Improvement) and next-generation Megakernels.**

WeaveProfiler fuses **automated low-overhead in-kernel instrumentation** with **same-run hardware utilization curves**, packaged natively as **`Skill + MCP + Hook`** for AI coding agents (Claude Code, Qwen Code, OpenAI Codex CLI, Antigravity, etc.). Across CUDA C++, HIP C++, CuTeDSL, and Triton—without touching protected source files or function signatures—it aligns 200 kHz chip-wide hardware curves, all-SM/CU waterfalls, per-warp async pipelines (in-flight memory/MMA depth strips, fixed-pixel issue pins `▼`, and exposed waits), and deterministic single-CTA micro-zooms into a single **four-panel PNG diagnostic view**.

---

## 1. Why RSI Needs WeaveProfiler

**Recursive Self-Improvement (RSI)** is shifting GPU kernel engineering toward autonomous loops where AI agents write, diagnose, and evolve complex fused **Megakernels** (MegaMoE, FlashAttention, persistent GEMM). End-to-end latency alone is blind: to autonomously break through microarchitectural ceilings, **every RSI loop needs a deep, semantically grounded diagnostic engine**.

Existing GPU profiling approaches force an impossible trade-off:

- **Automatic vendor profilers (`nsys`, `rocprof-sys`, etc.) → automated, but semantically blind.** They capture chip-wide aggregate hardware utilization (Compute / HBM / NVLink / xGMI), yet treat every kernel launch as an opaque box—showing a utilization trough without revealing which SM, warp role, async copy, or barrier caused it. (`ncu` peers inside kernels via multi-pass replay, discarding the microsecond time axis and breaking on persistent or multi-GPU collective megakernels.)
- **Manual in-kernel instrumentation → semantically rich, but complex, fragile, and impossible to automate.** Hand-writing timing code inside register- and SMEM-saturated inner loops easily triggers spills or scheduler distortions; pairing asynchronous hardware pipelines (`cp.async`, TMA, `wgmma`, `tcgen05`) by hand is error-prone and cannot scale inside an autonomous RSI loop.

**WeaveProfiler unifies both worlds into a native RSI building block:**

1. **Hand-crafted semantic fidelity × fully automated SW+HW alignment.** Automatically identifies stages and injects atomic-free, signature-preserving probes in an isolated temporary copy (including a PTX-level async pass for Triton that peels out compiler-hidden `cp.async`/`wgmma` waits), rigidly aligned with same-run 200 kHz hardware utilization curves in a **four-panel PNG diagnostic view**.
2. **Four deterministic blocking gates.** Disassembly/zero-spill, bitwise output equality, event completeness, and `<= 10%` steady-state overhead are enforced automatically—auto-coarsening hot probes on failure so corrupted measurements never mislead the next RSI round.
3. **Native `Skill + MCP + Hook` integration for RSI frameworks.** `Skill` teaches agents how to diagnose hardware–software bottlenecks from four-panel views; `MCP` exposes clean isolated profiling and cross-round comparison tools; `Hook` enforces clean source files, profile-before-edit discipline, and PNG-only delivery across every tool call—making WeaveProfiler the go-to diagnostic engine for autonomous Megakernel development.

---

## 2. System Architecture & Key Definitions

### 2.1 GPU Execution Hierarchy Across Platforms

Table 1 establishes the hardware and execution terminology used throughout this document and maps terms across NVIDIA and AMD GPUs.

| Concept | Term Used Here | NVIDIA Term | AMD Term | Definition |
| --- | --- | --- | --- | --- |
| Device program | kernel | CUDA `__global__` / Triton / CuTeDSL kernel | HIP `__global__` / AMD Triton kernel | A parallel GPU computation launched from the host. |
| Physical core | SM / CU | Streaming Multiprocessor (SM) | Compute Unit (CU) | Hardware core that schedules thread blocks and executes compute/memory instructions. |
| Thread block | thread block (CTA) | Cooperative Thread Array (CTA / Block) | Workgroup | Group of threads resident on one SM/CU sharing on-chip scratchpad memory (SMEM / LDS). |
| Hardware lockstep unit | warp / wave | Warp (32 threads) | Wavefront (64 threads on CDNA3) | Smallest hardware scheduling unit; the recorder defaults to physical warp/wave granularity. |
| Single thread in unit | lane | `lane 0..31` | `lane 0..63` | One execution lane within a warp/wave; by default only `lane 0` emits records. |
| Vendor profiler | hardware profiler | Nsight Systems (`nsys`) | `rocprof-sys` / `rocprofv3` | Vendor tool capturing kernel dispatch windows and hardware performance counters. |

### 2.2 Core System Terminology

The prose below uses consistent conceptual terms rather than raw source identifiers (Appendix F maps each concept to its implementation identifier):

- **Stage**: A semantically meaningful execution interval inside a kernel, such as tiled memory loading, matrix multiply-accumulate, online softmax rescaling, or epilogue writeback.
- **Probe**: A matched begin/end pair of in-kernel timing calls bracketing one stage.
- **Probe plan**: A read-only structured JSON specification produced by the stage identifier, declaring intra-CTA warp role predicates, descriptive stage labels, and begin/end source line numbers.
- **In-kernel recorder**: Lightweight device-side timing and buffer-append code injected into the temporary kernel copy, emitting fixed 16-byte event records (timestamp, tag/flags, auxiliary word).
- **Isolated temporary copy**: A temporary workspace copy of the kernel source created by the verifier; all probe injection, compilation, and measurement run on this copy without touching protected source files.
- **Software timeline**: Microsecond-resolution per-SM/CU, per-warp/wave stage intervals reconstructed from in-kernel recorder dumps.
- **Hardware timeline**: Chip-wide utilization curves (compute pipes, HBM bandwidth, intra-node/inter-node interconnect bandwidth, and active core fraction) collected by the vendor hardware profiler during the same run.
- **Three-panel aligned view**: A single PNG figure vertically aligning the hardware timeline, all-SM/CU waterfall, and representative-SM/CU per-warp/wave timeline on one shared microsecond axis.
- **Stage identifier**: Component that reads kernel source and emits a probe plan.
- **Verifier**: Separate deterministic component that injects probes, compiles baseline and instrumented binaries, benchmarks on GPU, executes the four checks, aligns timelines, renders the view, and signs seals.
- **Four checks**: The four blocking criteria enforced before delivery—assembly & resource preservation, output equality, event completeness & semantic coverage, and steady-state overhead.
- **Seal**: An HMAC-SHA256 signature computed by the verifier using an out-of-workspace session key over the protected source hash, artifact hashes, and passing check outcomes.
- **Protected file**: A kernel source file registered during session setup and governed by the file-locking state machine.
- **Supervisor**: The enforcement layer comprising the out-of-workspace session key, clean source backups, and lifecycle hooks that prevent bypassing profiling or leaving probes in production files.

### 2.3 System Components and Data Flow

WeaveProfiler coordinates five components: the **stage identifier**, **deterministic injection engine**, **verifier**, **timeline aligner & renderer**, and **supervisor**:

```text
Protected source (read-only) ──► Stage Identifier (external LLM API -> probe plan JSON)
                                        │
                                        ▼
                             Deterministic Injection (injects probes into isolated temp copy)
                                        │
                                        ▼
                             Verifier (compile baseline & instrumented -> GPU run -> 4 checks)
                               ├── Any check fails ──► return structured failure & hotspots -> refine
                               └── All 4 pass ───────► align HW curves -> render view -> sign seal
```

- **External LLM invocation disclosure**: Automated stage identification sends target kernel source excerpts and route patterns over HTTPS to an OpenAI-compatible LLM endpoint (default model `qwen4-max`; both endpoint and model are configurable via environment variables) and receives only a structured JSON probe plan. An API key is required for automated identification (Section 6). When a user supplies an existing probe plan, the verifier runs entirely locally with no network calls.
- **Deterministic injection and gating**: Code injection, compilation, disassembly comparison, output verification, event decoding, overhead gating, and seal signing are fixed Python code that an LLM cannot rewrite or bypass.

### 2.4 In-Kernel Recorder Design

The recorder targets register-bound Tensor Core and Matrix Core kernels with minimal measurement perturbation:

1. **Per-warp private buffers and streaming stores**: Device memory is pre-partitioned into disjoint per-`(CTA, physical warp/wave)` slices. Only `lane 0` of each warp/wave sequentially appends 16-byte records, eliminating **all atomic operations on the hot path** (an atomic flag is set only on the cold overflow path). On NVIDIA GPUs, records are written with 128-bit streaming-hint global stores (`st.global.cs.v4.u32`, hinting evict-first cache treatment to minimize data-cache interference); on AMD CDNA3, records use non-temporal 128-bit stores (`global_store_dwordx4 ... nt`).
2. **Recorder state placement and spill avoidance**: Each recording lane tracks its slot index, write cursor, and overflow state ("recorder context").
   - **CUDA C++ single-register mode (default)**: Buffer base address and capacity reside in read-only constant memory while the hot loop packs slot, 15-bit cursor, and overflow bit into **one 32-bit register** (`0 B` shared memory). Because the cursor is 15 bits wide, each warp lane holds at most **32,767 records** in this mode.
   - **CUDA C++ shared-memory and no-inline fallbacks**: When a kernel is at its hardware register limit and a single extra register triggers stack spilling, compiler-feedback selection falls back to static shared-memory context storage (`32 × 16 = 512 B` per CTA at default warp granularity, or `16 B` at per-CTA granularity) or a non-inlined helper function to keep zero extra live registers in the hot loop.
   - **HIP C++**: Uses a multi-field register struct context backed by per-lane cursor slots to continue recording across device helper calls.
3. **Dual clock modes and adjacent boundary fusion**: Supports chip-wide synchronized wall clocks (`%globaltimer` on NVIDIA, `s_memrealtime` on AMD) as well as dual-anchor mode—capturing paired wall-clock and local cycle counters (`%clock64` / `s_memtime`) at warp entry/exit while reading only the lighter local cycle counter inside loops for offline linear conversion. On NVIDIA CUDA C++ and single-role Triton, an end immediately followed by a begin merges into one clock read plus two contiguous 16-byte writes.
4. **Unchanged function signatures**: CUDA C++ and HIP C++ bind the buffer descriptor via a translation-unit-internal device symbol (`cudaMemcpyToSymbol` / `hipMemcpyToSymbol` or dynamic `.cubin` lookup), leaving `__global__` parameter lists and host callers untouched. Triton and CuTeDSL inject buffer arguments only inside a same-name transparent launcher proxy (stripping `@triton.autotune` from the instrumented copy to reuse the baseline winning configuration while preserving `@triton.heuristics`), requiring zero caller modifications.

### 2.5 Hardware Utilization and Bus-Physics Rules

Hardware and software timelines come from the **same physical run**, rigidly aligned via the kernel dispatch window without splicing across runs, and follow hardware bus physics strictly:

- **Compute utilization**: Reads Tensor Core / MMA pipe activity on NVIDIA; on AMD MI308X (`gfx942`, 4 XCC × 20 CU × 4 SIMD), derives Matrix Core utilization from same-run `rocprof-sys` counters via `100 × SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE × 80)`, kept separate from vector SIMD utilization (`SIMD_UTILIZATION`).
- **HBM bandwidth utilization (shared bus — summed)**: HBM reads and writes share one memory bus, so read and write throughput percentages are summed per timestamp (capped at `100%`) while retaining separate directional read/write series.
- **NVLink utilization (independent full-duplex paths — never merged)**: NVLink egress (`TX`) and ingress (`RX`) are **two independent physical pathways**. Request and response user data are summed **within each direction only** (`NVLink TX = TX Requests + TX Responses`; `NVLink RX = RX Requests + RX Responses`). **`TX` and `RX` are never summed or collapsed into one curve** and are always plotted as two separate lines in Panel (A).
- **Intra-node vs. inter-node separation**: Intra-node NVLink/xGMI counters measure single-node GPU-to-GPU traffic only and are never presented as cross-node RDMA/InfiniBand utilization. Inter-node traffic requires dedicated NIC hardware counters; when unavailable due to hardware counter bypass, the figure displays an explicit footnote rather than fabricating zeros.

---

## 3. End-to-End Workflow & Reading the View

### 3.1 Closed-Loop Workflow

1. **Create a protected session**: Run the installer specifying the workspace, target GPU card, and protected kernel files. The installer configures agent tools, lifecycle hooks, and skill docs, and stores clean backups plus a session key outside the workspace. Protected files start **locked**.
2. **Baseline profiling and initial seal**: The agent invokes profiling. The stage identifier reads the source and emits a probe plan; the verifier injects probes into an isolated temporary copy, compiles baseline and instrumented variants, runs on GPU, and evaluates the four checks. If any check fails (e.g., inner-loop overhead exceeds the limit), structured diagnostics and timeline hotspots return to the stage identifier for automatic hoisting or merging. Once all four checks pass, the verifier aligns hardware curves, renders the three-panel aligned view, and signs a baseline seal.
3. **Unlock and optimize**: After a valid baseline seal exists, protected files unlock so the agent can refactor the kernel guided by the timeline and bottleneck attribution.
4. **Re-profile and compare rounds**: Each optimization round is re-verified and sealed, then compared against the previous round to confirm speedup and bottleneck removal or trigger rollback.
5. **Pre-stop check**: Before the agent finishes, the supervisor checks that live protected source bytes match a validly signed passing seal and contain zero leftover probe statements.

### 3.2 Protected-File State Machine

```text
[install-agent --protect]
          │
          ▼
   (1) Locked ───────► Agent edits to protected files blocked; out-of-band shell mutations restored
          │
          │  First all-pass baseline run; verifier issues valid seal
          ▼
   (2) Unlocked ─────► Agent may edit protected files; writing probe headers/macros directly stays blocked
          │
          │  Re-verify after edits to issue a fresh seal
          ▼
   (3) Stop Check ───► Before exit: live source SHA-256 == a valid seal SHA-256 AND zero probe macros
```

### 3.3 How to Read the Four-Panel Aligned View

WeaveProfiler defaults to PNG-only visual delivery (`overview.png`) and blocks stray `.perfetto.json.gz` dumps unless explicitly requested. Panels (A)–(C) share **one locked microsecond (`µs`) horizontal axis**, and Panel (D) automatically adds a deterministic single-CTA micro-zoom whenever full-kernel event spacing is denser than `12 px`:

- **Panel (A) — Hardware Utilization Curves**: Same-run chip-wide metrics (0–100%, 200 kHz `nsys` on NVIDIA, `rocprof-sys` on AMD), showing compute pipeline activity (Tensor / Matrix Core), HBM Read+Write bandwidth, separate intra-node interconnect links (`NVLink TX` and `NVLink RX` on NVIDIA; `xGMI` on AMD), optional inter-node RDMA NIC utilization on multi-node runs, and active SM/CU percentage.
- **Panel (B) — All-SM / All-CU Execution Waterfall**: Every physical core on the vertical axis, displaying per-block stage spans, tail waves, cross-core load skew, and exposed wait clusters.
- **Panel (C) — Representative SM / CU Per-Warp / Per-Wave Timeline**: Each warp uses a **two-part lane: an upper main execution track (`MMA` / amber hatched `Exposed Wait` / `Store`) plus 0, 1, or 2 on-demand thin async engine sub-tracks (`MEM` emerald depth strip + magenta issue pins `▼`; `MMA` azure depth strip + violet issue pins `▼`)**, compressing arbitrary pipeline stage counts into a 3-shade depth band without exploding into N stage sub-lanes.
- **Panel (D) — Deterministic Single-CTA Micro-Zoom**: Auto-triggered on dense timelines; picks the complete CTA with highest `Exposed Wait` ratio (or steady-state p50 CTA when zero waits exist), caps long loops to a crisp 32-iteration window (`--zoom-cta` / `--zoom-window` overrides supported), and isolates that single CTA's issue pins, depth ramp/drain, and wait bubbles while highlighting the exact same window across Panels (A)–(C).

---

## 4. Deterministic Verification & Safeguards

### 4.1 Why Verification Runs Outside the Agent Workspace

When coding agents automate GPU profiling and tuning, failures are overwhelmingly practical engineering mistakes: editing kernels before measuring a baseline, leaving timing macros in production source, placing probes inside spin loops, or skipping real runs. WeaveProfiler keeps the verifier process, clean source backups, session state, and signing keys outside the workspace under `~/.cache/weaveprof/sessions/` and enforces simple lifecycle hooks (`PreToolUse` / `PostToolUse` / `Stop`) so source files stay clean and every completed run delivers a verified PNG timeline.

### 4.2 Four Blocking Checks

Every isolated-copy run must pass all four checks below before the verifier issues a passing seal:

| # | Check | Property Verified | Exact Criterion in Code | On Failure |
| --- | --- | --- | --- | --- |
| 1 | Assembly & resource check | Instrumentation preserves core compute/memory code generation with no added register spills or shared memory | Comparing baseline vs. instrumented SASS/ISA: `core_ratio >= 0.95` or (`core_multiset_ratio >= 0.98` and `core_ratio >= 0.85`), with overall sequence similarity `lcs_ratio >= 0.60`; `delta_spill_bytes <= 0`; `extra_smem_bytes <= 0`; no surge in local stack spill ops (`STL`/`LDL` or `scratch_*`). | Reject; recommend shared-memory or no-inline context mode when register-bound. |
| 2 | Output check | Instrumentation does not alter numerical results | Complete output buffers from baseline and instrumented runs on identical inputs are **bitwise identical**, non-empty, and distinct files. | Reject; report mismatch or missing output evidence. |
| 3 | Event completeness & semantic coverage check | Recorded events are intact, non-dropped, and semantically meaningful | Five independent sub-checks: (1) every begin/end pair closes properly (`t_end >= t_begin`, zero orphans or cross-nesting); (2) overflow bitfield is `0` (zero lost events); (3) every declared probe fires `> 0` times; (4) by default all physical warps/waves in each CTA are covered; (5) outer macro-wrappers enclosing inner stages, coarse collapses, and missing core operator stages (e.g., load/MMA/softmax/epilogue on attention) are rejected. | Reject; return missing-stage, stub, or nesting diagnostics. |
| 4 | Steady-state overhead check | Measurement does not distort steady-state execution time | After at least 1 warmup launch and at least 3 paired steady-state launches, median relative change satisfies `|t_inst / t_base - 1| <= 10%` (default limit `0.10`, non-relaxable via tool args). Slowdowns above `+10%` fail; **speedups beyond `-10%` also fail as compiler scheduling perturbations**. Interleaved ABBA ordering (baseline, instrumented, instrumented, baseline) is supported in replay mode to cancel clock drift. | Reject; return probe fire counts and enclosing loop lines for coarsening. |

*(Note: Full warp coverage and stage-granularity checks expose policy flags in the low-level Python API, enabled by default; the public MCP audit surface rejects caller-supplied policy overrides.)*

### 4.3 Failure Feedback and Prior-Guided Parallel Search

When any check fails, the verifier emits structured failure feedback (failed checks, per-probe fire counts, enclosing loop line spans, timeline hotspots, and semantic warnings) to guide automatic probe hoisting or merging. For compile-heavy C++ header libraries, the backend provides a prior-guided parallel search workflow: it places only the 1–2 modified headers into a minimal `-I` overlay directory, compiles candidate placements and context modes in parallel across CPU cores to screen out spills and disassembly regressions before touching the GPU, and persists outcomes as local priors to accelerate future rounds.

### 4.4 Seal Binding and Rejection of Caller-Supplied Results

The verifier accepts instructions to measure, never measurement results. Through the public MCP audit tool (`weaveprof_megakernel_audit`), an agent can submit only the kernel source and benchmark configuration; the verifier process itself performs instrumentation, compilation, GPU execution, and the four checks. This interface explicitly rejects five caller-supplied fields that could fabricate results:

- **Pre-recorded event dump path (`dump_path`)**: raw software-timeline data not produced by the verifier's own run;
- **Self-reported runtime metrics (`runtime_metrics`)**: caller-provided output-equality flags, timing arrays, or resource counts;
- **Custom event name maps (`names`)**, **custom profiler command overrides (`nsys_cmd`)**, and **verification policy overrides (`policy`)**.

Only when all four checks pass does the verifier compute an HMAC-SHA256 seal using the out-of-workspace session key over the protected source hash, artifact hashes, and check results; immutable session policy fields carry an independent policy HMAC, and keys are stripped from build/benchmark child process environments. The low-level Python API retains a pre-captured mode for offline re-rendering and testing, but lifecycle hooks reject pre-captured seals for unlocking protected files.

### 4.5 Lifecycle Hooks and Per-Agent Guarantee Matrix

On supported coding agents, the supervisor enforces rules at key tool lifecycle points (Table 2 explains each hook point; Table 3 summarizes support across the four agents):

| Hook Point | Plain-Language Meaning | Supervisor Action |
| --- | --- | --- |
| Before tool call | Intercepts file reads, edits, and shell commands before execution | Blocks reading session keys and hook/MCP guard scripts; blocks editing protected files before a valid baseline seal exists; after unlock, still blocks writing recorder headers or probe macros into production files; blocks modifying or deleting seals and hook configs. |
| After tool call | Runs immediately after a tool call completes | Re-hashes protected files and hook/MCP configs; restores exact out-of-workspace backups on unauthorized shell mutations or config tampering; blocks if probe code entered production source. |
| After failed tool call | Runs even when a tool/command exits with an error | Catches partial edits or config tampering left behind by failed shell commands and restores backups. |
| Before session stop | Runs when the main agent attempts to finish and report to the user | Requires live protected source bytes to match a validly signed passing seal with zero leftover probe code. |
| Before subagent stop | Runs when a spawned subagent completes its subtask | Checks for unauthorized mutations, config tampering, or leftover probe macros. |

| Agent | Integration | Registered Hook Points (Appendix C maps raw event names) | Extra Isolation |
| --- | --- | --- | --- |
| **Claude Code** | MCP + Hooks + Skill | Before tool, after tool, after failed tool, before stop, before subagent stop (5) | Optional Linux `guard-exec` (`bubblewrap` read-only mounts) |
| **Qwen Code (`qwen`)** | MCP + native JSON hooks + Skill | Before tool, after tool, after failed tool, before stop, before subagent stop (5) | Approve project hooks or trust workspace on first launch |
| **Antigravity (`agy`)** | Native workspace plugin (MCP + Hooks + Rules + Skill) | Before tool, after tool, post-invocation, before stop (4) | Same as above |
| **OpenAI Codex CLI** | MCP + Hooks + Instructions | Before tool, after tool, before stop (3; no post-failure or subagent-stop hooks) | Linux `guard-exec` recommended |

---

## 5. Evaluation & Case Studies

### 5.1 Experimental Setup and Metric Definitions

- **Hardware & software**: Measurements below were collected on NVIDIA H20 GPUs (Hopper architecture, 78 physical SMs; card profile target `sm_90`, supporting `sm_90a` WGMMA instructions; multi-GPU case on an 8×H20 NVLink full-mesh node). Hardware counters were captured via `nsys` configured at `200,000 Hz` (~`200.3 kHz` observed effective rate, ~5 µs per sample).
- **Metric definitions**:
  - **Steady-state overhead**: `(t_inst - t_base) / t_base`, comparing warm median latencies of instrumented vs. unmodified baseline kernels; acceptance requires `|overhead| <= 10%`.
  - **SW–HW boundary delta**: `|T_hw - T_sw| / T_hw`, relative difference between hardware profiler kernel duration `T_hw` and outermost in-kernel recorded software span `T_sw`.
  - **Register spill delta**: `spill_bytes_inst - spill_bytes_base`, measuring added local-stack spilling caused by instrumentation (acceptance requires `<= 0 B`).
- **Workload notation**: `B` batch size, `H` attention heads, `S` / `T` sequence length, `D` head dimension, `KV` KV-cache length, `EP` expert-parallel degree, `E` total experts.
- **Measurement transparency**: To keep this repository lightweight, raw binary dumps and multi-trial variance logs are not bundled; all numbers below reflect representative single-campaign silicon measurements for each configuration.

### 5.2 Zero-Signature Profiling of Open-Source Kernels

Without changing function signatures, template parameters, or caller code in unmodified open-source libraries, WeaveProfiler completed closed-loop profiling on three kernels (Table 4). Figures 1–3 are raw three-panel aligned views rendered directly on H20 (see Section 3.3 for reading guide).

| Open-Source Library & Kernel | Frontend & Shape (H20) | Physical Warp Coverage | Closed-Loop Result & Steady-State Overhead |
| --- | --- | :---: | --- |
| **FlashInfer 0.6.8 Hopper FA3** (`prefill_sm90.cuh` + 2 mainloop headers) | CUDA C++, `B=5, H=16, S=512, D=128` (also long-seq `S=4096`) | 12 / 12 (`W00..W11`) | Passed with zero signature edits; long-seq overhead `-1.62%` (discussed below), SW–HW boundary delta `0.33%` |
| **FlashInfer 0.6.8 Hopper FlashMLA** (`mla_hopper.cuh`, DeepSeek-V3 paged MLA) | CUDA C++, `B=64, H=128, KV=512` (also long-seq `KV=4096`) | 8 / 8 (`W00..W07`) | Round 1 fine probes (`+14.55%`) rejected -> Round 2 auto-hoisted PASS: `+8.25%` (`KV=512`) / `+6.11%` (`KV=4096`), `0 B` register spill |
| **vLLM FLA FlashKDA** (`chunk_delta_h.py`, linear attention recurrence) | Triton 3.6, `B=1, T=512, H=64, D=128` (also long-seq `T=8192`) | 4 / 4 (`W00..W03`) | Proxy pins baseline winning autotune config; auto-hoisted & boundary-fused PASS: `+6.74%` (`T=512`) / `+8.14%` (`T=8192`), `0 B` register spill |

**Note on negative overhead (`-1.62%`)**: FA3 exhibits a `-1.62%` change at `S=4096`, which passes within the `±10%` bound alongside clean disassembly and output checks. Negative overhead does not mean instrumentation accelerates execution; rather, extra instructions slightly perturbed compiler instruction scheduling. Negative perturbations exceeding `-10%` are rejected by the verifier as scheduling distortion.

- **Figure 1 — FlashInfer 0.6.8 Hopper FA3 Prefill** (CUDA C++ across 3 headers, `B=5, H=16, S=512, D=128`, all 12 physical warps: `W00` TMA prefetch, `W01..W03` early-exit idle, `W04..W11` two compute warpgroups running WGMMA and online softmax):

![FlashInfer 0.6.8 Hopper FA3 H20 Three-Panel View](docs/images/case_h20_flashinfer_fa3.png)

- **Figure 2 — FlashInfer 0.6.8 Hopper FlashMLA** (CUDA C++ `mla_hopper.cuh`, `B=64, H=128, KV=512`, all 8 physical warps; 128 thread blocks on 78 SMs produce the `128 = 78 + 50` two-wave step visible in both Panel B and Panel A Tensor utilization):

![FlashInfer 0.6.8 Hopper FlashMLA H20 Three-Panel View](docs/images/case_h20_flashmla.png)

- **Figure 3 — vLLM FLA FlashKDA** (Triton 3.6 `chunk_delta_h.py`, `B=1, T=512, H=64, D=128`, all 4 physical warps, separating initial state load, 8 chunked matrix-multiply/decay iterations, and final state store):

![vLLM FLA FlashKDA H20 Three-Panel View](docs/images/case_h20_flashkda.png)

### 5.3 Case Study: 8×H20 Production DeepGEMM MegaMoE Optimization (`1.47×` Speedup)

On an 8×NVIDIA H20 NVLink full-mesh node running production **Qwen3.5-397B DeepGEMM MegaMoE** (`EP=8, S=1024, E=512`; consisting of first-stage gate/up projection `GEMM1`, second-stage down projection `GEMM2`, and cross-GPU `Scatter` with specialized `Disp`, `TMA`, `WG0–1`, and `WG2–3` warp roles), the three-panel aligned view exposed two root bottlenecks (full walkthrough in [docs/case_studies/h20_megamoe_optimization.md](docs/case_studies/h20_megamoe_optimization.md)):

![8×H20 DeepGEMM MegaMoE Before/After Three-Panel View](docs/images/case_h20_megamoe_1p47x.png)

| Variant | End-to-End Latency | FP8 Tensor Core (Avg / Peak) | HBM BW (Avg / Peak) | Diagnosed Bottlenecks & Changes |
| --- | ---: | ---: | ---: | --- |
| Original split-kernel DeepGEMM (a) | `613.2 µs` | `27.9%` / `37.1%` | `36.4%` / `47.6%` | ① ~`65 µs` inter-kernel idle gap between `GEMM1` and `GEMM2` under static expert assignment; ② Synchronous `Scatter` blocks compute warpgroups during `GEMM2` while `Disp` warps sit idle |
| Optimized fused persistent kernel (b) | `416.4 µs` (`1.47×`) | `41.8%` / `62.6%` | `55.7%` / `81.5%` | ① Single-kernel persistent fusion + dynamic expert queue (`94.0%` mean active SMs); ② Asynchronous `Scatter` offloaded to idle `Disp` warps overlapping matrix compute |

*(Note: Variant (b) reflects the combined effect of both changes without an isolated ablation of change ① alone; mean utilization × latency remains approximately conserved across both Tensor Core `+1.7%` and HBM `+3.9%`.)*

### 5.4 Limitations

1. **External LLM endpoint & source confidentiality**: Automated stage identification (`weaveprof_profile` / `wp.identify`) sends kernel source excerpts to the configured HTTP LLM endpoint. Environments with strict source-egress policies should point `OPENAI_BASE_URL` to a private endpoint or supply a local probe plan directly to the verifier.
2. **In-place edit subcommand vs. default isolation**: The default profiling loop edits only isolated temporary copies; however, the low-level CLI subcommand `run.py instrument --apply` edits target files in place by design, and optimization rounds intentionally edit protected files after baseline sealing unlocks them.
3. **Hook enforcement limits**: To avoid infinite agent retry loops, the stop hook allows exit with a warning after 3 consecutive blocks. Pre-tool read blocking protects session keys, the hook script, and the MCP server script; read-only protection of deeper verifier modules (`backend.py`, `harness.py`) relies on Linux `guard-exec` (`bubblewrap`). Windows lacks `bubblewrap` and full POSIX `0600` key enforcement.
4. **Single-register record cap**: CUDA C++ single-register context mode uses a 15-bit cursor, capping each physical warp lane at `32,767` records; longer loops require probe hoisting or shared-memory/standard context modes.
5. **AMD hardware counter resolution**: `rocprof-sys` raw PM counters are per-dispatch aggregates (flat bands across one kernel window rather than microsecond curves); intra-kernel AMD hardware sampling requires `rocprofiler-sdk` device counting service (DCS).

---

## 6. Quick Start & Repository Map

### 6.1 Support Matrix & Prerequisites

| Dimension | Support |
| --- | --- |
| Hardware & targets | NVIDIA H20 (`sm_90` / `sm_90a`), NVIDIA B300 (`sm_100a` / `sm_103a`), AMD Instinct MI308X (`gfx942`) |
| Languages | CUDA C++, HIP C++, CuTeDSL (NVIDIA only), Triton (NVIDIA PTX & AMD CDNA3) |
| Coding agents | Native Claude Code, OpenAI Codex CLI, Qwen Code (`qwen`), Antigravity (`agy`); any CLI agent via `guard-exec` |
| Prerequisites | Python 3.10+, vendor GPU toolchain (`nvcc`/`nsys` or `hipcc`/`rocprof-sys`), `matplotlib` for local PNG rendering; `OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, or `BAILIAN_API_KEY` (optional `OPENAI_BASE_URL`) for automated stage identification |

### 6.2 Installation & Entry Points

Three entry points share the same backend:
- **`run.py`**: Direct CLI script runnable straight from a git clone (18 subcommands covering `install-agent`, `guard-exec`, and staged `scan/instrument/capture/validate/deliver`);
- **`weaveprof` CLI**: Installed console script via `pip install -e .`;
- **Python API (`import weaveprof as wp`)**: Programmatic interface (`wp.identify`, `wp.verify`, `wp.profile`).

Wire WeaveProfiler into your kernel repository in two commands:

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git
python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_repo \
  --protect kernel.cu \
  --card h20
```

Options:
- `--workspace`: Root directory of your kernel repository.
- `--protect`: Register a protected kernel source file (repeatable). **Note**: Registered files are immediately locked until the first passing baseline seal is issued.
- `--card`: Target GPU profile (`h20`, `b300`, or `mi308x`).
- `--agent` (optional): Wires all 4 agents by default, or pass `claude|codex|qwen|agy`; see Appendix A for exact files written.

For non-interactive CLI runs with Linux `bubblewrap` read-only mounts:

```bash
python3 -B WeaveProfiler/run.py guard-exec \
  --workspace /path/to/your_kernel_repo \
  -- codex exec "Profile this kernel and diagnose cross-SM load imbalance"
```

### 6.3 Repository Layout & Offline Tests

```text
WeaveProfiler/
├── README.md              # Simplified Chinese documentation
├── README_EN.md           # English documentation
├── SKILL.md               # Agent operating specification (Agent Skill)
├── pyproject.toml         # Python package config & weaveprof CLI entry
├── plugin.json            # Agent plugin manifest (6 tools)
├── run.py                 # Unified CLI entry point
├── recorders/             # In-kernel recorders (CUDA C++ / HIP C++ / CuTeDSL / Triton NV & AMD)
├── weaveprof/             # Core backend, stage identifier, injection, verifier, MCP server, hooks, viz
├── configs/               # Card profiles, hardware metric rules, operator patterns & exemplars
├── docs/                  # Case studies, design notes, and platform capture docs
├── examples/              # Minimal per-language examples, host wrappers & verified references
└── tests/                 # Offline unit, contract, and adversarial security tests
```

Run offline verification without a GPU:

```bash
python3 -B tests/package_structure.py
python3 -B -m unittest tests/test_mcp_hook_installer.py tests/test_wp_megakernel.py tests/test_hardware_utilization.py
bash tests/run_all.sh
```

---

## Appendix: Engineering Reference

### Appendix A: Files Installed Per Agent

| Agent | Files Written to Workspace |
| --- | --- |
| Claude Code | `.mcp.json`, `.claude/settings.json`, `.claude/settings.local.json`, `.claude/skills/weaveprofiler/SKILL.md`, `CLAUDE.md` |
| OpenAI Codex CLI | `.codex/config.toml`, `.codex/hooks.json`, `.agents/skills/weaveprofiler/SKILL.md`, `AGENTS.md` |
| Qwen Code (`qwen`) | `.qwen/settings.json` (`mcpServers` + native JSON hooks), `.qwen/skills/weaveprofiler/SKILL.md`, `QWEN.md` (approve project hooks or run `/trust` on first launch; verify `active` via `/hooks`) |
| Antigravity (`agy`) | `.agents/plugins/weaveprofiler/` (`plugin.json`, `mcp_config.json`, `hooks.json`, `rules/AGENTS.md`, `skills/weaveprofiler/SKILL.md`) and `GEMINI.md` |

### Appendix B: MCP Tools (`weaveprof/mcp_server.py`)

| Tool | Primary Inputs | Uses LLM | When to Use |
| --- | --- | :---: | --- |
| `weaveprof_status` | Workspace path | No | Inspect protected files, baseline/live SHA-256 hashes, valid seals, and best latency. |
| `weaveprof_profile` | Source path, benchmark command, card | Yes | End-to-end automated profiling from scratch (stage identification -> isolated inject & 4 checks -> auto-coarsen -> aligned PNG -> seal). |
| `weaveprof_megakernel_audit` | Source & benchmark config (rejects self-reported inputs) | No | Isolated-copy build, GPU run, 4 checks, cross-SM skew analysis, PNG render, and seal signing for instrumented megakernels. |
| `weaveprof_plan_context` | Source path & register/SMEM headroom | No | Recommend recorder context storage mode and compiler flags. |
| `weaveprof_analyze` | Aligned SW+HW trace | No | Compute/bandwidth summary, cross-SM skew attribution, and ranked optimization ideas. |
| `weaveprof_compare_rounds` | Previous & current round summaries | No | Compare rounds to confirm bottleneck removal or advise rollback on regression. |
| `weaveprof_verify_seal` | Seal file path | No | Verify HMAC-SHA256 signature and bound artifact hashes against the out-of-workspace session key. |

### Appendix C: Hook Event Names by Agent (`weaveprof/guard_hook.py`)

| Lifecycle Point | Raw Hook Event Name | Supporting Agents |
| --- | --- | --- |
| Before tool call | `PreToolUse` | Claude Code, Codex CLI, Qwen Code, Antigravity |
| After tool call | `PostToolUse` | Claude Code, Codex CLI, Qwen Code, Antigravity |
| After failed tool call | `PostToolUseFailure` | Claude Code, Qwen Code |
| Post-invocation | `PostInvocation` | Antigravity |
| Before session stop | `Stop` | Claude Code, Codex CLI, Qwen Code, Antigravity |
| Before subagent stop | `SubagentStop` | Claude Code, Qwen Code |

### Appendix D: Recorder Frontends & Build Switches

| Language | Recorder File | Key Switches & Binding |
| --- | --- | --- |
| CUDA C++ | `recorders/cuda_cpp/wp_lite.cuh` | `-DWP_LEVEL=0\|1\|2`, `-DWP_GRAN=0\|1`, `-DWP_TIMER=0\|1`; context switches: `-DWP_CTX_REG` (1×32-bit reg, 0 B SMEM, 32,767 records/lane cap), `-DWP_CTX_SMEM` (512 B SMEM at warp granularity / 16 B at CTA granularity), `-DWP_CTX_REG_NOINLINE`; host symbol binding via `cudaMemcpyToSymbol` / `cudaLibraryGetGlobal` (`wp_fi_c`, alias `wp_g`). |
| HIP C++ | `recorders/hip_cpp/wp_lite_hip.h` | `-DWP_LEVEL=0\|1\|2`, `-DWP_GRAN=0\|1`, `-DWP_TIMER=0\|1` (default 1: cycle counter + `s_memrealtime` anchors); `global_store_dwordx4 ... nt`; `hipMemcpyToSymbol`. |
| Triton (NVIDIA) | `recorders/triton/wp_triton.py` | Transparent launcher proxy strips `@triton.autotune` from the instrumented copy and pins the baseline winning config; PTX `st.global.cs.v4.u32`; fused transitions via `WP_TRANS`. |
| Triton (AMD) | `recorders/triton/wp_triton_amd.py` | Targets `gfx942` wave64 with 16-byte packed stores (`fi_open_16b` etc.) and `HW_ID`/`XCC_ID` physical CU decoding. |
| CuTeDSL (NVIDIA) | `recorders/cutedsl/wp_cute.py` | Value-style SSA context with transparent function/descriptor proxies threading buffer tensors through `@cute.jit` launchers into `@cute.kernel`. |

### Appendix E: Artifact & Concept-to-Identifier Reference

| Concept in Prose | Artifact / Code Identifier | Purpose |
| --- | --- | --- |
| Probe plan | `probe.json` | Structured warp-role and stage-line specification from the stage identifier |
| Raw in-kernel event dump | `capture.fi02.bin` (magic `FILITE02`) | Compact 16-byte binary records emitted by all recorders |
| Failure diagnostics | `feedback.json` | Failed-check diagnostics and hotspots returned for refinement |
| Verification & seal records | `verify.json`, `seal.json` | Check metrics and HMAC-SHA256 signature bound to artifact hashes |
| Session state & policy | `session.json` (`policy_hmac`) | Protected file hashes, valid seals, and signed immutable policy fields |
| Three-panel aligned view | `overview.png` | Local PNG rendered by `weaveprof/viz.py` |
| Perfetto trace (optional) | `*.perfetto-trace` / `*.perfetto.json.gz` | Optional interactive Perfetto export |

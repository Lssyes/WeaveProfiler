# Refactor scope (proposal, needs an owner decision)

Baseline: tag `baseline/pre-refactor` = `ae39cc4`. Core code today: 78,296 non-blank lines in 105 files
(`python3 -B tests/quality/code_metrics.py`). Goal of the refactor: core under 10,000 lines without losing
any behavior that the kept scope promises.

Line counts below are non-blank lines from `tests/quality/code_metrics.py`.

## Tier A: keep (the main path, covered by the golden tests)

The megakernel audit, as called through MCP `weaveprof_megakernel_audit` or
`weaveprof.megakernel.run_megakernel_audit`.

| Component | Lines | Notes |
|---|---:|---|
| `weaveprof/megakernel.py` | 1,775 | audit driver, gates, nsys reader, trace merge |
| `weaveprof/harness.py` | 1,767 | policy, gate checks, GPU region lock, seal; only part of it is used by the audit |
| `weaveprof/causal.py` | 1,624 | SW+HW causal analysis (advisory) |
| `weaveprof/viz.py` | 2,021 | four-panel PNG |
| `weaveprof/pipeline/validate/tools/{fi_lite_decode,wp_decode}.py` | ~1,400 | dump decoder |
| `recorders/cuda_cpp/wp_lite.cuh` | 1,260 | device recorder |
| `weaveprof/mcp_server.py` | 668 | MCP tools |
| `weaveprof/workspace.py`, `cli.py` (render, seal-check), `__init__.py` | ~350 | |
| **Tier A total** | **~10,900** | still has to shrink internally to reach < 10k |

Golden coverage of this tier: see `docs/refactor/coverage_main_path.md`.

## Tier B: decide (keep only if you use it)

| Component | Lines | Keep if ... |
|---|---:|---|
| Triton recorder `recorders/triton/` + `pipeline/instrument/triton_ptx_async.py` | ~2,000 | you profile Triton kernels |
| CuTeDSL recorder `recorders/cutedsl/` | 752 | you profile CuTeDSL kernels |
| HIP recorder `recorders/hip_cpp/` + rocprof capture (`pipeline/capture/*rocm*`, `*rocprof*`) | ~5,000 | you profile on MI308X (mail-1/3/21 are MI308X) |
| Agent integration: `agent_installer.py`, `guard_hook.py`, `pipeline/validate/harness/`, `aka_adapter.py`, `atrex_plugin.py`, `plugin.json` | ~4,100 | agents must be installed/guarded by this package rather than by your own setup |
| LLM probe placement: `api.py` (`identify`), `pipeline/scan/llm_probe_placer.py`, `direct_probe_agent.py` | ~3,100 | you want WeaveProfiler itself to call an LLM to place probes (today the coding agent does it) |
| `backend.py` (`wp.profile` / `wp.verify`) | 651 | you need the second audit path that compiles itself and checks SASS similarity |

## Tier C: delete candidates (parallel or legacy paths)

| Component | Lines | Why |
|---|---:|---|
| Staged pipeline `run.py scan / instrument / capture / validate / check / deliver` and most of `weaveprof/pipeline/{scan,instrument,capture,validate,core}` | ~45,000 | a second, older end-to-end path with its own nsys reader, alignment, scale rules and gates |
| `gemm-probe`, `auto-gemm`, `auto-profile`, `fast` (`fast_search.py`, `pipeline/capture/{gemm_probe,auto_gemm}.py`) | ~4,000 | special-purpose GEMM probe paths |
| `preview.py`, `replay.py`, `contract.py` | ~1,160 | used only by the paths above (to be confirmed by coverage) |
| Modules with no reference from non-test code (`detect_gates.py`, `build_full_k_viewer.py`, `direct_probe_agent.py`, `quick_scan.py`, `fi_binsim_triton.py`, `render_repair_prompt.py`, `compact_exact_perfetto.py`, `validate_capture_implementation.py`) | ~5,800 | dead code |
| `examples/host/` copies of pipeline modules | (not core) | diverged duplicates |

Deleting Tier C before touching Tier A is the cheapest and safest first round: the golden tests pin the
main path, so any accidental dependency of Tier A on Tier C shows up as an import error or a golden diff.

## What I need from you

For each Tier B row: keep or delete. For Tier C: confirm deletion (it stays recoverable from the
`baseline/pre-refactor` tag).

WeaveProfiler provides deterministic, isolated-copy intra-kernel GPU profiling and SW+HW causal bottleneck attribution across NVIDIA H20 (`sm_90a`), B300 (`sm_100a`/`sm_103a`), and AMD MI308X (`gfx942`).

Call tools through AKA's plugin runtime (`python3 tools/plugin.py call weaveprofiler.<tool> --input req.json` in managed Light drafts, or `python -m aka.resources.cli call weaveprofiler.<tool> --input req.json` in Legacy):
- `weaveprofiler.plan-context`: choose low-overhead recorder context (`-DWP_CTX_REG` single-register bit-packed default vs `-DWP_CTX_SMEM` / `-DWP_CTX_REG_NOINLINE` fallback and `{pre,post}_setmaxnreg`).
- `weaveprofiler.profile`: run decoupled probe injection + 4-gate verification (SASS/spill, bitwise output, event pairing, overhead) + HMAC-sealed receipt + SW+HW causal report.
- `weaveprofiler.analyze`: run calibrated SW+HW bottleneck attribution on an existing trace/summary JSON.
- `weaveprofiler.compare-rounds`: compute round-to-round bottleneck deltas, `revert_previous_change`, and `next_round_directive`.
- `weaveprofiler.verify-seal`: verify an episode seal fail-closed before citing it.

Hard boundary: keep all instrumented artifacts under `scratch/`. Never hand off or promote an instrumented snapshot to `episode-report`; final promotion always uses a probe-free candidate verified by AKA's official Evaluator and native ABBA.

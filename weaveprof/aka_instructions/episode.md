During an optimization episode on {{PLATFORM}} ({{ARCH}}, {{FRAMEWORK}}):
1. Only invoke `weaveprofiler` when aggregate kernel latency or NCU/NSYS summaries cannot identify which warp role, barrier, or serial compute/comm window is causing underutilization.
2. Inspect `verdict`, `causal_opportunities` (ranked by counterfactual speedup upper bound), and `structural_blockers` (negative-ROI guardrails) from `weaveprofiler.analyze` / `weaveprofiler.profile`.
3. After revising a candidate, call `weaveprofiler.compare-rounds` against the incumbent's causal report to verify whether the targeted bubble disappeared before spending full ABBA evaluation budget.
4. Always restore the clean probe-free candidate before submitting `episode-report`.

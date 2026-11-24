# Refactor process

The code base is too large for line-by-line human review, so correctness is held by tests that pin observable
behavior, and every refactor round is small and mechanically checked.

## Safety net (Phase 1)

| Suite | Command | What it pins | Runs |
|---|---|---|---|
| Golden end-to-end | `python -B -m unittest tests/golden/test_golden_audit.py` | full audit output on 3 real H20 captures (gates, decode summary, skew, causal report, nsys curves, merged trace, PNG validity) | every PR (CPU, ~3 min) |
| Public surface | `python -B -m unittest tests/golden/test_public_surface.py` | CLI commands, Python API, MCP / AKA tool schemas, audit signature, recorder macros and host API | every PR |
| Code-metrics ratchet | `python -B -m unittest tests/quality/test_code_metrics_ratchet.py` | core lines, long functions, broad / swallowed excepts, prints never grow | every PR |
| Existing offline suite | `bash tests/ci_offline.sh` | the 41 legacy offline tests, with the known pre-existing failures listed in `tests/known_failures.txt` | every PR |
| GPU smoke | `WEAVEPROF_GPU_SMOKE=1 python -B -m unittest tests/gpu/test_recorder_smoke.py` | device recorder: known-duration spans, completeness, context/timer invariance, overflow | any PR touching `recorders/`, the decoder or capture code (on an H20, under the GPU lock) |

Golden fixtures are built from real captures with `tests/golden/make_fixture.py` (see its docstring).

## Rules for every round

1. One theme per PR: "delete X", "merge the nsys readers", "split decode()". Not "make it smaller".
2. Pure refactor and behavior change never share a PR.
   - Refactor PR: all golden cases identical, public surface identical.
   - Behavior PR (bug fix, new gate): regenerate with `WEAVEPROF_GOLDEN_UPDATE=1`, and explain every changed
     field in the PR description.
3. A refactor PR must not modify `tests/golden/**` or `tests/gpu/**`. Changes there go in their own PR, reviewed
   by a human first.
4. Ratchet: metrics may only go down. Lower `tests/quality/baseline_metrics.json` in the same PR
   (`WEAVEPROF_METRICS_UPDATE=1`). Raising a value needs a written reason.
5. Use two agents: one implements; a second (preferably a different model) reviews the diff only against these
   rules and the public surface.
6. Human review focuses on: test/fixture diffs, the list of deleted files and symbols, public-surface diffs.

## Suggested order

| Phase | Work |
|---|---|
| 2 | Delete Tier C (see `SCOPE.md`), guided by `coverage_main_path.md` |
| 3 | Merge duplicates (one nsys reader, one alignment, one scale table); split functions > 300 lines |
| 4 | Behavior fixes from the review (alignment, self-reported gate inputs, LDL metric, name-based exemptions), each with new tests |

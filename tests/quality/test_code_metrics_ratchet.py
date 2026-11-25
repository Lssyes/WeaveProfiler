#!/usr/bin/env python3
"""Ratchet: no core-code metric may grow past tests/quality/baseline_metrics.json.

When a refactor round improves a metric, lower the baseline in the same PR:

    WEAVEPROF_METRICS_UPDATE=1 python -B -m unittest tests/quality/test_code_metrics_ratchet.py

Raising a baseline value requires an explicit justification in the PR description.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import code_metrics  # noqa: E402

BASELINE = HERE / "baseline_metrics.json"


class CodeMetricsRatchetTest(unittest.TestCase):
    def test_no_metric_grows(self):
        metrics, _ = code_metrics.collect()
        if os.environ.get("WEAVEPROF_METRICS_UPDATE") == "1" or not BASELINE.is_file():
            if os.environ.get("WEAVEPROF_METRICS_UPDATE") != "1":
                self.fail(f"{BASELINE} missing; generate it with WEAVEPROF_METRICS_UPDATE=1")
            BASELINE.write_text(json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            return
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        grown = {k: (base[k], v) for k, v in metrics.items() if k in base and k != "py_functions" and v > base[k]}
        self.assertEqual(grown, {}, f"metrics grew past the baseline (baseline, now): {grown}")
        improved = {k: (base[k], v) for k, v in metrics.items() if k in base and v < base[k]}
        if improved:
            print(f"\n[ratchet] improved, lower the baseline in this PR: {improved}", file=sys.stderr)


if __name__ == "__main__":
    unittest.main()

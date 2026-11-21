#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_PATH = ROOT / "weaveprof/pipeline/scan" / "scan.py"
spec = importlib.util.spec_from_file_location("weaveprofiler_fast_scan", SCAN_PATH)
scan_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(scan_mod)


class FastScanTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="weaveprofiler_scan_")
        self.root = Path(self.td.name)
        self.source = self.root / "kernel.cu"
        self.source.write_text(
            "__global__ void kernel(float* out, const float* in) {\n"
            "  float x = cp.async.load(in);\n"
            "  wgmma.mma_async(x);\n"
            "  __syncthreads();\n"
            "  st.global.f32(out, x);\n"
            "}\n", encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def run_scan(self, refresh=False):
        return scan_mod.scan(self.root, card="h20", arch="sm_90", platform="nv",
                             language="cuda_cpp", sources=["kernel.cu"], globs=[],
                             cache_dir=self.root / ".weaveprofiler/cache/scan", refresh=refresh)

    def test_canonical_candidates_and_target(self):
        result = self.run_scan()
        ids = [item["semantic_id"] for item in result["candidates"]]
        self.assertIn("load", ids)
        self.assertIn("mma", ids)
        self.assertIn("barrier", ids)
        self.assertIn("store", ids)
        self.assertTrue(all(item["display_name"] == item["semantic_id"] for item in result["candidates"]))
        self.assertEqual(result["target"]["card"], "h20")
        self.assertEqual(result["language"], "cuda_cpp")
        self.assertFalse(result["delivery_evidence"])
        self.assertEqual(result["unknown"], [])

    def test_cache_hit_and_source_invalidation(self):
        first = self.run_scan()
        self.assertFalse(first["cache"]["hit"])
        second = self.run_scan()
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(first["cache_key"], second["cache_key"])
        self.source.write_text(self.source.read_text(encoding="utf-8") + "// changed\n", encoding="utf-8")
        third = self.run_scan()
        self.assertFalse(third["cache"]["hit"])
        self.assertNotEqual(first["cache_key"], third["cache_key"])

    def test_ambiguity_is_explicit(self):
        self.source.write_text("y = exp2(x)\n", encoding="utf-8")
        result = self.run_scan()
        self.assertTrue(result["ambiguities"])
        row = result["ambiguities"][0]
        self.assertIn("softmax", row["candidates"])
        self.assertIn("compute", row["candidates"])

    def test_cache_is_not_delivery_evidence(self):
        result = self.run_scan()
        cache_file = Path(result["cache"]["path"])
        self.assertTrue(cache_file.is_file())
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        self.assertFalse(cached["delivery_evidence"])
        self.assertTrue(cached["fast_path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

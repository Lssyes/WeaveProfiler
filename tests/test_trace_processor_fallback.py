#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "weaveprof/pipeline/scan" / "get_trace_processor.py"
spec = importlib.util.spec_from_file_location("weaveprofiler_get_trace_processor", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load get_trace_processor.py")
gtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gtp)


class TraceProcessorFallbackTests(unittest.TestCase):
    def test_single_statement_import_query_parses_v47_csv(self):
        sql = gtp._g24_sql()
        self.assertEqual(sql.count(";"), 1)
        self.assertNotIn(";\nSELECT", sql)
        parsed = gtp._parse_import_result(
            '"forbidden_stat_count","forbidden_stats","imported_slices"\n'
            '0,"[NULL]",5120\n'
        )
        self.assertEqual(parsed, (0, "[NULL]", 5120))
        self.assertEqual(gtp._imported_slices(
            '"forbidden_stat_count","forbidden_stats","imported_slices"\n'
            '0,"[NULL]",5120\n'), 5120)

    def test_pinned_v47_fallback_is_exact_and_linux_x86_64(self):
        entry = gtp._pinned_fallback_entry()
        self.assertIsNotNone(entry)
        self.assertEqual(entry["version"], "v47.0-c74251226")
        self.assertEqual(entry["file_size"], 10363488)
        self.assertEqual(
            entry["sha256"],
            "832425c3c7934904d1e0ec1721beb51423de7dbcf399a899973f2b6b464603fa")
        self.assertIn("v47.0/linux-amd64/trace_processor_shell", entry["url"])
        self.assertEqual(len(entry["sha256"]), 64)

    def test_import_selfcheck_rejects_empty_slice_table(self):
        # A zero-slice import is not a successful validation even when the
        # process exits zero and all forbidden stats are zero.
        parsed = gtp._parse_import_result(
            '"forbidden_stat_count","forbidden_stats","imported_slices"\n'
            '0,"[NULL]",0\n'
        )
        self.assertEqual(parsed[2], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

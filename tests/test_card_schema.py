#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-module regression for the SPEC-1 hardware-card filename schema."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
for rel in ("weaveprof/pipeline/capture",):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import merge_hw_into_perfetto  # noqa: E402


BASE = ("trace__demomoe_decode__card-{card}__bs8_sq4096_kv16384__fp8e4m3"
        "__grid78_warp12_ph6__sw.perfetto.json.gz")


class FilenameSchemaTests(unittest.TestCase):
    def test_all_three_cards_match_every_spec1_regex(self):
        for card in ("h20", "b300", "mi308x", "sm86"):
            name = BASE.format(card=card)
            match = merge_hw_into_perfetto.TRACE_FILENAME_RE.match(name)
            self.assertIsNotNone(match, (name, merge_hw_into_perfetto.TRACE_FILENAME_RE.pattern))
            self.assertEqual(match.group("card"), card)

    def test_old_and_unknown_card_names_fail_closed(self):
        bad_names = (
            BASE.format(card="h20").replace("__card-h20", ""),
            BASE.format(card="h100"),
            BASE.format(card="MI308X"),
            BASE.format(card="b300").replace("_ph6", "_ph"),
        )
        for name in bad_names:
            self.assertIsNone(
                merge_hw_into_perfetto.TRACE_FILENAME_RE.match(name),
                (name, merge_hw_into_perfetto.TRACE_FILENAME_RE.pattern),
            )

    def test_receipt_suffix_inherits_card_name(self):
        name = BASE.format(card="mi308x")
        self.assertTrue((name + ".delivery_receipt.json").endswith(
            "card-mi308x__bs8_sq4096_kv16384__fp8e4m3"
            "__grid78_warp12_ph6__sw.perfetto.json.gz.delivery_receipt.json"))


class TargetCardTests(unittest.TestCase):
    def test_b300_is_not_locked_to_one_architecture(self):
        self.assertEqual(merge_hw_into_perfetto.CARD_TARGETS["b300"]["arches"],
                         frozenset(("sm_100a", "sm_103a")))


class GeneratorTests(unittest.TestCase):
    def test_merge_inherits_card_and_rejects_legacy_names(self):
        for card in ("h20", "b300", "mi308x", "sm86"):
            sw_name = BASE.format(card=card)
            merged = merge_hw_into_perfetto.build_merged_trace_filename(Path(sw_name))
            self.assertEqual(merged, sw_name.replace("__sw.perfetto", "__sw_hw.perfetto"))
            self.assertEqual(merge_hw_into_perfetto.TRACE_FILENAME_RE.match(merged)
                             .group("card"), card)
        old = BASE.format(card="h20").replace("__card-h20", "")
        with self.assertRaises(ValueError):
            merge_hw_into_perfetto.build_merged_trace_filename(Path(old))
        hw = BASE.format(card="h20").replace("__sw.perfetto", "__hw.perfetto")
        with self.assertRaises(ValueError):
            merge_hw_into_perfetto.build_merged_trace_filename(Path(hw))

        trace = {"traceEvents": [], "otherData": {}}
        good_input = Path(BASE.format(card="h20"))
        good_output = BASE.format(card="h20").replace(
            "__sw.perfetto", "__sw_hw.perfetto")
        self.assertEqual(
            merge_hw_into_perfetto.validate_merge_identity(
                good_input, trace, good_output),
            good_output)
        with self.assertRaises(ValueError):
            merge_hw_into_perfetto.validate_merge_identity(
                Path(old), trace, good_output)
        with self.assertRaises(ValueError):
            merge_hw_into_perfetto.validate_merge_identity(
                good_input, trace, good_output.replace("card-h20", "card-b300"))
        with self.assertRaises(ValueError):
            merge_hw_into_perfetto.validate_merge_identity(
                good_input, trace, BASE.format(card="h20"))

    def test_merge_rejects_embedded_target_card_or_arch_conflict(self):
        trace = {"otherData": {"target": {
            "card": "h20", "platform": "nv", "arch": "sm_90"}}}
        merge_hw_into_perfetto.validate_embedded_target(trace, "h20")
        with self.assertRaises(ValueError):
            merge_hw_into_perfetto.validate_embedded_target(trace, "mi308x")
        trace["otherData"]["target"]["arch"] = "sm_100a"
        with self.assertRaises(ValueError):
            merge_hw_into_perfetto.validate_embedded_target(trace, "h20")


if __name__ == "__main__":
    unittest.main(verbosity=2)

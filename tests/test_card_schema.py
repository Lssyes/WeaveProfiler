#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-module regression for the SPEC-1 hardware-card filename schema."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
for rel in ("weaveprof/pipeline/validate", "weaveprof/pipeline/scan", "weaveprof/pipeline/validate/harness", "weaveprof/pipeline/capture"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import deliver_gate  # noqa: E402
import hooklib  # noqa: E402
import merge_hw_into_perfetto  # noqa: E402
import verify_receipt  # noqa: E402


BASE = ("trace__demomoe_decode__card-{card}__bs8_sq4096_kv16384__fp8e4m3"
        "__grid78_warp12_ph6__sw.perfetto.json.gz")


def contract(card="h20", platform="nv", arch="sm_90"):
    return {
        "target": {
            "model": "demo_moe",
            "stage": "decode",
            "card": card,
            "platform": platform,
            "arch": arch,
        },
        "input": {
            "batch": 8,
            "seqlen_q": 4096,
            "seqlen_kv": 16384,
            "dtype": "fp8e4m3",
        },
        "build_and_launch": {"grid": [78, 1, 1], "warps_per_block": 12, "phases": 6},
        "capture": {"information_bundle": "sw"},
    }


def failure_text(failures):
    return "\n".join("{}: {}".format(item[0], item[1]) for item in failures)


class FilenameSchemaTests(unittest.TestCase):
    def test_all_three_cards_match_every_spec1_regex(self):
        for card in ("h20", "b300", "mi308x", "sm86"):
            name = BASE.format(card=card)
            for regex in (deliver_gate.CANONICAL_TRACE_FILENAME_RE,
                          verify_receipt.SPEC1, hooklib.SPEC1,
                          merge_hw_into_perfetto.TRACE_FILENAME_RE):
                self.assertIsNotNone(regex.match(name), (name, regex.pattern))
            self.assertEqual(deliver_gate.parse_trace_filename(name)["card"], card)
            self.assertEqual(verify_receipt.SPEC1.match(name).group("card"), card)
            self.assertTrue(hooklib.SPEC1_IN_TEXT.search("cp x " + name))

    def test_old_and_unknown_card_names_fail_closed(self):
        bad_names = (
            BASE.format(card="h20").replace("__card-h20", ""),
            BASE.format(card="h100"),
            BASE.format(card="MI308X"),
            BASE.format(card="b300").replace("_ph6", "_ph"),
        )
        for name in bad_names:
            for regex in (deliver_gate.CANONICAL_TRACE_FILENAME_RE,
                          verify_receipt.SPEC1, hooklib.SPEC1,
                          merge_hw_into_perfetto.TRACE_FILENAME_RE):
                self.assertIsNone(regex.match(name), (name, regex.pattern))
            failures, _ = deliver_gate.check_trace_filename_schema(Path(name), contract())
            self.assertTrue(failures, name)
            if "__card-" not in name:
                self.assertTrue(hooklib.SPEC1_IN_TEXT.search("cp x " + name),
                                "legacy trace must enter the guarded namespace")

    def test_receipt_suffix_inherits_card_name(self):
        name = BASE.format(card="mi308x")
        self.assertTrue((name + ".delivery_receipt.json").endswith(
            "card-mi308x__bs8_sq4096_kv16384__fp8e4m3"
            "__grid78_warp12_ph6__sw.perfetto.json.gz.delivery_receipt.json"))


class TargetCardTests(unittest.TestCase):
    def assert_valid(self, doc, card):
        canonical, failures = deliver_gate.validate_target_card(doc)
        self.assertEqual(failures, [], failure_text(failures))
        self.assertEqual(canonical, card)
        errors = []
        self.assertEqual(verify_receipt.check_target_card(errors, doc), card)
        self.assertEqual(errors, [])

    def test_supported_cards_and_b300_architectures(self):
        self.assert_valid(contract("h20", "nv", "sm_90"), "h20")
        self.assert_valid(contract("b300", "nv", "sm_100a"), "b300")
        self.assert_valid(contract("b300", "nv", "sm_103a"), "b300")
        self.assert_valid(contract("mi308x", "amd", "gfx942"), "mi308x")
        self.assert_valid(contract("sm86", "nv", "sm_86"), "sm86")

    def test_unknown_missing_and_cross_platform_are_rejected(self):
        cases = (
            ({}, "target_missing"),
            ({"target": {"platform": "nv", "arch": "sm_90"}}, "target_card_missing"),
            (contract("h100", "nv", "sm_90"), "target_card_invalid"),
            (contract("h20", "amd", "gfx942"), "target_platform_mismatch"),
            (contract("h20", "nv", "sm_100a"), "target_arch_mismatch"),
            (contract("b300", "nv", "sm_90"), "target_arch_mismatch"),
            (contract("mi308x", "nv", "gfx942"), "target_platform_mismatch"),
            (contract("mi308x", "amd", "sm_90"), "target_arch_mismatch"),
        )
        for doc, expected in cases:
            _card, failures = deliver_gate.validate_target_card(doc)
            self.assertIn(expected, failure_text(failures), (doc, failures))
            errors = []
            verify_receipt.check_target_card(errors, doc)
            self.assertTrue(errors, doc)

    def test_b300_is_not_locked_to_one_architecture(self):
        self.assertEqual(deliver_gate.CARD_TARGETS["b300"]["arches"],
                         frozenset(("sm_100a", "sm_103a")))
        self.assertEqual(merge_hw_into_perfetto.CARD_TARGETS["b300"]["arches"],
                         frozenset(("sm_100a", "sm_103a")))


class ReconciliationTests(unittest.TestCase):
    def test_filename_card_must_match_contract(self):
        name = BASE.format(card="mi308x")
        doc = contract("h20", "nv", "sm_90")
        failures, _ = deliver_gate.check_trace_filename_schema(Path(name), doc)
        self.assertIn("target.card", failure_text(failures))
        errors = []
        verify_receipt.check_filename(errors, name, doc)
        self.assertTrue(any("target.card" in item for item in errors), errors)

    def test_valid_cards_reconcile_and_bundle_grid_still_checked(self):
        cases = (
            ("h20", "nv", "sm_90", "sw"),
            ("b300", "nv", "sm_100a", "sw"),
            ("b300", "nv", "sm_103a", "sw"),
            ("mi308x", "amd", "gfx942", "sw"),
            ("sm86", "nv", "sm_86", "sw"),
        )
        for card, platform, arch, bundle in cases:
            name = BASE.format(card=card)
            doc = contract(card, platform, arch)
            self.assertEqual(deliver_gate.check_trace_filename_schema(Path(name), doc)[0], [])
            errors = []
            verify_receipt.check_filename(errors, name, doc)
            self.assertEqual(errors, [])
            self.assertEqual(bundle, "sw")

        name = BASE.format(card="h20").replace("grid78", "grid39")
        errors = []
        verify_receipt.check_filename(errors, name, contract())
        self.assertTrue(any("contradicts contract grid" in item for item in errors), errors)

    def test_missing_target_card_fails_every_contract_aware_gate(self):
        doc = contract()
        doc.pop("target")
        name = BASE.format(card="h20")
        failures, _ = deliver_gate.check_trace_filename_schema(Path(name), doc)
        self.assertIn("target_missing", failure_text(failures))
        errors = []
        verify_receipt.check_filename(errors, name, doc)
        self.assertTrue(any("contract target missing" in item for item in errors), errors)


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

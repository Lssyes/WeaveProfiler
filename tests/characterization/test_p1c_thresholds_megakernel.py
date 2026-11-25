#!/usr/bin/env python3
"""Characterization tests pinning threshold boundaries in ``weaveprof/megakernel.py`` (P1c R21).

Covers:
  1. ``inspect.signature`` defaults on ``analyze_megakernel_skew`` and ``run_megakernel_audit`` (D2).
  2. Second-layer audit gate wiring (``_gate_output``, ``_gate_compiler``, ``_gate_events``,
     ``_gate_overhead``) with default ``HarnessPolicy()``, including the ``* 1000.0`` us->ns
     conversion at ``megakernel.py:967``.
  3. All threshold boundaries in ``analyze_megakernel_skew`` (M06–M15), ``resolve_tensor_active_scale``
     (M16), and ``_tensor_plateau_suspect`` (M17).

Run:  python -B -m unittest tests/characterization/test_p1c_thresholds_megakernel.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import inspect
from pathlib import Path
import tempfile
import unittest
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.harness import HarnessPolicy, PolicyGuard  # noqa: E402
from weaveprof.megakernel import (  # noqa: E402
    _gate_compiler,
    _gate_events,
    _gate_output,
    _gate_overhead,
    _tensor_plateau_suspect,
    analyze_megakernel_skew,
    resolve_tensor_active_scale,
    run_megakernel_audit,
)


def _ev(
    name: str,
    ts: float,
    dur: float,
    *,
    slot: int = 0,
    warp: int | None = None,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {"wave_slot": slot}
    if warp is not None:
        args["warp"] = warp
    return {"ph": "X", "pid": 0, "tid": slot, "ts": ts, "dur": dur, "name": name, "args": args}


def _clean_events_trace() -> Dict[str, Any]:
    return {
        "traceEvents": [
            _ev("mma", 0.0, 20.0, slot=0),
            _ev("load", 20.0, 20.0, slot=0),
            _ev("mma", 40.0, 20.0, slot=0),
        ],
        "otherData": {"gran": 1},
    }


def _clean_summary(*, overflow: int = 0, errors: Dict[str, int] | None = None, spans: int = 3) -> Dict[str, Any]:
    return {"gran": 1, "wpc": 1, "overflow": overflow, "errors": errors or {}, "spans": spans}


def _hint_codes(res: Dict[str, Any]) -> List[str]:
    return [h["code"] for h in res.get("hints") or []]


class MegakernelDefaultsPinTest(unittest.TestCase):
    """D2: Pin ``analyze_megakernel_skew`` and ``run_megakernel_audit`` parameter defaults."""

    def test_d2_megakernel_signature_defaults(self) -> None:
        sig_skew = inspect.signature(analyze_megakernel_skew).parameters
        self.assertIsNone(sig_skew["trace"].default)
        self.assertIsNone(sig_skew["total_sms"].default)
        self.assertEqual(sig_skew["idle_threshold"].default, 0.15)
        self.assertEqual(sig_skew["spread_threshold"].default, 0.20)
        self.assertEqual(sig_skew["role_skew_threshold"].default, 0.15)

        sig_audit = inspect.signature(run_megakernel_audit).parameters
        self.assertEqual(sig_audit["nsys_freq_hz"].default, 200000)
        self.assertEqual(sig_audit["limit"].default, 0.10)
        self.assertIsNone(sig_audit["policy"].default)


class GateWiringThresholdsTest(unittest.TestCase):
    """M01–M05 plus second-layer ``_gate_*`` coverage of H04–H07 and H14–H38."""

    def test_m01_gate_output_empty_file_zero_bytes_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "a0.bin").write_bytes(b"")
            (base / "b0.bin").write_bytes(b"")
            r_empty = _gate_output({"output_files": [["a0.bin", "b0.bin"]]}, base)
            self.assertEqual((base / "a0.bin").stat().st_size, 0)
            self.assertIs(r_empty["ok"], False)
            self.assertEqual(r_empty["mismatched"], ["b0.bin"])

            (base / "a1.bin").write_bytes(b"x")
            (base / "b1.bin").write_bytes(b"x")
            r_nonempty = _gate_output({"output_files": [["a1.bin", "b1.bin"]]}, base)
            self.assertEqual((base / "a1.bin").stat().st_size, 1)
            self.assertIs(r_nonempty["ok"], True)
            self.assertEqual(r_nonempty["mismatched"], [])

    def test_gate_compiler_all_thresholds_via_default_policy(self) -> None:
        pol = HarnessPolicy()
        for key, code, ok_val, fail_val in (
            ("spill_stores", "SPILL_STORE_INCREASE", 0, 1),
            ("spill_loads", "SPILL_LOAD_INCREASE", 0, 1),
            ("stack_bytes", "STACK_FRAME_INCREASE", 0, 1),
            ("stl", "SASS_STL_EXPLOSION", 4, 5),
            ("ldl", "SASS_LDL_EXPLOSION", 4, 5),
        ):
            with self.subTest(metric=key):
                r_ok = _gate_compiler({"l0_stats": {key: 10}, "inst_stats": {key: 10 + ok_val}}, pol)
                self.assertIs(r_ok["ok"], True)
                self.assertEqual(r_ok["violations"], [])

                r_fail = _gate_compiler({"l0_stats": {key: 10}, "inst_stats": {key: 10 + fail_val}}, pol)
                self.assertIs(r_fail["ok"], False)
                self.assertEqual(r_fail["violations"][0]["code"], code)
                self.assertEqual(r_fail["violations"][0]["delta"], fail_val)

        r_blind_t = _gate_compiler(
            {"l0_stats": {"regs": 96, "spill_stores": 0}, "inst_stats": {"regs": 96, "spill_stores": 4}, "smem_free_bytes": 16},
            pol,
        )
        self.assertIs(r_blind_t["register_count_blind_spot"], True)
        self.assertEqual(
            r_blind_t["remediation"],
            ["enable_WP_CTX_SMEM (-DWP_CTX_SMEM, 16B SMEM + __noinline__ cold path)"],
        )

        r_blind_f = _gate_compiler(
            {"l0_stats": {"regs": 96, "spill_stores": 0}, "inst_stats": {"regs": 97, "spill_stores": 4}, "smem_free_bytes": 15},
            pol,
        )
        self.assertIs(r_blind_f["register_count_blind_spot"], False)
        self.assertEqual(r_blind_f["remediation"], ["coarsen_probes_or_hoist_outside_hot_loop"])

    def test_gate_events_stub_us_to_ns_1000x_conversion_and_subconditions(self) -> None:
        pol = HarnessPolicy()
        sm = _clean_summary(spans=10)

        # H22 via megakernel.py:967 (dur_us * 1000.0 <= 128.0): catches *1000.0 -> *1001.0 mutation!
        evs_stub = [_ev("mma", float(i), 0.128) for i in range(8)] + [_ev("load", 20.0 + i, 5.0) for i in range(4)]
        self.assertEqual(0.128 * 1000.0, 128.0)
        g_stub = _gate_events(sm, {"traceEvents": evs_stub, "otherData": {"gran": 1}}, {}, pol)
        self.assertIs(g_stub["ok"], False)
        self.assertIs(g_stub["stub_gate"]["ok"], False)
        self.assertEqual(g_stub["stub_gate"]["stubs"][0]["median_ns"], 128.0)

        evs_ok = [_ev("mma", float(i), 0.1280625) for i in range(8)] + [_ev("load", 20.0 + i, 5.0) for i in range(4)]
        self.assertEqual(0.1280625 * 1000.0, 128.0625)
        g_ok = _gate_events(sm, {"traceEvents": evs_ok, "otherData": {"gran": 1}}, {}, pol)
        self.assertIs(g_ok["ok"], True)
        self.assertIs(g_ok["stub_gate"]["ok"], True)

        # H21 via _gate_events (sass_body_insns 0 vs 1)
        tr_clean = _clean_events_trace()
        g_sass0 = _gate_events(sm, tr_clean, {"sass_body_insns": {"mma": 0}}, pol)
        self.assertIs(g_sass0["ok"], False)
        self.assertEqual(g_sass0["stub_gate"]["stubs"][0]["reasons"], ["zero_non_probe_sass_instructions"])
        g_sass1 = _gate_events(sm, tr_clean, {"sass_body_insns": {"mma": 1}}, pol)
        self.assertIs(g_sass1["ok"], True)

        # H23 via _gate_events (fires 4 vs 3)
        evs_f4 = [_ev("mma", float(i), 0.064) for i in range(4)] + [_ev("load", 20.0 + i, 5.0) for i in range(4)]
        self.assertIs(_gate_events(sm, {"traceEvents": evs_f4, "otherData": {"gran": 1}}, {}, pol)["ok"], False)
        evs_f3 = [_ev("mma", float(i), 0.064) for i in range(3)] + [_ev("load", 20.0 + i, 5.0) for i in range(4)]
        self.assertIs(_gate_events(sm, {"traceEvents": evs_f3, "otherData": {"gran": 1}}, {}, pol)["ok"], True)

        # H24 via _gate_events (p90_ns 256.0 vs 256.125)
        evs_p90_stub = (
            [_ev("mma", float(i), 0.064) for i in range(9)]
            + [_ev("mma", 10.0, 0.256), _ev("mma", 11.0, 0.512)]
            + [_ev("load", 20.0 + i, 5.0) for i in range(4)]
        )
        g_p90_stub = _gate_events(sm, {"traceEvents": evs_p90_stub, "otherData": {"gran": 1}}, {}, pol)
        self.assertIs(g_p90_stub["ok"], False)
        self.assertEqual(g_p90_stub["stub_gate"]["stubs"][0]["p90_ns"], 256.0)

        evs_p90_ok = (
            [_ev("mma", float(i), 0.064) for i in range(9)]
            + [_ev("mma", 10.0, 0.256125), _ev("mma", 11.0, 0.512)]
            + [_ev("load", 20.0 + i, 5.0) for i in range(4)]
        )
        g_p90_ok = _gate_events(sm, {"traceEvents": evs_p90_ok, "otherData": {"gran": 1}}, {}, pol)
        self.assertIs(g_p90_ok["ok"], True)

    def test_gate_events_granularity_and_warp_completeness_via_default_policy(self) -> None:
        pol = HarnessPolicy()
        sm_gran1 = _clean_summary(spans=10)

        def _check_gran_pair(name: str, tr_fail: Dict[str, Any], expected_code: str, tr_ok: Dict[str, Any]) -> None:
            with self.subTest(gran_threshold=name):
                g_fail = _gate_events(sm_gran1, tr_fail, {}, pol)
                self.assertIs(g_fail["ok"], False)
                self.assertEqual(g_fail["granularity_gate"]["code"], expected_code)
                g_ok = _gate_events(sm_gran1, tr_ok, {}, pol)
                self.assertIs(g_ok["ok"], True)
                self.assertIsNone(g_ok["granularity_gate"]["code"])

        # H25: lane interval cap 64 vs 65
        _check_gran_pair(
            "H25_cap_64",
            {"traceEvents": [_ev("mma", 0.0, 100.0), _ev("epilogue", 100.0, 10.0)] + [_ev("load", 1.0 + i, 0.5) for i in range(62)], "otherData": {"gran": 1}},
            "NESTED_WRAPPER_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 100.0), _ev("epilogue", 100.0, 10.0)] + [_ev("load", 1.0 + i, 0.5) for i in range(63)], "otherData": {"gran": 1}},
        )
        # H26a/b/c: enclosure start tol 1e-6, end tol 1e-6, span ratio 1.05
        _check_gran_pair(
            "H26a_start_tol",
            {"traceEvents": [_ev("mma", 10.000001, 90.0), _ev("load", 10.0, 20.0), _ev("epilogue", 101.0, 5.0)], "otherData": {"gran": 1}},
            "NESTED_WRAPPER_COLLAPSE",
            {"traceEvents": [_ev("mma", 10.0000010005, 90.0), _ev("load", 10.0, 20.0), _ev("epilogue", 101.0, 5.0)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H26b_end_tol",
            {"traceEvents": [_ev("mma", 0.0, 100.0), _ev("load", 25.0, 75.000001), _ev("epilogue", 101.0, 10.0)], "otherData": {"gran": 1}},
            "NESTED_WRAPPER_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 100.0), _ev("load", 25.0, 75.000002), _ev("epilogue", 101.0, 10.0)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H26c_ratio_1p05",
            {"traceEvents": [_ev("mma", 0.0, 84.0625), _ev("load", 0.0, 80.0), _ev("epilogue", 100.0, 5.0)], "otherData": {"gran": 1}},
            "NESTED_WRAPPER_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 84.0), _ev("load", 0.0, 80.0), _ev("epilogue", 100.0, 5.0)], "otherData": {"gran": 1}},
        )
        # H27/H28/H29: nested wrapper spans/lane <= 1.5, per_lane_us >= 20.0, union_share >= 0.85
        evs_h27_base = (
            [_ev("load", 1.0, 20.0, slot=s) for s in range(2000)]
            + [_ev("mma", 0.0, 100.0, slot=s) for s in range(1000)]
            + [_ev("mma", 0.0, 50.0, slot=s) for s in range(1000, 2000)]
            + [_ev("mma", 50.0, 50.0, slot=s) for s in range(1000, 2000)]
        )
        _check_gran_pair(
            "H27_nested_spl_1p5",
            {"traceEvents": evs_h27_base, "otherData": {"gran": 1}},
            "NESTED_WRAPPER_COLLAPSE",
            {"traceEvents": evs_h27_base + [_ev("mma", 100.0, 1.0, slot=0), _ev("mma", 101.0, 1.0, slot=1)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H28_nested_20us",
            {"traceEvents": [_ev("mma", 0.0, 20.0), _ev("load", 1.0, 10.0)], "otherData": {"gran": 1}},
            "NESTED_WRAPPER_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 19.984375), _ev("load", 1.0, 10.0)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H29_nested_0p85",
            {"traceEvents": [_ev("mma", 0.0, 85.0), _ev("load", 1.0, 20.0), _ev("epilogue", 100.0, 15.0)], "otherData": {"gran": 1}},
            "NESTED_WRAPPER_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 84.9375), _ev("load", 1.0, 20.0), _ev("epilogue", 100.0, 15.0625)], "otherData": {"gran": 1}},
        )
        # H30/H31a/H31b/H32: single-stage collapse
        _check_gran_pair(
            "H30_single_20us",
            {"traceEvents": [_ev("mma", 0.0, 20.0), _ev("load", 20.0, 0.5)], "otherData": {"gran": 1}},
            "SINGLE_STAGE_LOOP_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 19.984375), _ev("load", 20.0, 0.5)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H31a_single_dom_share_0p95",
            {"traceEvents": [_ev("mma", 0.0, 950.0), _ev("load", 950.0, 50.0)], "otherData": {"gran": 1}},
            "SINGLE_STAGE_LOOP_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 949.5), _ev("load", 949.5, 50.5)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H31b_single_union_share_0p95",
            {"traceEvents": [_ev("mma", 0.0, 950.0), _ev("load", 500.0, 500.0)], "otherData": {"gran": 1}},
            "SINGLE_STAGE_LOOP_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 949.5), _ev("load", 500.0, 500.0)], "otherData": {"gran": 1}},
        )
        evs_h32_fail = [_ev("mma", 0.0, 96.0, slot=s) for s in range(20)] + [_ev("load", 100.0, 1.0, slot=s) for s in range(20)] + [_ev("mma", 96.0, 1.0, slot=0)]
        evs_h32_ok = [_ev("mma", 0.0, 96.0, slot=s) for s in range(2000)] + [_ev("load", 100.0, 1.0, slot=s) for s in range(2000)] + [_ev("mma", 96.0, 1.0, slot=s) for s in range(102)]
        _check_gran_pair(
            "H32_single_spl_1p05",
            {"traceEvents": evs_h32_fail, "otherData": {"gran": 1}},
            "SINGLE_STAGE_LOOP_COLLAPSE",
            {"traceEvents": evs_h32_ok, "otherData": {"gran": 1}},
        )
        # H33a/b/c/d: multi_coarse_no_mem (use "gemm" since "softmax" matches "tma" in _MEM_SEM_RE)
        _check_gran_pair(
            "H33a_coarse_overall_60us",
            {"traceEvents": [_ev("mma", 0.0, 51.0), _ev("gemm", 51.0, 9.0)], "otherData": {"gran": 1}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 51.0), _ev("gemm", 51.0, 8.96875)], "otherData": {"gran": 1}},
        )
        evs_h33b_base = [_ev("mma", 0.0, 60.0, slot=s) for s in range(250)] + [_ev("gemm", 60.0 + 5.0 * k, 5.0, slot=s) for s in range(250) for k in range(7)]
        _check_gran_pair(
            "H33b_coarse_overall_spl_8",
            {"traceEvents": evs_h33b_base, "otherData": {"gran": 1}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": evs_h33b_base + [_ev("gemm", 95.0, 5.0, slot=0)], "otherData": {"gran": 1}},
        )
        evs_h33c_base = (
            [_ev("mma", 0.0, 60.0, slot=s) for s in range(1000)]
            + [_ev("mma", 0.0, 30.0, slot=s) for s in range(1000, 2000)]
            + [_ev("mma", 30.0, 30.0, slot=s) for s in range(1000, 2000)]
            + [_ev("gemm", 60.0, 30.0, slot=s) for s in range(2000)]
        )
        _check_gran_pair(
            "H33c_coarse_dom_spl_1p5",
            {"traceEvents": evs_h33c_base, "otherData": {"gran": 1}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": evs_h33c_base + [_ev("mma", 90.0, 1.0, slot=0), _ev("mma", 91.0, 1.0, slot=1)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H33d_coarse_dom_50us",
            {"traceEvents": [_ev("mma", 0.0, 50.0), _ev("gemm", 50.0, 30.0)], "otherData": {"gran": 1}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 49.96875), _ev("gemm", 49.96875, 30.03125)], "otherData": {"gran": 1}},
        )
        # H34a/b/c/d/e/f: multi_coarse_long_warp
        _check_gran_pair(
            "H34a_long_warp_gran",
            {"traceEvents": [_ev("mma", 0.0, 300.0), _ev("load", 300.0, 200.0)], "otherData": {"gran": 0}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 300.0), _ev("load", 300.0, 200.0)], "otherData": {"gran": 1}},
        )
        _check_gran_pair(
            "H34b_long_warp_450us",
            {"traceEvents": [_ev("mma", 0.0, 270.0), _ev("load", 270.0, 180.0)], "otherData": {"gran": 0}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 270.0), _ev("load", 270.0, 179.75)], "otherData": {"gran": 0}},
        )
        _check_gran_pair(
            "H34c_long_warp_0p40",
            {"traceEvents": [_ev("mma", 0.0, 400.0), _ev("load", 400.0, 300.0), _ev("stage_c", 700.0, 300.0)], "otherData": {"gran": 0}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 399.75), _ev("load", 399.75, 300.125), _ev("stage_c", 699.875, 300.125)], "otherData": {"gran": 0}},
        )
        _check_gran_pair(
            "H34d_long_warp_250us",
            {"traceEvents": [_ev("mma", 0.0, 250.0), _ev("load", 250.0, 125.0), _ev("stage_c", 375.0, 125.0)], "otherData": {"gran": 0}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": [_ev("mma", 0.0, 249.875), _ev("load", 249.875, 125.0625), _ev("stage_c", 374.9375, 125.0625)], "otherData": {"gran": 0}},
        )
        evs_h34e_base = (
            [_ev("mma", 0.0, 300.0, slot=s) for s in range(1000)]
            + [_ev("mma", 0.0, 150.0, slot=s) for s in range(1000, 2000)]
            + [_ev("mma", 150.0, 150.0, slot=s) for s in range(1000, 2000)]
            + [_ev("load", 300.0, 200.0, slot=s) for s in range(2000)]
        )
        _check_gran_pair(
            "H34e_long_warp_dom_spl_1p5",
            {"traceEvents": evs_h34e_base, "otherData": {"gran": 0}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": evs_h34e_base + [_ev("mma", 500.0, 1.0, slot=0), _ev("mma", 501.0, 1.0, slot=1)], "otherData": {"gran": 0}},
        )
        evs_h34f_base = [_ev("mma", 0.0, 300.0, slot=s) for s in range(200)] + [_ev("load", 300.0 + 40.0 * k, 40.0, slot=s) for s in range(200) for k in range(5)]
        _check_gran_pair(
            "H34f_long_warp_overall_spl_6",
            {"traceEvents": evs_h34f_base, "otherData": {"gran": 0}},
            "COARSE_PHASE_ONLY_COLLAPSE",
            {"traceEvents": evs_h34f_base + [_ev("load", 500.0, 1.0, slot=0)], "otherData": {"gran": 0}},
        )

        # H35: warp completeness wpc=1 vs wpc=2 missing warp
        tr_w0_only = {"traceEvents": [_ev("mma", 0.0, 15.0, slot=0, warp=0), _ev("load", 15.0, 15.0, slot=0, warp=0)], "otherData": {"gran": 0}}
        self.assertIs(_gate_events({"gran": 0, "wpc": 1, "overflow": 0, "errors": {}, "spans": 2, "lane_bounds_ns": {"0": [0, 50000]}}, tr_w0_only, {}, pol)["ok"], True)
        g_wmiss = _gate_events({"gran": 0, "wpc": 2, "overflow": 0, "errors": {}, "spans": 2, "lane_bounds_ns": {"0": [0, 50000]}}, tr_w0_only, {}, pol)
        self.assertIs(g_wmiss["ok"], False)
        self.assertEqual(g_wmiss["warp_gate"]["code"], "WARP_COVERAGE_INCOMPLETE")

        # H36: hollow lifetime 49.96us vs 50.0us via _gate_events
        tr_warp_life = {
            "traceEvents": [
                _ev("mma", 0.0, 1.0, slot=0, warp=0),
                _ev("mma", 10.0, 10.0, slot=1, warp=1),
                _ev("load", 20.0, 10.0, slot=1, warp=1),
            ],
            "otherData": {"gran": 0},
        }
        g_life_ok = _gate_events(
            {"gran": 0, "wpc": 2, "overflow": 0, "errors": {}, "spans": 3, "lane_bounds_ns": {"0": [0, 49960], "1": [0, 100000]}},
            tr_warp_life,
            {},
            pol,
        )
        self.assertIs(g_life_ok["ok"], True)
        g_life_fail = _gate_events(
            {"gran": 0, "wpc": 2, "overflow": 0, "errors": {}, "spans": 3, "lane_bounds_ns": {"0": [0, 50000], "1": [0, 100000]}},
            tr_warp_life,
            {},
            pol,
        )
        self.assertIs(g_life_fail["ok"], False)
        self.assertEqual(g_life_fail["warp_gate"]["code"], "HOLLOW_WARP_LANE")

        # H37: hollow spans_per_cta 1.05 vs 1.051 via _gate_events (give warp 1 two spans/lane so granularity_gate is True)
        lb_w20 = {str(k): [0, 100000] for k in range(40)}
        evs_w20 = (
            [_ev("mma", 0.0, 0.5, slot=2 * c, warp=0) for c in range(20)]
            + [_ev("mma", 1.0, 0.5, slot=0, warp=0)]
            + [_ev("load", 0.0, 10.0, slot=2 * c + 1, warp=1) for c in range(20)]
            + [_ev("load", 10.0, 10.0, slot=2 * c + 1, warp=1) for c in range(20)]
        )
        g_h37_fail = _gate_events({"gran": 0, "wpc": 2, "overflow": 0, "errors": {}, "spans": len(evs_w20), "lane_bounds_ns": lb_w20}, {"traceEvents": evs_w20, "otherData": {"gran": 0}}, {}, pol)
        self.assertIs(g_h37_fail["granularity_gate"]["ok"], True)
        self.assertIs(g_h37_fail["ok"], False)
        self.assertEqual(g_h37_fail["warp_gate"]["code"], "HOLLOW_WARP_LANE")

        lb_w2000 = {str(k): [0, 100000] for k in range(4000)}
        evs_w2000 = (
            [_ev("mma", 0.0, 0.5, slot=2 * c, warp=0) for c in range(2000)]
            + [_ev("mma", 1.0, 0.5, slot=2 * c, warp=0) for c in range(102)]
            + [_ev("load", 0.0, 10.0, slot=2 * c + 1, warp=1) for c in range(2000)]
            + [_ev("load", 10.0, 10.0, slot=2 * c + 1, warp=1) for c in range(2000)]
        )
        g_h37_ok = _gate_events({"gran": 0, "wpc": 2, "overflow": 0, "errors": {}, "spans": len(evs_w2000), "lane_bounds_ns": lb_w2000}, {"traceEvents": evs_w2000, "otherData": {"gran": 0}}, {}, pol)
        self.assertIs(g_h37_ok["ok"], True)

        # H38: cov_ratio 0.04996875 vs 0.05 via _gate_events
        sm_cov = {"gran": 0, "wpc": 2, "overflow": 0, "errors": {}, "spans": 3, "lane_bounds_ns": {"0": [0, 1000000], "1": [0, 1000000]}}
        tr_cov_fail = {
            "traceEvents": [
                _ev("mma", 0.0, 49.96875, slot=0, warp=0),
                _ev("mma", 0.0, 100.0, slot=1, warp=1),
                _ev("load", 100.0, 100.0, slot=1, warp=1),
            ],
            "otherData": {"gran": 0},
        }
        g_cov_fail = _gate_events(sm_cov, tr_cov_fail, {}, pol)
        self.assertIs(g_cov_fail["ok"], False)
        self.assertEqual(g_cov_fail["warp_gate"]["code"], "HOLLOW_WARP_LANE")

        tr_cov_ok = {
            "traceEvents": [
                _ev("mma", 0.0, 50.0, slot=0, warp=0),
                _ev("mma", 0.0, 100.0, slot=1, warp=1),
                _ev("load", 100.0, 100.0, slot=1, warp=1),
            ],
            "otherData": {"gran": 0},
        }
        g_cov_ok = _gate_events(sm_cov, tr_cov_ok, {}, pol)
        self.assertIs(g_cov_ok["ok"], True)

    def test_m02_m03_m04_gate_events_overflow_pairing_spans_boundaries(self) -> None:
        pol = HarnessPolicy()
        tr = _clean_events_trace()

        # M02: overflow == 0 vs 1
        g_ov0 = _gate_events(_clean_summary(overflow=0), tr, {}, pol)
        self.assertIs(g_ov0["ok"], True)
        self.assertEqual(g_ov0["overflow"], 0)

        g_ov1 = _gate_events(_clean_summary(overflow=1), tr, {}, pol)
        self.assertIs(g_ov1["ok"], False)
        self.assertEqual(g_ov1["overflow"], 1)

        # M03: pairing == 0 vs 1
        g_pair0 = _gate_events(_clean_summary(errors={}), tr, {}, pol)
        self.assertIs(g_pair0["ok"], True)
        self.assertEqual(g_pair0["pairing_errors"], 0)

        g_pair1 = _gate_events(_clean_summary(errors={"unmatched_close": 1}), tr, {}, pol)
        self.assertIs(g_pair1["ok"], False)
        self.assertEqual(g_pair1["pairing_errors"], 1)

        # M04: spans > 0 (1 vs 0)
        g_sp1 = _gate_events(_clean_summary(spans=1), tr, {}, pol)
        self.assertIs(g_sp1["ok"], True)
        self.assertEqual(g_sp1["spans"], 1)

        g_sp0 = _gate_events(_clean_summary(spans=0), tr, {}, pol)
        self.assertIs(g_sp0["ok"], False)
        self.assertEqual(g_sp0["spans"], 0)

    def test_m05_gate_overhead_positive_negative_and_protocol_boundaries(self) -> None:
        pol = HarnessPolicy()
        guard = PolicyGuard(pol)
        default_limit = inspect.signature(run_megakernel_audit).parameters["limit"].default

        # Positive boundary: ratio = 0.100000 vs 0.100001 (quantized by summarize_abba round(..., 6))
        g_pos_ok = _gate_overhead(
            guard,
            {"warmup": 1, "baseline_ns": [1000000.0] * 3, "instrumented_ns": [1100000.0] * 3},
            default_limit,
        )
        self.assertIs(g_pos_ok["ok"], True)
        self.assertEqual(g_pos_ok["overhead_ratio"], 0.1)
        self.assertEqual(g_pos_ok["errors"], [])

        g_pos_fail = _gate_overhead(
            guard,
            {"warmup": 1, "baseline_ns": [1000000.0] * 3, "instrumented_ns": [1100001.0] * 3},
            default_limit,
        )
        self.assertIs(g_pos_fail["ok"], False)
        self.assertEqual(g_pos_fail["overhead_ratio"], 0.100001)
        self.assertEqual(g_pos_fail["errors"], ["warm overhead 10.00% > limit +10.00%"])

        # Negative boundary: ratio = -0.100000 vs -0.100001
        g_neg_ok = _gate_overhead(
            guard,
            {"warmup": 1, "baseline_ns": [1000000.0] * 3, "instrumented_ns": [900000.0] * 3},
            default_limit,
        )
        self.assertIs(g_neg_ok["ok"], True)
        self.assertEqual(g_neg_ok["overhead_ratio"], -0.1)
        self.assertEqual(g_neg_ok["errors"], [])

        g_neg_fail = _gate_overhead(
            guard,
            {"warmup": 1, "baseline_ns": [1000000.0] * 3, "instrumented_ns": [899999.0] * 3},
            default_limit,
        )
        self.assertIs(g_neg_fail["ok"], False)
        self.assertEqual(g_neg_fail["overhead_ratio"], -0.100001)
        self.assertTrue(g_neg_fail["errors"][0].startswith("NEGATIVE_OVERHEAD_PERTURBATION:"))

        # Protocol boundaries via _gate_overhead (H04 warmup 1 vs 0, H05 paired samples 3 vs 2, H06 limit ceiling+1e-12)
        g_w0 = _gate_overhead(guard, {"warmup": 0, "baseline_ns": [1000.0] * 3, "instrumented_ns": [1000.0] * 3}, default_limit)
        self.assertIs(g_w0["ok"], False)
        self.assertEqual(g_w0["errors"], ["warmup=0 < policy min_warmup=1"])

        g_p2 = _gate_overhead(guard, {"warmup": 1, "baseline_ns": [1000.0] * 2, "instrumented_ns": [1000.0] * 2}, default_limit)
        self.assertIs(g_p2["ok"], False)
        self.assertEqual(g_p2["errors"], ["paired samples mismatch: baseline=2, instrumented=2 (min=3)"])

        rm_ok = {"warmup": 1, "baseline_ns": [1000.0] * 3, "instrumented_ns": [1000.0] * 3}
        self.assertIs(_gate_overhead(guard, rm_ok, 0.10000000000100001)["ok"], True)
        g_lim_fail = _gate_overhead(guard, rm_ok, 0.1000000000011)
        self.assertIs(g_lim_fail["ok"], False)
        self.assertEqual(g_lim_fail["errors"], ["limit=0.1000000000011 exceeds policy ceiling 0.1"])

        # H07 via _gate_overhead: sample > 0 vs <= 0
        self.assertIs(
            _gate_overhead(guard, {"warmup": 1, "baseline_ns": [0.125] * 3, "instrumented_ns": [0.125] * 3}, default_limit)["ok"],
            True,
        )
        g_v0 = _gate_overhead(
            guard, {"warmup": 1, "baseline_ns": [0.0, 0.125, 0.125], "instrumented_ns": [0.125] * 3}, default_limit
        )
        self.assertIs(g_v0["ok"], False)
        self.assertEqual(g_v0["errors"], ["no valid paired baseline_ns/instrumented_ns samples"])


class SkewAndTensorScaleThresholdsTest(unittest.TestCase):
    """M06–M17: ``analyze_megakernel_skew``, ``resolve_tensor_active_scale``, ``_tensor_plateau_suspect``."""

    def test_m06_skew_min_two_sms_for_partition_boundary(self) -> None:
        r_one = analyze_megakernel_skew({"wpc": 1, "cta_sm": {"0": 0}, "lane_bounds_ns": {"0": [0, 1000000]}})
        self.assertEqual(r_one["observed_sms"], 1)
        self.assertEqual(r_one["split_mode"], "none")
        self.assertIsNone(r_one["split_cut_us"])

        r_two = analyze_megakernel_skew(
            {"wpc": 1, "cta_sm": {"0": 0, "1": 1}, "lane_bounds_ns": {"0": [0, 800000], "1": [0, 1000000]}}
        )
        self.assertEqual(r_two["observed_sms"], 2)
        self.assertEqual(r_two["split_mode"], "bimodal_gap")
        self.assertEqual(r_two["split_cut_us"], 900.0)

    def test_m07_skew_bimodal_gap_0p10_boundary(self) -> None:
        r_bimodal = analyze_megakernel_skew(
            {"wpc": 1, "cta_sm": {"0": 0, "1": 1}, "lane_bounds_ns": {"0": [0, 1800000], "1": [0, 2000000]}}
        )
        self.assertEqual((2000.0 - 1800.0) / 2000.0, 0.10)
        self.assertEqual(r_bimodal["split_mode"], "bimodal_gap")
        self.assertEqual(r_bimodal["split_cut_us"], 1900.0)
        self.assertEqual(r_bimodal["stragglers"]["sm_ids"], [1])
        self.assertEqual(r_bimodal["early_finishing"]["sm_ids"], [0])

        r_tail = analyze_megakernel_skew(
            {"wpc": 1, "cta_sm": {"0": 0, "1": 1}, "lane_bounds_ns": {"0": [0, 1800125], "1": [0, 2000000]}}
        )
        self.assertEqual((2000.0 - 1800.125) / 2000.0, 0.0999375)
        self.assertEqual(r_tail["split_mode"], "tail_threshold")
        self.assertIsNone(r_tail["split_cut_us"])

    def test_m08_skew_tail_straggler_0p95_max_busy_boundary(self) -> None:
        r_in = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0, "1": 1, "2": 2},
                "lane_bounds_ns": {"0": [0, 920000], "1": [0, 950000], "2": [0, 1000000]},
            }
        )
        self.assertEqual(r_in["split_mode"], "tail_threshold")
        self.assertEqual(r_in["per_sm"][1]["busy_us"], 950.0)
        self.assertEqual(r_in["stragglers"]["sm_ids"], [1, 2])
        self.assertEqual(r_in["stragglers"]["count"], 2)

        r_out = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0, "1": 1, "2": 2},
                "lane_bounds_ns": {"0": [0, 920000], "1": [0, 949950], "2": [0, 1000000]},
            }
        )
        self.assertEqual(r_out["split_mode"], "tail_threshold")
        self.assertEqual(r_out["per_sm"][1]["busy_us"], 949.95)
        self.assertEqual(r_out["stragglers"]["sm_ids"], [2])
        self.assertEqual(r_out["stragglers"]["count"], 1)

    def test_m09_skew_tail_early_0p85_max_busy_boundary(self) -> None:
        r_in = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0, "1": 1, "2": 2},
                "lane_bounds_ns": {"0": [0, 850000], "1": [0, 925000], "2": [0, 1000000]},
            }
        )
        self.assertEqual(r_in["split_mode"], "tail_threshold")
        self.assertEqual(r_in["per_sm"][0]["busy_us"], 850.0)
        self.assertEqual(r_in["early_finishing"]["sm_ids"], [0])
        self.assertEqual(r_in["early_finishing"]["count"], 1)

        r_out = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0, "1": 1, "2": 2},
                "lane_bounds_ns": {"0": [0, 850050], "1": [0, 925000], "2": [0, 1000000]},
            }
        )
        self.assertEqual(r_out["split_mode"], "tail_threshold")
        self.assertEqual(r_out["per_sm"][0]["busy_us"], 850.05)
        self.assertEqual(r_out["early_finishing"]["sm_ids"], [])
        self.assertEqual(r_out["early_finishing"]["count"], 0)

    def test_m10_skew_barrier_starved_wait_0p10_life_boundary(self) -> None:
        sm = {"wpc": 1, "cta_sm": {"0": 0, "1": 1}, "lane_bounds_ns": {"0": [0, 500000], "1": [0, 1000000]}}
        r_starved = analyze_megakernel_skew(sm, {"traceEvents": [_ev("barrier_wait", 0.0, 50.0, slot=0)]})
        self.assertEqual(r_starved["per_sm"][0]["wait_us"], 50.0)
        self.assertEqual(r_starved["early_finishing"]["barrier_starved"], 1)

        r_not = analyze_megakernel_skew(sm, {"traceEvents": [_ev("barrier_wait", 0.0, 49.96875, slot=0)]})
        self.assertEqual(r_not["per_sm"][0]["wait_us"], 49.969)
        self.assertEqual(49.96875 / 500.0, 0.0999375)
        self.assertEqual(r_not["early_finishing"]["barrier_starved"], 0)

    def test_m11a_skew_idle_sm_fraction_0p15_with_sm_spread_zero(self) -> None:
        def _build(observed: int) -> Dict[str, Any]:
            cta_sm = {str(i): i for i in range(observed)}
            lb = {str(i): [0, 1000000] for i in range(observed)}
            return analyze_megakernel_skew({"wpc": 1, "cta_sm": cta_sm, "lane_bounds_ns": lb}, total_sms=10000)

        r_hit = _build(8500)
        self.assertEqual(r_hit["sm_spread_ratio"], 0.0)
        self.assertEqual(r_hit["idle_sm_fraction"], 0.15)
        self.assertIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", _hint_codes(r_hit))

        r_miss = _build(8501)
        self.assertEqual(r_miss["sm_spread_ratio"], 0.0)
        self.assertEqual(r_miss["idle_sm_fraction"], 0.1499)
        self.assertNotIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", _hint_codes(r_miss))

    def test_m11b_skew_sm_spread_ratio_0p20_with_idle_sm_fraction_below_0p15(self) -> None:
        cta_sm = {str(i): i for i in range(10)}
        lb_hit = {str(i): [0, 800000 if i == 0 else 1000000] for i in range(10)}
        r_hit = analyze_megakernel_skew({"wpc": 1, "cta_sm": cta_sm, "lane_bounds_ns": lb_hit}, total_sms=10)
        self.assertEqual(r_hit["idle_sm_fraction"], 0.10)
        self.assertEqual(r_hit["sm_spread_ratio"], 0.20)
        self.assertIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", _hint_codes(r_hit))

        lb_miss = {str(i): [0, 800100 if i == 0 else 1000000] for i in range(10)}
        r_miss = analyze_megakernel_skew({"wpc": 1, "cta_sm": cta_sm, "lane_bounds_ns": lb_miss}, total_sms=10)
        self.assertEqual(r_miss["idle_sm_fraction"], 0.10)
        self.assertEqual(r_miss["sm_spread_ratio"], 0.1999)
        self.assertNotIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", _hint_codes(r_miss))

    def test_m12a_skew_role_skew_ratio_0p15_with_role_tail_zero(self) -> None:
        r_hit = analyze_megakernel_skew(
            {
                "wpc": 2,
                "span_us": 1000.0,
                "cta_sm": {"0": 0},
                "lane_bounds_ns": {"0": [0, 1000000, 0], "1": [0, 850000, 1]},
            }
        )
        self.assertEqual(r_hit["max_role_tail_us"], 0.0)
        self.assertEqual(r_hit["role_skew_ratio"], 0.15)
        self.assertIn("WARP_GROUP_REBALANCE", _hint_codes(r_hit))

        r_miss = analyze_megakernel_skew(
            {
                "wpc": 2,
                "span_us": 1000.0,
                "cta_sm": {"0": 0},
                "lane_bounds_ns": {"0": [0, 1000000, 0], "1": [0, 850100, 1]},
            }
        )
        self.assertEqual(r_miss["max_role_tail_us"], 0.0)
        self.assertEqual(r_miss["role_skew_ratio"], 0.1499)
        self.assertNotIn("WARP_GROUP_REBALANCE", _hint_codes(r_miss))

    def test_m12b_skew_max_role_tail_0p20_kernel_us_with_role_skew_zero(self) -> None:
        cta_sm = {"0": 0, "1": 1, "2": 2}
        r_hit = analyze_megakernel_skew(
            {
                "wpc": 2,
                "span_us": 1000.0,
                "cta_sm": cta_sm,
                "lane_bounds_ns": {
                    "0": [0, 800000, 0], "1": [0, 800000, 1],
                    "2": [0, 800000, 0], "3": [0, 800000, 1],
                    "4": [0, 1000000, 0], "5": [0, 800000, 1],
                },
            }
        )
        self.assertEqual(r_hit["role_skew_ratio"], 0.0)
        self.assertEqual(r_hit["max_role_tail_us"], 200.0)
        self.assertIn("WARP_GROUP_REBALANCE", _hint_codes(r_hit))

        r_miss = analyze_megakernel_skew(
            {
                "wpc": 2,
                "span_us": 1000.0,
                "cta_sm": cta_sm,
                "lane_bounds_ns": {
                    "0": [0, 800000, 0], "1": [0, 800000, 1],
                    "2": [0, 800000, 0], "3": [0, 800000, 1],
                    "4": [0, 999875, 0], "5": [0, 800000, 1],
                },
            }
        )
        self.assertEqual(r_miss["role_skew_ratio"], 0.0)
        self.assertEqual(r_miss["max_role_tail_us"], 199.875)
        self.assertNotIn("WARP_GROUP_REBALANCE", _hint_codes(r_miss))

    def test_m12c_skew_role_tail_requires_two_roles_boundary(self) -> None:
        cta_sm = {"0": 0, "1": 1, "2": 2}
        r_one_role = analyze_megakernel_skew(
            {
                "wpc": 1,
                "span_us": 1000.0,
                "cta_sm": cta_sm,
                "lane_bounds_ns": {"0": [0, 700000, 0], "1": [0, 700000, 0], "2": [0, 1000000, 0]},
            }
        )
        self.assertEqual(len(r_one_role["role_stats"]), 1)
        self.assertEqual(r_one_role["max_role_tail_us"], 300.0)
        self.assertNotIn("WARP_GROUP_REBALANCE", _hint_codes(r_one_role))

        r_two_roles = analyze_megakernel_skew(
            {
                "wpc": 2,
                "span_us": 1000.0,
                "cta_sm": cta_sm,
                "lane_bounds_ns": {
                    "0": [0, 700000, 0], "1": [0, 700000, 1],
                    "2": [0, 700000, 0], "3": [0, 700000, 1],
                    "4": [0, 1000000, 0], "5": [0, 700000, 1],
                },
            }
        )
        self.assertEqual(len(r_two_roles["role_stats"]), 2)
        self.assertEqual(r_two_roles["role_skew_ratio"], 0.0)
        self.assertEqual(r_two_roles["max_role_tail_us"], 300.0)
        self.assertIn("WARP_GROUP_REBALANCE", _hint_codes(r_two_roles))

    def test_m13a_skew_wait_share_0p20_with_starved_zero(self) -> None:
        r_hit = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0},
                "lane_bounds_ns": {"0": [0, 1000000]},
                "events": {
                    "barrier_wait": {"n": 10, "total_us": 200.0, "p50_us": 20.0, "max_us": 20.0},
                    "mma": {"n": 10, "total_us": 800.0, "p50_us": 80.0, "max_us": 80.0},
                },
            }
        )
        self.assertEqual(r_hit["early_finishing"]["barrier_starved"], 0)
        self.assertEqual(r_hit["wait_share"], 0.20)
        self.assertIn("GRID_SYNC_WAIT_DOMINANT", _hint_codes(r_hit))

        r_miss = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0},
                "lane_bounds_ns": {"0": [0, 1000000]},
                "events": {
                    "barrier_wait": {"n": 10, "total_us": 199.9, "p50_us": 20.0, "max_us": 20.0},
                    "mma": {"n": 10, "total_us": 800.1, "p50_us": 80.0, "max_us": 80.0},
                },
            }
        )
        self.assertEqual(r_miss["early_finishing"]["barrier_starved"], 0)
        self.assertEqual(r_miss["wait_share"], 0.1999)
        self.assertNotIn("GRID_SYNC_WAIT_DOMINANT", _hint_codes(r_miss))

    def test_m13b_skew_starved_fraction_0p10_with_wait_share_zero(self) -> None:
        def _build(observed: int) -> Dict[str, Any]:
            cta_sm = {str(i): i for i in range(observed)}
            lb = {str(i): [0, 500000 if i < 1000 else 1000000] for i in range(observed)}
            evs = [_ev("barrier_wait", 0.0, 100.0, slot=i) for i in range(1000)]
            return analyze_megakernel_skew(
                {
                    "wpc": 1,
                    "cta_sm": cta_sm,
                    "lane_bounds_ns": lb,
                    "events": {"mma": {"n": 10, "total_us": 1000.0, "p50_us": 100.0, "max_us": 100.0}},
                },
                {"traceEvents": evs},
            )

        r_hit = _build(10000)
        self.assertEqual(r_hit["wait_share"], 0.0)
        self.assertEqual(r_hit["early_finishing"]["barrier_starved"], 1000)
        self.assertEqual(1000 / 10000.0, 0.10)
        self.assertIn("GRID_SYNC_WAIT_DOMINANT", _hint_codes(r_hit))

        r_miss = _build(10001)
        self.assertEqual(r_miss["wait_share"], 0.0)
        self.assertEqual(r_miss["early_finishing"]["barrier_starved"], 1000)
        self.assertLess(1000 / 10001.0, 0.10)
        self.assertNotIn("GRID_SYNC_WAIT_DOMINANT", _hint_codes(r_miss))

    def test_m14_skew_unused_sms_fraction_0p10_boundary(self) -> None:
        cta_sm = {str(i): i for i in range(9000)}
        lb = {str(i): [0, 1000000] for i in range(9000)}
        r_hit = analyze_megakernel_skew({"wpc": 1, "cta_sm": cta_sm, "lane_bounds_ns": lb}, total_sms=10000)
        self.assertEqual(r_hit["unused_sms"], 1000)
        self.assertEqual(1000 / 10000.0, 0.10)
        self.assertIn("FILL_IDLE_SMS", _hint_codes(r_hit))

        r_miss = analyze_megakernel_skew({"wpc": 1, "cta_sm": cta_sm, "lane_bounds_ns": lb}, total_sms=9999)
        self.assertEqual(r_miss["unused_sms"], 999)
        self.assertLess(999 / 9999.0, 0.10)
        self.assertNotIn("FILL_IDLE_SMS", _hint_codes(r_miss))

    def test_m15_skew_dominant_compute_share_50pct_boundary(self) -> None:
        r_hit = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0},
                "lane_bounds_ns": {"0": [0, 1000000]},
                "events": {
                    "mma": {"n": 10, "total_us": 500.0, "p50_us": 50.0, "max_us": 50.0},
                    "load": {"n": 10, "total_us": 300.0, "p50_us": 30.0, "max_us": 30.0},
                    "epilogue": {"n": 10, "total_us": 200.0, "p50_us": 20.0, "max_us": 20.0},
                },
            }
        )
        self.assertEqual(r_hit["stages"][0]["share_pct"], 50.0)
        self.assertIn("OPTIMIZE_DOMINANT_STAGE", _hint_codes(r_hit))

        r_miss = analyze_megakernel_skew(
            {
                "wpc": 1,
                "cta_sm": {"0": 0},
                "lane_bounds_ns": {"0": [0, 1000000]},
                "events": {
                    "mma": {"n": 10, "total_us": 499.9, "p50_us": 50.0, "max_us": 50.0},
                    "load": {"n": 10, "total_us": 300.1, "p50_us": 30.0, "max_us": 30.0},
                    "epilogue": {"n": 10, "total_us": 200.0, "p50_us": 20.0, "max_us": 20.0},
                },
            }
        )
        self.assertEqual(r_miss["stages"][0]["share_pct"], 49.99)
        self.assertNotIn("OPTIMIZE_DOMINANT_STAGE", _hint_codes(r_miss))

    def test_m16_resolve_tensor_active_scale_h20_ceiling_29pct_boundary(self) -> None:
        hw_at = {
            "gpu_name": "NVIDIA H20",
            "kernel": {"duration_us": 100.0},
            "hw_series": {"tensor_active": {"t_us": [10.0, 50.0], "val": [20.0, 29.0]}},
        }
        r_at = resolve_tensor_active_scale(hw_at)
        self.assertEqual(r_at["raw_ceiling_pct"], 29.0)
        self.assertEqual(r_at["raw_max_pct"], 29.0)
        self.assertIs(r_at["applied"], True)
        self.assertEqual(r_at["scale"], 4.0)

        hw_above = {
            "gpu_name": "NVIDIA H20",
            "kernel": {"duration_us": 100.0},
            "hw_series": {"tensor_active": {"t_us": [10.0, 50.0], "val": [20.0, 29.015625]}},
        }
        r_above = resolve_tensor_active_scale(hw_above)
        self.assertEqual(r_above["raw_ceiling_pct"], 29.0)
        self.assertEqual(r_above["raw_max_pct"], 29.015625)
        self.assertIs(r_above["applied"], False)
        self.assertEqual(r_above["scale"], 1.0)

    def test_m17_tensor_plateau_suspect_all_subconditions(self) -> None:
        scale_unapplied = {"applied": False, "source": "test"}

        def _hw(tv: List[float], sv: List[float]) -> Dict[str, Any]:
            return {
                "kernel": {"duration_us": float(max(len(tv), len(sv), 1000))},
                "hw_series": {
                    "tensor_active": {"t_us": [float(i) for i in range(len(tv))], "val": tv},
                    "sm_active": {"t_us": [float(i) for i in range(len(sv))], "val": sv},
                },
            }

        # 1) len(tv) < 10: 9 vs 10
        self.assertIsNone(_tensor_plateau_suspect(_hw([25.0] * 9, [100.0] * 9), scale_unapplied))
        self.assertIsNotNone(_tensor_plateau_suspect(_hw([25.0] * 10, [100.0] * 10), scale_unapplied))

        # 2) max(sv) < 99.0: 98.9375 vs 99.0
        self.assertIsNone(_tensor_plateau_suspect(_hw([25.0] * 10, [98.9375] * 10), scale_unapplied))
        self.assertIsNotNone(_tensor_plateau_suspect(_hw([25.0] * 10, [99.0] * 10), scale_unapplied))

        # 3) abs(vmax - 25.0) <= 1.5 (upper 26.5 vs 26.515625, lower 23.5 vs 23.484375)
        self.assertIn("25%", _tensor_plateau_suspect(_hw([26.5] * 10, [100.0] * 10), scale_unapplied) or "")
        self.assertIsNone(_tensor_plateau_suspect(_hw([26.515625] * 10, [100.0] * 10), scale_unapplied))
        self.assertIn("25%", _tensor_plateau_suspect(_hw([23.5] * 10, [100.0] * 10), scale_unapplied) or "")
        self.assertIsNone(_tensor_plateau_suspect(_hw([23.484375] * 10, [100.0] * 10), scale_unapplied))

        # 4) abs(vmax - 50.0) <= 1.5 (upper 51.5 vs 51.53125, lower 48.5 vs 48.46875)
        self.assertIn("50%", _tensor_plateau_suspect(_hw([51.5] * 10, [100.0] * 10), scale_unapplied) or "")
        self.assertIsNone(_tensor_plateau_suspect(_hw([51.53125] * 10, [100.0] * 10), scale_unapplied))
        self.assertIn("50%", _tensor_plateau_suspect(_hw([48.5] * 10, [100.0] * 10), scale_unapplied) or "")
        self.assertIsNone(_tensor_plateau_suspect(_hw([48.46875] * 10, [100.0] * 10), scale_unapplied))

        # 5a) v >= vmax - 1.0 (24.0 vs 23.984375 when vmax=25.0, 10 samples -> need 2 >= 24.0)
        self.assertIsNotNone(_tensor_plateau_suspect(_hw([25.0, 24.0] + [10.0] * 8, [100.0] * 10), scale_unapplied))
        self.assertIsNone(_tensor_plateau_suspect(_hw([25.0, 23.984375] + [10.0] * 8, [100.0] * 10), scale_unapplied))

        # 5b) count >= 0.2 * len(tv) (2000/10000 = 0.20 vs 1999/10000 = 0.1999, rel dist 0.05%)
        tv_2000 = [25.0] * 2000 + [10.0] * 8000
        self.assertEqual(2000 / 10000.0, 0.20)
        self.assertIsNotNone(_tensor_plateau_suspect(_hw(tv_2000, [100.0] * 10), scale_unapplied))

        tv_1999 = [25.0] * 1999 + [10.0] * 8001
        self.assertEqual(1999 / 10000.0, 0.1999)
        self.assertIsNone(_tensor_plateau_suspect(_hw(tv_1999, [100.0] * 10), scale_unapplied))


if __name__ == "__main__":
    unittest.main()

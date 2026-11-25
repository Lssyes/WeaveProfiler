#!/usr/bin/env python3
"""Characterization tests pinning threshold boundaries in ``weaveprof/harness.py`` (P1c R21).

Pins observable pass/fail decisions, violation codes, and intermediate compared quantities
at both sides of every threshold in ``weaveprof/harness.py`` using default ``HarnessPolicy()``
and default function parameter signatures (never overriding thresholds in test calls).

Run:  python -B -m unittest tests/characterization/test_p1c_thresholds_harness.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import inspect
from pathlib import Path
import statistics
import tempfile
import unittest
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.harness import (  # noqa: E402
    SEAL_SCHEMA,
    HarnessPolicy,
    PolicyGuard,
    PolicyViolation,
    RoundStateMachine,
    canonical_sha256,
    run_compiler_feedback_selection,
    summarize_abba,
    verify_compiler_spill_gate,
    verify_seal,
    verify_semantic_body_and_stubs,
    verify_stage_granularity,
    verify_sw_hw_coverage,
    verify_warp_completeness,
)
from weaveprof.workspace import write_json  # noqa: E402


def _ev(
    name: str,
    ts: float,
    dur: float,
    *,
    slot: int = 0,
    warp: int | None = None,
    semantic_id: str | None = None,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {"wave_slot": slot}
    if warp is not None:
        args["warp"] = warp
    if semantic_id is not None:
        args["semantic_id"] = semantic_id
    return {"ph": "X", "pid": 0, "tid": slot, "ts": ts, "dur": dur, "name": name, "args": args}


def _write_unsigned_seal(tmp: Path, *, limit: float, overhead_ratio: float) -> Path:
    src = tmp / "kernel.cu"
    fb = tmp / "feedback.json"
    tr = tmp / "trace.json"
    src.write_bytes(b"int main() { return 0; }\n")
    fb.write_bytes(b"{}\n")
    tr.write_bytes(b"{}\n")
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    fb_sha = hashlib.sha256(fb.read_bytes()).hexdigest()
    tr_sha = hashlib.sha256(tr.read_bytes()).hexdigest()
    pol = HarnessPolicy()
    doc: Dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "sealed_at": 1700000000.0,
        "round": 1,
        "prev_seal_sha256": None,
        "status": "PASS",
        "fsm_state": "VERIFY_AND_SEAL",
        "escalation": None,
        "limit": limit,
        "policy": pol.to_dict(),
        "policy_digest": pol.digest(),
        "bench_cmd_sha256": None,
        "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
        "failed": [],
        "overhead_ratio": overhead_ratio,
        "region_lock": {"gpu_id": "gpu_0"},
        "trace_processor": None,
        "artifacts": {
            "source": {"path": str(src), "sha256": src_sha, "bytes": src.stat().st_size},
            "feedback": {"path": str(fb), "sha256": fb_sha, "bytes": fb.stat().st_size},
            "trace": {"path": str(tr), "sha256": tr_sha, "bytes": tr.stat().st_size},
        },
    }
    doc["seal_sha256"] = canonical_sha256(doc)
    seal_path = tmp / "seal.json"
    write_json(seal_path, doc)
    return seal_path


class HarnessDefaultsPinTest(unittest.TestCase):
    """D1: Pin ``HarnessPolicy()`` fields and ``harness.py`` function parameter defaults."""

    def test_d1_harness_policy_and_function_signature_defaults(self) -> None:
        pol = HarnessPolicy()
        self.assertEqual(pol.max_overhead_limit, 0.10)
        self.assertEqual(pol.min_paired_samples, 3)
        self.assertEqual(pol.min_warmup, 1)
        self.assertIs(pol.require_abba_order, False)
        self.assertIs(pol.allow_loop_sampling, False)
        self.assertIs(pol.allow_mlir, False)
        self.assertIs(pol.require_source_immutable, True)
        self.assertIs(pol.enforce_trace_processor, False)
        self.assertIs(pol.enable_region_lock, True)
        self.assertEqual(pol.region_lock_timeout_s, 120.0)
        self.assertEqual(pol.command_timeout_s, 600.0)
        self.assertIs(pol.escalation_on_stagnation, True)
        self.assertIs(pol.export_failed_timeline, True)
        self.assertEqual(pol.max_timeline_hotspots, 6)
        self.assertIs(pol.allow_single_stage_collapse, False)
        self.assertIs(pol.allow_partial_warps, False)
        self.assertEqual(pol.max_delta_spill_bytes, 0)
        self.assertEqual(pol.max_delta_stack_bytes, 0)
        self.assertEqual(pol.max_delta_stl, 4)
        self.assertEqual(pol.max_delta_ldl, 4)
        self.assertIs(pol.require_output_files, False)

        sig_spill = inspect.signature(verify_compiler_spill_gate).parameters
        self.assertEqual(sig_spill["max_delta_spill_bytes"].default, 0)
        self.assertEqual(sig_spill["max_delta_stack_bytes"].default, 0)
        self.assertEqual(sig_spill["max_delta_stl"].default, 4)
        self.assertEqual(sig_spill["max_delta_ldl"].default, 4)
        self.assertIs(sig_spill["has_setmaxnreg"].default, False)
        self.assertIsNone(sig_spill["smem_free_bytes"].default)

        sig_sel = inspect.signature(run_compiler_feedback_selection).parameters
        self.assertEqual(sig_sel["l0_smem_bytes"].default, 0)
        self.assertIs(sig_sel["try_fast_first"].default, True)
        self.assertEqual(sig_sel["max_delta_stl"].default, 4)
        self.assertEqual(sig_sel["max_delta_ldl"].default, 4)

        sig_stub = inspect.signature(verify_semantic_body_and_stubs).parameters
        self.assertEqual(sig_stub["min_median_ns"].default, 128.0)
        self.assertIsNone(sig_stub["sass_body_insns"].default)
        self.assertEqual(sig_stub["min_fires_for_stub"].default, 4)

        sig_gran = inspect.signature(verify_stage_granularity).parameters
        self.assertEqual(sig_gran["max_single_stage_share"].default, 0.95)
        self.assertEqual(sig_gran["min_active_us"].default, 20.0)
        self.assertEqual(sig_gran["max_collapsed_spans_per_lane"].default, 1.05)

        sig_warp = inspect.signature(verify_warp_completeness).parameters
        self.assertIsNone(sig_warp["expected_warps"].default)
        self.assertEqual(sig_warp["min_hollow_lifetime_us"].default, 50.0)
        self.assertEqual(sig_warp["min_warp_coverage_ratio"].default, 0.05)

        sig_cov = inspect.signature(verify_sw_hw_coverage).parameters
        self.assertIsNone(sig_cov["in_window_samples"].default)
        self.assertEqual(sig_cov["expected_freq_hz"].default, 200000.0)
        self.assertEqual(sig_cov["max_span_delta_ratio"].default, 0.05)
        self.assertEqual(sig_cov["min_pmu_density"].default, 0.90)


class PolicyAndSealThresholdsTest(unittest.TestCase):
    """H01–H13: PolicyGuard, summarize_abba, verify_seal, and RoundStateMachine."""

    def test_h01_session_start_nonnegative_limit_boundary(self) -> None:
        guard = PolicyGuard()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "k.cu"
            src.write_bytes(b"int main() { return 0; }\n")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            limit_ok = 0.0
            self.assertEqual(limit_ok, 0.0)
            self.assertIsNone(guard.validate_session_start(src, sha, limit_ok, env={}))

            limit_bad = -1e-12
            self.assertEqual(limit_bad, -1e-12)
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_session_start(src, sha, limit_bad, env={})
            self.assertEqual(ctx.exception.code, "INVALID_LIMIT")

    def test_h02_session_start_ceiling_plus_1e12_boundary(self) -> None:
        guard = PolicyGuard()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "k.cu"
            src.write_bytes(b"int main() { return 0; }\n")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            at_ceiling = 0.10000000000100001
            self.assertEqual(at_ceiling, 0.10 + 1e-12)
            self.assertIsNone(guard.validate_session_start(src, sha, at_ceiling, env={}))

            above_ceiling = 0.1000000000011
            self.assertGreater(above_ceiling, 0.10000000000100001)
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_session_start(src, sha, above_ceiling, env={})
            self.assertEqual(ctx.exception.code, "LIMIT_RELAXATION_BLOCKED")
            self.assertEqual(ctx.exception.detail, {"requested_limit": 0.1000000000011, "policy_ceiling": 0.10})

    def test_h03_session_start_loop_sampling_env_boundary(self) -> None:
        guard = PolicyGuard()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "k.cu"
            src.write_bytes(b"int main() { return 0; }\n")
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
            self.assertIsNone(guard.validate_session_start(src, sha, 0.10, env={"WP_SAMPLE": "1"}))
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_session_start(src, sha, 0.10, env={"WP_SAMPLE": "2"})
            self.assertEqual(ctx.exception.code, "LOOP_SAMPLING_FORBIDDEN")
            self.assertEqual(ctx.exception.detail, {"env_var": "WP_SAMPLE", "value": "2"})

    def test_h04_runtime_protocol_min_warmup_boundary(self) -> None:
        guard = PolicyGuard()
        ok_pass, abba_pass, err_pass = guard.validate_runtime_protocol(
            {"warmup": 1, "baseline_ns": [1000.0, 1000.0, 1000.0], "instrumented_ns": [1050.0, 1050.0, 1050.0]},
            0.10,
        )
        self.assertIs(ok_pass, True)
        self.assertEqual(err_pass, [])
        self.assertEqual(abba_pass["paired_samples"], 3)

        ok_fail, _abba_fail, err_fail = guard.validate_runtime_protocol(
            {"warmup": 0, "baseline_ns": [1000.0, 1000.0, 1000.0], "instrumented_ns": [1050.0, 1050.0, 1050.0]},
            0.10,
        )
        self.assertIs(ok_fail, False)
        self.assertEqual(err_fail, ["warmup=0 < policy min_warmup=1"])

    def test_h05a_runtime_protocol_min_paired_samples_with_equal_lengths(self) -> None:
        guard = PolicyGuard()
        ok_pass, abba_pass, err_pass = guard.validate_runtime_protocol(
            {"warmup": 1, "baseline_ns": [1000.0] * 3, "instrumented_ns": [1050.0] * 3},
            0.10,
        )
        self.assertIs(ok_pass, True)
        self.assertEqual(err_pass, [])
        self.assertEqual(abba_pass["paired_samples"], 3)

        ok_fail, abba_fail, err_fail = guard.validate_runtime_protocol(
            {"warmup": 1, "baseline_ns": [1000.0] * 2, "instrumented_ns": [1050.0] * 2},
            0.10,
        )
        self.assertIs(ok_fail, False)
        self.assertEqual(abba_fail["paired_samples"], 2)
        self.assertEqual(err_fail, ["paired samples mismatch: baseline=2, instrumented=2 (min=3)"])

    def test_h05b_runtime_protocol_equal_length_with_min_samples_satisfied(self) -> None:
        guard = PolicyGuard()
        ok_pass, abba_pass, err_pass = guard.validate_runtime_protocol(
            {"warmup": 1, "baseline_ns": [1000.0] * 3, "instrumented_ns": [1050.0] * 3},
            0.10,
        )
        self.assertIs(ok_pass, True)
        self.assertEqual(err_pass, [])
        self.assertEqual(abba_pass["paired_samples"], 3)

        ok_fail, abba_fail, err_fail = guard.validate_runtime_protocol(
            {"warmup": 1, "baseline_ns": [1000.0] * 3, "instrumented_ns": [1050.0] * 4},
            0.10,
        )
        self.assertIs(ok_fail, False)
        self.assertEqual(abba_fail, {})
        self.assertEqual(err_fail, ["paired samples mismatch: baseline=3, instrumented=4 (min=3)"])

    def test_h06_runtime_protocol_ceiling_plus_1e12_boundary(self) -> None:
        guard = PolicyGuard()
        rm = {"warmup": 1, "baseline_ns": [1000.0] * 3, "instrumented_ns": [1050.0] * 3}
        ok_pass, _abba_pass, err_pass = guard.validate_runtime_protocol(rm, 0.10000000000100001)
        self.assertIs(ok_pass, True)
        self.assertEqual(err_pass, [])

        ok_fail, _abba_fail, err_fail = guard.validate_runtime_protocol(rm, 0.1000000000011)
        self.assertIs(ok_fail, False)
        self.assertEqual(err_fail, ["limit=0.1000000000011 exceeds policy ceiling 0.1"])

    def test_h07_summarize_abba_positive_sample_boundary(self) -> None:
        res_pos = summarize_abba([0.125, 0.125, 0.125], [0.125, 0.125, 0.125])
        self.assertEqual(res_pos["paired_samples"], 3)
        self.assertEqual(res_pos["median_overhead_ratio"], 0.0)

        res_zero = summarize_abba([0.0, 0.125, 0.125], [0.125, 0.125, 0.125])
        self.assertEqual(res_zero, {})

    def test_h08_verify_seal_nonnegative_limit_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d_ok = Path(td) / "ok"
            d_ok.mkdir()
            s_ok = _write_unsigned_seal(d_ok, limit=0.0, overhead_ratio=0.0)
            r_ok = verify_seal(s_ok, require_hmac=False)
            self.assertIs(r_ok["ok"], True)
            self.assertEqual(r_ok["errors"], [])
            self.assertEqual(r_ok["seal"]["limit"], 0.0)

            d_bad = Path(td) / "bad"
            d_bad.mkdir()
            s_bad = _write_unsigned_seal(d_bad, limit=-1e-12, overhead_ratio=0.0)
            r_bad = verify_seal(s_bad, require_hmac=False)
            self.assertIs(r_bad["ok"], False)
            self.assertEqual(r_bad["errors"], ["Invalid seal limit: -1e-12"])

    def test_h09_verify_seal_limit_ceiling_plus_1e12_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d_ok = Path(td) / "ok"
            d_ok.mkdir()
            s_ok = _write_unsigned_seal(d_ok, limit=0.10000000000100001, overhead_ratio=0.05)
            r_ok = verify_seal(s_ok, require_hmac=False)
            self.assertIs(r_ok["ok"], True)
            self.assertEqual(r_ok["errors"], [])
            self.assertEqual(r_ok["seal"]["limit"], 0.10000000000100001)

            d_bad = Path(td) / "bad"
            d_bad.mkdir()
            s_bad = _write_unsigned_seal(d_bad, limit=0.1000000000011, overhead_ratio=0.05)
            r_bad = verify_seal(s_bad, require_hmac=False)
            self.assertIs(r_bad["ok"], False)
            self.assertEqual(r_bad["errors"], ["Seal limit 0.1000000000011 exceeds policy ceiling 0.1"])

    def test_h10_verify_seal_overhead_ratio_ceiling_plus_1e12_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d_ok = Path(td) / "ok"
            d_ok.mkdir()
            s_ok = _write_unsigned_seal(d_ok, limit=0.10, overhead_ratio=0.10000000000100001)
            r_ok = verify_seal(s_ok, require_hmac=False)
            self.assertIs(r_ok["ok"], True)
            self.assertEqual(r_ok["errors"], [])
            self.assertEqual(r_ok["seal"]["overhead_ratio"], 0.10000000000100001)

            d_bad = Path(td) / "bad"
            d_bad.mkdir()
            s_bad = _write_unsigned_seal(d_bad, limit=0.10, overhead_ratio=0.1000000000011)
            r_bad = verify_seal(s_bad, require_hmac=False)
            self.assertIs(r_bad["ok"], False)
            self.assertEqual(r_bad["errors"], ["Measured overhead_ratio=0.1000000000011 exceeds ceiling=0.1"])

    def test_h11_round_fsm_max_rounds_boundary(self) -> None:
        fsm_cont = RoundStateMachine(max_rounds=3)
        fsm_cont.transition("POLICY_CHECK")
        fsm_cont.transition("CRITICAL_REGION_MEASURE")
        act_2, rsn_2 = fsm_cont.decide_after_round(
            2, "fp_a", {"status": "FAIL", "failed": ["overhead"], "metrics": {"overhead_ratio": 0.25}}
        )
        self.assertEqual((act_2, rsn_2), ("REFINE_LLM", "failed_gates_refinement"))
        self.assertEqual(fsm_cont.state, "REFINE_LLM")

        fsm_halt = RoundStateMachine(max_rounds=3)
        fsm_halt.transition("POLICY_CHECK")
        fsm_halt.transition("CRITICAL_REGION_MEASURE")
        act_3, rsn_3 = fsm_halt.decide_after_round(
            3, "fp_a", {"status": "FAIL", "failed": ["overhead"], "metrics": {"overhead_ratio": 0.25}}
        )
        self.assertEqual((act_3, rsn_3), ("HALT_BLOCKED", "max_rounds_exhausted"))
        self.assertEqual(fsm_halt.state, "HALT_BLOCKED")

    def test_h12_round_fsm_stagnation_delta_0p005_boundary_without_oscillation_or_final_retry(self) -> None:
        def _run_two_rounds(ovh_round2: float) -> tuple[str, str, int]:
            fsm = RoundStateMachine(max_rounds=5)
            fsm.transition("POLICY_CHECK")
            fsm.transition("CRITICAL_REGION_MEASURE")
            fsm.decide_after_round(
                1,
                "fp_1",
                {"status": "FAIL", "failed": ["overhead"], "enclosing_loop": [10, 20], "metrics": {"overhead_ratio": 0.25}},
            )
            fsm.transition("POLICY_CHECK")
            fsm.transition("CRITICAL_REGION_MEASURE")
            act, rsn = fsm.decide_after_round(
                2,
                "fp_2",
                {"status": "FAIL", "failed": ["overhead"], "enclosing_loop": [10, 20], "metrics": {"overhead_ratio": ovh_round2}},
            )
            return act, rsn, fsm._stagnation_count

        self.assertEqual(0.25 - 0.005, 0.245)
        self.assertEqual(_run_two_rounds(0.245), ("ESCALATE_DETERMINISTIC", "stagnation_detected", 1))
        self.assertEqual(_run_two_rounds(0.244875), ("REFINE_LLM", "failed_gates_refinement", 0))

    def test_h13a_round_fsm_final_retry_round_index_with_max_rounds_ge_3(self) -> None:
        fsm_r2 = RoundStateMachine(max_rounds=4)
        fsm_r2.transition("POLICY_CHECK")
        fsm_r2.transition("CRITICAL_REGION_MEASURE")
        act_r2, rsn_r2 = fsm_r2.decide_after_round(
            2,
            "fp_unique_2",
            {"status": "FAIL", "failed": ["overhead"], "enclosing_loop": [10, 20], "metrics": {"overhead_ratio": 0.20}},
        )
        self.assertEqual((act_r2, rsn_r2), ("REFINE_LLM", "failed_gates_refinement"))

        fsm_r3 = RoundStateMachine(max_rounds=4)
        fsm_r3.transition("POLICY_CHECK")
        fsm_r3.transition("CRITICAL_REGION_MEASURE")
        act_r3, rsn_r3 = fsm_r3.decide_after_round(
            3,
            "fp_unique_3",
            {"status": "FAIL", "failed": ["overhead"], "enclosing_loop": [10, 20], "metrics": {"overhead_ratio": 0.20}},
        )
        self.assertEqual((act_r3, rsn_r3), ("ESCALATE_DETERMINISTIC", "final_round_guaranteed_hoist"))

    def test_h13b_round_fsm_final_retry_min_max_rounds_3_boundary(self) -> None:
        fsm_m2 = RoundStateMachine(max_rounds=2)
        fsm_m2.transition("POLICY_CHECK")
        fsm_m2.transition("CRITICAL_REGION_MEASURE")
        act_m2, rsn_m2 = fsm_m2.decide_after_round(
            1,
            "fp_m2",
            {"status": "FAIL", "failed": ["overhead"], "enclosing_loop": [10, 20], "metrics": {"overhead_ratio": 0.20}},
        )
        self.assertEqual((act_m2, rsn_m2), ("REFINE_LLM", "failed_gates_refinement"))

        fsm_m3 = RoundStateMachine(max_rounds=3)
        fsm_m3.transition("POLICY_CHECK")
        fsm_m3.transition("CRITICAL_REGION_MEASURE")
        act_m3, rsn_m3 = fsm_m3.decide_after_round(
            2,
            "fp_m3",
            {"status": "FAIL", "failed": ["overhead"], "enclosing_loop": [10, 20], "metrics": {"overhead_ratio": 0.20}},
        )
        self.assertEqual((act_m3, rsn_m3), ("ESCALATE_DETERMINISTIC", "final_round_guaranteed_hoist"))


class CompilerSpillThresholdsTest(unittest.TestCase):
    """H14–H20: ``verify_compiler_spill_gate`` and ``run_compiler_feedback_selection``."""

    def test_h14_spill_store_delta_zero_boundary(self) -> None:
        r_ok = verify_compiler_spill_gate({"spill_stores": 16}, {"spill_stores": 16})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["metrics"]["delta_spill_stores_bytes"], 0)
        self.assertEqual(r_ok["violations"], [])

        r_fail = verify_compiler_spill_gate({"spill_stores": 16}, {"spill_stores": 17})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["metrics"]["delta_spill_stores_bytes"], 1)
        self.assertEqual(r_fail["violations"][0]["code"], "SPILL_STORE_INCREASE")
        self.assertEqual(r_fail["violations"][0]["delta"], 1)
        self.assertEqual(r_fail["violations"][0]["limit"], 0)

    def test_h15_spill_load_delta_zero_boundary(self) -> None:
        r_ok = verify_compiler_spill_gate({"spill_loads": 8}, {"spill_loads": 8})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["metrics"]["delta_spill_loads_bytes"], 0)
        self.assertEqual(r_ok["violations"], [])

        r_fail = verify_compiler_spill_gate({"spill_loads": 8}, {"spill_loads": 9})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["metrics"]["delta_spill_loads_bytes"], 1)
        self.assertEqual(r_fail["violations"][0]["code"], "SPILL_LOAD_INCREASE")
        self.assertEqual(r_fail["violations"][0]["delta"], 1)
        self.assertEqual(r_fail["violations"][0]["limit"], 0)

    def test_h16_stack_frame_delta_zero_boundary(self) -> None:
        r_ok = verify_compiler_spill_gate({"stack_bytes": 32}, {"stack_bytes": 32})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["metrics"]["delta_stack_bytes"], 0)
        self.assertEqual(r_ok["violations"], [])

        r_fail = verify_compiler_spill_gate({"stack_bytes": 32}, {"stack_bytes": 33})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["metrics"]["delta_stack_bytes"], 1)
        self.assertEqual(r_fail["violations"][0]["code"], "STACK_FRAME_INCREASE")
        self.assertEqual(r_fail["violations"][0]["delta"], 1)
        self.assertEqual(r_fail["violations"][0]["limit"], 0)

    def test_h17_sass_stl_delta_four_boundary(self) -> None:
        r_ok = verify_compiler_spill_gate({"stl": 10}, {"stl": 14})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["metrics"]["delta_sass_stl"], 4)
        self.assertEqual(r_ok["violations"], [])

        r_fail = verify_compiler_spill_gate({"stl": 10}, {"stl": 15})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["metrics"]["delta_sass_stl"], 5)
        self.assertEqual(r_fail["violations"][0]["code"], "SASS_STL_EXPLOSION")
        self.assertEqual(r_fail["violations"][0]["delta"], 5)
        self.assertEqual(r_fail["violations"][0]["limit"], 4)

    def test_h18_sass_ldl_delta_four_boundary(self) -> None:
        r_ok = verify_compiler_spill_gate({"ldl": 10}, {"ldl": 14})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["metrics"]["delta_sass_ldl"], 4)
        self.assertEqual(r_ok["violations"], [])

        r_fail = verify_compiler_spill_gate({"ldl": 10}, {"ldl": 15})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["metrics"]["delta_sass_ldl"], 5)
        self.assertEqual(r_fail["violations"][0]["code"], "SASS_LDL_EXPLOSION")
        self.assertEqual(r_fail["violations"][0]["delta"], 5)
        self.assertEqual(r_fail["violations"][0]["limit"], 4)

    def test_h19_register_count_blind_spot_boundary(self) -> None:
        r_blind = verify_compiler_spill_gate({"regs": 128, "spill_stores": 0}, {"regs": 128, "spill_stores": 4})
        self.assertIs(r_blind["ok"], False)
        self.assertIs(r_blind["register_count_blind_spot"], True)
        self.assertEqual(r_blind["metrics"]["l0_regs"], 128)
        self.assertEqual(r_blind["metrics"]["inst_regs"], 128)

        r_visible = verify_compiler_spill_gate({"regs": 128, "spill_stores": 0}, {"regs": 129, "spill_stores": 4})
        self.assertIs(r_visible["ok"], False)
        self.assertIs(r_visible["register_count_blind_spot"], False)
        self.assertEqual(r_visible["metrics"]["l0_regs"], 128)
        self.assertEqual(r_visible["metrics"]["inst_regs"], 129)

    def test_h20_smem_free_16_bytes_boundary_in_gate_and_feedback_selection(self) -> None:
        r_16 = verify_compiler_spill_gate(
            {"spill_stores": 0}, {"spill_stores": 4}, smem_free_bytes=16
        )
        self.assertIs(r_16["ok"], False)
        self.assertEqual(
            r_16["remediation"],
            ["enable_WP_CTX_SMEM (-DWP_CTX_SMEM, 16B SMEM + __noinline__ cold path)"],
        )

        r_15 = verify_compiler_spill_gate(
            {"spill_stores": 0}, {"spill_stores": 4}, smem_free_bytes=15
        )
        self.assertIs(r_15["ok"], False)
        self.assertEqual(r_15["remediation"], ["coarsen_probes_or_hoist_outside_hot_loop"])

        # Note: causal.plan_context_and_search_space checks smem_free >= 16 * max_warps_per_cta (default 32 -> 512 B)
        # while run_compiler_feedback_selection checks smem_free >= 16 B directly when try_fast_first=True.
        def _failing_fast_compile(arm: Dict[str, Any]) -> Dict[str, Any]:
            if arm["mode"] == "WP_CTX_REG":
                return {"regs": 64, "spill_bytes": 8, "stack_bytes": 0, "stl_count": 0, "ldl_count": 0}
            return {"regs": 64, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 0, "ldl_count": 0}

        sel_16 = run_compiler_feedback_selection(
            _failing_fast_compile,
            "__global__ void k() {}",
            baseline_metrics={"regs": 64, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 0, "ldl_count": 0},
            l0_smem_bytes=1008,
            smem_limit_bytes=1024,
        )
        self.assertEqual(sel_16["smem_free_bytes"], 16)
        self.assertEqual([a["mode"] for a in sel_16["attempts"]], ["WP_CTX_REG", "WP_CTX_SMEM", "WP_CTX_REG_NOINLINE"])
        self.assertEqual(sel_16["compile_calls"], 3)

        sel_15 = run_compiler_feedback_selection(
            _failing_fast_compile,
            "__global__ void k() {}",
            baseline_metrics={"regs": 64, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 0, "ldl_count": 0},
            l0_smem_bytes=1009,
            smem_limit_bytes=1024,
        )
        self.assertEqual(sel_15["smem_free_bytes"], 15)
        self.assertEqual([a["mode"] for a in sel_15["attempts"]], ["WP_CTX_REG", "WP_CTX_REG_NOINLINE"])
        self.assertEqual(sel_15["compile_calls"], 2)


class SemanticStubAndGranularityThresholdsTest(unittest.TestCase):
    """H21–H38: ``verify_semantic_body_and_stubs``, ``verify_stage_granularity``, ``verify_warp_completeness``."""

    def test_h21_stub_sass_body_insns_zero_boundary_with_self_latency_false(self) -> None:
        r_stub = verify_semantic_body_and_stubs({"s0": [512.0] * 8}, sass_body_insns={"s0": 0})
        self.assertIs(r_stub["ok"], False)
        self.assertEqual(r_stub["stub_count"], 1)
        self.assertEqual(r_stub["stubs"][0]["sass_body_insns"], 0)
        self.assertEqual(r_stub["stubs"][0]["reasons"], ["zero_non_probe_sass_instructions"])

        r_ok = verify_semantic_body_and_stubs({"s0": [512.0] * 8}, sass_body_insns={"s0": 1})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["stub_count"], 0)
        self.assertEqual(r_ok["valid_stages"], ["s0"])

    def test_h22_stub_median_128ns_boundary_with_fires_and_p90_satisfied(self) -> None:
        r_stub = verify_semantic_body_and_stubs({"s0": [128.0] * 8})
        self.assertIs(r_stub["ok"], False)
        self.assertEqual(r_stub["stub_count"], 1)
        self.assertEqual(r_stub["stubs"][0]["median_ns"], 128.0)
        self.assertEqual(r_stub["stubs"][0]["p90_ns"], 128.0)

        durs_ok = [128.0625] * 8
        self.assertEqual(statistics.median(durs_ok), 128.0625)
        r_ok = verify_semantic_body_and_stubs({"s0": durs_ok})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["stub_count"], 0)
        self.assertEqual(r_ok["valid_stages"], ["s0"])

    def test_h23_stub_min_fires_4_boundary_with_median_and_p90_satisfied(self) -> None:
        r_stub = verify_semantic_body_and_stubs({"s0": [64.0] * 4})
        self.assertIs(r_stub["ok"], False)
        self.assertEqual(r_stub["stubs"][0]["fires"], 4)
        self.assertEqual(r_stub["stubs"][0]["median_ns"], 64.0)

        durs_ok = [64.0] * 3
        self.assertEqual(len(durs_ok), 3)
        r_ok = verify_semantic_body_and_stubs({"s0": durs_ok})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["stub_count"], 0)
        self.assertEqual(r_ok["valid_stages"], ["s0"])

    def test_h24_stub_p90_256ns_boundary_with_median_and_fires_satisfied(self) -> None:
        durs_stub = [64.0] * 9 + [256.0, 512.0]
        r_stub = verify_semantic_body_and_stubs({"s0": durs_stub})
        self.assertIs(r_stub["ok"], False)
        self.assertEqual(r_stub["stubs"][0]["median_ns"], 64.0)
        self.assertEqual(r_stub["stubs"][0]["p90_ns"], 256.0)

        durs_ok = [64.0] * 9 + [256.125, 512.0]
        p90_ok = sorted(durs_ok)[min(len(durs_ok) - 1, int(round(0.90 * (len(durs_ok) - 1))))]
        self.assertEqual(p90_ok, 256.125)
        r_ok = verify_semantic_body_and_stubs({"s0": durs_ok})
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["stub_count"], 0)
        self.assertEqual(r_ok["valid_stages"], ["s0"])

    def test_h25_granularity_nested_lane_interval_cap_64_boundary(self) -> None:
        def _trace_with_n_inner(n_inner: int) -> Dict[str, Any]:
            evs = [_ev("mma", 0.0, 100.0), _ev("epilogue", 100.0, 10.0)]
            for i in range(n_inner):
                evs.append(_ev("load", 1.0 + float(i), 0.5))
            return {"traceEvents": evs, "otherData": {"gran": 1}}

        tr_64 = _trace_with_n_inner(62)
        self.assertEqual(len(tr_64["traceEvents"]), 64)
        r_64 = verify_stage_granularity(tr_64)
        self.assertIs(r_64["ok"], False)
        self.assertEqual(r_64["code"], "NESTED_WRAPPER_COLLAPSE")

        tr_65 = _trace_with_n_inner(63)
        self.assertEqual(len(tr_65["traceEvents"]), 65)
        r_65 = verify_stage_granularity(tr_65)
        self.assertIs(r_65["ok"], True)
        self.assertIsNone(r_65["code"])

    def test_h26a_granularity_enclosure_start_tolerance_1e6_boundary(self) -> None:
        tr_enclosed = {
            "traceEvents": [
                _ev("mma", 10.000001, 90.0),
                _ev("load", 10.0, 20.0),
                _ev("epilogue", 101.0, 5.0),
            ],
            "otherData": {"gran": 1},
        }
        self.assertEqual(10.0 + 1e-6, 10.000001)
        r_fail = verify_stage_granularity(tr_enclosed)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "NESTED_WRAPPER_COLLAPSE")

        tr_not_enclosed = {
            "traceEvents": [
                _ev("mma", 10.0000010005, 90.0),
                _ev("load", 10.0, 20.0),
                _ev("epilogue", 101.0, 5.0),
            ],
            "otherData": {"gran": 1},
        }
        self.assertGreater(10.0000010005, 10.000001)
        r_ok = verify_stage_granularity(tr_not_enclosed)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])

    def test_h26b_granularity_enclosure_end_tolerance_1e6_boundary(self) -> None:
        eb_in = 25.0 + 75.000001
        self.assertEqual(eb_in, 100.000001)
        tr_in = {
            "traceEvents": [
                _ev("mma", 0.0, 100.0),
                _ev("load", 25.0, 75.000001),
                _ev("epilogue", 101.0, 10.0),
            ],
            "otherData": {"gran": 1},
        }
        r_fail = verify_stage_granularity(tr_in)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "NESTED_WRAPPER_COLLAPSE")

        eb_out = 25.0 + 75.000002
        self.assertEqual(eb_out, 100.000002)
        tr_out = {
            "traceEvents": [
                _ev("mma", 0.0, 100.0),
                _ev("load", 25.0, 75.000002),
                _ev("epilogue", 101.0, 10.0),
            ],
            "otherData": {"gran": 1},
        }
        r_ok = verify_stage_granularity(tr_out)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])

    def test_h26c_granularity_enclosure_span_ratio_1p05_boundary(self) -> None:
        self.assertEqual(80.0 * 1.05, 84.0)
        tr_at = {
            "traceEvents": [
                _ev("mma", 0.0, 84.0),
                _ev("load", 0.0, 80.0),
                _ev("epilogue", 100.0, 5.0),
            ],
            "otherData": {"gran": 1},
        }
        r_ok = verify_stage_granularity(tr_at)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])

        tr_above = {
            "traceEvents": [
                _ev("mma", 0.0, 84.0625),
                _ev("load", 0.0, 80.0),
                _ev("epilogue", 100.0, 5.0),
            ],
            "otherData": {"gran": 1},
        }
        self.assertEqual(84.0625 / 80.0, 1.05078125)
        r_fail = verify_stage_granularity(tr_above)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "NESTED_WRAPPER_COLLAPSE")

    def test_h27_granularity_nested_wrapper_spans_per_lane_1p5_boundary(self) -> None:
        def _build(extra_spans: int) -> Dict[str, Any]:
            evs: List[Dict[str, Any]] = []
            for slot in range(2000):
                evs.append(_ev("load", 1.0, 20.0, slot=slot))
                if slot < 1000:
                    evs.append(_ev("mma", 0.0, 100.0, slot=slot))
                else:
                    evs.append(_ev("mma", 0.0, 50.0, slot=slot))
                    evs.append(_ev("mma", 50.0, 50.0, slot=slot))
            for k in range(extra_spans):
                evs.append(_ev("mma", 100.0 + float(k), 1.0, slot=k))
            return {"traceEvents": evs, "otherData": {"gran": 1}}

        r_fail = verify_stage_granularity(_build(0))
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "NESTED_WRAPPER_COLLAPSE")
        self.assertEqual(r_fail["spans_per_lane"], 1.5)

        r_ok = verify_stage_granularity(_build(2))
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(3002 / 2000.0, 1.501)
        self.assertEqual(r_ok["spans_per_lane"], 1.501)

    def test_h28_granularity_nested_wrapper_min_active_20us_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [_ev("mma", 0.0, 20.0), _ev("load", 1.0, 10.0)],
            "otherData": {"gran": 1},
        }
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "NESTED_WRAPPER_COLLAPSE")
        self.assertEqual(r_fail["per_lane_us"], 20.0)

        tr_ok = {
            "traceEvents": [_ev("mma", 0.0, 19.984375), _ev("load", 1.0, 10.0)],
            "otherData": {"gran": 1},
        }
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["per_lane_us"], 19.984)
        self.assertEqual(19.984375, 20.0 - 0.015625)

    def test_h29_granularity_nested_wrapper_union_share_0p85_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [
                _ev("mma", 0.0, 85.0),
                _ev("load", 1.0, 20.0),
                _ev("epilogue", 100.0, 15.0),
            ],
            "otherData": {"gran": 1},
        }
        self.assertEqual(85.0 / 100.0, 0.85)
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "NESTED_WRAPPER_COLLAPSE")

        tr_ok = {
            "traceEvents": [
                _ev("mma", 0.0, 84.9375),
                _ev("load", 1.0, 20.0),
                _ev("epilogue", 100.0, 15.0625),
            ],
            "otherData": {"gran": 1},
        }
        self.assertEqual(84.9375 / 100.0, 0.849375)
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])

    def test_h30_granularity_single_collapsed_min_active_20us_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [_ev("mma", 0.0, 20.0), _ev("load", 20.0, 0.5)],
            "otherData": {"gran": 1},
        }
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "SINGLE_STAGE_LOOP_COLLAPSE")
        self.assertEqual(r_fail["per_lane_us"], 20.0)

        tr_ok = {
            "traceEvents": [_ev("mma", 0.0, 19.984375), _ev("load", 20.0, 0.5)],
            "otherData": {"gran": 1},
        }
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["per_lane_us"], 19.984)

    def test_h31a_granularity_single_collapsed_disjoint_dom_share_0p95_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [_ev("mma", 0.0, 950.0), _ev("load", 950.0, 50.0)],
            "otherData": {"gran": 1},
        }
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "SINGLE_STAGE_LOOP_COLLAPSE")
        self.assertEqual(r_fail["dominant_share_pct"], 95.0)

        tr_ok = {
            "traceEvents": [_ev("mma", 0.0, 949.5), _ev("load", 949.5, 50.5)],
            "otherData": {"gran": 1},
        }
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["dominant_share_pct"], 94.95)

    def test_h31b_granularity_single_collapsed_dom_union_share_0p95_with_dom_share_below_0p95(self) -> None:
        # Partial overlap [0, 950] and [500, 1000]: not enclosed, total_us=1450 (dom_share=65.52% < 95%), union=1000.
        tr_fail = {
            "traceEvents": [_ev("mma", 0.0, 950.0), _ev("load", 500.0, 500.0)],
            "otherData": {"gran": 1},
        }
        self.assertEqual(950.0 / 1000.0, 0.95)
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "SINGLE_STAGE_LOOP_COLLAPSE")
        self.assertEqual(r_fail["dominant_share_pct"], 65.52)

        tr_ok = {
            "traceEvents": [_ev("mma", 0.0, 949.5), _ev("load", 500.0, 500.0)],
            "otherData": {"gran": 1},
        }
        self.assertEqual(949.5 / 1000.0, 0.9495)
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["dominant_share_pct"], 65.51)

    def test_h32_granularity_single_collapsed_spans_per_lane_1p05_boundary(self) -> None:
        evs_fail: List[Dict[str, Any]] = []
        for slot in range(20):
            evs_fail.append(_ev("mma", 0.0, 96.0, slot=slot))
            evs_fail.append(_ev("load", 100.0, 1.0, slot=slot))
        evs_fail.append(_ev("mma", 96.0, 1.0, slot=0))
        r_fail = verify_stage_granularity({"traceEvents": evs_fail, "otherData": {"gran": 1}})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "SINGLE_STAGE_LOOP_COLLAPSE")
        self.assertEqual(r_fail["spans_per_lane"], 1.05)

        evs_ok: List[Dict[str, Any]] = []
        for slot in range(2000):
            evs_ok.append(_ev("mma", 0.0, 96.0, slot=slot))
            evs_ok.append(_ev("load", 100.0, 1.0, slot=slot))
            if slot < 102:
                evs_ok.append(_ev("mma", 96.0, 1.0, slot=slot))
        self.assertEqual(2102 / 2000.0, 1.051)
        r_ok = verify_stage_granularity({"traceEvents": evs_ok, "otherData": {"gran": 1}})
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["spans_per_lane"], 1.051)

    def test_h33a_granularity_multi_coarse_no_mem_overall_per_lane_60us_boundary(self) -> None:
        # Characterization note: "softmax" contains substring "tma" and sets has_memory_stage=True via _MEM_SEM_RE;
        # use "gemm" (pure compute without "tma" substring) so has_memory_stage is False.
        r_softmax_quirk = verify_stage_granularity(
            {"traceEvents": [_ev("mma", 0.0, 51.0), _ev("softmax", 51.0, 9.0)], "otherData": {"gran": 1}}
        )
        self.assertIs(r_softmax_quirk["has_memory_stage"], True)
        self.assertIs(r_softmax_quirk["ok"], True)

        tr_fail = {
            "traceEvents": [_ev("mma", 0.0, 51.0), _ev("gemm", 51.0, 9.0)],
            "otherData": {"gran": 1},
        }
        self.assertEqual(51.0 + 9.0, 60.0)
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["has_memory_stage"], False)
        self.assertIs(r_fail["has_compute_stage"], True)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")

        tr_ok = {
            "traceEvents": [_ev("mma", 0.0, 51.0), _ev("gemm", 51.0, 8.96875)],
            "otherData": {"gran": 1},
        }
        self.assertEqual(51.0 + 8.96875, 59.96875)
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])

    def test_h33b_granularity_multi_coarse_no_mem_overall_spans_per_lane_8_boundary(self) -> None:
        def _build(extra: bool) -> Dict[str, Any]:
            evs: List[Dict[str, Any]] = []
            for slot in range(250):
                evs.append(_ev("mma", 0.0, 60.0, slot=slot))
                for k in range(7):
                    evs.append(_ev("gemm", 60.0 + 5.0 * k, 5.0, slot=slot))
            if extra:
                evs.append(_ev("gemm", 95.0, 5.0, slot=0))
            return {"traceEvents": evs, "otherData": {"gran": 1}}

        r_fail = verify_stage_granularity(_build(False))
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertEqual(r_fail["overall_spans_per_lane"], 8.0)

        r_ok = verify_stage_granularity(_build(True))
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["overall_spans_per_lane"], 8.004)

    def test_h33c_granularity_multi_coarse_no_mem_dom_spans_per_lane_1p5_boundary(self) -> None:
        def _build(extra_spans: int) -> Dict[str, Any]:
            evs: List[Dict[str, Any]] = []
            for slot in range(2000):
                if slot < 1000:
                    evs.append(_ev("mma", 0.0, 60.0, slot=slot))
                else:
                    evs.append(_ev("mma", 0.0, 30.0, slot=slot))
                    evs.append(_ev("mma", 30.0, 30.0, slot=slot))
                evs.append(_ev("gemm", 60.0, 30.0, slot=slot))
            for k in range(extra_spans):
                evs.append(_ev("mma", 90.0 + float(k), 1.0, slot=k))
            return {"traceEvents": evs, "otherData": {"gran": 1}}

        r_fail = verify_stage_granularity(_build(0))
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertEqual(r_fail["spans_per_lane"], 1.5)

        r_ok = verify_stage_granularity(_build(2))
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(3002 / 2000.0, 1.501)
        self.assertEqual(r_ok["spans_per_lane"], 1.501)

    def test_h33d_granularity_multi_coarse_no_mem_dom_per_lane_50us_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [_ev("mma", 0.0, 50.0), _ev("gemm", 50.0, 30.0)],
            "otherData": {"gran": 1},
        }
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertEqual(r_fail["per_lane_us"], 50.0)

        tr_ok = {
            "traceEvents": [_ev("mma", 0.0, 49.96875), _ev("gemm", 49.96875, 30.03125)],
            "otherData": {"gran": 1},
        }
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["per_lane_us"], 49.969)
        self.assertEqual(49.96875, 50.0 - 0.03125)

    def test_h34a_granularity_long_warp_gran_zero_vs_one_boundary(self) -> None:
        evs = [_ev("mma", 0.0, 300.0), _ev("load", 300.0, 200.0)]
        r_fail = verify_stage_granularity({"traceEvents": evs, "otherData": {"gran": 0}})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")

        r_ok = verify_stage_granularity({"traceEvents": evs, "otherData": {"gran": 1}})
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])

    def test_h34b_granularity_long_warp_overall_per_lane_450us_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [_ev("mma", 0.0, 270.0), _ev("load", 270.0, 180.0)],
            "otherData": {"gran": 0},
        }
        self.assertEqual(270.0 + 180.0, 450.0)
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")

        tr_ok = {
            "traceEvents": [_ev("mma", 0.0, 270.0), _ev("load", 270.0, 179.75)],
            "otherData": {"gran": 0},
        }
        self.assertEqual(270.0 + 179.75, 449.75)
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])

    def test_h34c_granularity_long_warp_dom_share_0p40_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [
                _ev("mma", 0.0, 400.0),
                _ev("load", 400.0, 300.0),
                _ev("stage_c", 700.0, 300.0),
            ],
            "otherData": {"gran": 0},
        }
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertEqual(r_fail["dominant_share_pct"], 40.0)

        tr_ok = {
            "traceEvents": [
                _ev("mma", 0.0, 399.75),
                _ev("load", 399.75, 300.125),
                _ev("stage_c", 699.875, 300.125),
            ],
            "otherData": {"gran": 0},
        }
        self.assertEqual(399.75 / 1000.0, 0.39975)
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["dominant_share_pct"], 39.98)

    def test_h34d_granularity_long_warp_dom_per_lane_250us_boundary(self) -> None:
        tr_fail = {
            "traceEvents": [
                _ev("mma", 0.0, 250.0),
                _ev("load", 250.0, 125.0),
                _ev("stage_c", 375.0, 125.0),
            ],
            "otherData": {"gran": 0},
        }
        r_fail = verify_stage_granularity(tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertEqual(r_fail["per_lane_us"], 250.0)

        tr_ok = {
            "traceEvents": [
                _ev("mma", 0.0, 249.875),
                _ev("load", 249.875, 125.0625),
                _ev("stage_c", 374.9375, 125.0625),
            ],
            "otherData": {"gran": 0},
        }
        r_ok = verify_stage_granularity(tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["per_lane_us"], 249.875)

    def test_h34e_granularity_long_warp_dom_spans_per_lane_1p5_boundary(self) -> None:
        def _build(extra_spans: int) -> Dict[str, Any]:
            evs: List[Dict[str, Any]] = []
            for slot in range(2000):
                if slot < 1000:
                    evs.append(_ev("mma", 0.0, 300.0, slot=slot))
                else:
                    evs.append(_ev("mma", 0.0, 150.0, slot=slot))
                    evs.append(_ev("mma", 150.0, 150.0, slot=slot))
                evs.append(_ev("load", 300.0, 200.0, slot=slot))
            for k in range(extra_spans):
                evs.append(_ev("mma", 500.0 + float(k), 1.0, slot=k))
            return {"traceEvents": evs, "otherData": {"gran": 0}}

        r_fail = verify_stage_granularity(_build(0))
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertEqual(r_fail["spans_per_lane"], 1.5)

        r_ok = verify_stage_granularity(_build(2))
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(3002 / 2000.0, 1.501)
        self.assertEqual(r_ok["spans_per_lane"], 1.501)

    def test_h34f_granularity_long_warp_overall_spans_per_lane_6_boundary(self) -> None:
        def _build(extra: bool) -> Dict[str, Any]:
            evs: List[Dict[str, Any]] = []
            for slot in range(200):
                evs.append(_ev("mma", 0.0, 300.0, slot=slot))
                for k in range(5):
                    evs.append(_ev("load", 300.0 + 40.0 * k, 40.0, slot=slot))
            if extra:
                evs.append(_ev("load", 500.0, 1.0, slot=0))
            return {"traceEvents": evs, "otherData": {"gran": 0}}

        r_fail = verify_stage_granularity(_build(False))
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertEqual(r_fail["overall_spans_per_lane"], 6.0)

        r_ok = verify_stage_granularity(_build(True))
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["overall_spans_per_lane"], 6.005)

    def test_h35_warp_completeness_wpc_and_missing_warp_boundary(self) -> None:
        tr_w0 = {"traceEvents": [_ev("mma", 0.0, 50.0, slot=0, warp=0)]}
        r_wpc1 = verify_warp_completeness({"gran": 0, "wpc": 1, "lane_bounds_ns": {"0": [0, 100000]}}, tr_w0)
        self.assertIs(r_wpc1["ok"], True)
        self.assertIsNone(r_wpc1["code"])

        r_missing = verify_warp_completeness({"gran": 0, "wpc": 2, "lane_bounds_ns": {"0": [0, 100000]}}, tr_w0)
        self.assertIs(r_missing["ok"], False)
        self.assertEqual(r_missing["code"], "WARP_COVERAGE_INCOMPLETE")
        self.assertEqual(r_missing["missing_warps"], [1])

        tr_both = {
            "traceEvents": [
                _ev("mma", 0.0, 50.0, slot=0, warp=0),
                _ev("load", 0.0, 50.0, slot=1, warp=1),
            ]
        }
        r_all = verify_warp_completeness(
            {"gran": 0, "wpc": 2, "lane_bounds_ns": {"0": [0, 100000], "1": [0, 100000]}},
            tr_both,
        )
        self.assertIs(r_all["ok"], True)
        self.assertIsNone(r_all["code"])
        self.assertEqual(r_all["missing_warps"], [])

    def test_h36_warp_completeness_hollow_lifetime_50us_boundary(self) -> None:
        tr = {
            "traceEvents": [
                _ev("mma", 0.0, 1.0, slot=0, warp=0),
                _ev("load", 0.0, 50.0, slot=1, warp=1),
            ]
        }
        r_early = verify_warp_completeness(
            {"gran": 0, "wpc": 2, "lane_bounds_ns": {"0": [0, 49960], "1": [0, 100000]}},
            tr,
        )
        self.assertEqual(49960 / 1000.0, 49.96)
        self.assertIs(r_early["ok"], True)
        self.assertIsNone(r_early["code"])
        self.assertEqual(r_early["hollow_warps"], [])

        r_hollow = verify_warp_completeness(
            {"gran": 0, "wpc": 2, "lane_bounds_ns": {"0": [0, 50000], "1": [0, 100000]}},
            tr,
        )
        self.assertIs(r_hollow["ok"], False)
        self.assertEqual(r_hollow["code"], "HOLLOW_WARP_LANE")
        self.assertEqual(r_hollow["hollow_warps"][0]["median_lifetime_us"], 50.0)

    def test_h37_warp_completeness_hollow_spans_per_cta_1p05_boundary_with_low_coverage(self) -> None:
        lb_20 = {str(2 * c): [0, 100000] for c in range(20)}
        lb_20.update({str(2 * c + 1): [0, 100000] for c in range(20)})
        evs_20 = [_ev("mma", 0.0, 0.5, slot=2 * c, warp=0) for c in range(20)]
        evs_20.append(_ev("mma", 1.0, 0.5, slot=0, warp=0))
        evs_20.extend(_ev("load", 0.0, 50.0, slot=2 * c + 1, warp=1) for c in range(20))
        r_fail = verify_warp_completeness({"gran": 0, "wpc": 2, "lane_bounds_ns": lb_20}, {"traceEvents": evs_20})
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "HOLLOW_WARP_LANE")
        self.assertEqual(r_fail["hollow_warps"][0]["spans_per_cta"], 1.05)

        lb_2000 = {str(2 * c): [0, 100000] for c in range(2000)}
        lb_2000.update({str(2 * c + 1): [0, 100000] for c in range(2000)})
        evs_2000 = [_ev("mma", 0.0, 0.5, slot=2 * c, warp=0) for c in range(2000)]
        evs_2000.extend(_ev("mma", 1.0, 0.5, slot=2 * c, warp=0) for c in range(101))
        evs_2000.extend(_ev("load", 0.0, 50.0, slot=2 * c + 1, warp=1) for c in range(2000))
        self.assertEqual(2101 / 2000.0, 1.0505)
        r_ok = verify_warp_completeness({"gran": 0, "wpc": 2, "lane_bounds_ns": lb_2000}, {"traceEvents": evs_2000})
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["hollow_warps"], [])

    def test_h38_warp_completeness_hollow_cov_ratio_0p05_boundary_with_low_spans(self) -> None:
        lb = {"0": [0, 1000000], "1": [0, 1000000]}
        tr_fail = {
            "traceEvents": [
                _ev("mma", 0.0, 49.96875, slot=0, warp=0),
                _ev("load", 0.0, 500.0, slot=1, warp=1),
            ]
        }
        self.assertEqual(49.96875 / 1000.0, 0.04996875)
        r_fail = verify_warp_completeness({"gran": 0, "wpc": 2, "lane_bounds_ns": lb}, tr_fail)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["code"], "HOLLOW_WARP_LANE")
        self.assertEqual(r_fail["hollow_warps"][0]["coverage_pct"], 5.0)

        tr_ok = {
            "traceEvents": [
                _ev("mma", 0.0, 50.0, slot=0, warp=0),
                _ev("load", 0.0, 500.0, slot=1, warp=1),
            ]
        }
        self.assertEqual(50.0 / 1000.0, 0.05)
        r_ok = verify_warp_completeness({"gran": 0, "wpc": 2, "lane_bounds_ns": lb}, tr_ok)
        self.assertIs(r_ok["ok"], True)
        self.assertIsNone(r_ok["code"])
        self.assertEqual(r_ok["hollow_warps"], [])


class SwHwCoverageThresholdsTest(unittest.TestCase):
    """H39–H42: ``verify_sw_hw_coverage``."""

    def test_h39_positive_hw_and_sw_duration_boundaries(self) -> None:
        r_hw0 = verify_sw_hw_coverage(100.0, 0.0)
        self.assertIs(r_hw0["ok"], False)
        self.assertEqual(r_hw0["violations"][0]["code"], "INVALID_HW_KERNEL_DURATION")

        r_sw0 = verify_sw_hw_coverage(0.0, 100.0)
        self.assertIs(r_sw0["ok"], False)
        self.assertEqual(r_sw0["violations"][0]["code"], "EMPTY_SW_SPAN")

        r_ok = verify_sw_hw_coverage(0.125, 0.125)
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["span_delta_ratio"], 0.0)

    def test_h40_span_delta_ratio_0p05_positive_and_negative_boundaries(self) -> None:
        r_pos_ok = verify_sw_hw_coverage(1050.0, 1000.0)
        self.assertIs(r_pos_ok["ok"], True)
        self.assertEqual(r_pos_ok["span_delta_ratio"], 0.05)
        self.assertEqual(r_pos_ok["violations"], [])

        r_pos_fail = verify_sw_hw_coverage(1050.03125, 1000.0)
        self.assertEqual(abs(1050.03125 - 1000.0) / 1000.0, 0.05003125)
        self.assertIs(r_pos_fail["ok"], False)
        self.assertEqual(r_pos_fail["span_delta_ratio"], 0.05003)
        self.assertEqual(r_pos_fail["violations"][0]["code"], "SW_HW_SPAN_TRUNCATION")

        r_neg_ok = verify_sw_hw_coverage(950.0, 1000.0)
        self.assertIs(r_neg_ok["ok"], True)
        self.assertEqual(r_neg_ok["span_delta_ratio"], 0.05)
        self.assertEqual(r_neg_ok["violations"], [])

        r_neg_fail = verify_sw_hw_coverage(949.96875, 1000.0)
        self.assertEqual(abs(949.96875 - 1000.0) / 1000.0, 0.05003125)
        self.assertIs(r_neg_fail["ok"], False)
        self.assertEqual(r_neg_fail["span_delta_ratio"], 0.05003)
        self.assertEqual(r_neg_fail["violations"][0]["code"], "SW_HW_SPAN_TRUNCATION")

    def test_h41a_pmu_in_window_samples_zero_boundary_with_expected_samples_below_5(self) -> None:
        r_zero = verify_sw_hw_coverage(10.0, 10.0, in_window_samples=0)
        self.assertEqual(r_zero["expected_samples"], 2.0)
        self.assertIs(r_zero["ok"], False)
        self.assertEqual(r_zero["violations"][0]["code"], "PMU_RING_BUFFER_DROPOUT")

        r_one = verify_sw_hw_coverage(10.0, 10.0, in_window_samples=1)
        self.assertEqual(r_one["expected_samples"], 2.0)
        self.assertEqual(r_one["pmu_density"], 0.5)
        self.assertIs(r_one["ok"], True)
        self.assertEqual(r_one["violations"], [])

    def test_h41b_pmu_expected_samples_5_boundary_with_positive_samples_and_low_density(self) -> None:
        r_fail = verify_sw_hw_coverage(25.0, 25.0, in_window_samples=1)
        self.assertEqual(r_fail["expected_samples"], 5.0)
        self.assertEqual(r_fail["pmu_density"], 0.2)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["violations"][0]["code"], "PMU_RING_BUFFER_DROPOUT")

        self.assertEqual(24.984375 * 1e-6 * 200000.0, 4.996875)
        r_ok = verify_sw_hw_coverage(24.984375, 24.984375, in_window_samples=1)
        self.assertEqual(r_ok["expected_samples"], 5.0)
        self.assertEqual(r_ok["pmu_density"], 0.2001)
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["violations"], [])

    def test_h42_pmu_density_0p90_boundary_with_expected_samples_above_5(self) -> None:
        r_ok = verify_sw_hw_coverage(10000.0, 10000.0, in_window_samples=1800)
        self.assertEqual(r_ok["expected_samples"], 2000.0)
        self.assertEqual(r_ok["pmu_density"], 0.90)
        self.assertIs(r_ok["ok"], True)
        self.assertEqual(r_ok["violations"], [])

        r_fail = verify_sw_hw_coverage(10000.0, 10000.0, in_window_samples=1799)
        self.assertEqual(r_fail["expected_samples"], 2000.0)
        self.assertEqual(r_fail["pmu_density"], 0.8995)
        self.assertIs(r_fail["ok"], False)
        self.assertEqual(r_fail["violations"][0]["code"], "PMU_RING_BUFFER_DROPOUT")


if __name__ == "__main__":
    unittest.main()

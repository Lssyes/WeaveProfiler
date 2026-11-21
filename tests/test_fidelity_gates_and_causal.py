#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline tests for Pillar 1 fidelity gates, Pillar 2 SW+HW causal attribution,
FI_SMEM_ZREG / FI_ROLE_SWITCH contracts, KREX pluggable lock, 0-Token remap,
Atrex v0.4.0 plugin adapter, generic OPENAI_BASE_URL, and CLI doctor/causal.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.api import build_agent_report, doctor, locate_or_remap_0token
from weaveprof.atrex_plugin import CAPABILITY_KEY, PLUGIN_NAME, apply as apply_atrex_plugin
from weaveprof.backend import validate_trace_product
from weaveprof.causal import (
    analyze_sw_hw_causality,
    calibrate_hw_series,
    compare_optimization_rounds,
    plan_context_and_search_space,
)
from weaveprof.cli import main as cli_main
from weaveprof.harness import (
    gpu_critical_region,
    list_lock_providers,
    register_lock_provider,
    run_compiler_feedback_selection,
    unregister_lock_provider,
    verify_compiler_spill_gate,
    verify_semantic_body_and_stubs,
    verify_sw_hw_coverage,
)


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DECODER = _load("weaveprof/pipeline/validate/tools/fi_lite_decode.py", "fg_decoder")
PLACER = _load("weaveprof/pipeline/scan/llm_probe_placer.py", "fg_placer")


class TestFidelityGatesAndCausal(unittest.TestCase):
    def test_fi_lite_cuh_smem_zreg_and_role_switch_decode(self) -> None:
        hdr = (ROOT / "recorders" / "cuda_cpp" / "fi_lite.cuh").read_text(encoding="utf-8")
        for token in (
            "FI_SMEM_ZREG",
            "struct FiSmemCtx",
            "fi_zreg_open",
            "fi_zreg_rec",
            "fi_zreg_trans",
            "fi_zreg_close",
            "FI_TAG_ROLE_SWITCH",
            "FI_ROLE_SWITCH",
            "WP_ROLE_SWITCH",
        ):
            self.assertIn(token, hdr)

        # Synthetic FILITE02 dump containing a dynamic role-switch record decodes cleanly with 0 pairing errors.
        with tempfile.TemporaryDirectory() as td:
            dump_p = Path(td) / "role_switch.fi02.bin"
            recs = [
                (1000, DECODER.OPEN | 1, 0),
                (1100, 4, 0),
                (1500, DECODER.END | 4, 0),
                (1600, DECODER.ROLE_SWITCH | DECODER.MARK | 2, 2),
                (1700, 3, 0),
                (2200, DECODER.END | 3, 0),
                (2300, DECODER.CLOSE, 0),
            ]
            meta = json.dumps({"gran": 0, "arch": "sm_103a", "backend": "nvidia_ptx"}).encode("utf-8")
            blob = bytearray(b"FILITE02")
            blob.extend(struct.pack("<6I", 1, len(recs), 1, 0, 0, len(meta)))
            blob.extend(meta)
            blob.extend(struct.pack("<1I", len(recs)))
            for r in recs:
                blob.extend(struct.pack("<QII", *r))
            dump_p.write_bytes(blob)

            dump = DECODER.read_dump(str(dump_p))
            names = {
                "events": {"3": "OutProj", "4": "Conv1D"},
                "semantics": {"3": "mma", "4": "compute"},
                "roles": {"1": "Conv", "2": "OutProj"},
            }
            trace, summary = DECODER.decode(dump, names)
            self.assertFalse(any(summary["errors"].values()), summary["errors"])
            self.assertEqual(summary["spans"], 2)
            self.assertEqual(summary["marks"], 1)
            self.assertEqual(summary["events_by_id"]["3"]["n"], 1)
            self.assertEqual(summary["events_by_id"]["4"]["n"], 1)
            self.assertTrue(trace["traceEvents"])

    def test_compiler_spill_gate_catches_reg_blind_spot(self) -> None:
        # B300 bs128 post-setmaxnreg failure: Used 72 regs unchanged while STL explodes 2 -> 134!
        bad = verify_compiler_spill_gate(
            {"regs": 72, "spill_stores": 16, "stack_bytes": 16, "stl": 2, "ldl": 2},
            {"regs": 72, "spill_stores": 760, "stack_bytes": 16, "stl": 134, "ldl": 140},
            has_setmaxnreg=True,
            smem_free_bytes=4096,
        )
        self.assertFalse(bad["ok"])
        self.assertTrue(bad["register_count_blind_spot"])
        codes = {v["code"] for v in bad["violations"]}
        self.assertIn("SPILL_STORE_INCREASE", codes)
        self.assertIn("SASS_STL_EXPLOSION", codes)
        self.assertTrue(any("setmaxnreg" in r for r in bad["remediation"]))

        # B300 v82 K2 FI_SMEM_ZREG clean arm: 0B spill delta, 0 STL delta!
        good = verify_compiler_spill_gate(
            {"regs": 128, "spill_stores": 0, "stack_bytes": 0, "stl": 27, "ldl": 25},
            {"regs": 128, "spill_stores": 0, "stack_bytes": 0, "stl": 27, "ldl": 25},
            smem_free_bytes=60000,
        )
        self.assertTrue(good["ok"])
        self.assertFalse(good["register_count_blind_spot"])

    def test_semantic_body_stub_gate_catches_64ns_empty_stub(self) -> None:
        res = verify_semantic_body_and_stubs(
            {
                "TASK_EPILOG_R2S": [64.0] * 160,
                "P4_MLA_Decode": [372000.0] * 50 + [190000.0] * 28,
            },
            sass_body_insns={"TASK_EPILOG_R2S": 0, "P4_MLA_Decode": 420},
        )
        self.assertFalse(res["ok"])
        self.assertEqual(res["stub_count"], 1)
        self.assertEqual(res["stubs"][0]["stage"], "TASK_EPILOG_R2S")
        self.assertIn("P4_MLA_Decode", res["valid_stages"])

    def test_sw_hw_coverage_gate_catches_truncation_and_pmu_blackout(self) -> None:
        # Silent CuTeDSL tag==0 break-on-zero truncation: 168 us / 741 us -> FAIL!
        trunc = verify_sw_hw_coverage(168.0, 741.0, in_window_samples=148, expected_freq_hz=200000.0)
        self.assertFalse(trunc["ok"])
        self.assertEqual(trunc["violations"][0]["code"], "SW_HW_SPAN_TRUNCATION")

        # 200kHz nsys ring-buffer blackout without cudaProfilerApi -> 0 in-window samples -> FAIL!
        blackout = verify_sw_hw_coverage(739.8, 741.0, in_window_samples=0, expected_freq_hz=200000.0)
        self.assertFalse(blackout["ok"])
        self.assertEqual(blackout["violations"][0]["code"], "PMU_RING_BUFFER_DROPOUT")

        # Clean lossless capture -> PASS!
        clean = verify_sw_hw_coverage(739.84, 740.99, in_window_samples=148, expected_freq_hz=200000.0)
        self.assertTrue(clean["ok"])
        self.assertLess(clean["span_delta_ratio"], 0.01)
        self.assertGreaterEqual(clean["pmu_density"], 0.99)

    def test_non_gemm_resource_class_product_gate(self) -> None:
        aligned = {
            "otherData": {
                "platform": "nv",
                "alignment": {
                    "warps_per_cta": 4,
                    "physical_units_observed": 8,
                    "has_compute_curve": True,
                    "has_memory_curve": True,
                    "has_sm_curve": True,
                    "kernel_duration_us": 100.0,
                },
            },
            "traceEvents": [
                {
                    "ph": "X", "pid": 100001, "tid": w, "ts": 0.0, "dur": 99.0,
                    "name": "RMSNorm", "args": {"cta": cta, "warp": w, "semantic_id": "compute"},
                }
                for cta in range(8) for w in range(4)
            ],
        }
        summary = {"wpc": 4, "n_ctas": 8, "fires": {4: 32}}
        spec = {"roles": {"0": "rmsnorm"}, "probes": [{"id": 4}]}
        src_rmsnorm = "__global__ void gated_rmsnorm(float* x) { float s = __shfl_xor_sync(0xffffffff, x[0], 16); }"
        rep_ok = validate_trace_product(aligned, summary, spec, src_rmsnorm, {"grid_ctas": 8})
        self.assertTrue(rep_ok["ok"], rep_ok["failures"])
        self.assertEqual(rep_ok["kernel_class"], "non_gemm")

        # A GEMM kernel missing 'load' must still fail!
        src_gemm = "__global__ void gemm_k() { wgmma.mma_async(); }"
        rep_bad = validate_trace_product(aligned, summary, spec, src_gemm, {"grid_ctas": 8})
        self.assertFalse(rep_bad["ok"])
        self.assertEqual(rep_bad["kernel_class"], "gemm_or_tensor")

    def test_pluggable_krex_lock_provider(self) -> None:
        events = []

        @contextmanager
        def fake_tre_krex(gpu_id, *, timeout_s, label, cpu_cores):
            events.append(("enter", gpu_id, label))
            yield {"gpu_id": gpu_id, "lock_path": f"tre://krex/{gpu_id}", "wait_ms": 1.25}
            events.append(("exit", gpu_id, label))

        register_lock_provider("tre_krex_test", fake_tre_krex)
        try:
            self.assertIn("tre_krex_test", list_lock_providers())
            with gpu_critical_region("gpu_2", label="unit_test", lock_provider="tre_krex_test") as lease:
                self.assertEqual(lease.provider, "tre_krex_test")
                self.assertEqual(lease.lock_path, "tre://krex/gpu_2")
            self.assertEqual(events, [("enter", "gpu_2", "unit_test"), ("exit", "gpu_2", "unit_test")])
        finally:
            unregister_lock_provider("tre_krex_test")

    def test_0token_incremental_probe_remap(self) -> None:
        old_src = "\n".join([
            "__global__ void k(float* a) {",
            "    int i = threadIdx.x;",
            "    load_tile(a, i);",
            "    mma_step(a, i);",
            "    store_tile(a, i);",
            "}",
        ]) + "\n"
        # Agent inserts 3 lines above load_tile without touching the probed anchors.
        new_src = "\n".join([
            "__global__ void k(float* a) {",
            "    // optimization candidate v2",
            "    __shared__ float smem[64];",
            "    int i = threadIdx.x;",
            "    load_tile(a, i);",
            "    mma_step(a, i);",
            "    store_tile(a, i);",
            "}",
        ]) + "\n"
        prior_spec = {
            "probes": [
                {"id": "load", "begin": {"role": "0", "line": 3}, "end": {"role": "0", "line": 3}},
                {"id": "mma", "begin": {"role": "0", "line": 4}, "end": {"role": "0", "line": 4}},
            ]
        }
        remapped, stats = locate_or_remap_0token(prior_spec, old_src, new_src)
        self.assertTrue(stats["reused_zero_token"])
        self.assertEqual(stats["endpoints_shifted"], 4)
        self.assertEqual(remapped["probes"][0]["begin"]["line"], 5)
        self.assertEqual(remapped["probes"][1]["begin"]["line"], 6)

        # If an anchor line is deleted, fail closed so caller falls back to LLM.
        broken_src = new_src.replace("    mma_step(a, i);\n", "    completely_new_op(a, i);\n")
        remapped_bad, stats_bad = locate_or_remap_0token(prior_spec, old_src, broken_src)
        self.assertIsNone(remapped_bad)
        self.assertFalse(stats_bad["reused_zero_token"])
        self.assertIn("mma", stats_bad["failed_probes"])

    def test_plan_context_and_search_space(self) -> None:
        p1 = plan_context_and_search_space(
            "__global__ void k() {}",
            arch="sm_103a", block_threads=512, l0_regs=128, l0_smem_bytes=160000,
        )
        self.assertEqual(p1["context_mode"], "FI_SMEM_ZREG")
        self.assertIn("-DFI_SMEM_ZREG", p1["compile_defines"])
        self.assertEqual(p1["reg_headroom"], 0)

        p2 = plan_context_and_search_space(
            'asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");',
            arch="sm_103a", block_threads=896, l0_regs=72, l0_smem_bytes=100000,
        )
        self.assertEqual(p2["placement_search_space"], ["pre_setmaxnreg", "post_setmaxnreg"])

        # Default register headroom picks 1-register bit-packed WpCtx1R (0 B SMEM) with fallback arms.
        p3 = plan_context_and_search_space(
            "__global__ void k() {}",
            arch="sm_90a", block_threads=256, l0_regs=64, l0_smem_bytes=232448,
        )
        self.assertEqual(p3["context_mode"], "WP_CTX_1R")
        self.assertIn("-DWP_CTX_1R", p3["compile_defines"])
        self.assertTrue(any(a["mode"] == "WP_CTX_1R_NOINLINE" for a in p3["fallback_arms"]))

        # Compiler-in-the-loop arm selector automatically falls back on spill/STL regression.
        from weaveprof.causal import select_compiler_feedback_arm
        sel = select_compiler_feedback_arm(
            [
                {"mode": "WP_CTX_1R", "defines": ["-DWP_CTX_1R"], "regs": 128, "spill_bytes": 128, "stl_count": 32, "overhead_pct": 24.0},
                {"mode": "FI_SMEM_ZREG", "defines": ["-DFI_SMEM_ZREG"], "regs": 128, "spill_bytes": 0, "stl_count": 0, "overhead_pct": 4.2},
            ],
            baseline_regs=128, baseline_spill_bytes=0, baseline_stl=0,
        )
        self.assertTrue(sel["fallback_triggered"])
        self.assertEqual(sel["winner"]["mode"], "FI_SMEM_ZREG")
        self.assertTrue(sel["winner"]["spill_clean"])

    def test_causal_engine_h20_calibration_straggler_and_perfetto(self) -> None:
        # 1. H20 Tensor×4 calibration + bimodal grid barrier straggler + Pearson correlation.
        t_axis = [float(i * 10) for i in range(60)]
        # Raw H20 tensor drops from ~22.5% (90% cal) to ~13.5% (54% cal) during t=[350, 530] us.
        raw_tensor = [22.5 if t < 350 else (13.5 if t <= 530 else 20.0) for t in t_axis]
        per_sm = []
        for cta in range(78):
            if cta < 50:
                spans = [[1, 0, 0.0, 150.0], [2, 0, 150.0, 527.0], [9, 0, 527.0, 530.0]]
            else:
                spans = [[1, 0, 0.0, 150.0], [2, 0, 150.0, 345.0], [9, 0, 345.0, 530.0]]
            per_sm.append({"cta": cta, "smid": cta, "open_us": 0.0, "close_us": 590.0, "spans": spans})
        doc = {
            "target": "synth_h20_mla",
            "gpu": "NVIDIA H20",
            "arch": "sm_90a",
            "sms": 78,
            "nsys_kernel_us": 590.0,
            "sw_span_us": 590.0,
            "phase_names": {"1": "P1_GEMM", "2": "P4_MLA", "9": "grid_barrier_wait"},
            "phase_stats_all78sm": {
                "P1_GEMM": {"median_us": 150.0, "min_us": 149.0, "max_us": 151.0, "skew_us": 2.0},
                "P4_MLA": {"median_us": 377.0, "min_us": 195.0, "max_us": 377.0, "skew_us": 182.0},
                "grid_barrier_wait": {"median_us": 3.0, "min_us": 3.0, "max_us": 185.0, "skew_us": 182.0},
            },
            "per_sm_w0": per_sm,
            "hw_series": {
                "sm_active": {"t_us": t_axis, "val": [100.0] * len(t_axis), "in_win_count": len(t_axis)},
                "tensor_active": {"t_us": t_axis, "val": raw_tensor, "in_win_count": len(t_axis)},
                "dram_read": {"t_us": t_axis, "val": [20.0] * len(t_axis), "in_win_count": len(t_axis)},
            },
        }
        rep = analyze_sw_hw_causality(doc)
        self.assertEqual(rep["tensor_calibration_factor"], 4.0)
        self.assertIn("P1_GEMM", rep["refinement_summary"]["saturated_stop_refining"])
        self.assertTrue(rep["causal_opportunities"])
        top = rep["causal_opportunities"][0]
        self.assertEqual(top["category"], "LOAD_IMBALANCE_GRID_BARRIER")
        self.assertGreater(top["sw_hw_pearson_r"], 0.8)
        self.assertGreater(top["counterfactual_speedup_upper_bound_pct"], 8.0)

        # 2. Perfetto JSON ingestion works directly too.
        perf_doc = {
            "otherData": {"arch": "sm_103a", "gpu": "B300", "alignment": {"kernel_duration_us": 100.0}},
            "traceEvents": [
                {"ph": "C", "name": "dram_read_bw", "ts": float(i * 10), "args": {"pct": 85.0}}
                for i in range(10)
            ] + [
                {"ph": "C", "name": "tensor_active", "ts": float(i * 10), "args": {"pct": 4.0}}
                for i in range(10)
            ] + [
                {"ph": "C", "name": "sm_active", "ts": float(i * 10), "args": {"pct": 99.0}}
                for i in range(10)
            ] + [
                {"ph": "X", "pid": 100001, "tid": 0, "ts": 0.0, "dur": 98.0, "name": "Rec_GDN",
                 "args": {"cta": c, "sm": c, "semantic_id": "compute"}}
                for c in range(16)
            ],
        }
        rep_p = analyze_sw_hw_causality(perf_doc)
        self.assertEqual(rep_p["global_hw_regime"], "DRAM_READ_BOUND")

    def test_h20_dual_roofline_padding_waste_blockers_and_overlap(self) -> None:
        # 1. Dual-denominator Roofline + MoE M=64 69% padding waste overrides fake saturation!
        moe_doc = {
            "target": "h20_megamoe_fp8_small_n",
            "gpu": "NVIDIA H20",
            "arch": "sm_90a",
            "sms": 78,
            "nsys_kernel_us": 500.0,
            "sw_span_us": 500.0,
            "workload": {
                "dtype": "fp8",
                "effective_flops": 31.0 * 1e9,
                "hw_executed_flops": 100.0 * 1e9,
                "reg_per_thread": 128,
                "smem_bytes": 232448,
                "math_warp_wait_pct": 2.4,
                "comm_pct": 4.0,
                "combine_pct": 1.2,
            },
            "phase_stats_all78sm": {
                "MoE_WGMMA_FP8": {"median_us": 320.0, "min_us": 318.0, "max_us": 322.0, "skew_us": 4.0},
                "MoE_Combine_HBM": {"median_us": 180.0, "min_us": 178.0, "max_us": 182.0, "skew_us": 4.0},
            },
            "hw_series": {
                "sm_active": {"t_us": [0.0, 250.0, 500.0], "val": [100.0, 100.0, 100.0], "in_win_count": 3},
                "tensor_active": {"t_us": [0.0, 160.0, 310.0, 360.0, 480.0], "val": [23.0, 23.0, 23.0, 2.0, 2.0], "in_win_count": 5},
                "dram_read": {"t_us": [0.0, 160.0, 310.0, 360.0, 480.0], "val": [15.0, 15.0, 15.0, 72.0, 75.0], "in_win_count": 5},
            },
        }
        rep = analyze_sw_hw_causality(moe_doc)
        self.assertAlmostEqual(rep["caliber"]["padding_waste_pct"], 69.0, places=1)
        self.assertGreater(rep["caliber"]["emp_rf_pct"], rep["caliber"]["nom_rf_pct"])
        self.assertTrue(any(o["category"] == "TENSOR_PADDING_WASTE" for o in rep["causal_opportunities"]))
        blocker_codes = {b["code"] for b in rep["structural_blockers"]}
        self.assertIn("LOW_MATH_WAIT_NO_LATENCY_HIDING", blocker_codes)
        self.assertIn("RESOURCE_WALL_NO_TILE_GROWTH_OR_CTA_FUSION", blocker_codes)
        self.assertIn("LOW_COMM_NO_DEDUP_REFACTOR", blocker_codes)
        self.assertNotIn("MoE_WGMMA_FP8", rep["refinement_summary"]["saturated_stop_refining"])
        # Because MoE_WGMMA_FP8 (COMP) and MoE_Combine_HBM (MEM) alternate while regs=128/smem=232KB,
        # engine must recommend TBO_TWO_MICROBATCH_OVERLAP!
        self.assertTrue(rep["overlap_recommendations"])
        self.assertEqual(rep["overlap_recommendations"][0]["strategy"], "TBO_TWO_MICROBATCH_OVERLAP")
        self.assertGreater(rep["interleaving_waste_us"], 100.0)

    def test_atrex_plugin_agent_report_cli_and_openai_url(self) -> None:
        provided = {}
        tools = {}

        class FakeAtrexCtx:
            def provide(self, key, svc):
                provided[key] = svc

            def register_tool(self, *, name, description, handler):
                tools[name] = (description, handler)

        ctx = FakeAtrexCtx()
        svc = apply_atrex_plugin(ctx, {"default_limit": 0.08, "card": "b300"})
        self.assertIn(CAPABILITY_KEY, provided)
        self.assertIs(provided[CAPABILITY_KEY], svc)
        self.assertIn("weaveprof_profile_for_agent", tools)
        self.assertEqual(PLUGIN_NAME, "aka-plugin-autoprof")
        self.assertEqual(svc.policy().max_overhead_limit, 0.08)

        # Compact agent report from build_agent_report
        agent_rep = build_agent_report({"status": "FAIL", "failed": ["overhead"], "metrics": {"overhead_ratio": 0.25}})
        self.assertEqual(agent_rep["status"], "FAIL")
        self.assertFalse(agent_rep["seal_verified"])
        self.assertAlmostEqual(agent_rep["overhead_pct"], 25.0)

        # Doctor API & CLI
        doc_res = doctor(check_gpu=False)
        self.assertTrue(doc_res["ready_for_offline_tests"])
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = cli_main(["doctor", "--no-gpu"])
        self.assertEqual(rc, 0)
        self.assertIn("weaveprof.doctor/v1", buf.getvalue())

        # Generic OPENAI_BASE_URL normalization
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.example.com/v1"}, clear=False):
            self.assertEqual(PLACER.get_llm_endpoint_url(), "https://api.example.com/v1/chat/completions")
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.example.com"}, clear=False):
            self.assertEqual(PLACER.get_llm_endpoint_url(), "https://api.example.com/v1/chat/completions")


def _fake_compiler(table):
    """Deterministic compile_fn stub keyed by (mode, placement); records every call."""
    calls = []

    def compile_fn(arm):
        calls.append((arm["mode"], arm["placement"], tuple(arm["defines"]), arm["stage"]))
        val = table.get((arm["mode"], arm["placement"]), table.get(arm["mode"]))
        if isinstance(val, Exception):
            raise val
        return dict(val)

    return compile_fn, calls


def _h20_round_doc(balanced: bool):
    """Synthetic H20 SW+HW capture: round N-1 has a bimodal grid-barrier straggler, round N is balanced."""
    t_axis = [float(i * 10) for i in range(60)]
    kus = 410.0 if balanced else 590.0
    raw_tensor = [22.5] * len(t_axis) if balanced else [
        22.5 if t < 350 else (13.5 if t <= 530 else 20.0) for t in t_axis
    ]
    per_sm = []
    for cta in range(78):
        if balanced:
            spans = [[1, 0, 0.0, 150.0], [2, 0, 150.0, 345.0], [9, 0, 345.0, 348.0]]
        elif cta < 50:
            spans = [[1, 0, 0.0, 150.0], [2, 0, 150.0, 527.0], [9, 0, 527.0, 530.0]]
        else:
            spans = [[1, 0, 0.0, 150.0], [2, 0, 150.0, 345.0], [9, 0, 345.0, 530.0]]
        per_sm.append({"cta": cta, "smid": cta, "open_us": 0.0, "close_us": kus, "spans": spans})
    p4 = ({"median_us": 195.0, "min_us": 194.0, "max_us": 196.0, "skew_us": 2.0} if balanced
          else {"median_us": 377.0, "min_us": 195.0, "max_us": 377.0, "skew_us": 182.0})
    gw = ({"median_us": 3.0, "min_us": 3.0, "max_us": 3.0, "skew_us": 0.0} if balanced
          else {"median_us": 3.0, "min_us": 3.0, "max_us": 185.0, "skew_us": 182.0})
    return {
        "target": "synth_h20_mla", "gpu": "NVIDIA H20", "arch": "sm_90a", "sms": 78,
        "nsys_kernel_us": kus, "sw_span_us": kus,
        "phase_names": {"1": "P1_GEMM", "2": "P4_MLA", "9": "grid_barrier_wait"},
        "phase_stats_all78sm": {
            "P1_GEMM": {"median_us": 150.0, "min_us": 149.0, "max_us": 151.0, "skew_us": 2.0},
            "P4_MLA": p4,
            "grid_barrier_wait": gw,
        },
        "per_sm_w0": per_sm,
        "hw_series": {
            "sm_active": {"t_us": t_axis, "val": [100.0] * len(t_axis), "in_win_count": len(t_axis)},
            "tensor_active": {"t_us": t_axis, "val": raw_tensor, "in_win_count": len(t_axis)},
            "dram_read": {"t_us": t_axis, "val": [20.0] * len(t_axis), "in_win_count": len(t_axis)},
        },
    }


class TestCompilerFeedbackSelectionD2(unittest.TestCase):
    BASE = {"regs": 128, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 27, "ldl_count": 25}

    def test_fast_arm_clean_compiles_once_without_fallback(self) -> None:
        fn, calls = _fake_compiler({"WP_CTX_1R": {"regs": 128, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 28, "ldl_count": 25}})
        res = run_compiler_feedback_selection(
            fn, "__global__ void k() {}", baseline_metrics=self.BASE,
            arch="sm_103a", block_threads=512, l0_smem_bytes=100000,
        )
        self.assertEqual(calls, [("WP_CTX_1R", "default_entry", ("-DWP_CTX_1R",), "fast")])
        self.assertFalse(res["fallback_triggered"])
        self.assertEqual(res["status"], "CLEAN")
        self.assertEqual(res["winner"]["mode"], "WP_CTX_1R")
        self.assertTrue(res["winner_is_fast_arm"])
        self.assertTrue(res["winner"]["gate_ok"])
        self.assertEqual(res["compile_calls"], 1)
        self.assertEqual(len(res["spill_gate_reports"]), 1)
        self.assertEqual(res["placement_search_space"], ["default_entry"])
        json.dumps(res)  # fully JSON-serializable for sealing/ledgers

    def test_spill_regression_falls_back_to_smem_zreg(self) -> None:
        # B300 v82 K2: Used 128 regs unchanged but WP_CTX_1R spills 220 B / STL 27 -> 97.
        fn, calls = _fake_compiler({
            "WP_CTX_1R": {"regs": 128, "spill_bytes": 220, "stack_bytes": 224, "stl_count": 97, "ldl_count": 90},
            "FI_SMEM_ZREG": {"regs": 128, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 27, "ldl_count": 25, "overhead_pct": 7.36},
            "WP_CTX_1R_NOINLINE": {"regs": 128, "spill_bytes": 32, "stack_bytes": 32, "stl_count": 31, "ldl_count": 29},
        })
        res = run_compiler_feedback_selection(
            fn, "__global__ void k() {}", baseline_metrics=self.BASE,
            arch="sm_103a", block_threads=512, l0_smem_bytes=160000,
        )
        self.assertEqual([c[0] for c in calls], ["WP_CTX_1R", "FI_SMEM_ZREG", "WP_CTX_1R_NOINLINE"])
        self.assertEqual([c[3] for c in calls], ["fast", "fallback", "fallback"])
        self.assertTrue(res["fallback_triggered"])
        self.assertFalse(res["winner_is_fast_arm"])
        self.assertEqual(res["winner"]["mode"], "FI_SMEM_ZREG")
        self.assertEqual(res["winner"]["compile_defines"], ["-DFI_SMEM_ZREG"])
        self.assertTrue(res["winner"]["spill_clean"])
        first_gate = res["spill_gate_reports"][0]["report"]
        self.assertFalse(first_gate["ok"])
        self.assertTrue(first_gate["register_count_blind_spot"])
        self.assertEqual([r["mode"] for r in res["ranked"]][0], "FI_SMEM_ZREG")
        # Deterministic: identical inputs -> identical output.
        fn2, _ = _fake_compiler({
            "WP_CTX_1R": {"regs": 128, "spill_bytes": 220, "stack_bytes": 224, "stl_count": 97, "ldl_count": 90},
            "FI_SMEM_ZREG": {"regs": 128, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 27, "ldl_count": 25, "overhead_pct": 7.36},
            "WP_CTX_1R_NOINLINE": {"regs": 128, "spill_bytes": 32, "stack_bytes": 32, "stl_count": 31, "ldl_count": 29},
        })
        res2 = run_compiler_feedback_selection(
            fn2, "__global__ void k() {}", baseline_metrics=self.BASE,
            arch="sm_103a", block_threads=512, l0_smem_bytes=160000,
        )
        self.assertEqual(json.dumps(res, sort_keys=True), json.dumps(res2, sort_keys=True))

    def test_smem_exhausted_skips_smem_zreg_and_uses_noinline(self) -> None:
        fn, calls = _fake_compiler({
            "WP_CTX_1R": {"regs": 64, "spill_bytes": 96, "stack_bytes": 96, "stl_count": 40, "ldl_count": 40},
            "WP_CTX_1R_NOINLINE": {"regs": 64, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 2, "ldl_count": 2},
        })
        res = run_compiler_feedback_selection(
            fn, "__global__ void k() {}",
            baseline_metrics={"regs": 64, "spill_bytes": 0, "stack_bytes": 0, "stl_count": 2, "ldl_count": 2},
            arch="sm_90a", block_threads=256, l0_smem_bytes=232448,
        )
        self.assertEqual(res["smem_free_bytes"], 0)
        self.assertNotIn("FI_SMEM_ZREG", [c[0] for c in calls])
        self.assertTrue(res["fallback_triggered"])
        self.assertEqual(res["winner"]["mode"], "WP_CTX_1R_NOINLINE")
        self.assertEqual(res["status"], "CLEAN")

    def test_setmaxnreg_searches_pre_post_placements(self) -> None:
        src = 'asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");'
        base = {"regs": 72, "spill_stores": 16, "stack_bytes": 16, "stl": 2, "ldl": 2}
        # bs128 decode: post placement explodes STL 2 -> 134 at unchanged 72 regs; pre is clean.
        fn, calls = _fake_compiler({
            ("WP_CTX_1R", "pre_setmaxnreg"): {"regs": 72, "spill_stores": 16, "stack_bytes": 16, "stl": 3, "ldl": 3},
            ("WP_CTX_1R", "post_setmaxnreg"): {"regs": 72, "spill_stores": 760, "stack_bytes": 16, "stl": 134, "ldl": 140},
        })
        res = run_compiler_feedback_selection(
            fn, src, baseline_metrics=base, arch="sm_103a", block_threads=896, l0_smem_bytes=100000,
        )
        self.assertTrue(res["has_setmaxnreg"])
        self.assertEqual(res["placement_search_space"], ["pre_setmaxnreg", "post_setmaxnreg"])
        self.assertEqual([(c[0], c[1]) for c in calls], [("WP_CTX_1R", "pre_setmaxnreg"), ("WP_CTX_1R", "post_setmaxnreg")])
        self.assertFalse(res["fallback_triggered"])
        self.assertEqual((res["winner"]["mode"], res["winner"]["placement"]), ("WP_CTX_1R", "pre_setmaxnreg"))

        # Both placements of the fast arm regress -> every fallback arm compiled at every placement.
        bad = {"regs": 72, "spill_stores": 760, "stack_bytes": 16, "stl": 134, "ldl": 140}
        fn2, calls2 = _fake_compiler({
            "WP_CTX_1R": bad,
            "FI_SMEM_ZREG": {"regs": 72, "spill_stores": 16, "stack_bytes": 16, "stl": 2, "ldl": 2, "overhead_pct": 3.61},
            "WP_CTX_1R_NOINLINE": {"regs": 72, "spill_stores": 64, "stack_bytes": 16, "stl": 9, "ldl": 9},
        })
        res2 = run_compiler_feedback_selection(
            fn2, src, baseline_metrics=base, arch="sm_103a", block_threads=896, l0_smem_bytes=100000,
        )
        self.assertTrue(res2["fallback_triggered"])
        self.assertEqual(res2["compile_calls"], 6)
        self.assertEqual(res2["winner"]["mode"], "FI_SMEM_ZREG")
        self.assertEqual(res2["winner"]["placement"], "pre_setmaxnreg")  # stable tie-break
        self.assertTrue(any(
            "setmaxnreg" in r for g in res2["spill_gate_reports"][:2] for r in g["report"]["remediation"]
        ))

    def test_build_failures_fail_closed_and_package_exports(self) -> None:
        fn, _ = _fake_compiler({
            "WP_CTX_1R": RuntimeError("ptxas fatal"),
            "FI_SMEM_ZREG": {"build_ok": False, "error": "smem overflow"},
            "WP_CTX_1R_NOINLINE": {"regs": 130, "spill_bytes": 64, "stack_bytes": 64, "stl_count": 60, "ldl_count": 60},
        })
        res = run_compiler_feedback_selection(
            fn, "__global__ void k() {}", baseline_metrics=self.BASE,
            arch="sm_103a", block_threads=512, l0_smem_bytes=100000,
        )
        self.assertEqual(res["status"], "NO_CLEAN_ARM")
        self.assertFalse(res["winner"]["gate_ok"])
        self.assertEqual(res["winner"]["mode"], "WP_CTX_1R_NOINLINE")  # only arm that built
        self.assertFalse(res["attempts"][0]["build_ok"])
        self.assertIn("ptxas fatal", res["attempts"][0]["error"])
        self.assertEqual(res["spill_gate_reports"][1]["report"]["violations"][0]["code"], "BUILD_FAILED")
        with self.assertRaises(TypeError):
            run_compiler_feedback_selection(None, "", baseline_metrics=self.BASE)

        import weaveprof
        self.assertIs(weaveprof.run_compiler_feedback_selection, run_compiler_feedback_selection)
        self.assertIn("run_compiler_feedback_selection", weaveprof.__all__)
        svc = apply_atrex_plugin({}, {})
        fn_ok, _ = _fake_compiler({"WP_CTX_1R": dict(self.BASE)})
        self.assertEqual(
            svc.select_compiler_arm(fn_ok, "", baseline_metrics=self.BASE, arch="sm_90a")["winner"]["mode"],
            "WP_CTX_1R",
        )


class TestRoundDiffFeedbackD3(unittest.TestCase):
    PREV = {
        "kernel_us": 590.0,
        "verdict": "CAUSE_EXPLAINED",
        "global_hw_regime": "UNDERUTILIZED_DIP",
        "hw_summary": {"tensor_active_cal_mean_pct": 77.6, "dram_read_mean_pct": 20.0, "sm_active_mean_pct": 100.0},
        "stage_validity": [
            {"stage": "P1_GEMM", "median_us": 150.0},
            {"stage": "P4_MLA", "median_us": 377.0},
            {"stage": "grid_barrier_wait", "median_us": 3.0},
        ],
        "causal_opportunities": [
            {"rank": 1, "category": "LOAD_IMBALANCE_GRID_BARRIER",
             "title": "Rebalance work before barrier 'grid_barrier_wait' (28/78 SMs idling)",
             "counterfactual_speedup_upper_bound_pct": 13.1},
            {"rank": 2, "category": "CHIP_SM_UNDERFILL", "title": "Eliminate 20/148 idle SMs",
             "counterfactual_speedup_upper_bound_pct": 10.0},
        ],
        "structural_blockers": [],
    }
    CURR = {
        "kernel_us": 420.0,
        "verdict": "CAUSE_EXPLAINED",
        "global_hw_regime": "UNDERUTILIZED_DIP",
        "hw_summary": {"tensor_active_cal_mean_pct": 88.0, "dram_read_mean_pct": 21.5, "sm_active_mean_pct": 100.0},
        "stage_validity": [
            {"stage": "P1_GEMM", "median_us": 150.0},
            {"stage": "P4_MLA", "median_us": 265.0},
            {"stage": "grid_barrier_wait", "median_us": 3.0},
            {"stage": "P5_Epilog", "median_us": 2.0},
        ],
        "causal_opportunities": [
            {"rank": 1, "category": "CHIP_SM_UNDERFILL", "title": "Eliminate 18/148 idle SMs",
             "counterfactual_speedup_upper_bound_pct": 12.0,
             "recommended_optimization": "Increase parallelism to occupy all SMs."},
            {"rank": 2, "category": "MEMORY_OR_SYNC_WAIT_EXPOSURE",
             "title": "Hide exposed wait stage 'P5_Epilog' (3.0% of stage time)",
             "counterfactual_speedup_upper_bound_pct": 3.0},
        ],
        "structural_blockers": [
            {"code": "LOW_MATH_WAIT_NO_LATENCY_HIDING",
             "blocked_actions": ["fine_grained_ready_signaling", "extra_load_latency_hiding"],
             "reason": "Math warp wait <5.5%."},
        ],
    }

    def test_round_diff_metrics_opportunity_sets_and_directive(self) -> None:
        d = compare_optimization_rounds(self.PREV, self.CURR)
        self.assertEqual(d["schema"], "weaveprof.round-diff/v1")
        self.assertAlmostEqual(d["latency_delta_us"], -170.0)
        self.assertAlmostEqual(d["speedup_ratio"], round(590.0 / 420.0, 4))
        self.assertTrue(d["improved"])
        self.assertFalse(d["regressed"])
        self.assertAlmostEqual(d["hw_deltas"]["tensor_active_cal_mean_pct"]["delta"], 10.4)
        self.assertAlmostEqual(d["hw_deltas"]["dram_read_mean_pct"]["delta"], 1.5)
        self.assertAlmostEqual(d["hw_deltas"]["sm_active_mean_pct"]["delta"], 0.0)
        sd = {s["stage"]: s for s in d["stage_deltas"]}
        self.assertAlmostEqual(sd["P4_MLA"]["delta_us"], -112.0)
        self.assertEqual(sd["P4_MLA"]["status"], "FASTER")
        self.assertEqual(sd["P1_GEMM"]["status"], "UNCHANGED")
        self.assertEqual(sd["P5_Epilog"]["status"], "NEW_STAGE")
        self.assertEqual([o["key"] for o in d["eliminated_opportunities"]],
                         ["LOAD_IMBALANCE_GRID_BARRIER::grid_barrier_wait"])
        self.assertEqual([o["key"] for o in d["persisting_opportunities"]], ["CHIP_SM_UNDERFILL"])
        self.assertAlmostEqual(d["persisting_opportunities"][0]["bound_delta_pct"], 2.0)
        self.assertEqual([o["key"] for o in d["newly_introduced_opportunities"]],
                         ["MEMORY_OR_SYNC_WAIT_EXPOSURE::P5_Epilog"])
        self.assertEqual([g["code"] for g in d["negative_roi_guardrails"]], ["LOW_MATH_WAIT_NO_LATENCY_HIDING"])
        nd = d["next_round_directive"]
        self.assertEqual(nd["action"], "PURSUE_TOP_OPPORTUNITY")
        self.assertEqual(nd["target_key"], "CHIP_SM_UNDERFILL")
        self.assertEqual(nd["opportunity_status"], "PERSISTING")
        self.assertFalse(nd["revert_previous_change"])
        self.assertIn("extra_load_latency_hiding", nd["text"])
        # Deterministic & JSON-serializable.
        self.assertEqual(json.dumps(d, sort_keys=True), json.dumps(compare_optimization_rounds(self.PREV, self.CURR), sort_keys=True))

    def test_round_diff_regression_and_saturated_stop(self) -> None:
        rev = compare_optimization_rounds(self.CURR, self.PREV)
        self.assertFalse(rev["improved"])
        self.assertTrue(rev["regressed"])
        self.assertTrue(rev["next_round_directive"]["revert_previous_change"])
        self.assertIn("revert", rev["next_round_directive"]["text"])
        self.assertEqual(sorted(s["stage"] for s in rev["stage_deltas"] if s["status"] == "REMOVED_STAGE"), ["P5_Epilog"])

        # Within the noise floor -> neither improved nor regressed.
        noisy = dict(self.PREV, kernel_us=593.0)
        nz = compare_optimization_rounds(self.PREV, noisy)
        self.assertFalse(nz["improved"])
        self.assertFalse(nz["regressed"])

        sat = dict(self.CURR, verdict="SATURATED_STOP_REFINING", global_hw_regime="TENSOR_SATURATED")
        s = compare_optimization_rounds(self.PREV, sat)
        self.assertEqual(s["next_round_directive"]["action"], "STOP_SATURATED")
        self.assertIsNone(s["next_round_directive"]["target_key"])

        empty = compare_optimization_rounds({}, {})
        self.assertIsNone(empty["latency_delta_us"])
        self.assertEqual(empty["next_round_directive"]["action"], "NO_OPPORTUNITY_REMAINING")

    def test_round_diff_end_to_end_causal_agent_report_and_plugin(self) -> None:
        prev = analyze_sw_hw_causality(_h20_round_doc(balanced=False))
        curr = analyze_sw_hw_causality(_h20_round_doc(balanced=True))
        d = compare_optimization_rounds(prev, curr)
        self.assertTrue(d["improved"])
        self.assertAlmostEqual(d["latency_delta_us"], -180.0)
        self.assertGreater(d["hw_deltas"]["tensor_active_cal_mean_pct"]["delta"], 10.0)
        self.assertIn("LOAD_IMBALANCE_GRID_BARRIER::grid_barrier_wait",
                      [o["key"] for o in d["eliminated_opportunities"]])
        self.assertEqual(d["curr_verdict"], "SATURATED_STOP_REFINING")
        self.assertEqual(d["next_round_directive"]["action"], "STOP_SATURATED")

        # build_agent_report threads the previous round through to `round_diff`.
        rep = build_agent_report({"status": "PASS", "metrics": {}}, causal_input=_h20_round_doc(balanced=True),
                                 prev_causal_report=prev)
        self.assertEqual(rep["kernel_us"], 410.0)
        self.assertIsNotNone(rep["round_diff"])
        self.assertEqual(rep["round_diff"]["next_round_directive"]["action"], "STOP_SATURATED")
        self.assertIsNone(build_agent_report({"status": "PASS", "metrics": {}},
                                             causal_input=_h20_round_doc(balanced=True))["round_diff"])
        # Previous agent report (compact form) is also accepted as the prior round.
        prev_agent = build_agent_report({"status": "PASS", "metrics": {}}, causal_input=_h20_round_doc(balanced=False))
        d_agent = compare_optimization_rounds(prev_agent, curr)
        self.assertAlmostEqual(d_agent["latency_delta_us"], -180.0)
        self.assertEqual([o["key"] for o in d_agent["eliminated_opportunities"]],
                         [o["key"] for o in d["eliminated_opportunities"]])

        import weaveprof
        self.assertIs(weaveprof.compare_optimization_rounds, compare_optimization_rounds)
        self.assertIn("compare_optimization_rounds", weaveprof.__all__)
        svc = apply_atrex_plugin({}, {})
        self.assertEqual(svc.compare_rounds(prev, curr), d)


if __name__ == "__main__":
    unittest.main()

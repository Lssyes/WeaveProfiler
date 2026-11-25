#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `weaveprof/aka_adapter.py` (B4-G1)."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import weaveprof.aka_adapter as aka_adapter  # noqa: E402
from weaveprof.api import identify  # noqa: E402
from weaveprof.harness import EpisodeSealer, HarnessPolicy  # noqa: E402
from weaveprof.workspace import digest as file_sha256  # noqa: E402


FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

ANALYZE_EXPECTED_KEYS = {
    "arch",
    "bottleneck_hint",
    "caliber",
    "causal_opportunities",
    "counterfactual_upper_bound_us",
    "global_hw_regime",
    "gpu",
    "hw_summary",
    "interleaving_waste_us",
    "kernel_us",
    "observed_ctas",
    "observed_sms",
    "overlap_recommendations",
    "phases",
    "refinement_summary",
    "schema",
    "stage_validity",
    "straggler_walls",
    "structural_blockers",
    "sw_span_us",
    "target",
    "tensor_calibration_factor",
    "total_sms",
    "verdict",
    "warp_tail_imbalance",
}

COMPARE_ROUNDS_EXPECTED_KEYS = {
    "curr_kernel_us",
    "curr_verdict",
    "eliminated_opportunities",
    "hw_deltas",
    "improved",
    "latency_delta_pct",
    "latency_delta_us",
    "negative_roi_guardrails",
    "newly_introduced_opportunities",
    "next_round_directive",
    "noise_floor_pct",
    "persisting_opportunities",
    "prev_kernel_us",
    "prev_verdict",
    "regressed",
    "schema",
    "speedup_ratio",
    "stage_deltas",
}

PLAN_CONTEXT_EXPECTED_KEYS = {
    "arch",
    "compile_defines",
    "context_mode",
    "fallback_arms",
    "has_grid_sync",
    "has_role_switch_macro",
    "has_setmaxnreg",
    "placement_search_space",
    "reasons",
    "recommended_granularity",
    "reg_budget_per_thread",
    "reg_headroom",
    "smem_free_bytes",
}

AGENT_REPORT_EXPECTED_KEYS = {
    "bottleneck_hint",
    "caliber",
    "causal_opportunities",
    "counterfactual_upper_bound_us",
    "failed",
    "fidelity_gates",
    "global_hw_regime",
    "hw_summary",
    "kernel_us",
    "ledger_path",
    "overhead_pct",
    "overlap_recommendations",
    "perfetto_json_gz",
    "phases",
    "preview_png",
    "refinement_advice",
    "round_diff",
    "schema",
    "seal_errors",
    "seal_path",
    "seal_sha256",
    "seal_verified",
    "stage_breakdown",
    "status",
    "structural_blockers",
    "trace_path",
    "verdict",
}

MEGAKERNEL_BLOCKED_EXPECTED_KEYS = {
    "causal_opportunities",
    "fidelity_gates",
    "kernel_us",
    "megakernel_skew",
    "mode",
    "out_dir",
    "overhead_pct",
    "report_bytes",
    "round_diff",
    "schema",
    "seal_path",
    "seal_sha256",
    "seal_verified",
    "status",
    "structural_blockers",
    "wall_clock_ms",
}


def _sample_perfetto_trace_doc() -> dict:
    return {
        "otherData": {
            "arch": "sm_103a",
            "gpu": "B300",
            "alignment": {"kernel_duration_us": 100.0},
        },
        "traceEvents": (
            [{"ph": "C", "name": "dram_read_bw", "ts": float(i * 10), "args": {"pct": 85.0}} for i in range(10)]
            + [{"ph": "C", "name": "tensor_active", "ts": float(i * 10), "args": {"pct": 4.0}} for i in range(10)]
            + [{"ph": "C", "name": "sm_active", "ts": float(i * 10), "args": {"pct": 99.0}} for i in range(10)]
            + [
                {
                    "ph": "X",
                    "pid": 100001,
                    "tid": 0,
                    "ts": 0.0,
                    "dur": 98.0,
                    "name": "Rec_GDN",
                    "args": {"cta": c, "sm": c, "semantic_id": "compute"},
                }
                for c in range(16)
            ]
        ),
    }


def _sample_prev_curr_reports() -> tuple[dict, dict]:
    prev_rep = {
        "kernel_us": 200.0,
        "verdict": "CAUSE_EXPLAINED",
        "global_hw_regime": "DRAM_READ_BOUND",
        "hw_summary": {
            "tensor_active_cal_mean_pct": 20.0,
            "dram_read_mean_pct": 80.0,
            "sm_active_mean_pct": 98.0,
        },
        "stage_validity": [{"stage": "Rec_GDN", "median_us": 190.0}],
        "causal_opportunities": [
            {
                "rank": 1,
                "category": "DRAM_BANDWIDTH_BOUND",
                "stage": "Rec_GDN",
                "title": "Reduce DRAM traffic",
                "counterfactual_speedup_upper_bound_pct": 25.0,
            }
        ],
        "structural_blockers": [],
    }
    curr_rep = {
        "kernel_us": 150.0,
        "verdict": "SATURATED_STOP_REFINING",
        "global_hw_regime": "TENSOR_SATURATED",
        "hw_summary": {
            "tensor_active_cal_mean_pct": 85.0,
            "dram_read_mean_pct": 30.0,
            "sm_active_mean_pct": 99.0,
        },
        "stage_validity": [{"stage": "Rec_GDN", "median_us": 145.0}],
        "causal_opportunities": [],
        "structural_blockers": [],
    }
    return prev_rep, curr_rep


def _write_synth_bench_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sqlite3
import struct

rdir = Path(os.environ["WEAVEPROF_ROUND_DIR"])
names = json.loads(Path(os.environ["WEAVEPROF_NAMES"]).read_text(encoding="utf-8"))
ovh_mode = os.environ.get("WP_CHAR_BENCH_MODE", "pass")

ops = ["MOV", "LDG", "FFMA", "FADD", "STG", "EXIT"]
sass = "Function : demo_kernel\\n" + "".join(f"/*{16 * i:04x}*/ {op} ;\\n" for i, op in enumerate(ops))
(rdir / "l0.sass").write_text(sass, encoding="utf-8")
(rdir / "inst.sass").write_text(sass, encoding="utf-8")
(rdir / "l0.bin").write_bytes(b"\\x01\\x02\\x03\\x04" * 8)
(rdir / "inst.bin").write_bytes(b"\\x01\\x02\\x03\\x04" * 8)

END = 1 << 31
OPEN = 1 << 30
CLOSE = 1 << 27
t0 = 1_000_000
recs = [(t0, OPEN, 0)]
t = t0 + 200
for k_str in sorted(names.get("probe_ids", {}), key=lambda x: int(x)):
    if ovh_mode == "no_events_high_ovh":
        break
    eid = int(k_str)
    recs.append((t, eid, 0))
    recs.append((t + 20000, END | eid, 0))
    t += 21000
recs.append((t + 200, CLOSE, 0))

meta = json.dumps({"gran": 0, "arch": "sm_90a", "backend": "nvidia_ptx"}).encode("utf-8")
blob = bytearray(b"FILITE02")
blob.extend(struct.pack("<6I", 1, len(recs), 1, 0, 0, len(meta)))
blob.extend(meta)
blob.extend(struct.pack("<1I", len(recs)))
for rec in recs:
    blob.extend(struct.pack("<QII", *rec))
(rdir / "capture.fi02.bin").write_bytes(blob)

db = sqlite3.connect(str(rdir / "capture.sqlite"))
db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)")
db.execute("INSERT INTO StringIds VALUES (1, 'demo_kernel')")
db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1)", (t0 - 500, t + 1000))
db.commit()
db.close()

inst_ns = (
    [54000, 54000, 54000, 54000]
    if ovh_mode in ("high_ovh", "no_events_high_ovh")
    else [41000, 41000, 41000, 41000]
)
proto = "interleaved" if ovh_mode == "non_abba" else "ABBA"
rt = {
    "kernel_name": "demo_kernel",
    "baseline_sass": "l0.sass",
    "instrumented_sass": "inst.sass",
    "baseline_spill_bytes": 0,
    "instrumented_spill_bytes": 0,
    "extra_smem_bytes": 0,
    "baseline_output": "l0.bin",
    "instrumented_output": "inst.bin",
    "dump": "capture.fi02.bin",
    "nsys_sqlite": "capture.sqlite",
    "warmup": 2,
    "timing_protocol": proto,
    "baseline_ns": [40000, 40000, 40000, 40000],
    "instrumented_ns": inst_ns,
}
(rdir / "runtime.json").write_text(json.dumps(rt), encoding="utf-8")
""",
        encoding="utf-8",
    )


def _normalize_usage(stderr_text: str) -> str:
    return re.sub(r"^usage:\s+\S+", "usage: <prog>", stderr_text.strip())


class AkaAdapterCliCharacterizationTests(unittest.TestCase):
    _ENV_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "FI_SAMPLE",
        "WP_SAMPLE",
        "WEAVEPROF_SAMPLE",
        "DASHSCOPE_API_KEY",
        "BAILIAN_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_KEYS}
        cls._root = Path(tempfile.mkdtemp(prefix="wp-r11-g1-"))
        for key, rel in (
            ("HOME", "home"),
            ("XDG_CACHE_HOME", "xdg_cache"),
            ("XDG_CONFIG_HOME", "xdg_config"),
            ("MPLCONFIGDIR", "mplconfig"),
            ("WEAVEPROF_SESSIONS_DIR", "sessions"),
            ("WEAVEPROF_CACHE", "wp_cache"),
            ("WEAVEPROF_LOCK_DIR", "wp_locks"),
            ("WEAVEPROF_WORKDIR", "wp_runs"),
        ):
            target = cls._root / rel
            target.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(target)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        for clear_key in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "FI_SAMPLE",
            "WP_SAMPLE",
            "WEAVEPROF_SAMPLE",
            "DASHSCOPE_API_KEY",
            "BAILIAN_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        ):
            os.environ.pop(clear_key, None)

    @classmethod
    def tearDownClass(cls) -> None:
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def _make_case_dir(self, label: str) -> Path:
        p = Path(tempfile.mkdtemp(prefix=f"{label}-", dir=str(self._root)))
        return p

    def _subproc_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if extra:
            env.update(extra)
        return env

    def _run_inproc_cli(self, argv: list[str], stdin_text: str) -> tuple[int, str, str]:
        in_buf = io.StringIO(stdin_text)
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        prev_stdin = sys.stdin
        try:
            sys.stdin = in_buf
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                rc = aka_adapter.main(argv)
        finally:
            sys.stdin = prev_stdin
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def _run_plugin_subproc(
        self,
        tool_name: str,
        stdin_text: str,
        cwd: Path,
        *,
        extra_Args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        plugin_doc = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        if extra_Args is not None:
            cmd = [sys.executable, "-B", str(ROOT / "weaveprof" / "aka_adapter.py"), *extra_Args]
        else:
            raw_cmd = plugin_doc["tools"][tool_name]["command"]
            cmd = [
                tok.format(python=sys.executable, plugin_root=str(ROOT))
                for tok in raw_cmd
            ]
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            env=self._subproc_env(extra_env),
            timeout=120,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def _create_sealed_episode(self, case_dir: Path) -> tuple[Path, Path, Path, str]:
        src = case_dir / "kernel.cu"
        src.write_text("__global__ void sealed_k(float* x) { x[0] = 1.0f; }\n", encoding="utf-8")
        src_sha = file_sha256(src)
        rdir = case_dir / "round-1"
        rdir.mkdir()
        fb_path = rdir / "feedback.json"
        ver_path = rdir / "verify.json"
        trace_path = rdir / "trace.perfetto-trace"
        fb_path.write_text('{"status": "PASS"}\n', encoding="utf-8")
        ver_path.write_text('{"status": "PASS"}\n', encoding="utf-8")
        trace_path.write_bytes(b"\x0a\x04test")
        policy = HarnessPolicy(max_overhead_limit=0.10)
        result = {
            "status": "PASS",
            "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
            "failed": [],
            "metrics": {"overhead_ratio": 0.03},
        }
        seal_path, _seal_doc = EpisodeSealer.seal_round(
            round_dir=rdir,
            round_number=1,
            source_path=src,
            source_sha256=src_sha,
            probe_path=None,
            instrumented_path=None,
            names_path=None,
            feedback_path=fb_path,
            verify_path=ver_path,
            trace_path=trace_path,
            failed_trace_path=None,
            result=result,
            policy=policy,
            limit=0.10,
            bench_cmd=["python", "bench.py"],
            fsm_state="PROMOTED",
            region_lock={"gpu_id": "h20_0", "provider": "local_file"},
        )
        return seal_path, src, trace_path, src_sha

    def test_plugin_manifest_and_cli_protocol_errors(self) -> None:
        plugin_doc = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
        self.assertEqual(plugin_doc["id"], "weaveprofiler")
        self.assertEqual(plugin_doc["api_version"], 1)
        expected_tools = {
            "analyze",
            "compare-rounds",
            "megakernel-audit",
            "plan-context",
            "profile",
            "verify-seal",
        }
        self.assertEqual(set(plugin_doc["tools"].keys()), expected_tools)

        cdir = self._make_case_dir("proto")
        for bad_argv in ([], ["unknown-subcmd"], ["analyze", "extra"]):
            rc, out, err = self._run_inproc_cli(bad_argv, "{}")
            self.assertEqual(rc, 2)
            self.assertEqual(out, "")
            self.assertEqual(
                _normalize_usage(err),
                "usage: <prog> <analyze|compare-rounds|megakernel-audit|plan-context|profile|verify-seal>",
            )

        rc_sub, out_sub, err_sub = self._run_plugin_subproc(
            "analyze", "{}", cdir, extra_Args=["unknown-subcmd"]
        )
        self.assertEqual(rc_sub, 2)
        self.assertEqual(out_sub, "")
        self.assertTrue(err_sub.startswith("usage: "))

        rc_bad_json, out_bad_json, err_bad_json = self._run_inproc_cli(["analyze"], "{bad_json")
        self.assertEqual(rc_bad_json, 1)
        self.assertEqual(out_bad_json, "")
        self.assertTrue(
            err_bad_json.startswith("weaveprofiler.analyze error: JSONDecodeError: "),
            err_bad_json,
        )

        rc_list, out_list, err_list = self._run_plugin_subproc("analyze", "[1, 2]", cdir)
        self.assertEqual(rc_list, 1)
        self.assertEqual(out_list, "")
        self.assertEqual(
            err_list,
            "weaveprofiler.analyze error: ValueError: Request payload must be a JSON object\n",
        )

        rc_type, out_type, err_type = self._run_inproc_cli(["analyze"], json.dumps({"data": 42}))
        self.assertEqual(rc_type, 1)
        self.assertEqual(out_type, "")
        self.assertEqual(
            err_type,
            "weaveprofiler.analyze error: ValueError: Expected JSON object or file path, got int\n",
        )

    def test_analyze_success_inline_and_path_plus_error_path(self) -> None:
        cdir = self._make_case_dir("analyze")
        trace_doc = _sample_perfetto_trace_doc()
        req_inline = {
            "data": trace_doc,
            "arch": "sm_103a",
            "total_sms": 148,
        }
        rc, out_txt, err_txt = self._run_plugin_subproc("analyze", json.dumps(req_inline), cdir)
        self.assertEqual(rc, 0, err_txt)
        self.assertEqual(err_txt, "")
        out = json.loads(out_txt)
        self.assertEqual(set(out.keys()), ANALYZE_EXPECTED_KEYS)
        self.assertEqual(out["schema"], "weaveprof.causal-report/v1")
        self.assertEqual(out["verdict"], "CAUSE_EXPLAINED")
        self.assertEqual(out["global_hw_regime"], "DRAM_READ_BOUND")
        self.assertEqual(out["arch"], "sm_103a")
        self.assertEqual(out["gpu"], "B300")
        self.assertEqual(out["total_sms"], 148)
        self.assertEqual(out["kernel_us"], 100.0)
        self.assertEqual(out["tensor_calibration_factor"], 1.0)

        trace_file = cdir / "trace.json"
        trace_file.write_text(json.dumps(trace_doc), encoding="utf-8")
        rc2, out_txt2, err_txt2 = self._run_inproc_cli(
            ["analyze"], json.dumps({"trace_json_path": str(trace_file), "total_sms": 148})
        )
        self.assertEqual(rc2, 0, err_txt2)
        out2 = json.loads(out_txt2)
        self.assertEqual(set(out2.keys()), ANALYZE_EXPECTED_KEYS)
        self.assertEqual(out2["global_hw_regime"], "DRAM_READ_BOUND")
        self.assertEqual(out2["observed_ctas"], 16)

        # Current behavior (`疑似 bug`): `aka_adapter.py:80-85` forwards `gpu_name` and `workload`
        # to `analyze_sw_hw_causality`, which accepts neither keyword argument.
        rc_kw, out_kw, err_kw = self._run_inproc_cli(
            ["analyze"],
            json.dumps({"data": trace_doc, "gpu_name": "NVIDIA B300", "workload": {"dtype": "bf16"}}),
        )
        self.assertEqual(rc_kw, 1)
        self.assertEqual(out_kw, "")
        self.assertEqual(
            err_kw,
            "weaveprofiler.analyze error: TypeError: analyze_sw_hw_causality() got an unexpected keyword argument 'gpu_name'\n",
        )

        rc_err, out_err, err_err = self._run_inproc_cli(["analyze"], "   ")
        self.assertEqual(rc_err, 1)
        self.assertEqual(out_err, "")
        self.assertEqual(
            err_err,
            "weaveprofiler.analyze error: ValueError: Either 'data' or 'trace_json_path' is required\n",
        )

    def test_compare_rounds_success_noise_floor_and_error_path(self) -> None:
        cdir = self._make_case_dir("cmp")
        prev_rep, curr_rep = _sample_prev_curr_reports()
        rc, out_txt, err_txt = self._run_plugin_subproc(
            "compare-rounds",
            json.dumps({"prev_report": prev_rep, "curr_report": curr_rep, "noise_floor_pct": 1.0}),
            cdir,
        )
        self.assertEqual(rc, 0, err_txt)
        out = json.loads(out_txt)
        self.assertEqual(set(out.keys()), COMPARE_ROUNDS_EXPECTED_KEYS)
        self.assertEqual(out["schema"], "weaveprof.round-diff/v1")
        self.assertIs(out["improved"], True)
        self.assertIs(out["regressed"], False)
        self.assertEqual(out["latency_delta_us"], -50.0)
        self.assertEqual(out["latency_delta_pct"], -25.0)
        self.assertEqual(out["speedup_ratio"], 1.3333)
        self.assertEqual(out["curr_verdict"], "SATURATED_STOP_REFINING")
        self.assertEqual(out["next_round_directive"]["action"], "STOP_SATURATED")
        self.assertEqual(out["noise_floor_pct"], 1.0)

        prev_path = cdir / "prev.json"
        curr_path = cdir / "curr.json"
        prev_path.write_text(json.dumps(prev_rep), encoding="utf-8")
        curr_path.write_text(json.dumps(curr_rep), encoding="utf-8")
        rc_nf, out_nf_txt, err_nf = self._run_inproc_cli(
            ["compare-rounds"],
            json.dumps({
                "prev_report_path": str(prev_path),
                "curr_report_path": str(curr_path),
                "noise_floor_pct": 30.0,
            }),
        )
        self.assertEqual(rc_nf, 0, err_nf)
        out_nf = json.loads(out_nf_txt)
        self.assertIs(out_nf["improved"], False)
        self.assertEqual(out_nf["noise_floor_pct"], 30.0)

        rc_err, out_err, err_err = self._run_inproc_cli(["compare-rounds"], "{}")
        self.assertEqual(rc_err, 1)
        self.assertEqual(out_err, "")
        self.assertEqual(
            err_err,
            "weaveprofiler.compare-rounds error: ValueError: Expected JSON object or file path, got NoneType\n",
        )

    def test_plan_context_success_and_error_path(self) -> None:
        cdir = self._make_case_dir("plan")
        src_text = "__global__ void gemm_k(float* x) { x[threadIdx.x] = 1.0f; }\n"
        req = {
            "source_text": src_text,
            "arch": "sm_90a",
            "gpu_name": "NVIDIA H20",
            "block_threads": 256,
            "l0_regs": 64,
            "l0_smem_bytes": 16384,
            "smem_limit_bytes": 232448,
            "max_warps_per_cta": 8,
        }
        rc, out_txt, err_txt = self._run_plugin_subproc("plan-context", json.dumps(req), cdir)
        self.assertEqual(rc, 0, err_txt)
        out = json.loads(out_txt)
        self.assertEqual(set(out.keys()), PLAN_CONTEXT_EXPECTED_KEYS)
        self.assertEqual(out["arch"], "sm_90a")
        self.assertEqual(out["context_mode"], "WP_CTX_REG")
        self.assertEqual(out["compile_defines"], ["-DWP_CTX_REG"])
        self.assertEqual(out["recommended_granularity"], "warp")
        self.assertEqual(out["reg_budget_per_thread"], 255)
        self.assertEqual(out["reg_headroom"], 191)
        self.assertEqual(out["smem_free_bytes"], 216064)
        self.assertIs(out["has_setmaxnreg"], False)

        src_file = cdir / "setmaxnreg.cu"
        src_file.write_text(
            '__global__ void k() { asm volatile("setmaxnreg.dec.sync.aligned.u32 48;\\n"); }\n',
            encoding="utf-8",
        )
        rc2, out_txt2, err_txt2 = self._run_inproc_cli(
            ["plan-context"],
            json.dumps({"source_path": str(src_file), "arch": "sm_103a", "block_threads": 512, "l0_regs": 128}),
        )
        self.assertEqual(rc2, 0, err_txt2)
        out2 = json.loads(out_txt2)
        self.assertEqual(set(out2.keys()), PLAN_CONTEXT_EXPECTED_KEYS)
        self.assertIs(out2["has_setmaxnreg"], True)
        self.assertEqual(out2["placement_search_space"], ["pre_setmaxnreg", "post_setmaxnreg"])

        rc_err, out_err, err_err = self._run_inproc_cli(["plan-context"], "{}")
        self.assertEqual(rc_err, 1)
        self.assertEqual(out_err, "")
        self.assertEqual(
            err_err,
            "weaveprofiler.plan-context error: ValueError: Either 'source_text' or 'source_path' is required\n",
        )

    def test_verify_seal_success_tamper_kwargs_and_error_path(self) -> None:
        cdir = self._make_case_dir("seal")
        seal_path, _src, trace_path, src_sha = self._create_sealed_episode(cdir)

        rc, out_txt, err_txt = self._run_plugin_subproc(
            "verify-seal",
            json.dumps({
                "seal_path": str(seal_path),
                "expected_source_sha": src_sha,
                "require_pass": True,
                "max_overhead_limit": 0.10,
            }),
            cdir,
        )
        self.assertEqual(rc, 0, err_txt)
        out = json.loads(out_txt)
        self.assertEqual(set(out.keys()), {"ok", "errors", "seal"})
        self.assertIs(out["ok"], True)
        self.assertEqual(out["errors"], [])
        self.assertEqual(out["seal"]["schema"], "weaveprof.episode-seal/v1")
        self.assertEqual(out["seal"]["status"], "PASS")

        rc_ceil, out_ceil_txt, _ = self._run_inproc_cli(
            ["verify-seal"],
            json.dumps({"seal_path": str(seal_path), "max_overhead_limit": 0.01}),
        )
        self.assertEqual(rc_ceil, 0)
        out_ceil = json.loads(out_ceil_txt)
        self.assertIs(out_ceil["ok"], False)
        self.assertTrue(
            any("exceeds policy ceiling 0.01" in e or "exceeds ceiling=0.01" in e for e in out_ceil["errors"]),
            out_ceil["errors"],
        )

        trace_path.write_bytes(b"tampered-trace-bytes")
        rc_tamp, out_tamp_txt, err_tamp = self._run_plugin_subproc(
            "verify-seal", json.dumps({"seal_path": str(seal_path)}), cdir
        )
        self.assertEqual(rc_tamp, 0, err_tamp)
        out_tamp = json.loads(out_tamp_txt)
        self.assertEqual(set(out_tamp.keys()), {"ok", "errors", "seal"})
        self.assertIs(out_tamp["ok"], False)
        self.assertEqual(len(out_tamp["errors"]), 1)
        self.assertTrue(
            out_tamp["errors"][0].startswith("Bound artifact 'trace' hash mismatch:"),
            out_tamp["errors"],
        )

        rc_err, out_err, err_err = self._run_inproc_cli(["verify-seal"], "{}")
        self.assertEqual(rc_err, 1)
        self.assertEqual(out_err, "")
        self.assertEqual(
            err_err,
            "weaveprofiler.verify-seal error: KeyError: 'seal_path'\n",
        )

    def test_megakernel_audit_protocol_arg_mapping_and_error_paths(self) -> None:
        cdir = self._make_case_dir("mk")

        rc_key, out_key, err_key = self._run_inproc_cli(["megakernel-audit"], "{}")
        self.assertEqual(rc_key, 1)
        self.assertEqual(out_key, "")
        self.assertEqual(
            err_key,
            "weaveprofiler.megakernel-audit error: KeyError: 'workspace'\n",
        )

        missing_ws = cdir / "no_such_ws"
        out_dir = cdir / "audit_out"
        rc_miss, out_miss_txt, err_miss = self._run_plugin_subproc(
            "megakernel-audit",
            json.dumps({
                "workspace": str(missing_ws),
                "entry_source": "kernel.cu",
                "out_dir": str(out_dir),
            }),
            cdir,
        )
        self.assertEqual(rc_miss, 0, err_miss)
        out_miss = json.loads(out_miss_txt)
        self.assertEqual(set(out_miss.keys()), MEGAKERNEL_BLOCKED_EXPECTED_KEYS)
        self.assertEqual(out_miss["schema"], "weaveprof.megakernel-audit/v1")
        self.assertEqual(out_miss["status"], "BLOCKED")
        self.assertEqual(out_miss["mode"], "precaptured")
        self.assertIs(out_miss["seal_verified"], False)
        self.assertEqual(len(out_miss["structural_blockers"]), 1)
        self.assertTrue(
            out_miss["structural_blockers"][0].startswith("WORKSPACE_MISSING: "),
            out_miss["structural_blockers"],
        )

        ws = cdir / "ws"
        ws.mkdir()
        (ws / "kernel.cu").write_text("__global__ void k(float* x) { x[0] = 1.0f; }\n", encoding="utf-8")

        rc_in_ws, out_in_ws_txt, _ = self._run_inproc_cli(
            ["megakernel-audit"],
            json.dumps({
                "workspace": str(ws),
                "entry_source": "kernel.cu",
                "bench_cmd": ["python", "bench.py"],
                "out_dir": str(ws / "nested_out"),
            }),
        )
        self.assertEqual(rc_in_ws, 0)
        out_in_ws = json.loads(out_in_ws_txt)
        self.assertEqual(out_in_ws["status"], "BLOCKED")
        self.assertEqual(out_in_ws["mode"], "shadow_build_run")
        self.assertTrue(
            out_in_ws["structural_blockers"][0].startswith("OUT_DIR_INSIDE_WORKSPACE: "),
            out_in_ws["structural_blockers"],
        )

        rc_lim, out_lim_txt, _ = self._run_inproc_cli(
            ["megakernel-audit"],
            json.dumps({
                "workspace": str(ws),
                "entry_source": "kernel.cu",
                "bench_cmd": ["python", "bench.py"],
                "limit": 0.25,
                "out_dir": str(cdir / "audit_lim_out"),
            }),
        )
        self.assertEqual(rc_lim, 0)
        out_lim = json.loads(out_lim_txt)
        self.assertEqual(out_lim["status"], "BLOCKED")
        self.assertEqual(out_lim["mode"], "shadow_build_run")
        self.assertEqual(
            out_lim["structural_blockers"],
            [
                "LIMIT_RELAXATION_BLOCKED: [LIMIT_RELAXATION_BLOCKED] Requested overhead limit 0.2500 exceeds hard policy ceiling 0.1000."
            ],
        )

    def test_profile_pass_fail_and_policy_error_paths(self) -> None:
        cdir = self._make_case_dir("prof")
        proj = cdir / "proj"
        proj.mkdir()
        src = proj / "kernel.cu"
        src.write_text(
            "__global__ void demo_kernel(float* out, int n) {\n"
            "    for (int i = 0; i < n; ++i) {\n"
            "        float v = out[i];\n"
            "        v = v * 2.0f + 1.0f;\n"
            "        out[i] = v;\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        identify(src, offline_heuristic=True, card="h20")

        bench_py = cdir / "bench_synth.py"
        _write_synth_bench_script(bench_py)
        prev_rep, _ = _sample_prev_curr_reports()
        prev_file = cdir / "prev_causal.json"
        prev_file.write_text(json.dumps(prev_rep), encoding="utf-8")

        out_trace = cdir / "out_pass" / "trace.perfetto-trace"
        rc_pass, out_pass_txt, err_pass = self._run_inproc_cli(
            ["profile"],
            json.dumps({
                "source": str(src),
                "bench": [sys.executable, "-B", str(bench_py)],
                "card": "h20",
                "limit": 0.10,
                "max_overhead_limit": 0.10,
                "require_abba_order": True,
                "max_rounds": 1,
                "out": str(out_trace),
                "prev_causal_report_path": str(prev_file),
            }),
        )
        self.assertEqual(rc_pass, 0, err_pass)
        out_pass = json.loads(out_pass_txt)
        self.assertEqual(set(out_pass.keys()), AGENT_REPORT_EXPECTED_KEYS)
        self.assertEqual(out_pass["schema"], "weaveprof.agent-report/v1")
        self.assertEqual(out_pass["status"], "PASS")
        self.assertIs(out_pass["seal_verified"], True)
        self.assertEqual(out_pass["seal_errors"], [])
        self.assertEqual(out_pass["overhead_pct"], 2.5)
        self.assertEqual(out_pass["failed"], [])
        self.assertEqual(
            out_pass["fidelity_gates"]["checks"],
            {"assembly": True, "output": True, "event": True, "events": True, "overhead": True},
        )
        # Current behavior (`疑似 bug`): `build_agent_report` always sets `"round_diff": None`,
        # so `"round_diff" not in res` at `aka_adapter.py:67` is False and leaves `round_diff` as None.
        self.assertIsNone(out_pass["round_diff"])

        out_fail_trace = cdir / "out_fail" / "trace.perfetto-trace"
        rc_fail, out_fail_txt, _err_fail = self._run_plugin_subproc(
            "profile",
            json.dumps({
                "source": str(src),
                "bench": [sys.executable, "-B", str(bench_py)],
                "card": "h20",
                "limit": 0.10,
                "max_overhead_limit": 0.10,
                "max_rounds": 1,
                "out": str(out_fail_trace),
                "prev_causal_report": prev_rep,
            }),
            cdir,
            extra_env={"WP_CHAR_BENCH_MODE": "no_events_high_ovh"},
        )
        self.assertEqual(rc_fail, 0)
        out_fail = json.loads(out_fail_txt)
        self.assertEqual(out_fail["schema"], "weaveprof.agent-report/v1")
        self.assertEqual(out_fail["status"], "FAIL")
        self.assertIs(out_fail["seal_verified"], True)
        self.assertEqual(out_fail["overhead_pct"], 35.0)
        self.assertEqual(out_fail["failed"], ["pairing", "overhead"])

        # Current behavior (`疑似 bug`): when a failed round has non-empty string `hotspots`,
        # `weaveprof/api.py:470-476` calls `hs.get("name")` on a `str` and raises `AttributeError`.
        rc_hs_bug, out_hs_bug, err_hs_bug = self._run_plugin_subproc(
            "profile",
            json.dumps({
                "source": str(src),
                "bench": [sys.executable, "-B", str(bench_py)],
                "card": "h20",
                "limit": 0.10,
                "max_overhead_limit": 0.10,
                "max_rounds": 1,
                "out": str(cdir / "out_hs_bug" / "trace.perfetto-trace"),
            }),
            cdir,
            extra_env={"WP_CHAR_BENCH_MODE": "high_ovh"},
        )
        self.assertEqual(rc_hs_bug, 1)
        self.assertEqual(out_hs_bug, "")
        self.assertEqual(
            err_hs_bug,
            "weaveprofiler.profile error: AttributeError: 'str' object has no attribute 'get'\n",
        )

        rc_pol, out_pol, err_pol = self._run_inproc_cli(
            ["profile"],
            json.dumps({
                "source": str(src),
                "bench": [sys.executable, "-B", str(bench_py)],
                "limit": 0.20,
                "max_overhead_limit": 0.10,
                "out": str(cdir / "out_pol" / "trace.perfetto-trace"),
            }),
        )
        self.assertEqual(rc_pol, 1)
        self.assertEqual(out_pol, "")
        self.assertTrue(
            err_pol.startswith(
                "weaveprofiler.profile error: PolicyViolation: [LIMIT_RELAXATION_BLOCKED]"
            ),
            err_pol,
        )


if __name__ == "__main__":
    unittest.main()

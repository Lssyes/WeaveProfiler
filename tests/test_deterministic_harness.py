#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit & integration tests for WeaveProfiler's Atrex+KREX Deterministic Outer Harness."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.api import profile
from weaveprof.backend import VerificationError, verify_round
from weaveprof.cli import main as cli_main
from weaveprof.harness import (
    EpisodeSealer,
    HarnessPolicy,
    IncumbentTracker,
    PolicyGuard,
    PolicyViolation,
    RoundStateMachine,
    SealedRoundRecord,
    abba_schedule,
    gpu_critical_region,
    summarize_abba,
    verify_seal,
)
from weaveprof.workspace import ShadowWorkspace


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DECODER = _load("4_validate/tools/fi_lite_decode.py", "dh_decoder")
ENGINE = _load("2_instrument/rule_engine.py", "dh_engine")
PRETOOL = ROOT / "4_validate" / "harness" / "hooks" / "fi_pretool_guard.py"
STOPGATE = ROOT / "4_validate" / "harness" / "hooks" / "fi_stop_gate.py"
MARKER = ROOT / "4_validate" / "harness" / "hooks" / "fi_session_marker.py"


def _sample_cuda_pipeline() -> str:
    lines = [""] * 73
    for number, code in {
        1: "__global__ void pipeline(float* out) {",
        2: "    int s = 0;",
        41: "    if (threadIdx.x / warpSize == 4) {",
        42: "        tma_load_async(s);",
        43: "    }",
        54: "    if (threadIdx.x / warpSize < 4) {",
        55: "        full_bar[s].wait();",
        58: "        for (int k=0; k<8192; ++k) {",
        60: "            // inner math body",
        61: "            float x = out[k];",
        62: "            x *= x;",
        63: "            out[k] = x;",
        70: "        }",
        71: "    }",
        73: "}",
    }.items():
        lines[number - 1] = code
    return "\n".join(lines) + "\n"


def _write_filite02(path: Path, lanes, wpc=5) -> None:
    meta = json.dumps({"gran": 0, "arch": "sm_90a", "backend": "nvidia_ptx"}).encode("utf-8")
    counts = [len(records) for records in lanes]
    blob = bytearray(b"FILITE02")
    blob.extend(struct.pack("<6I", len(lanes), max(counts), wpc, 0, 0, len(meta)))
    blob.extend(meta)
    blob.extend(struct.pack(f"<{len(lanes)}I", *counts))
    for records in lanes:
        for rec in records:
            blob.extend(struct.pack("<QII", *rec))
    path.write_bytes(blob)


def _write_fake_nsys_sqlite(path: Path, kernel_name: str, start_ns: int, end_ns: int) -> None:
    db = sqlite3.connect(str(path))
    try:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)")
        db.execute("INSERT INTO StringIds VALUES (1, ?)", (kernel_name,))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1)", (start_ns, end_ns))
        db.commit()
    finally:
        db.close()


def _write_synthetic_round_artifacts(
    directory: Path,
    spec,
    *,
    hot_inner_loop: bool,
    timing_protocol: str = "ABBA",
) -> None:
    base_ops = ["MOV", "LDG", "HMMA", "IADD", "IMAD", "MUL", "STG", "FADD", "EXIT"]
    if hot_inner_loop:
        # Triggers Check 1 assembly failure (core retained but heavy churn) + Check 4 (+38.4%)
        inst_ops = ["MOV", "NOP", "NOP", "NOP", "NOP", "HMMA", "NOP", "NOP", "STG", "EXIT"]
        base_ns = [42000, 42100, 41950, 42050, 42000, 42020]
        inst_ns = [58100, 58200, 58050, 58150, 58100, 58120]
        mma_iters = 128
        mma_dur = 25
    else:
        inst_ops = [*base_ops]
        base_ns = [42000, 42100, 41950, 42050, 42000, 42020]
        inst_ns = [42750, 42820, 42700, 42790, 42760, 42770]
        mma_iters = 1
        mma_dur = 35000

    for name, ops in (("l0", base_ops), ("inst", inst_ops)):
        sass = "Function : pipeline\n" + "".join(
            f"/*{16 * i:04x}*/ {op} ;\n" for i, op in enumerate(ops)
        )
        (directory / f"{name}.sass").write_text(sass, encoding="utf-8")

    (directory / "l0.bin").write_bytes(b"\x01\x02\x03\x04" * 16)
    (directory / "inst.bin").write_bytes(b"\x01\x02\x03\x04" * 16)

    names = json.loads((directory / "names.json").read_text(encoding="utf-8"))
    probe_eids = {p["id"]: p["event_id"] for p in spec["probes"]}
    roles_rev = {v: int(k) for k, v in names["roles"].items()}
    dma_rid = roles_rev.get("dma", 0)
    math_rid = roles_rev.get("math", 0)
    load_eid = probe_eids.get("load")
    mma_eid = probe_eids.get("mma")

    # Build a clean multi-warp FILITE02 capture
    t0 = 1_000_000
    lanes = []
    if dma_rid != math_rid and load_eid is not None:
        # 4 math warps + 1 dma warp
        for w in range(4):
            recs = [(t0, DECODER.OPEN | math_rid, 0), (t0 + 500 + 10 * w, DECODER.END | load_eid, 0)]
            t = t0 + 1000
            if mma_eid is not None:
                for _ in range(mma_iters):
                    recs.append((t, mma_eid, 0))
                    recs.append((t + mma_dur, DECODER.END | mma_eid, 0))
                    t += mma_dur + 15
            recs.append((t + 500, DECODER.CLOSE, 0))
            lanes.append(recs)
        lanes.append([(t0, DECODER.OPEN | dma_rid, 0), (t0 + 200, load_eid, 0), (t + 500, DECODER.CLOSE, 0)])
        wpc = 5
    else:
        recs = [(t0, DECODER.OPEN, 0)]
        t = t0 + 200
        for p in spec["probes"]:
            eid = p["event_id"]
            iters = mma_iters if p["id"] == "mma" else 1
            for _ in range(iters):
                recs.append((t, eid, 0))
                recs.append((t + mma_dur, DECODER.END | eid, 0))
                t += mma_dur + 15
        recs.append((t + 300, DECODER.CLOSE, 0))
        lanes.append(recs)
        wpc = 1

    _write_filite02(directory / "capture.fi02.bin", lanes, wpc=wpc)
    _write_fake_nsys_sqlite(directory / "capture.sqlite", "pipeline", t0 - 1000, t0 + 60000)

    runtime = {
        "kernel_name": "pipeline",
        "baseline_sass": "l0.sass",
        "instrumented_sass": "inst.sass",
        "baseline_spill_bytes": 0,
        "instrumented_spill_bytes": 0,
        "extra_smem_bytes": 0,
        "baseline_output": "l0.bin",
        "instrumented_output": "inst.bin",
        "warmup": 4,
        "timing_protocol": timing_protocol,
        "baseline_ns": base_ns,
        "instrumented_ns": inst_ns,
    }
    (directory / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


class DeterministicHarnessTests(unittest.TestCase):
    def test_policy_guard_blocks_anti_cheat_violations(self):
        with tempfile.TemporaryDirectory() as td:
            proj = Path(td) / "proj"
            proj.mkdir()
            src = proj / "k.cu"
            src.write_text(_sample_cuda_pipeline(), encoding="utf-8")
            sha = ShadowWorkspace(src, location=Path(td) / "ws").sha
            guard = PolicyGuard(HarnessPolicy(max_overhead_limit=0.10))

            # 1. Limit relaxation blocked
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_session_start(src, sha, 0.25, ["python", "bench.py"])
            self.assertEqual(ctx.exception.code, "LIMIT_RELAXATION_BLOCKED")

            # 2. Loop sampling blocked
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_session_start(src, sha, 0.10, ["FI_SAMPLE=8", "python", "bench.py"])
            self.assertEqual(ctx.exception.code, "LOOP_SAMPLING_FORBIDDEN")

            # 3. MLIR blocked
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_session_start(src, sha, 0.10, ["mlir-opt", "k.mlir"])
            self.assertEqual(ctx.exception.code, "MLIR_FORBIDDEN")

            # 4. Source immutability enforced
            src.write_text(_sample_cuda_pipeline() + "// tampered\n", encoding="utf-8")
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_source_immutable(src, sha)
            self.assertEqual(ctx.exception.code, "SOURCE_MUTATED")

            # 5. Spin-loop static policy catch
            spin_src = (
                "__global__ void spin_k(volatile int* flag, float* out) {\n"
                "    while (*flag == 0) {\n"
                "        __nanosleep(32);\n"
                "    }\n"
                "    out[0] = 1.0f;\n"
                "}\n"
            )
            bad_spec = {"roles": {"w": "all"}, "probes": [{"id": "wait", "begin": {"role": "w", "line": 3}, "end": {"role": "w", "line": 3}}]}
            issues = guard.validate_probe_spec(bad_spec, spin_src)
            self.assertTrue(any(i["code"] == "SPIN_LOOP_PROBE_POLICY" for i in issues), issues)

    def test_abba_schedule_and_drift_summary(self):
        seq = abba_schedule(3)
        self.assertEqual(seq, ["baseline", "instrumented", "instrumented", "baseline"] * 3)
        summary = summarize_abba([1000, 1020, 1040, 1060], [1020, 1040, 1060, 1080], protocol="ABBA")
        self.assertEqual(summary["protocol"], "ABBA")
        self.assertAlmostEqual(summary["median_overhead_ratio"], 1050.0 / 1030.0 - 1.0, places=5)
        self.assertGreater(summary["baseline_half_drift_pct"], 0.0)

        strict_guard = PolicyGuard(HarnessPolicy(require_abba_order=True))
        ok, _, errs = strict_guard.validate_runtime_protocol(
            {"warmup": 2, "baseline_ns": [100, 101, 102], "instrumented_ns": [102, 103, 104], "timing_protocol": "naive"},
            0.10,
        )
        self.assertFalse(ok)
        self.assertTrue(any("ABBA" in e for e in errs))

    def test_krex_region_lock_concurrency_and_zero_overlap(self):
        """Off-region work runs in parallel across 4 workers; in-region timing is 100% mutually exclusive."""
        with tempfile.TemporaryDirectory() as td:
            lock_dir = Path(td) / "locks"
            intervals = []

            def worker(idx: int):
                # Off-region work (simulating LLM + compile + AST rewrite): runs concurrently
                time.sleep(0.12)
                # In-region GPU measurement: strictly exclusive per GPU ID
                with gpu_critical_region(gpu_id="h20_0", timeout_s=15.0, lock_dir=lock_dir, label=f"w{idx}") as lease:
                    t_start = time.perf_counter()
                    time.sleep(0.05)
                    t_end = time.perf_counter()
                return idx, t_start, t_end, lease

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(worker, range(4)))
            total_wall = time.perf_counter() - t0

            for _, s, e, lease in results:
                intervals.append((s, e))
                self.assertGreaterEqual(lease.hold_ms, 35.0)
            intervals.sort()
            # Assert ZERO overlap between critical regions
            for (_, prev_end), (next_start, _) in zip(intervals, intervals[1:]):
                self.assertGreaterEqual(next_start, prev_end - 1e-4, "KREX critical regions overlapped!")
            # Serial would take 4 * (0.12 + 0.05) = 0.68s; concurrent off-region should finish well under 0.55s
            self.assertLess(total_wall, 0.58, f"Off-region concurrency too slow: {total_wall:.3f}s")

    def test_failed_timeline_export_sealing_and_fsm_escalation_end_to_end(self):
        """Full closed-loop Supervisor test:
        Round 1: Hot inner-loop probe fails Check 1+4 -> exports failed_trace.perfetto-trace +
                 timeline_summary + sealed round-1/seal.json.
        Round 2: Stubborn LLM repeats the exact same failing probe -> FSM catches duplicate/oscillation
                 and deterministically escalates to hoist `mma` to enclosing loop L58-L70 -> PASSES ->
                 promotes sealed `out.perfetto-trace` + `out.perfetto-trace.seal.json` + `episode_ledger.json`.
        """
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            proj = td_p / "proj"
            proj.mkdir()
            src = proj / "pipeline.cu"
            src.write_text(_sample_cuda_pipeline(), encoding="utf-8")
            out_dir = td_p / "out"
            out_dir.mkdir()
            out_trace = out_dir / "pipeline.perfetto-trace"

            hot_spec = {
                "roles": {"dma": "warp 4", "math": "warps 0-3"},
                "probes": [
                    {"id": "load", "begin": {"role": "dma", "line": 42}, "end": {"role": "math", "line": 55}, "slot": "s"},
                    {"id": "mma", "begin": {"role": "math", "line": 61}, "end": {"role": "math", "line": 63}},
                ],
            }

            llm_calls = []

            def stubborn_llm(model, messages, **kwargs):
                llm_calls.append(messages[-1]["content"])
                # Stubbornly echoes the hot inner-loop spec even on Round 2 refinement!
                return {"ok": True, "elapsed_s": 0.1, "content": json.dumps(hot_spec), "usage": {"total_tokens": 100}}

            from weaveprof.api import _Session
            old_cache = os.environ.get("WEAVEPROF_CACHE")
            os.environ["WEAVEPROF_CACHE"] = str(td_p / "cache")
            try:
                policy = HarnessPolicy(
                    max_overhead_limit=0.10,
                    require_abba_order=True,
                    enable_region_lock=True,
                    escalation_on_stagnation=True,
                )
                sess = _Session(
                    src, 0.10, out_trace, hot_spec, None, "qwen4-max", 3, "h20",
                    policy=policy, gpu_id="test_gpu_0", bench_cmd=["python", "bench.py"],
                    llm_refine_first=True, llm_caller=stubborn_llm,
                )

                def fake_measure(prepared):
                    directory, _base, _inst = prepared
                    cur_spec = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
                    # Normalize to get event_ids
                    from weaveprof.contract import normalize_probe_spec
                    norm = normalize_probe_spec(cur_spec, src.read_text(encoding="utf-8"))
                    mma_p = next(p for p in norm["probes"] if p["id"] == "mma")
                    is_hot = mma_p["begin"]["line"] >= 60
                    _write_synthetic_round_artifacts(directory, norm, hot_inner_loop=is_hot, timing_protocol="ABBA")

                res = sess.execute(fake_measure)
            finally:
                if old_cache is None:
                    os.environ.pop("WEAVEPROF_CACHE", None)
                else:
                    os.environ["WEAVEPROF_CACHE"] = old_cache
            self.assertEqual(res["status"], "PASS")
            self.assertTrue(out_trace.is_file())
            seal_path = Path(res["seal_path"])
            self.assertTrue(seal_path.is_file())
            ledger_path = Path(res["ledger_path"])
            self.assertTrue(ledger_path.is_file())

            # 1. Round 1 must have exported failed_trace.perfetto-trace + timeline_summary into LLM Round 2 prompt!
            self.assertEqual(len(llm_calls), 1)
            self.assertIn("Failed-run timeline binary summary", llm_calls[0])
            self.assertIn("inside loop L58-L70", llm_calls[0])

            r1_dir = Path(sess.tracker.records[0].round_dir)
            self.assertTrue((r1_dir / "failed_trace.perfetto-trace").is_file())
            r1_fb = json.loads((r1_dir / "feedback.json").read_text(encoding="utf-8"))
            self.assertIn("timeline_summary", r1_fb)
            self.assertTrue(r1_fb["timeline_summary"]["hotspots"])

            # 2. Ledger must record deterministic escalation after stubborn duplicate proposal
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(ledger["status"], "PASS")
            self.assertEqual(ledger["promoted_round"], 2)
            self.assertTrue(any(t["to"] == "ESCALATE_DETERMINISTIC" for t in ledger["fsm_transitions"]))

            # 3. Promoted seal passes verify_seal() AND `weaveprof seal-check` CLI
            check_ok = verify_seal(seal_path, require_pass=True)
            self.assertTrue(check_ok["ok"], check_ok["errors"])
            self.assertEqual(cli_main(["seal-check", str(seal_path)]), 0)

            # 4. Tampering 1 byte of promoted trace immediately fails verify_seal() & CLI!
            orig_bytes = out_trace.read_bytes()
            out_trace.write_bytes(orig_bytes + b"\x00")
            check_bad = verify_seal(seal_path, require_pass=True)
            self.assertFalse(check_bad["ok"])
            self.assertTrue(any("hash mismatch" in e for e in check_bad["errors"]))
            self.assertEqual(cli_main(["seal-check", str(seal_path)]), 1)
            out_trace.write_bytes(orig_bytes)

    def test_external_hooks_enforce_r4_policy_and_sealed_stop_gate(self):
        """Test real stdin/stdout hook wire protocol for PreToolUse (R4) and Stop Gate (sealed trace)."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            state_dir = td_p / "state"
            state_dir.mkdir()
            work_dir = td_p / "work"
            work_dir.mkdir()
            env = dict(os.environ, FI_STATE_DIR=str(state_dir))
            session_id = "test_harness_sess_01"

            def run_hook(script: Path, payload: dict):
                return subprocess.run(
                    [sys.executable, "-B", str(script)],
                    input=json.dumps(payload),
                    capture_output=True, text=True, env=env, timeout=30,
                )

            # Start session marker
            r_start = run_hook(MARKER, {"hook_event_name": "SessionStart", "session_id": session_id, "cwd": str(work_dir)})
            self.assertEqual(r_start.returncode, 0)

            # R4: Block --limit relaxation (> 0.10)
            r_lim = run_hook(PRETOOL, {
                "hook_event_name": "PreToolUse", "session_id": session_id, "cwd": str(work_dir),
                "tool_name": "Bash",
                "tool_input": {"command": "python -m weaveprof profile k.cu --limit 0.35 -- python bench.py"},
            })
            self.assertIn('"permissionDecision": "deny"', r_lim.stdout)
            self.assertIn("R4", r_lim.stdout)

            # R4: Block FI_SAMPLE loop sampling
            r_samp = run_hook(PRETOOL, {
                "hook_event_name": "PreToolUse", "session_id": session_id, "cwd": str(work_dir),
                "tool_name": "Bash",
                "tool_input": {"command": "FI_SAMPLE=8 python -m weaveprof profile k.cu --limit 0.10 -- python bench.py"},
            })
            self.assertIn('"permissionDecision": "deny"', r_samp.stdout)
            self.assertIn("FI_SAMPLE", r_samp.stdout)

            # R3: Block shell tampering of *.seal.json
            r_tamper = run_hook(PRETOOL, {
                "hook_event_name": "PreToolUse", "session_id": session_id, "cwd": str(work_dir),
                "tool_name": "Bash",
                "tool_input": {"command": "echo '{}' > out.perfetto-trace.seal.json"},
            })
            self.assertIn('"permissionDecision": "deny"', r_tamper.stdout)
            self.assertIn("R3", r_tamper.stdout)

            # Valid weaveprof invocation allowed and arms session
            r_ok = run_hook(PRETOOL, {
                "hook_event_name": "PreToolUse", "session_id": session_id, "cwd": str(work_dir),
                "tool_name": "Bash",
                "tool_input": {"command": "python -m weaveprof profile k.cu --limit 0.10 -- python bench.py"},
            })
            self.assertEqual(r_ok.stdout.strip(), "")

            # Stop Gate: unsealed .perfetto-trace MUST be blocked!
            unsealed_trace = work_dir / "demo.perfetto-trace"
            unsealed_trace.write_bytes(b"FAKE_TRACE_BYTES")
            r_stop_block = run_hook(STOPGATE, {
                "hook_event_name": "Stop", "session_id": session_id, "cwd": str(work_dir),
            })
            self.assertIn('"decision": "block"', r_stop_block.stdout)

            # Now create a genuine sealed trace via EpisodeSealer -> Stop Gate MUST allow!
            src = work_dir / "k.cu"
            src.write_text("__global__ void k() {}\n", encoding="utf-8")
            fb = work_dir / "feedback.json"
            fb.write_text('{"status":"PASS"}\n', encoding="utf-8")
            EpisodeSealer.seal_round(
                round_dir=work_dir, round_number=1, source_path=src,
                source_sha256=ShadowWorkspace(src, location=td_p / "ws2").sha,
                probe_path=None, instrumented_path=None, names_path=None,
                feedback_path=fb, verify_path=None, trace_path=unsealed_trace,
                failed_trace_path=None,
                result={"status": "PASS", "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
                        "failed": [], "metrics": {"overhead_ratio": 0.02}},
                policy=HarnessPolicy(), limit=0.10,
                region_lock={"gpu_id": "gpu_0", "wait_ms": 0.1, "hold_ms": 50.0},
                out_seal_path=Path(str(unsealed_trace) + ".seal.json"),
            )
            r_stop_allow = run_hook(STOPGATE, {
                "hook_event_name": "Stop", "session_id": session_id, "cwd": str(work_dir),
                "stop_hook_active": False,
            })
            self.assertEqual(r_stop_allow.stdout.strip(), "", f"Expected allow, got: {r_stop_allow.stdout}")

    def test_red_team_audit_fixes_bypasses_and_hardening(self):
        """Regression tests for every P0/P1 vulnerability found by live Bailian run + red-team audit."""
        from weaveprof.contract import normalize_probe_spec
        from weaveprof.backend import assess

        # 1. Live Bailian bugs: `"slot": null` and Python floor-div `"k0 // BLOCK_K"` normalize cleanly
        src = _sample_cuda_pipeline()
        spec_null_slot = {
            "roles": {"math": "warps 0-3"},
            "probes": [{"id": "mma", "begin": {"role": "math", "line": 61}, "end": {"role": "math", "line": 63}, "slot": None}],
        }
        norm = normalize_probe_spec(spec_null_slot, src)
        self.assertNotIn("slot", norm["probes"][0])
        tri_norm = normalize_probe_spec(
            {
                "language": "triton",
                "roles": {"w": "all"},
                "probes": [{"id": "load", "begin": {"role": "w", "line": 1}, "end": {"role": "w", "line": 1}, "slot": "k0 // BLOCK_K"}],
            },
            "a = tl.load(p)\n",
            language="triton",
        )
        self.assertEqual(tri_norm["probes"][0]["aux_expr"], "k0 // BLOCK_K")

        # 2. PolicyGuard blocks FI_SAMPLE injected purely via environment variable
        with tempfile.TemporaryDirectory() as td:
            proj = Path(td) / "proj"
            proj.mkdir()
            sf = proj / "k.cu"
            sf.write_text(src, encoding="utf-8")
            sha = ShadowWorkspace(sf, location=Path(td) / "ws").sha
            guard = PolicyGuard(HarnessPolicy(max_overhead_limit=0.10))
            with self.assertRaises(PolicyViolation) as ctx:
                guard.validate_session_start(sf, sha, 0.10, ["python", "bench.py"], env={"FI_SAMPLE": "0x4"})
            self.assertEqual(ctx.exception.code, "LOOP_SAMPLING_FORBIDDEN")

            # 3. Assembly gate rejects identical baseline_sass == instrumented_sass
            sass = proj / "same.sass"
            sass.write_text("Function : pipeline\n/*0000*/ MOV ;\n/*0010*/ EXIT ;\n", encoding="utf-8")
            checks, _, errs, *_ = assess(proj, spec_null_slot, 0.10, {
                "kernel_name": "pipeline", "baseline_sass": "same.sass", "instrumented_sass": "same.sass",
                "baseline_spill_bytes": 0, "instrumented_spill_bytes": 0, "extra_smem_bytes": 0,
            })
            self.assertFalse(checks["assembly"])
            self.assertIn("baseline_sass == instrumented_sass", errs.get("assembly", ""))

            # 4. Forged unkeyed self-hashed seal + missing overhead_ratio + malformed seal all fail-closed
            tr = proj / "t.perfetto-trace"
            tr.write_bytes(b"TRACE")
            fb = proj / "feedback.json"
            fb.write_text('{"status":"PASS"}\n', encoding="utf-8")
            seal_p, seal_doc = EpisodeSealer.seal_round(
                round_dir=proj, round_number=1, source_path=sf, source_sha256=sha,
                probe_path=None, instrumented_path=None, names_path=None,
                feedback_path=fb, verify_path=None, trace_path=tr, failed_trace_path=None,
                result={"status": "PASS", "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
                        "failed": [], "metrics": {"overhead_ratio": 0.02}},
                policy=HarnessPolicy(), limit=0.10,
                region_lock={"gpu_id": "gpu_0", "wait_ms": 0.0, "hold_ms": 10.0},
            )
            self.assertTrue(verify_seal(seal_p, require_pass=True)["ok"])

            # Forgery: strip HMAC and re-hash -> MUST FAIL
            forged = dict(seal_doc)
            forged.pop("seal_hmac", None)
            forged_p = proj / "forged.seal.json"
            forged_p.write_text(json.dumps(forged), encoding="utf-8")
            chk_forged = verify_seal(forged_p, require_pass=True)
            self.assertFalse(chk_forged["ok"])
            self.assertTrue(any("HMAC" in e for e in chk_forged["errors"]), chk_forged["errors"])

            # Missing overhead_ratio on PASS -> MUST FAIL even with valid HMAC
            no_ovh_p, _ = EpisodeSealer.seal_round(
                round_dir=proj, round_number=1, source_path=sf, source_sha256=sha,
                probe_path=None, instrumented_path=None, names_path=None,
                feedback_path=fb, verify_path=None, trace_path=tr, failed_trace_path=None,
                result={"status": "PASS", "checks": {"assembly": True, "output": True, "events": True, "overhead": True},
                        "failed": [], "metrics": {}},
                policy=HarnessPolicy(), limit=0.10,
                region_lock={"gpu_id": "gpu_0", "wait_ms": 0.0, "hold_ms": 10.0},
                out_seal_path=proj / "no_ovh.seal.json",
            )
            chk_no_ovh = verify_seal(no_ovh_p, require_pass=True)
            self.assertFalse(chk_no_ovh["ok"])
            self.assertTrue(any("overhead_ratio" in e for e in chk_no_ovh["errors"]), chk_no_ovh["errors"])

            # Malformed seal JSON -> fails closed without raising
            bad_p = proj / "bad.seal.json"
            bad_p.write_text('{"schema": "weaveprof.episode-seal/v1", "artifacts": {"x": {"path": 123}}}', encoding="utf-8")
            chk_bad = verify_seal(bad_p, require_pass=True)
            self.assertFalse(chk_bad["ok"])

        # 5. FSM rejects illegal transitions (PROMOTED -> POLICY_CHECK)
        fsm = RoundStateMachine(max_rounds=3)
        fsm.transition("POLICY_CHECK")
        fsm.transition("CRITICAL_REGION_MEASURE")
        fsm.transition("PROMOTED")
        with self.assertRaises(PolicyViolation) as ctx_fsm:
            fsm.transition("POLICY_CHECK")
        self.assertEqual(ctx_fsm.exception.code, "INVALID_FSM_TRANSITION")

        # 6. PreToolUse Hook blocks Write/Edit bypasses, comment smuggling, newlines, $(...), scientific limits
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td) / "state"
            state_dir.mkdir()
            env = dict(os.environ, FI_STATE_DIR=str(state_dir))

            def run_pre(payload):
                return subprocess.run(
                    [sys.executable, "-B", str(PRETOOL)],
                    input=json.dumps(payload), capture_output=True, text=True, env=env, timeout=30,
                )

            bypasses = [
                # Audit #1: Write/Edit tool targeting evidence/seal files
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Write", "tool_input": {"file_path": f"{td}/out.perfetto-trace.seal.json", "content": "{}"}},
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "edit_file", "tool_input": {"Path": f"{td}/round-1/seal.json"}},
                # Audit #2: comment smuggling `# run_measurement.py` & `find -exec`
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "cp fake.json out.seal.json # run_measurement.py"}},
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "find . -name x -exec cp fake ALL_GATES.json {} ;"}},
                # Audit #3: newline, single `&`, `$(...)`
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "cat x\ncp fake ALL_GATES.json"}},
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "cat x & cp fake round-1/seal.json"}},
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "cat $(cp fake round-1/seal.json)"}},
                # Audit #14: scientific / decimal `--limit` & python env sampling
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "python -m weaveprof profile k.cu --limit .5 -- python b.py"}},
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "python -m weaveprof profile k.cu --limit 5e-1 -- python b.py"}},
                {"hook_event_name": "PreToolUse", "session_id": "rt1", "cwd": td,
                 "tool_name": "Bash", "tool_input": {"command": "python -c \"os.environ['FI_SAMPLE']='4'\""}},
            ]
            for idx, pay in enumerate(bypasses):
                r = run_pre(pay)
                self.assertIn('"permissionDecision": "deny"', r.stdout, f"Bypass #{idx} slipped through: {pay} -> {r.stdout}")

    def test_incumbent_tracker_never_regresses_on_worse_round(self):
        tracker = IncumbentTracker()
        r1 = SealedRoundRecord(1, "fp1", "FAIL", {"assembly": True, "output": True, "events": True, "overhead": False}, ["overhead"], 0.14, "s1", "h1", "d1")
        r2 = SealedRoundRecord(2, "fp2", "FAIL", {"assembly": False, "output": True, "events": False, "overhead": False}, ["assembly", "events", "overhead"], 0.42, "s2", "h2", "d2")
        r3 = SealedRoundRecord(3, "fp3", "PASS", {"assembly": True, "output": True, "events": True, "overhead": True}, [], 0.018, "s3", "h3", "d3")
        self.assertTrue(tracker.record(r1))
        self.assertFalse(tracker.record(r2))  # Regression must NOT replace incumbent!
        self.assertEqual(tracker.incumbent.round_number, 1)
        self.assertIsNone(tracker.promoted)
        self.assertTrue(tracker.record(r3))
        self.assertEqual(tracker.promoted.round_number, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

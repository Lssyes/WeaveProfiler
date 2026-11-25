#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `weaveprof/backend.py` bench execution, process-tree cleanup & live `verify_round` (B6-G2)."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import select
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.api import verify as api_verify  # noqa: E402
from weaveprof.backend import (  # noqa: E402
    VerificationError,
    _terminate_process_tree,
    run_benchmark,
    run_cmd,
    verify_round,
)
from weaveprof.contract import normalize_probe_spec  # noqa: E402
from weaveprof.harness import HarnessPolicy, verify_seal  # noqa: E402
from weaveprof.workspace import ShadowWorkspace  # noqa: E402


FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

EXPECTED_REGION_LOCK_KEYS = {
    "acquired_at",
    "cpu_cores_pinned",
    "gpu_id",
    "hold_ms",
    "host",
    "label",
    "lock_path",
    "pid",
    "provider",
    "released_at",
    "wait_ms",
}

EXPECTED_VERIFY_METRICS_BASE_KEYS = {
    "abba",
    "alignment",
    "assembly_change_rate",
    "baseline_median_ns",
    "baseline_spill_bytes",
    "compiler_spill_gate",
    "core_multiset_ratio",
    "core_retention_ratio",
    "delta_spill_bytes",
    "extra_smem_bytes",
    "granularity_gate",
    "instrumented_median_ns",
    "instrumented_spill_bytes",
    "lcs_ratio",
    "order_change_ratio",
    "output_bytes",
    "overhead_ratio",
    "overflow",
    "paired_samples",
    "pairing_errors",
    "preview_png",
    "warp_gate",
}


def _sample_cuda_source() -> str:
    lines = [""] * 24
    for ln, text in {
        1: "__global__ void pipeline(float* out) {",
        2: "    int s = 0;",
        6: "    if (threadIdx.x / warpSize == 4) {",
        7: "        tma_load_async(s);",
        8: "    }",
        12: "    if (threadIdx.x / warpSize < 4) {",
        13: "        full_bar[s].wait();",
        15: "        for (int k = 0; k < 16; ++k) {",
        16: "            out[k] = out[k] * 2.0f;",
        17: "        }",
        19: "    }",
        22: "}",
    }.items():
        lines[ln - 1] = text
    return "\n".join(lines) + "\n"


def _raw_probe_spec() -> dict:
    return {
        "card": "h20",
        "roles": {"dma": "warp 4", "math": "warps 0-3"},
        "probes": [
            {
                "id": "load",
                "begin": {"role": "dma", "line": 7},
                "end": {"role": "math", "line": 13},
                "slot": "s",
            },
            {
                "id": "mma",
                "begin": {"role": "math", "line": 15},
                "end": {"role": "math", "line": 17},
            },
        ],
    }


# Short (<255 byte) `-c` snippet executed via `[sys.executable, "-c", SHORT_LIVE_BENCH_CODE]`
# inside `run_benchmark`. Keep under 255 bytes because `ShadowWorkspace.command` (`workspace.py:160`)
# calls `Path(token).exists()`, which raises `OSError(ENAMETOOLONG)` on >255-byte single-segment strings.
SHORT_LIVE_BENCH_CODE = (
    "import json as j,os;"
    "d=os.environ['WEAVEPROF_ROUND_DIR'];"
    "r=j.load(open(d+'/rt_seed.json'));"
    "r.update(pid=os.getpid(),cwd=os.getcwd(),hm='WEAVEPROF_SUPERVISOR_HMAC_KEY'in os.environ);"
    "j.dump(r,open(d+'/runtime.json','w'));print('ok')"
)


def _seed_prepared_round_artifacts(directory: Path) -> None:
    """Pre-stage SASS/output/FILITE02/SQLite + `rt_seed.json` (NOT `runtime.json`);
    live `run_benchmark` child creates `runtime.json` during measurement."""
    ops = ["MOV", "LDG", "HMMA", "IADD", "IMAD", "MUL", "STG", "FADD", "EXIT"]
    sass = "Function : pipeline\n" + "".join(f"/*{16 * i:04x}*/ {op} ;\n" for i, op in enumerate(ops))
    (directory / "l0.sass").write_text(sass, encoding="utf-8")
    (directory / "inst.sass").write_text(sass, encoding="utf-8")
    (directory / "l0.bin").write_bytes(b"\x01\x02\x03\x04" * 16)
    (directory / "inst.bin").write_bytes(b"\x01\x02\x03\x04" * 16)

    names = json.loads((directory / "names.json").read_text(encoding="utf-8"))
    probe = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
    eids = {p["id"]: int(p.get("event_id") or idx) for idx, p in enumerate(probe["probes"], start=1)}
    roles_rev = {v: int(k) for k, v in names["roles"].items()}
    dma_rid = roles_rev["dma"]
    math_rid = roles_rev["math"]
    load_eid = eids["load"]
    mma_eid = eids["mma"]

    tag_open, tag_end, tag_close = 0x40000000, 0x80000000, 0x08000000
    t0 = 1_000_000
    lanes = []
    for w in range(4):
        lanes.append([
            (t0, tag_open | math_rid, 0),
            (t0 + 500 + 10 * w, tag_end | load_eid, 0),
            (t0 + 1_000, mma_eid, 0),
            (t0 + 36_000, tag_end | mma_eid, 0),
            (t0 + 36_500, tag_close, 0),
        ])
    lanes.append([
        (t0, tag_open | dma_rid, 0),
        (t0 + 200, load_eid, 0),
        (t0 + 36_500, tag_close, 0),
    ])

    meta = json.dumps({"gran": 0, "arch": "sm_90a", "backend": "nvidia_ptx"}).encode("utf-8")
    counts = [len(r) for r in lanes]
    blob = bytearray(b"FILITE02")
    blob.extend(struct.pack("<6I", len(lanes), max(counts), 5, 0, 0, len(meta)))
    blob.extend(meta)
    blob.extend(struct.pack(f"<{len(lanes)}I", *counts))
    for r in lanes:
        for rec in r:
            blob.extend(struct.pack("<QII", *rec))
    (directory / "capture.fi02.bin").write_bytes(blob)

    db = sqlite3.connect(str(directory / "capture.sqlite"))
    try:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER)")
        db.execute("INSERT INTO StringIds VALUES (1, 'pipeline')")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1)", (t0 - 1000, t0 + 60000))
        db.commit()
    finally:
        db.close()

    rt_seed = {
        "kernel_name": "pipeline",
        "baseline_sass": "l0.sass",
        "instrumented_sass": "inst.sass",
        "baseline_spill_bytes": 0,
        "instrumented_spill_bytes": 0,
        "extra_smem_bytes": 0,
        "baseline_output": "l0.bin",
        "instrumented_output": "inst.bin",
        "warmup": 4,
        "timing_protocol": "ABBA",
        "baseline_ns": [42000, 42100, 41950, 42050, 42000, 42020],
        "instrumented_ns": [42750, 42820, 42700, 42790, 42760, 42770],
    }
    (directory / "rt_seed.json").write_text(json.dumps(rt_seed, indent=2) + "\n", encoding="utf-8")

TREE_CHILD_SCRIPT = """\
import os, pathlib, signal, subprocess, sys

w = int(sys.argv[1])
ignore_sigterm = sys.argv[2] == "1"
gc_pid_path = pathlib.Path(sys.argv[3])

if ignore_sigterm:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

gc_code = (
    "import os, pathlib, signal, sys; "
    "w = int(sys.argv[1]); "
    "ig = sys.argv[2] == '1'; "
    "p = pathlib.Path(sys.argv[3]); "
    "(signal.signal(signal.SIGTERM, signal.SIG_IGN) if ig else None); "
    "p.write_text(str(os.getpid()), encoding='utf-8'); "
    "os.write(w, b'G'); "
    "r2, w2 = os.pipe(); "
    "os.read(r2, 1)"
)
subprocess.Popen(
    [sys.executable, "-c", gc_code, str(w), sys.argv[2], str(gc_pid_path)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    pass_fds=(w,),
)
os.write(w, b"C")
r1, w1 = os.pipe()
os.read(r1, 1)
"""


class BackendBenchCharacterizationTests(unittest.TestCase):
    _ENV_SAVE_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "TMPDIR",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "FI_SAMPLE",
        "WP_SAMPLE",
        "WEAVEPROF_SAMPLE",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}

        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_b6_g2_iso_")).resolve()
        for sub in ("home", "xdg_cache", "xdg_config", "mpl", "sessions", "cache", "locks", "workdir", "tmp", "cwd"):
            (cls._iso_root / sub).mkdir(parents=True, exist_ok=True)

        cls._lock_dir = cls._iso_root / "locks"
        os.environ["HOME"] = str(cls._iso_root / "home")
        os.environ["XDG_CACHE_HOME"] = str(cls._iso_root / "xdg_cache")
        os.environ["XDG_CONFIG_HOME"] = str(cls._iso_root / "xdg_config")
        os.environ["MPLCONFIGDIR"] = str(cls._iso_root / "mpl")
        os.environ["WEAVEPROF_SESSIONS_DIR"] = str(cls._iso_root / "sessions")
        os.environ["WEAVEPROF_CACHE"] = str(cls._iso_root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._lock_dir)
        os.environ["WEAVEPROF_WORKDIR"] = str(cls._iso_root / "workdir")
        os.environ["TMPDIR"] = str(cls._iso_root / "tmp")
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        for unset_key in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "FI_SAMPLE",
            "WP_SAMPLE",
            "WEAVEPROF_SAMPLE",
        ):
            os.environ.pop(unset_key, None)

        tempfile.tempdir = str(cls._iso_root / "tmp")
        os.chdir(cls._iso_root / "cwd")
        cls.spawned_records: list[dict] = []

    @classmethod
    def tearDownClass(cls) -> None:
        pid_log = os.environ.get("WP_B6_PID_LOG")
        if pid_log:
            Path(pid_log).write_text(
                json.dumps(cls.spawned_records, indent=2) + "\n",
                encoding="utf-8",
            )
        os.chdir(cls._saved_cwd)
        tempfile.tempdir = cls._saved_tempdir
        for k, val in cls._saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        shutil.rmtree(cls._iso_root, ignore_errors=True)

    def _new_case_dir(self, tag: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(self._iso_root / "tmp"))).resolve()

    def _child_env(self) -> dict[str, str]:
        return dict(os.environ)

    def _read_exact_bytes(self, fd: int, count: int) -> bytes:
        buf = bytearray()
        while len(buf) < count:
            ready, _, _ = select.select([fd], [], [], 20.0)
            self.assertEqual(len(ready), 1, "Safety select timed out waiting for child ready byte")
            chunk = os.read(fd, count - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def _read_eof(self, fd: int) -> bytes:
        ready, _, _ = select.select([fd], [], [], 20.0)
        self.assertEqual(len(ready), 1, "Safety select timed out waiting for pipe EOF after process tree kill")
        return os.read(fd, 16)

    def test_verification_error_attributes_and_string(self) -> None:
        case_dir = self._new_case_dir("ver_err")
        round_dir = case_dir / "round-1"
        feedback_payload = {"status": "FAIL", "failed": ["overhead"]}
        err = VerificationError(feedback_payload, str(round_dir))

        self.assertIsInstance(err, RuntimeError)
        self.assertEqual(err.feedback, {"status": "FAIL", "failed": ["overhead"]})
        self.assertEqual(err.directory, round_dir)
        self.assertIsInstance(err.directory, Path)
        self.assertEqual(
            str(err).replace(str(case_dir), "<TMP>"),
            "Verification failed in <TMP>/round-1: {'status': 'FAIL', 'failed': ['overhead']}",
        )

    def test_run_cmd_success_and_nonzero_exit(self) -> None:
        case_dir = self._new_case_dir("run_cmd_basic")
        ok_log = case_dir / "ok_bench.log"
        res = run_cmd(
            [
                sys.executable,
                "-c",
                (
                    "import os, pathlib, sys; "
                    f"pathlib.Path({str(case_dir / 'ok.pid')!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                    "sys.stdout.write('stdout_ok\\n'); sys.stdout.flush(); "
                    "sys.stderr.write('stderr_note\\n'); sys.stderr.flush()"
                ),
            ],
            case_dir,
            self._child_env(),
            ok_log,
            timeout_s=20,
        )
        self.assertIsNone(res)
        self.assertEqual(ok_log.read_text(encoding="utf-8"), "stdout_ok\nstderr_note\n")
        self.spawned_records.append(
            {
                "case": "run_cmd_success",
                "pid": int((case_dir / "ok.pid").read_text(encoding="utf-8")),
                "exit_state": "rc=0",
            }
        )

        err_log = case_dir / "err_bench.log"
        with self.assertRaises(RuntimeError) as err_ctx:
            run_cmd(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os, pathlib, sys; "
                        f"pathlib.Path({str(case_dir / 'err.pid')!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                        "sys.stderr.write('fatal_bench_err\\n'); sys.stderr.flush(); "
                        "sys.exit(7)"
                    ),
                ],
                case_dir,
                self._child_env(),
                err_log,
                timeout_s=20,
            )
        self.assertEqual(
            str(err_ctx.exception).replace(str(case_dir), "<TMP>"),
            f"{Path(sys.executable).name} exited 7 (<TMP>/err_bench.log): fatal_bench_err\n",
        )
        self.spawned_records.append(
            {
                "case": "run_cmd_nonzero_exit",
                "pid": int((case_dir / "err.pid").read_text(encoding="utf-8")),
                "exit_state": "rc=7",
            }
        )

    def test_run_cmd_timeout_raises_runtime_error_with_empty_log(self) -> None:
        case_dir = self._new_case_dir("run_cmd_timeout")
        log_path = case_dir / "bench.log"
        argv = [sys.executable, "-c", "import os; r, w = os.pipe(); os.read(r, 1)"]

        with self.assertRaises(RuntimeError) as timeout_ctx:
            run_cmd(argv, case_dir, self._child_env(), log_path, timeout_s=1)

        normalized = str(timeout_ctx.exception).replace(str(case_dir), "<TMP>")
        self.assertEqual(
            normalized,
            f"{Path(sys.executable).name} timed out after 1s (<TMP>/bench.log): ",
        )
        self.assertTrue(log_path.is_file())
        self.assertEqual(log_path.stat().st_size, 0)
        self.assertEqual(log_path.read_bytes(), b"")

    def test_terminate_process_tree_noop_on_exited_and_sigterm_and_sigkill_trees(self) -> None:
        case_dir = self._new_case_dir("term_tree")

        # 0) Already-exited process returns immediately and preserves returncode == 0
        done_proc = subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(0)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(case_dir),
            env=self._child_env(),
        )
        self.assertEqual(done_proc.wait(timeout=20), 0)
        self.assertIsNone(_terminate_process_tree(done_proc))
        self.assertEqual(done_proc.returncode, 0)
        self.spawned_records.append(
            {"case": "terminate_already_exited", "pid": done_proc.pid, "exit_state": "rc=0"}
        )

        # 1) Standard child + grandchild group killed via `os.killpg(..., SIGTERM)`
        r1, w1 = os.pipe()
        gc1_pid_path = case_dir / "gc_sigterm.pid"
        proc_term = subprocess.Popen(
            [sys.executable, "-c", TREE_CHILD_SCRIPT, str(w1), "0", str(gc1_pid_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(w1,),
            cwd=str(case_dir),
            env=self._child_env(),
        )
        os.close(w1)
        try:
            ready1 = self._read_exact_bytes(r1, 2)
            self.assertEqual(len(ready1), 2)
            self.assertEqual(set(ready1), {ord("C"), ord("G")})

            _terminate_process_tree(proc_term)
            self.assertEqual(proc_term.returncode, -signal.SIGTERM)
            self.assertEqual(self._read_eof(r1), b"")
            self.spawned_records.extend(
                [
                    {
                        "case": "terminate_sigterm_child",
                        "pid": proc_term.pid,
                        "exit_state": f"returncode={proc_term.returncode} (-SIGTERM)",
                    },
                    {
                        "case": "terminate_sigterm_grandchild",
                        "pid": int(gc1_pid_path.read_text(encoding="utf-8")),
                        "exit_state": "pipe_write_fd_closed_eof (-SIGTERM)",
                    },
                ]
            )
        finally:
            if proc_term.poll() is None:
                try:
                    os.killpg(proc_term.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc_term.wait(timeout=5)
            os.close(r1)

        # 2) `SIG_IGN` child + grandchild escalates after 3s wait to `os.killpg(..., SIGKILL)`
        r2, w2 = os.pipe()
        gc2_pid_path = case_dir / "gc_sigkill.pid"
        proc_kill = subprocess.Popen(
            [sys.executable, "-c", TREE_CHILD_SCRIPT, str(w2), "1", str(gc2_pid_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(w2,),
            cwd=str(case_dir),
            env=self._child_env(),
        )
        os.close(w2)
        try:
            ready2 = self._read_exact_bytes(r2, 2)
            self.assertEqual(len(ready2), 2)
            self.assertEqual(set(ready2), {ord("C"), ord("G")})

            _terminate_process_tree(proc_kill)
            self.assertEqual(proc_kill.returncode, -signal.SIGKILL)
            self.assertEqual(self._read_eof(r2), b"")
            self.spawned_records.extend(
                [
                    {
                        "case": "terminate_sigkill_child",
                        "pid": proc_kill.pid,
                        "exit_state": f"returncode={proc_kill.returncode} (-SIGKILL)",
                    },
                    {
                        "case": "terminate_sigkill_grandchild",
                        "pid": int(gc2_pid_path.read_text(encoding="utf-8")),
                        "exit_state": "pipe_write_fd_closed_eof (-SIGKILL)",
                    },
                ]
            )
        finally:
            if proc_kill.poll() is None:
                try:
                    os.killpg(proc_kill.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc_kill.wait(timeout=5)
            os.close(r2)

    def test_run_benchmark_executes_in_shadow_workspace_and_checks_exit_code(self) -> None:
        case_dir = self._new_case_dir("run_bench")
        proj = case_dir / "proj"
        proj.mkdir()
        src = proj / "pipeline.cu"
        source_text = _sample_cuda_source()
        src.write_text(source_text, encoding="utf-8")
        spec = normalize_probe_spec(_raw_probe_spec(), source_text)

        ws = ShadowWorkspace(src, location=case_dir / "ws")
        prepared1 = ws.prepare(1, spec)
        round1_dir, _base1, _inst1 = prepared1
        _seed_prepared_round_artifacts(round1_dir)
        self.assertFalse((round1_dir / "runtime.json").exists())

        bench_ok = [sys.executable, "-c", SHORT_LIVE_BENCH_CODE]
        self.assertLess(len(SHORT_LIVE_BENCH_CODE.encode("utf-8")), 255)
        self.assertIsNone(run_benchmark(ws, prepared1, bench_ok, spec, timeout_s=20))

        rt = json.loads((round1_dir / "runtime.json").read_text(encoding="utf-8"))
        self.assertIs(rt["hm"], False)
        self.assertEqual(Path(rt["cwd"]).resolve(), (round1_dir / "L_inst").resolve())
        self.assertEqual((round1_dir / "bench.log").read_text(encoding="utf-8"), "ok\n")
        self.spawned_records.append(
            {"case": "run_benchmark_pass", "pid": int(rt["pid"]), "exit_state": "rc=0"}
        )

        # Pin BASE `workspace.py:160` behavior: >255-byte single-segment `-c` token triggers ENAMETOOLONG
        overlong_code = "raise SystemExit(0) #" + ("x" * 260)
        with self.assertRaises(OSError) as long_ctx:
            run_benchmark(ws, prepared1, [sys.executable, "-c", overlong_code], spec, timeout_s=20)
        self.assertEqual(long_ctx.exception.errno, errno.ENAMETOOLONG)

        # Non-zero exit code from bench must raise `RuntimeError`
        prepared2 = ws.prepare(2, spec)
        round2_dir, _, _ = prepared2
        bench_fail = [
            sys.executable,
            "-c",
            (
                "import os, pathlib, sys; "
                f"pathlib.Path({str(case_dir / 'bench_fail.pid')!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                "sys.stderr.write('bench_nonzero_rc\\n'); sys.stderr.flush(); "
                "sys.exit(5)"
            ),
        ]
        with self.assertRaises(RuntimeError) as fail_ctx:
            run_benchmark(ws, prepared2, bench_fail, spec, timeout_s=20)
        self.assertEqual(
            str(fail_ctx.exception).replace(str(round2_dir), "<ROUND2>"),
            f"{Path(sys.executable).name} exited 5 (<ROUND2>/bench.log): bench_nonzero_rc\n",
        )
        self.spawned_records.append(
            {
                "case": "run_benchmark_fail",
                "pid": int((case_dir / "bench_fail.pid").read_text(encoding="utf-8")),
                "exit_state": "rc=5",
            }
        )

    def test_verify_round_live_bench_region_lock_preview_png_and_png_only_out(self) -> None:
        case_dir = self._new_case_dir("verify_round_live")
        proj = case_dir / "proj"
        proj.mkdir()
        src = proj / "pipeline.cu"
        source_text = _sample_cuda_source()
        src.write_text(source_text, encoding="utf-8")
        spec = normalize_probe_spec(_raw_probe_spec(), source_text)

        # Clean any prior files in `_lock_dir` before testing lock file names
        for existing in self._lock_dir.iterdir():
            existing.unlink()

        ws = ShadowWorkspace(src, location=case_dir / "ws")
        prepared1 = ws.prepare(1, spec)
        round1_dir, _, inst1 = prepared1
        _seed_prepared_round_artifacts(round1_dir)
        out_dir = case_dir / "out"
        out_dir.mkdir()
        out_trace = out_dir / "pipeline.perfetto-trace"
        # Seed a stale file at `out_trace` to pin Audit #8 unlinking before measurement
        out_trace.write_bytes(b"STALE_PREVIOUS_TRACE")
        custom_preview = out_dir / "custom_preview.png"
        feedback_path = out_dir / "feedback.json"
        verify_json_path = out_dir / "verify.json"

        bench_argv = [sys.executable, "-c", SHORT_LIVE_BENCH_CODE]
        res = verify_round(
            round1_dir,
            spec,
            source_text,
            0.10,
            out_trace,
            feedback_path,
            lambda: run_benchmark(ws, prepared1, bench_argv, spec, timeout_s=20),
            verify_json=verify_json_path,
            preview_png=custom_preview,
            policy=HarnessPolicy(enable_region_lock=True),
            gpu_id="auto",
            round_number=1,
            source_path=src,
            source_sha256=ws.sha,
            probe_path=round1_dir / "probe.json",
            instrumented_path=inst1,
            bench_cmd=bench_argv,
        )

        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["failed"], [])
        self.assertEqual(
            res["checks"],
            {"assembly": True, "output": True, "event": True, "events": True, "overhead": True},
        )

        region_lock = res["metrics"]["region_lock"]
        self.assertEqual(set(region_lock.keys()), EXPECTED_REGION_LOCK_KEYS)
        self.assertEqual(region_lock["gpu_id"], "gpu_0")
        self.assertEqual(region_lock["label"], "verify_round_1")
        self.assertEqual(region_lock["provider"], "local_file")
        self.assertIsNone(region_lock["cpu_cores_pinned"])
        self.assertEqual(Path(region_lock["lock_path"]).resolve(), (self._lock_dir / "gpu_0.lock").resolve())
        # After releasing the lock, `gpu_0.owner.json` is unlinked (`harness.py:588-589`), leaving only `gpu_0.lock`
        self.assertEqual(sorted(os.listdir(self._lock_dir)), ["gpu_0.lock"])

        self.assertEqual(
            set(res["metrics"].keys()),
            EXPECTED_VERIFY_METRICS_BASE_KEYS | {"region_lock", "trace_processor"},
        )
        self.assertEqual(res["metrics"]["preview_png"], str(custom_preview))
        self.assertTrue(custom_preview.is_file())
        self.assertEqual(custom_preview.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue(out_trace.is_file())
        self.assertNotEqual(out_trace.read_bytes(), b"STALE_PREVIOUS_TRACE")

        seal_check = verify_seal(res["seal_path"], require_pass=True, expected_source_sha=ws.sha)
        self.assertIs(seal_check["ok"], True)
        self.assertEqual(seal_check["errors"], [])

        rt1 = json.loads((round1_dir / "runtime.json").read_text(encoding="utf-8"))
        self.spawned_records.append(
            {"case": "verify_round_live_locked", "pid": int(rt1["pid"]), "exit_state": "rc=0"}
        )

        # Variant with `enable_region_lock=False` and `out=".../overview.png"` (`png_only_out=True`)
        prepared2 = ws.prepare(2, spec)
        round2_dir, _, inst2 = prepared2
        _seed_prepared_round_artifacts(round2_dir)
        png_out = out_dir / "overview.png"
        res_nolock = verify_round(
            round2_dir,
            spec,
            source_text,
            0.10,
            png_out,
            out_dir / "feedback_r2.json",
            lambda: run_benchmark(ws, prepared2, bench_argv, spec, timeout_s=20),
            verify_json=out_dir / "verify_r2.json",
            policy=HarnessPolicy(enable_region_lock=False),
            gpu_id="auto",
            round_number=2,
            source_path=src,
            source_sha256=ws.sha,
            probe_path=round2_dir / "probe.json",
            instrumented_path=inst2,
            bench_cmd=bench_argv,
        )
        self.assertEqual(res_nolock["status"], "PASS")
        self.assertEqual(res_nolock["failed"], [])
        self.assertEqual(set(res_nolock["metrics"].keys()), EXPECTED_VERIFY_METRICS_BASE_KEYS)
        self.assertNotIn("region_lock", res_nolock["metrics"])
        self.assertNotIn("trace_processor", res_nolock["metrics"])
        self.assertEqual(res_nolock["metrics"]["preview_png"], str(png_out))
        self.assertTrue(png_out.is_file())
        self.assertEqual(png_out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

        rt2 = json.loads((round2_dir / "runtime.json").read_text(encoding="utf-8"))
        self.spawned_records.append(
            {"case": "verify_round_live_unlocked_png", "pid": int(rt2["pid"]), "exit_state": "rc=0"}
        )

    def test_verify_round_live_bench_failure_and_api_verify_raises_verification_error(self) -> None:
        case_dir = self._new_case_dir("verify_round_fail")
        proj = case_dir / "proj"
        proj.mkdir()
        src = proj / "pipeline.cu"
        source_text = _sample_cuda_source()
        src.write_text(source_text, encoding="utf-8")
        spec = normalize_probe_spec(_raw_probe_spec(), source_text)

        ws = ShadowWorkspace(src, location=case_dir / "ws")
        prepared = ws.prepare(1, spec)
        round_dir, _, inst_src = prepared
        out_dir = case_dir / "out"
        out_dir.mkdir()
        out_trace = out_dir / "fail_round.perfetto-trace"

        fail_bench = [
            sys.executable,
            "-c",
            (
                "import os, pathlib, sys; "
                f"pathlib.Path({str(case_dir / 'vr_fail.pid')!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                "sys.stderr.write('simulated_bench_crash\\n'); sys.stderr.flush(); "
                "sys.exit(9)"
            ),
        ]
        res_fail = verify_round(
            round_dir,
            spec,
            source_text,
            0.10,
            out_trace,
            out_dir / "feedback_fail.json",
            lambda: run_benchmark(ws, prepared, fail_bench, spec, timeout_s=20),
            verify_json=out_dir / "verify_fail.json",
            policy=HarnessPolicy(enable_region_lock=True),
            gpu_id="auto",
            round_number=1,
            source_path=src,
            source_sha256=ws.sha,
            probe_path=round_dir / "probe.json",
            instrumented_path=inst_src,
            bench_cmd=fail_bench,
        )
        self.assertEqual(res_fail["status"], "FAIL")
        self.assertEqual(
            res_fail["checks"],
            {"assembly": False, "output": False, "event": False, "events": False, "overhead": False},
        )
        self.assertEqual(sorted(res_fail["failed"]), ["assembly", "output", "overhead", "pairing"])
        self.assertEqual(
            res_fail["metrics"]["errors"]["measurement"].replace(str(round_dir), "<ROUND>"),
            f"{Path(sys.executable).name} exited 9 (<ROUND>/bench.log): simulated_bench_crash\n",
        )
        self.assertFalse(out_trace.exists())
        self.spawned_records.append(
            {
                "case": "verify_round_measurement_exception",
                "pid": int((case_dir / "vr_fail.pid").read_text(encoding="utf-8")),
                "exit_state": "rc=9",
            }
        )

        # Public `api.verify` wraps `verify_round` + `run_benchmark` and raises `VerificationError` on FAIL
        api_fail_bench = [
            sys.executable,
            "-c",
            (
                "import os, pathlib, sys; "
                f"pathlib.Path({str(case_dir / 'api_verify_fail.pid')!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                "sys.stderr.write('api_verify_rc6\\n'); sys.stderr.flush(); "
                "sys.exit(6)"
            ),
        ]
        with self.assertRaises(VerificationError) as ver_ctx:
            api_verify(
                _raw_probe_spec(),
                source=src,
                limit=0.10,
                feedback=out_dir / "api_feedback.json",
                verify_json=out_dir / "api_verify.json",
                out=out_dir / "api_fail.perfetto-trace",
                bench=api_fail_bench,
            )
        err = ver_ctx.exception
        self.assertEqual(err.feedback["status"], "FAIL")
        self.assertEqual(sorted(err.feedback["failed"]), ["assembly", "output", "overhead", "pairing"])
        self.assertEqual(err.directory.name, "round-1")
        self.assertEqual(
            err.feedback["metrics"]["errors"]["measurement"].replace(str(err.directory), "<API_ROUND>"),
            f"{Path(sys.executable).name} exited 6 (<API_ROUND>/bench.log): api_verify_rc6\n",
        )
        self.spawned_records.append(
            {
                "case": "api_verify_raises_verification_error",
                "pid": int((case_dir / "api_verify_fail.pid").read_text(encoding="utf-8")),
                "exit_state": "rc=6",
            }
        )


if __name__ == "__main__":
    unittest.main()

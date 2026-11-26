#!/usr/bin/env python3
"""Characterization tests pinning fixed M2 behavior when requested nsys capture fails.

Drives ``weaveprof.megakernel.run_megakernel_audit`` through Workflow A (``shadow_build_run``) on CPU
with runtime-generated fake ``build_cmd``, fake ``bench_cmd``, and fake ``nsys`` scripts, pinning:
  1. ``test_m2b_nsys_capture_failure_without_overview_png_fails_in_all_four_places``:
     ``capture_nsys=True, render_overview_png=False`` when ``nsys profile`` fails (``rc=2``, no ``.nsys-rep``):
     returned ``report['status']``, ``seal.json['status']``, ``verify.json['status']``, and
     ``feedback.json['status']`` are all ``'FAIL'`` (修复前的行为见 PR-3).
  2. ``test_m2a_nsys_capture_failure_default_overview_png_fails_in_all_four_places``:
     same failing ``nsys profile`` under default ``render_overview_png=True``: returned ``report['status']``,
     ``seal.json['status']``, ``verify.json['status']``, and ``feedback.json['status']`` are all ``'FAIL'``
     with 3 ``structural_blockers`` (修复前的行为见 PR-3).
  3. ``test_control_...``: identical inputs to M2b (``capture_nsys=True, render_overview_png=False``) with
     succeeding fake ``nsys``: returned ``report``, ``seal.json``, ``verify.json``, and ``feedback.json`` all
     consistently report ``status='PASS'`` with extracted hardware curves and no ``NSYS_CAPTURE_FAILED``.

Run:  python -B -m unittest tests/characterization/test_p1c_m2_nsys_capture_failure.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import lzma
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO_ROOT)]:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.pop("WEAVEPROF_GOLDEN_UPDATE", None)
os.environ.pop("WEAVEPROF_METRICS_UPDATE", None)

from weaveprof import megakernel as mk  # noqa: E402
from weaveprof.harness import verify_seal  # noqa: E402

_ENV_KEYS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "WEAVEPROF_SESSIONS_DIR",
    "WEAVEPROF_CACHE",
    "WEAVEPROF_LOCK_DIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "MPLCONFIGDIR",
    "WEAVEPROF_SUPERVISOR_HMAC_KEY",
    "WEAVEPROF_SUPERVISOR_KEY_FILE",
    "QWEN_PROJECT_DIR",
    "QWEN_SESSION_ID",
    "WEAVEPROF_LOCK_PROVIDER",
    "WEAVEPROF_KEEP_SHADOW",
    "WEAVEPROF_GOLDEN_UPDATE",
    "WEAVEPROF_METRICS_UPDATE",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "P1C_INVOCATION_LOG",
    "P1C_NSYS_FAIL_RC",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
)

_VERIFY_JSON_KEYS = {
    "schema",
    "status",
    "checks",
    "failed",
    "limit",
    "overhead_ratio",
    "source_sha256",
    "fidelity_gates",
    "region_lock",
}

_FAKE_BUILD_BODY = """from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import fcntl
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--l0", required=True)
ap.add_argument("--inst", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tree", required=True)
args = ap.parse_args()

l0_p = Path(args.l0).resolve()
inst_p = Path(args.inst).resolve()
tree_p = Path(args.tree).resolve()
role = "L0" if tree_p == l0_p else ("L_inst" if tree_p == inst_p else "other")

log_file = os.environ.get("P1C_INVOCATION_LOG")
if log_file:
    rec = {
        "kind": "build",
        "role": role,
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

(tree_p / "fake_compiled_kernel.cubin").write_bytes(f"FAKE_CUBIN_V1:{role}\\n".encode("ascii"))
"""

_FAKE_BENCH_BODY = """from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

STAGE_DIR = Path(__STAGE_DIR__)

ap = argparse.ArgumentParser()
ap.add_argument("--require-build", action="store_true")
ap.add_argument("--l0", required=True)
ap.add_argument("--inst", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tree", required=True)
args = ap.parse_args()

log_file = os.environ.get("P1C_INVOCATION_LOG")
if log_file:
    lock_dir = os.environ.get("WEAVEPROF_LOCK_DIR")
    owner_present = bool(lock_dir and (Path(lock_dir) / "gpu_0.owner.json").is_file())
    rec = {
        "kind": "bench",
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "lock_owner_present": owner_present,
        "nsys_capture_env": os.environ.get("WEAVEPROF_NSYS_CAPTURE"),
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

if args.require_build:
    l0_prod = Path(args.l0) / "fake_compiled_kernel.cubin"
    inst_prod = Path(args.inst) / "fake_compiled_kernel.cubin"
    if not l0_prod.is_file() or l0_prod.read_bytes() != b"FAKE_CUBIN_V1:L0\\n":
        sys.stderr.write("missing or invalid L0 build artifact\\n")
        raise SystemExit(126)
    if not inst_prod.is_file() or inst_prod.read_bytes() != b"FAKE_CUBIN_V1:L_inst\\n":
        sys.stderr.write("missing or invalid L_inst build artifact\\n")
        raise SystemExit(127)

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
for fname in ("capture.fi02.bin", "runtime.json", "names.json"):
    shutil.copy2(STAGE_DIR / fname, out_dir / fname)
"""

_FAKE_NSYS_BODY = """from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

STAGE_DIR = Path(__STAGE_DIR__)

def _log(rec):
    log_file = os.environ.get("P1C_INVOCATION_LOG")
    if not log_file:
        return
    with open(log_file, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

subcmd = sys.argv[1] if len(sys.argv) > 1 else ""
if subcmd == "profile":
    fail_rc_raw = os.environ.get("P1C_NSYS_FAIL_RC")
    if fail_rc_raw is not None:
        fail_rc = int(fail_rc_raw)
        _log({
            "kind": "nsys_profile",
            "mode": "fail_rc",
            "rc": fail_rc,
            "pid": os.getpid(),
            "argv": sys.argv,
            "cwd": os.getcwd(),
        })
        sys.stderr.write(f"fake nsys profile simulated failure (rc={fail_rc})\\n")
        raise SystemExit(fail_rc)

    _log({
        "kind": "nsys_profile",
        "mode": "ok",
        "rc": 0,
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    })
    idx = sys.argv.index("-o")
    rep_base = sys.argv[idx + 1]
    child_argv = sys.argv[idx + 2:]
    child = subprocess.run(child_argv, env=os.environ.copy(), check=False)
    if child.returncode != 0:
        raise SystemExit(child.returncode)
    Path(rep_base + ".nsys-rep").write_bytes(b"FAKE_NSYS_REP_V1\\n")
    raise SystemExit(0)

if subcmd == "export":
    _log({
        "kind": "nsys_export",
        "rc": 0,
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    })
    out_idx = sys.argv.index("--output")
    sqlite_dst = Path(sys.argv[out_idx + 1])
    rep_src = Path(sys.argv[out_idx + 2])
    if not rep_src.is_file():
        sys.stderr.write(f"missing rep file: {rep_src}\\n")
        raise SystemExit(2)
    shutil.copy2(STAGE_DIR / "nsys.sqlite", sqlite_dst)
    raise SystemExit(0)

sys.stderr.write(f"unsupported fake nsys subcommand: {subcmd!r}\\n")
raise SystemExit(2)
"""


def _unxz(src: Path, dst: Path) -> Path:
    with lzma.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    return dst


def _norm_paths(obj: Any, pairs: List[Tuple[str, str]]) -> Any:
    if isinstance(obj, str):
        for real, token in pairs:
            if real in obj:
                obj = obj.replace(real, token)
        return obj
    if isinstance(obj, dict):
        return {k: _norm_paths(v, pairs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm_paths(v, pairs) for v in obj]
    return obj


def _make_pairs(*items: Tuple[Path, str]) -> List[Tuple[str, str]]:
    raw = set()
    for p, tok in items:
        raw.add((str(p), tok))
        raw.add((os.path.realpath(str(p)), tok))
    return sorted(raw, key=lambda kv: -len(kv[0]))


def _make_network_guard():
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    inet_families = {
        fam
        for fam in (getattr(socket, "AF_INET", None), getattr(socket, "AF_INET6", None))
        if fam is not None
    }

    def _guarded_connect(sock: socket.socket, address: Any) -> Any:
        if sock.family in inet_families:
            raise RuntimeError(f"Outbound network connect blocked in P1c tests: {address!r}")
        return real_connect(sock, address)

    def _guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family in inet_families:
            raise RuntimeError(f"Outbound network connect_ex blocked in P1c tests: {address!r}")
        return real_connect_ex(sock, address)

    def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"Outbound socket.create_connection blocked in P1c tests: {address!r}")

    return (
        mock.patch.object(socket.socket, "connect", _guarded_connect),
        mock.patch.object(socket.socket, "connect_ex", _guarded_connect_ex),
        mock.patch("socket.create_connection", side_effect=_guarded_create_connection),
    )


class M2NsysCaptureFailureCharacterizationTest(unittest.TestCase):
    PID_RECORDS: List[Dict[str, Any]] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls.PID_RECORDS = []
        cls._net_patches = _make_network_guard()
        for p in cls._net_patches:
            p.start()

        cls._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        cls._saved_tempdir = tempfile.tempdir
        cls._root = Path(tempfile.mkdtemp(prefix="wp")).resolve()
        for env_k, sub in (
            ("HOME", "home"),
            ("USERPROFILE", "home"),
            ("XDG_CACHE_HOME", "xdg_cache"),
            ("XDG_CONFIG_HOME", "xdg_config"),
            ("WEAVEPROF_SESSIONS_DIR", "sessions"),
            ("WEAVEPROF_CACHE", "wp_cache"),
            ("WEAVEPROF_LOCK_DIR", "locks"),
            ("TMPDIR", "tmp"),
            ("TEMP", "tmp"),
            ("TMP", "tmp"),
            ("MPLCONFIGDIR", "mpl"),
        ):
            d = cls._root / sub
            d.mkdir(parents=True, exist_ok=True)
            os.environ[env_k] = str(d)
        tempfile.tempdir = None
        for clear_k in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "WEAVEPROF_GOLDEN_UPDATE",
            "WEAVEPROF_METRICS_UPDATE",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "P1C_NSYS_FAIL_RC",
        ):
            os.environ.pop(clear_k, None)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "weaveprof-golden-test-key"
        os.environ["PYTHONPATH"] = str(REPO_ROOT)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        cls._case_dir = REPO_ROOT / "tests" / "golden" / "cases" / "h20_pingpong_v1"
        cls._stage_dir = cls._root / "golden_stage"
        cls._stage_dir.mkdir(parents=True, exist_ok=True)
        _unxz(cls._case_dir / "capture.fi02.bin.xz", cls._stage_dir / "capture.fi02.bin")
        _unxz(cls._case_dir / "nsys.sqlite.xz", cls._stage_dir / "nsys.sqlite")
        shutil.copy2(cls._case_dir / "runtime.json", cls._stage_dir / "runtime.json")
        shutil.copy2(cls._case_dir / "names.json", cls._stage_dir / "names.json")

        shebang = f"#!{sys.executable} -B\n"
        cls._tools_dir = cls._root / "fake_tools"
        cls._tools_dir.mkdir(parents=True, exist_ok=True)
        cls._fake_build_py = cls._tools_dir / "fake_build.py"
        cls._fake_build_py.write_text(shebang + _FAKE_BUILD_BODY, encoding="utf-8")
        cls._fake_build_py.chmod(0o755)

        cls._fake_bench_py = cls._tools_dir / "fake_bench.py"
        cls._fake_bench_py.write_text(
            shebang + _FAKE_BENCH_BODY.replace("__STAGE_DIR__", repr(str(cls._stage_dir))),
            encoding="utf-8",
        )
        cls._fake_bench_py.chmod(0o755)

        cls._bin_dir = cls._root / "bin"
        cls._bin_dir.mkdir(parents=True, exist_ok=True)
        cls._fake_nsys = cls._bin_dir / "nsys"
        cls._fake_nsys.write_text(
            shebang + _FAKE_NSYS_BODY.replace("__STAGE_DIR__", repr(str(cls._stage_dir))),
            encoding="utf-8",
        )
        cls._fake_nsys.chmod(0o755)

        orig_path = cls._saved_env.get("PATH") or ""
        os.environ["PATH"] = f"{cls._bin_dir}{os.pathsep}{orig_path}"

        cls._build_cmd = [
            sys.executable,
            "-B",
            str(cls._fake_build_py),
            "--l0",
            "{l0}",
            "--inst",
            "{inst}",
            "--out",
            "{out}",
            "--tree",
            "{tree}",
        ]
        cls._bench_cmd = [
            sys.executable,
            "-B",
            str(cls._fake_bench_py),
            "--require-build",
            "--l0",
            "{l0}",
            "--inst",
            "{inst}",
            "--out",
            "{out}",
            "--tree",
            "{tree}",
        ]

        cls._cwd = cls._root / "cwd"
        cls._cwd.mkdir(parents=True, exist_ok=True)
        cls._env = dict(os.environ)

        probe = subprocess.run(
            [sys.executable, "-B", "-c", "import weaveprof; print(weaveprof.__file__)"],
            cwd=str(cls._cwd),
            env=cls._env,
            capture_output=True,
            text=True,
            check=True,
        )
        rel_init = Path(probe.stdout.strip()).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        if rel_init != "weaveprof/__init__.py":
            raise AssertionError(f"weaveprof imported from outside repo under test: {probe.stdout!r}")

    @classmethod
    def tearDownClass(cls) -> None:
        pid_report = os.environ.get("P1C_PID_REPORT_PATH")
        if pid_report:
            Path(pid_report).write_text(json.dumps(cls.PID_RECORDS, indent=2), encoding="utf-8")
        tempfile.tempdir = cls._saved_tempdir
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for p in reversed(cls._net_patches):
            p.stop()
        shutil.rmtree(cls._root, ignore_errors=True)

    def _run_with_isolated_env(self, cache_dir: Path, inv_log: Path, extra_env: Dict[str, str | None] | None = None):
        cache_dir.mkdir(parents=True, exist_ok=True)
        inv_log.unlink(missing_ok=True)
        updates: Dict[str, str | None] = {
            "WEAVEPROF_CACHE": str(cache_dir),
            "P1C_INVOCATION_LOG": str(inv_log),
            "P1C_NSYS_FAIL_RC": None,
        }
        if extra_env:
            updates.update(extra_env)
        prev = {k: os.environ.get(k) for k in updates}
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return prev

    @staticmethod
    def _restore_env(prev: Dict[str, str | None]) -> None:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _read_invocations(self, inv_log: Path, test_tag: str) -> List[Dict[str, Any]]:
        if not inv_log.is_file():
            return []
        rows = [json.loads(line) for line in inv_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in rows:
            self.__class__.PID_RECORDS.append(
                {
                    "pid": int(r["pid"]),
                    "label": f"{test_tag}:{r['kind']}",
                    "rc": int(r.get("rc", 0)),
                }
            )
        return rows

    def test_m2b_nsys_capture_failure_without_overview_png_fails_in_all_four_places(self) -> None:
        """capture_nsys=True, render_overview_png=False 下 nsys 采集失败时，四处结论均为 FAIL。修复前的行为见 PR-3。"""
        resolved_nsys = mk.find_nsys_binary()
        self.assertIsNotNone(resolved_nsys)
        self.assertEqual(Path(str(resolved_nsys)).resolve(), self._fake_nsys.resolve())

        ws = self._root / "ws_m2b"
        shutil.copytree(self._case_dir / "workspace", ws)
        out_b = self._root / "b"
        inv_b = self._root / "inv_m2b.jsonl"
        pairs = _make_pairs((self._root, "<TMP>"), (REPO_ROOT, "<REPO>"))

        prev = self._run_with_isolated_env(
            self._root / "cache_m2b",
            inv_b,
            {"P1C_NSYS_FAIL_RC": "2"},
        )
        try:
            raw_rep = mk.run_megakernel_audit(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=True,
                render_overview_png=False,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_b,
            )
        finally:
            self._restore_env(prev)

        rep = _norm_paths(raw_rep, pairs)
        self.assertEqual(rep["schema"], "weaveprof.megakernel-audit/v1")
        self.assertEqual(rep["status"], "FAIL")
        self.assertEqual(rep["mode"], "shadow_build_run")
        self.assertIs(rep["seal_verified"], True)
        self.assertIs(rep["seal_integrity_only"], True)
        self.assertEqual(rep["seal_path"], "<TMP>/b/seal.json")
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", str(rep["seal_sha256"])))
        self.assertEqual(rep["overhead_pct"], 3.95)
        self.assertEqual(rep["kernel_us"], 1066.5)
        self.assertEqual(rep["hw_quartiles"], {})
        self.assertIsNone(rep["tensor_active_scale"])
        self.assertEqual(
            rep["structural_blockers"],
            [
                "NSYS_CAPTURE_FAILED: nsys profile exited 2 (see <TMP>/b/nsys_profile.log)",
                "LOW_MATH_WAIT_NO_LATENCY_HIDING",
            ],
        )
        self.assertEqual(rep["artifacts"], {"out_dir": "<TMP>/b", "files": ["feedback.json", "verify.json"]})

        self.assertFalse((out_b / "overview.png").exists())
        self.assertFalse((out_b / "nsys_hw.json").exists())
        self.assertFalse((out_b / "nsys_cap.sqlite").exists())
        self.assertTrue((out_b / "nsys_profile.log").is_file())

        seal_doc = _norm_paths(json.loads((out_b / "seal.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(seal_doc["status"], "FAIL")
        self.assertEqual(seal_doc["mode"], "shadow_build_run")
        self.assertEqual(seal_doc["failed"], ["structural"])
        self.assertEqual(seal_doc["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertIsNone(seal_doc["artifacts"]["overview_png"])
        self.assertIsNone(seal_doc["artifacts"]["nsys_hw"])
        self.assertIsNone(seal_doc["artifacts"]["trace"])
        self.assertEqual(seal_doc["artifacts"]["failed_trace"]["path"], "<TMP>/b/bench/capture.fi02.bin")
        self.assertIs(verify_seal(out_b / "seal.json", require_pass=True)["ok"], False)
        self.assertIs(verify_seal(out_b / "seal.json", require_pass=False)["ok"], True)

        verify_doc = _norm_paths(json.loads((out_b / "verify.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(set(verify_doc.keys()), _VERIFY_JSON_KEYS)
        self.assertEqual(verify_doc["status"], "FAIL")
        self.assertEqual(verify_doc["failed"], [])
        self.assertEqual(verify_doc["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertEqual(verify_doc["overhead_ratio"], 0.039501)
        self.assertIsNone(verify_doc["fidelity_gates"]["sw_hw_coverage"])
        self.assertEqual(verify_doc["region_lock"]["provider"], "local_file")

        feedback_doc = _norm_paths(json.loads((out_b / "feedback.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(feedback_doc["status"], "FAIL")
        self.assertEqual(feedback_doc["mode"], "shadow_build_run")
        self.assertIsNone(feedback_doc["nsys"])
        self.assertEqual(
            feedback_doc["structural_blockers"],
            [
                "NSYS_CAPTURE_FAILED: nsys profile exited 2 (see <TMP>/b/nsys_profile.log)",
                "LOW_MATH_WAIT_NO_LATENCY_HIDING",
            ],
        )

        recs = self._read_invocations(inv_b, "m2b")
        self.assertEqual(len(recs), 4)
        self.assertEqual({recs[0]["kind"], recs[1]["kind"]}, {"build"})
        self.assertEqual({recs[0]["role"], recs[1]["role"]}, {"L0", "L_inst"})
        self.assertEqual([r["kind"] for r in recs[2:]], ["bench", "nsys_profile"])
        self.assertEqual(recs[3]["mode"], "fail_rc")
        self.assertEqual(recs[3]["rc"], 2)

    def test_m2a_nsys_capture_failure_default_overview_png_fails_in_all_four_places(self) -> None:
        """默认参数（render_overview_png=True）下 nsys 采集失败时，四处结论均为 FAIL。修复前的行为见 PR-3。"""
        resolved_nsys = mk.find_nsys_binary()
        self.assertIsNotNone(resolved_nsys)
        self.assertEqual(Path(str(resolved_nsys)).resolve(), self._fake_nsys.resolve())

        ws = self._root / "ws_m2a"
        shutil.copytree(self._case_dir / "workspace", ws)
        out_a = self._root / "a"
        inv_a = self._root / "inv_m2a.jsonl"
        pairs = _make_pairs((self._root, "<TMP>"), (REPO_ROOT, "<REPO>"))

        prev = self._run_with_isolated_env(
            self._root / "cache_m2a",
            inv_a,
            {"P1C_NSYS_FAIL_RC": "2"},
        )
        try:
            raw_rep = mk.run_megakernel_audit(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=True,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_a,
            )
        finally:
            self._restore_env(prev)

        rep = _norm_paths(raw_rep, pairs)
        self.assertEqual(rep["schema"], "weaveprof.megakernel-audit/v1")
        self.assertEqual(rep["status"], "FAIL")
        self.assertEqual(rep["mode"], "shadow_build_run")
        self.assertIs(rep["seal_verified"], True)
        self.assertIs(rep["seal_integrity_only"], True)
        self.assertEqual(rep["seal_path"], "<TMP>/a/seal.json")
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", str(rep["seal_sha256"])))
        self.assertEqual(rep["overhead_pct"], 3.95)
        self.assertEqual(rep["kernel_us"], 1066.5)
        self.assertEqual(rep["hw_quartiles"], {})
        self.assertIsNone(rep["tensor_active_scale"])
        self.assertEqual(
            rep["structural_blockers"],
            [
                "NSYS_CAPTURE_FAILED: nsys profile exited 2 (see <TMP>/a/nsys_profile.log)",
                "LOW_MATH_WAIT_NO_LATENCY_HIDING",
                "OVERVIEW_PNG_FAILED: ValueError: MISSING_HARDWARE_UTILIZATION_CURVES: Panel (A) requires same-run hardware profiler utilization curves (200 ",
            ],
        )
        self.assertEqual(rep["artifacts"], {"out_dir": "<TMP>/a", "files": ["feedback.json", "verify.json"]})
        self.assertFalse((out_a / "overview.png").exists())
        self.assertFalse((out_a / "nsys_hw.json").exists())

        seal_doc = _norm_paths(json.loads((out_a / "seal.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(seal_doc["status"], "FAIL")
        self.assertEqual(seal_doc["mode"], "shadow_build_run")
        self.assertEqual(seal_doc["failed"], ["structural"])
        self.assertEqual(seal_doc["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertIsNone(seal_doc["artifacts"]["overview_png"])
        self.assertIsNone(seal_doc["artifacts"]["nsys_hw"])
        self.assertIsNone(seal_doc["artifacts"]["trace"])
        self.assertEqual(seal_doc["artifacts"]["failed_trace"]["path"], "<TMP>/a/bench/capture.fi02.bin")
        self.assertIs(verify_seal(out_a / "seal.json", require_pass=True)["ok"], False)
        self.assertIs(verify_seal(out_a / "seal.json", require_pass=False)["ok"], True)

        verify_doc = _norm_paths(json.loads((out_a / "verify.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(set(verify_doc.keys()), _VERIFY_JSON_KEYS)
        self.assertEqual(verify_doc["status"], "FAIL")
        self.assertEqual(verify_doc["failed"], [])
        self.assertEqual(verify_doc["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertEqual(verify_doc["overhead_ratio"], 0.039501)
        self.assertIsNone(verify_doc["fidelity_gates"]["sw_hw_coverage"])

        feedback_doc = _norm_paths(json.loads((out_a / "feedback.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(feedback_doc["status"], "FAIL")
        self.assertEqual(feedback_doc["mode"], "shadow_build_run")
        self.assertIsNone(feedback_doc["nsys"])
        self.assertEqual(
            feedback_doc["structural_blockers"],
            [
                "NSYS_CAPTURE_FAILED: nsys profile exited 2 (see <TMP>/a/nsys_profile.log)",
                "LOW_MATH_WAIT_NO_LATENCY_HIDING",
                "OVERVIEW_PNG_FAILED: ValueError: MISSING_HARDWARE_UTILIZATION_CURVES: Panel (A) requires same-run hardware profiler utilization curves (200 kHz nsys on NVIDIA o",
            ],
        )

        recs = self._read_invocations(inv_a, "m2a")
        self.assertEqual(len(recs), 4)
        self.assertEqual({recs[0]["kind"], recs[1]["kind"]}, {"build"})
        self.assertEqual({recs[0]["role"], recs[1]["role"]}, {"L0", "L_inst"})
        self.assertEqual([r["kind"] for r in recs[2:]], ["bench", "nsys_profile"])
        self.assertEqual(recs[3]["mode"], "fail_rc")
        self.assertEqual(recs[3]["rc"], 2)

    def test_control_nsys_capture_success_without_overview_png_consistent_pass(self) -> None:
        """Control: identical parameters to M2b (capture_nsys=True, render_overview_png=False) when fake nsys succeeds."""
        resolved_nsys = mk.find_nsys_binary()
        self.assertIsNotNone(resolved_nsys)
        self.assertEqual(Path(str(resolved_nsys)).resolve(), self._fake_nsys.resolve())

        ws = self._root / "ws_control"
        shutil.copytree(self._case_dir / "workspace", ws)
        out_c = self._root / "c"
        inv_c = self._root / "inv_control.jsonl"
        pairs = _make_pairs((self._root, "<TMP>"), (REPO_ROOT, "<REPO>"))

        prev = self._run_with_isolated_env(self._root / "cache_control", inv_c)
        try:
            raw_rep = mk.run_megakernel_audit(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=True,
                render_overview_png=False,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_c,
            )
        finally:
            self._restore_env(prev)

        rep = _norm_paths(raw_rep, pairs)
        self.assertEqual(rep["schema"], "weaveprof.megakernel-audit/v1")
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["mode"], "shadow_build_run")
        self.assertIs(rep["seal_verified"], True)
        self.assertNotIn("seal_integrity_only", rep)
        self.assertEqual(rep["seal_path"], "<TMP>/c/seal.json")
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", str(rep["seal_sha256"])))
        self.assertEqual(rep["overhead_pct"], 3.95)
        self.assertEqual(rep["kernel_us"], 1067.939)
        self.assertEqual(rep["tensor_active_scale"], 4.0)
        self.assertEqual(
            rep["hw_quartiles"],
            {
                "dram_read": [7.1, 5.1, 5.2, 3.1],
                "dram_write": [3.4, 1.6, 1.5, 1.3],
                "sm_active": [100.0, 100.0, 100.0, 74.0],
                "tensor_active": [96.5, 96.1, 96.1, 69.2],
                "warp_issue": [12.7, 12.9, 12.9, 9.4],
            },
        )
        self.assertEqual(rep["structural_blockers"], ["LOW_MATH_WAIT_NO_LATENCY_HIDING"])
        self.assertEqual(rep["artifacts"], {"out_dir": "<TMP>/c", "files": ["feedback.json", "verify.json"]})

        self.assertFalse((out_c / "overview.png").exists())
        self.assertTrue((out_c / "nsys_hw.json").is_file())
        self.assertTrue((out_c / "nsys_cap.sqlite").is_file())

        seal_doc = _norm_paths(json.loads((out_c / "seal.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(seal_doc["status"], "PASS")
        self.assertEqual(seal_doc["mode"], "shadow_build_run")
        self.assertEqual(seal_doc["failed"], [])
        self.assertEqual(seal_doc["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertIsNone(seal_doc["artifacts"]["overview_png"])
        self.assertEqual(seal_doc["artifacts"]["nsys_hw"]["path"], "<TMP>/c/nsys_hw.json")
        self.assertEqual(seal_doc["artifacts"]["trace"]["path"], "<TMP>/c/nsys_run/capture.fi02.bin")
        self.assertIsNone(seal_doc["artifacts"]["failed_trace"])

        verify_doc = _norm_paths(json.loads((out_c / "verify.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(set(verify_doc.keys()), _VERIFY_JSON_KEYS)
        self.assertEqual(verify_doc["status"], "PASS")
        self.assertEqual(verify_doc["failed"], [])
        self.assertEqual(verify_doc["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertIs(verify_doc["fidelity_gates"]["sw_hw_coverage"]["ok"], True)

        feedback_doc = _norm_paths(json.loads((out_c / "feedback.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(feedback_doc["status"], "PASS")
        self.assertEqual(feedback_doc["mode"], "shadow_build_run")
        self.assertEqual(feedback_doc["nsys"]["tensor_active_scale"]["scale"], 4.0)
        self.assertEqual(feedback_doc["structural_blockers"], ["LOW_MATH_WAIT_NO_LATENCY_HIDING"])

        recs = self._read_invocations(inv_c, "control")
        self.assertEqual(len(recs), 6)
        self.assertEqual({recs[0]["kind"], recs[1]["kind"]}, {"build"})
        self.assertEqual({recs[0]["role"], recs[1]["role"]}, {"L0", "L_inst"})
        self.assertEqual([r["kind"] for r in recs[2:]], ["bench", "nsys_profile", "bench", "nsys_export"])


if __name__ == "__main__":
    unittest.main()

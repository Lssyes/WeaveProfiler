#!/usr/bin/env python3
"""Characterization tests pinning Workflow A (shadow build + locked bench + nsys) directly (P1c R23).

Drives ``weaveprof.megakernel.run_megakernel_audit`` through Workflow A (``shadow_build_run``) on CPU
using runtime-generated fake ``build_cmd``, fake ``bench_cmd``, and fake ``nsys`` scripts, verifying:
  1. Full ``h20_pingpong_v1`` golden equivalence via a ``run_case`` Workflow-A converter (with only
     the 3 pre-registered Workflow-A diff paths ``$.report.mode``, ``$.feedback.mode``, and
     ``$.feedback.l0_cache``), plus exact call order, placeholder expansion, environment isolation,
     ``find_nsys_binary()`` resolution, and local KREX GPU lock metadata.
  2. L0 content-addressed build cache miss (2 builds, ``hit=False``) followed by cache hit (1 build
     for ``L_inst`` only, ``hit=True``, cached product restored into ``L0``).
  3. Fail-closed error reporting when ``build_cmd`` exits non-zero (``pre-build failed``), when
     ``bench_cmd`` exits non-zero (``bench_cmd exited``), and when ``nsys profile`` exits 0 without
     emitting ``.nsys-rep`` (``nsys profile exited 0``).

Run:  python -B -m unittest tests/characterization/test_p1c_workflow_a_direct.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO_ROOT)]:
    sys.path.insert(0, str(REPO_ROOT))

# Clear golden update flags before importing the golden module (N4), and import only the module
# object so unittest does not re-collect GoldenAuditTest (N1).
os.environ.pop("WEAVEPROF_GOLDEN_UPDATE", None)
os.environ.pop("WEAVEPROF_METRICS_UPDATE", None)
golden = importlib.import_module("tests.golden.test_golden_audit")

from weaveprof import megakernel as mk  # noqa: E402

_REAL_RUN_MEGAKERNEL_AUDIT = mk.run_megakernel_audit

# Pre-determined exact JSON leaf/field paths that legitimately differ between Workflow B
# ("precaptured", recorded in golden expected.json) and Workflow A ("shadow_build_run").
WORKFLOW_A_EXPECTED_DIFF_PATHS: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    (
        ("report", "mode"),
        "$.report.mode",
        "Workflow A sets report['mode'] = 'shadow_build_run' vs 'precaptured' (weaveprof/megakernel.py:1673,2049).",
    ),
    (
        ("feedback", "mode"),
        "$.feedback.mode",
        "Workflow A writes feedback['mode'] = 'shadow_build_run' vs 'precaptured' (weaveprof/megakernel.py:1673,1943).",
    ),
    (
        ("feedback", "l0_cache"),
        "$.feedback.l0_cache",
        "Workflow A populates shadow_info['l0_cache'] instead of None (weaveprof/megakernel.py:1129-1134,1194,1230,1948).",
    ),
)

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
    "P1C_BUILD_FAIL_L0_RC",
    "P1C_BENCH_FAIL_RC",
    "P1C_NSYS_OMIT_REP",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
)

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
        "env": {
            "WEAVEPROF_ROOT": os.environ.get("WEAVEPROF_ROOT"),
            "WEAVEPROF_L0_DIR": os.environ.get("WEAVEPROF_L0_DIR"),
            "WEAVEPROF_INST_DIR": os.environ.get("WEAVEPROF_INST_DIR"),
            "WEAVEPROF_OUT_DIR": os.environ.get("WEAVEPROF_OUT_DIR"),
            "WEAVEPROF_L0_CACHE_DIR_SET": bool(os.environ.get("WEAVEPROF_L0_CACHE_DIR")),
            "WP_LEVEL": os.environ.get("WP_LEVEL"),
            "FI_LEVEL": os.environ.get("FI_LEVEL"),
            "has_hmac_key": "WEAVEPROF_SUPERVISOR_HMAC_KEY" in os.environ,
            "has_key_file": "WEAVEPROF_SUPERVISOR_KEY_FILE" in os.environ,
        },
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(json.dumps(rec) + "\\n")
        fh.flush()
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

fail_rc = os.environ.get("P1C_BUILD_FAIL_L0_RC")
if fail_rc and role == "L0":
    raise SystemExit(int(fail_rc))

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
        "env": {
            "WEAVEPROF_L0_DIR": os.environ.get("WEAVEPROF_L0_DIR"),
            "WEAVEPROF_INST_DIR": os.environ.get("WEAVEPROF_INST_DIR"),
            "WEAVEPROF_OUT_DIR": os.environ.get("WEAVEPROF_OUT_DIR"),
            "WEAVEPROF_NSYS_CAPTURE": os.environ.get("WEAVEPROF_NSYS_CAPTURE"),
            "has_hmac_key": "WEAVEPROF_SUPERVISOR_HMAC_KEY" in os.environ,
            "has_key_file": "WEAVEPROF_SUPERVISOR_KEY_FILE" in os.environ,
        },
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

fail_rc = os.environ.get("P1C_BENCH_FAIL_RC")
if fail_rc:
    raise SystemExit(int(fail_rc))
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
    _log({
        "kind": "nsys_profile",
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": {
            "WEAVEPROF_OUT_DIR": os.environ.get("WEAVEPROF_OUT_DIR"),
            "WEAVEPROF_NSYS_CAPTURE": os.environ.get("WEAVEPROF_NSYS_CAPTURE"),
            "has_hmac_key": "WEAVEPROF_SUPERVISOR_HMAC_KEY" in os.environ,
            "has_key_file": "WEAVEPROF_SUPERVISOR_KEY_FILE" in os.environ,
        },
    })
    idx = sys.argv.index("-o")
    rep_base = sys.argv[idx + 1]
    child_argv = sys.argv[idx + 2:]
    child = subprocess.run(child_argv, env=os.environ.copy(), check=False)
    if child.returncode != 0:
        raise SystemExit(child.returncode)
    if os.environ.get("P1C_NSYS_OMIT_REP") != "1":
        Path(rep_base + ".nsys-rep").write_bytes(b"FAKE_NSYS_REP_V1\\n")
    raise SystemExit(0)

if subcmd == "export":
    _log({
        "kind": "nsys_export",
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


def _pop_nested(doc: Dict[str, Any], key_path: Tuple[str, ...]) -> Any:
    cur: Any = doc
    for part in key_path[:-1]:
        cur = cur[part]
    return cur.pop(key_path[-1])


class WorkflowADirectTest(unittest.TestCase):
    PID_RECORDS: List[Tuple[int, str, int | None]] = []

    @classmethod
    def setUpClass(cls) -> None:
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
            "P1C_BUILD_FAIL_L0_RC",
            "P1C_BENCH_FAIL_RC",
            "P1C_NSYS_OMIT_REP",
        ):
            os.environ.pop(clear_k, None)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "weaveprof-golden-test-key"
        os.environ["PYTHONPATH"] = str(REPO_ROOT)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        # Stage uncompressed golden fixture payloads once for fake bench_cmd & fake nsys
        cls._case_dir = REPO_ROOT / "tests" / "golden" / "cases" / "h20_pingpong_v1"
        cls._stage_dir = cls._root / "golden_stage"
        cls._stage_dir.mkdir(parents=True, exist_ok=True)
        _unxz(cls._case_dir / "capture.fi02.bin.xz", cls._stage_dir / "capture.fi02.bin")
        _unxz(cls._case_dir / "nsys.sqlite.xz", cls._stage_dir / "nsys.sqlite")
        shutil.copy2(cls._case_dir / "runtime.json", cls._stage_dir / "runtime.json")
        shutil.copy2(cls._case_dir / "names.json", cls._stage_dir / "names.json")

        # Write runtime fake build_cmd, fake bench_cmd, and fake nsys scripts inside temp root
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
        tempfile.tempdir = cls._saved_tempdir
        for k, v in cls._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls._root, ignore_errors=True)

    def _run_with_isolated_env(self, cache_dir: Path, inv_log: Path, extra_env: Dict[str, str] | None = None):
        cache_dir.mkdir(parents=True, exist_ok=True)
        inv_log.unlink(missing_ok=True)
        updates = {
            "WEAVEPROF_CACHE": str(cache_dir),
            "P1C_INVOCATION_LOG": str(inv_log),
            "P1C_BUILD_FAIL_L0_RC": None,
            "P1C_BENCH_FAIL_RC": None,
            "P1C_NSYS_OMIT_REP": None,
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

    def _read_invocations(self, inv_log: Path) -> List[Dict[str, Any]]:
        if not inv_log.is_file():
            return []
        rows = [json.loads(line) for line in inv_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        for r in rows:
            self.__class__.PID_RECORDS.append((int(r["pid"]), f"direct:{r['kind']}", 0))
        return rows

    def test_1_workflow_a_golden_conversion_and_invocation_contract(self) -> None:
        # Must resolve to our fake nsys script, never any system /usr/local/bin/nsys or /opt/nvidia nsys
        resolved_nsys = mk.find_nsys_binary()
        self.assertIsNotNone(resolved_nsys)
        self.assertEqual(Path(str(resolved_nsys)).resolve(), self._fake_nsys.resolve())

        case_tmp = self._root / "golden_wf_a_case"
        case_tmp.mkdir(parents=True, exist_ok=True)
        cache_dir = self._root / "cache_test_1"
        inv_log = self._root / "invocations_test_1.jsonl"
        converted_calls: List[Dict[str, Any]] = []

        def _workflow_a_converter(workspace: str | Path, **kwargs: Any) -> Dict[str, Any]:
            conv = dict(kwargs)
            for precap_key in ("dump_path", "names", "runtime_metrics", "nsys_sqlite"):
                self.assertIn(precap_key, conv)
                conv.pop(precap_key)
            conv["build_cmd"] = list(self._build_cmd)
            conv["bench_cmd"] = list(self._bench_cmd)
            conv["capture_nsys"] = True
            converted_calls.append(dict(conv))
            return _REAL_RUN_MEGAKERNEL_AUDIT(workspace, **conv)

        prev = self._run_with_isolated_env(cache_dir, inv_log)
        try:
            with mock.patch("weaveprof.megakernel.run_megakernel_audit", side_effect=_workflow_a_converter):
                obs = golden.run_case(self._case_dir, case_tmp)
        finally:
            self._restore_env(prev)

        self.assertEqual(len(converted_calls), 1)
        self.assertEqual(obs["png"], {"checked": True, "valid": True})

        # Concrete literal assertions on core Workflow A report fields before diffing against expected.json
        rep = obs["report"]
        self.assertEqual(rep["schema"], "weaveprof.megakernel-audit/v1")
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual(rep["mode"], "shadow_build_run")
        self.assertIs(rep["seal_verified"], True)
        self.assertEqual(rep["overhead_pct"], 3.95)
        self.assertEqual(rep["kernel_us"], 1067.939)
        self.assertEqual(rep["tensor_active_scale"], 4.0)
        self.assertEqual(rep["causal_opportunities"], [])
        self.assertEqual(rep["structural_blockers"], ["LOW_MATH_WAIT_NO_LATENCY_HIDING"])
        self.assertIsNone(rep["round_diff"])
        self.assertEqual(
            rep["artifacts"]["files"],
            ["feedback.json", "verify.json", "overview.png", "trace.perfetto.json.gz"],
        )
        self.assertEqual(
            rep["fidelity_gates"],
            {
                "compiler_spill": {"codes": [], "ok": True, "reg_blind_spot": False},
                "granularity": {"ok": True},
                "output": {"ok": True},
                "overflow": 0,
                "overhead": {"ok": True, "ratio": 0.0395},
                "pairing_errors": 0,
                "stub": {"ok": True, "stubs": []},
                "warps": {"ok": True},
            },
        )

        # Assert each pre-determined Workflow-A diff path, then delete from both obs and expected
        expected = json.loads((self._case_dir / "expected.json").read_text(encoding="utf-8"))
        self.assertEqual(expected["report"]["mode"], "precaptured")
        self.assertEqual(expected["feedback"]["mode"], "precaptured")
        self.assertIsNone(expected["feedback"]["l0_cache"])

        self.assertEqual(obs["feedback"]["mode"], "shadow_build_run")
        l0_cache = obs["feedback"]["l0_cache"]
        self.assertIsInstance(l0_cache, dict)
        self.assertEqual(set(l0_cache.keys()), {"enabled", "hit", "key", "stored_products"})
        self.assertIs(l0_cache["enabled"], True)
        self.assertIs(l0_cache["hit"], False)
        self.assertEqual(l0_cache["stored_products"], 1)
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{16}", str(l0_cache["key"])))

        obs_cmp = json.loads(golden._canon(obs))
        for key_path, _json_path, _reason in WORKFLOW_A_EXPECTED_DIFF_PATHS:
            _pop_nested(obs_cmp, key_path)
            _pop_nested(expected, key_path)
        diff = golden._first_diff(expected, obs_cmp)
        self.assertIsNone(diff, f"Unexpected Workflow A diff outside registered paths: {diff}")

        # Verify local KREX GPU lock metadata in verify.json (stripped by golden._VOLATILE during golden._norm)
        pairs = _make_pairs((case_tmp, "<CASE>"), (self._root, "<TMP>"), (REPO_ROOT, "<REPO>"))
        raw_verify = _norm_paths(
            json.loads((case_tmp / "out" / "verify.json").read_text(encoding="utf-8")),
            pairs,
        )
        rlock = raw_verify["region_lock"]
        self.assertEqual(rlock["provider"], "local_file")
        self.assertEqual(rlock["gpu_id"], "gpu_0")
        self.assertEqual(rlock["label"], "megakernel_audit")
        self.assertEqual(rlock["lock_path"], "<TMP>/locks/gpu_0.lock")
        self.assertTrue((self._root / "locks" / "gpu_0.lock").is_file())
        # gpu_critical_region unlinks gpu_0.owner.json in finally upon releasing the lock (harness.py:588-589)
        self.assertFalse((self._root / "locks" / "gpu_0.owner.json").exists())

        # Shadow workspace was reclaimed while instrumented source copy was preserved and workspace stayed clean
        self.assertFalse((case_tmp / "out" / "shadow").exists())
        self.assertTrue((case_tmp / "out" / "instrumented_matmul-pingpong-v1.cu").is_file())
        self.assertEqual(
            golden._sha((case_tmp / "workspace" / "matmul-pingpong-v1.cu").read_bytes()),
            "6cea508de081153f26edfd9aa4d1269ebd7d3556e48168efad354fba9991e976",
        )

        # Verify exact call count, ordering, normalized arguments, and child environments across all 6 invocations
        records = _norm_paths(self._read_invocations(inv_log), pairs)
        self.assertEqual(len(records), 6)
        self.assertEqual({records[0]["kind"], records[1]["kind"]}, {"build"})
        self.assertEqual({records[0]["role"], records[1]["role"]}, {"L0", "L_inst"})
        self.assertEqual([r["kind"] for r in records[2:]], ["bench", "nsys_profile", "bench", "nsys_export"])

        builds_by_role = {r["role"]: r for r in records[:2]}
        self.assertEqual(
            builds_by_role["L0"]["argv"],
            [
                "<TMP>/fake_tools/fake_build.py",
                "--l0",
                "<CASE>/out/shadow/L0",
                "--inst",
                "<CASE>/out/shadow/L_inst",
                "--out",
                "<CASE>/out/bench",
                "--tree",
                "<CASE>/out/shadow/L0",
            ],
        )
        self.assertEqual(builds_by_role["L0"]["cwd"], "<CASE>/out/shadow/L0")
        self.assertEqual(
            builds_by_role["L0"]["env"],
            {
                "WEAVEPROF_ROOT": "<REPO>",
                "WEAVEPROF_L0_DIR": "<CASE>/out/shadow/L0",
                "WEAVEPROF_INST_DIR": "<CASE>/out/shadow/L_inst",
                "WEAVEPROF_OUT_DIR": "<CASE>/out/bench",
                "WEAVEPROF_L0_CACHE_DIR_SET": True,
                "WP_LEVEL": "0",
                "FI_LEVEL": "0",
                "has_hmac_key": False,
                "has_key_file": False,
            },
        )
        self.assertEqual(
            builds_by_role["L_inst"]["argv"],
            [
                "<TMP>/fake_tools/fake_build.py",
                "--l0",
                "<CASE>/out/shadow/L0",
                "--inst",
                "<CASE>/out/shadow/L_inst",
                "--out",
                "<CASE>/out/bench",
                "--tree",
                "<CASE>/out/shadow/L_inst",
            ],
        )
        self.assertEqual(builds_by_role["L_inst"]["cwd"], "<CASE>/out/shadow/L_inst")
        self.assertEqual(
            builds_by_role["L_inst"]["env"],
            {
                "WEAVEPROF_ROOT": "<REPO>",
                "WEAVEPROF_L0_DIR": "<CASE>/out/shadow/L0",
                "WEAVEPROF_INST_DIR": "<CASE>/out/shadow/L_inst",
                "WEAVEPROF_OUT_DIR": "<CASE>/out/bench",
                "WEAVEPROF_L0_CACHE_DIR_SET": True,
                "WP_LEVEL": None,
                "FI_LEVEL": None,
                "has_hmac_key": False,
                "has_key_file": False,
            },
        )

        bench_abba = records[2]
        self.assertIs(bench_abba["lock_owner_present"], True)
        self.assertEqual(
            bench_abba["argv"],
            [
                "<TMP>/fake_tools/fake_bench.py",
                "--require-build",
                "--l0",
                "<CASE>/out/shadow/L0",
                "--inst",
                "<CASE>/out/shadow/L_inst",
                "--out",
                "<CASE>/out/bench",
                "--tree",
                "<CASE>/out/shadow/L_inst",
            ],
        )
        self.assertEqual(bench_abba["cwd"], "<CASE>/out/shadow/L_inst")
        self.assertEqual(
            bench_abba["env"],
            {
                "WEAVEPROF_L0_DIR": "<CASE>/out/shadow/L0",
                "WEAVEPROF_INST_DIR": "<CASE>/out/shadow/L_inst",
                "WEAVEPROF_OUT_DIR": "<CASE>/out/bench",
                "WEAVEPROF_NSYS_CAPTURE": None,
                "has_hmac_key": False,
                "has_key_file": False,
            },
        )

        nsys_prof = records[3]
        py_norm = _norm_paths(sys.executable, pairs)
        self.assertEqual(
            nsys_prof["argv"],
            [
                "<TMP>/bin/nsys",
                "profile",
                "--force-overwrite=true",
                "--stats=false",
                "--trace=cuda",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--gpu-metrics-devices=cuda-visible",
                "--gpu-metrics-frequency=200000",
                "-o",
                "<CASE>/out/nsys_cap",
                py_norm,
                "-B",
                "<TMP>/fake_tools/fake_bench.py",
                "--require-build",
                "--l0",
                "<CASE>/out/shadow/L0",
                "--inst",
                "<CASE>/out/shadow/L_inst",
                "--out",
                "<CASE>/out/nsys_run",
                "--tree",
                "<CASE>/out/shadow/L_inst",
            ],
        )
        self.assertEqual(nsys_prof["cwd"], "<CASE>/out/shadow/L_inst")
        self.assertEqual(
            nsys_prof["env"],
            {
                "WEAVEPROF_OUT_DIR": "<CASE>/out/nsys_run",
                "WEAVEPROF_NSYS_CAPTURE": "1",
                "has_hmac_key": False,
                "has_key_file": False,
            },
        )

        bench_nsys = records[4]
        self.assertIs(bench_nsys["lock_owner_present"], True)
        self.assertEqual(
            bench_nsys["argv"],
            [
                "<TMP>/fake_tools/fake_bench.py",
                "--require-build",
                "--l0",
                "<CASE>/out/shadow/L0",
                "--inst",
                "<CASE>/out/shadow/L_inst",
                "--out",
                "<CASE>/out/nsys_run",
                "--tree",
                "<CASE>/out/shadow/L_inst",
            ],
        )
        self.assertEqual(bench_nsys["cwd"], "<CASE>/out/shadow/L_inst")
        self.assertEqual(
            bench_nsys["env"],
            {
                "WEAVEPROF_L0_DIR": "<CASE>/out/shadow/L0",
                "WEAVEPROF_INST_DIR": "<CASE>/out/shadow/L_inst",
                "WEAVEPROF_OUT_DIR": "<CASE>/out/nsys_run",
                "WEAVEPROF_NSYS_CAPTURE": "1",
                "has_hmac_key": False,
                "has_key_file": False,
            },
        )

        nsys_exp = records[5]
        self.assertEqual(
            nsys_exp["argv"],
            [
                "<TMP>/bin/nsys",
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                "--output",
                "<CASE>/out/nsys_cap.sqlite",
                "<CASE>/out/nsys_cap.nsys-rep",
            ],
        )
        self.assertEqual(nsys_exp["cwd"], "<CASE>/out")

    def test_2_l0_build_cache_miss_then_hit(self) -> None:
        ws = self._root / "ws_l0_cache"
        shutil.copytree(self._case_dir / "workspace", ws)
        shared_cache = self._root / "shared_l0_cache"
        inv_miss = self._root / "inv_l0_miss.jsonl"
        inv_hit = self._root / "inv_l0_hit.jsonl"
        out_miss = self._root / "out_l0_miss"
        out_hit = self._root / "out_l0_hit"

        # Audit 1: cold L0 cache -> hit == False, 2 builds (L0 + L_inst)
        prev1 = self._run_with_isolated_env(shared_cache, inv_miss)
        try:
            rep1 = _REAL_RUN_MEGAKERNEL_AUDIT(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=False,
                render_overview_png=False,
                write_perfetto_gz=False,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_miss,
            )
        finally:
            self._restore_env(prev1)

        self.assertEqual(rep1["status"], "PASS")
        self.assertEqual(rep1["mode"], "shadow_build_run")
        fb1 = json.loads((out_miss / "feedback.json").read_text(encoding="utf-8"))
        c1 = fb1["l0_cache"]
        self.assertEqual(set(c1.keys()), {"enabled", "hit", "key", "stored_products"})
        self.assertIs(c1["enabled"], True)
        self.assertIs(c1["hit"], False)
        self.assertEqual(c1["stored_products"], 1)
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{16}", str(c1["key"])))

        recs1 = self._read_invocations(inv_miss)
        self.assertEqual(len(recs1), 3)
        self.assertEqual({recs1[0]["kind"], recs1[1]["kind"]}, {"build"})
        self.assertEqual({recs1[0]["role"], recs1[1]["role"]}, {"L0", "L_inst"})
        self.assertEqual(recs1[2]["kind"], "bench")

        # Audit 2: same WEAVEPROF_CACHE + same tree/build_cmd -> hit == True, 1 build (L_inst only)
        prev2 = self._run_with_isolated_env(shared_cache, inv_hit)
        try:
            rep2 = _REAL_RUN_MEGAKERNEL_AUDIT(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=False,
                render_overview_png=False,
                write_perfetto_gz=False,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_hit,
            )
        finally:
            self._restore_env(prev2)

        self.assertEqual(rep2["status"], "PASS")
        self.assertEqual(rep2["mode"], "shadow_build_run")
        fb2 = json.loads((out_hit / "feedback.json").read_text(encoding="utf-8"))
        c2 = fb2["l0_cache"]
        self.assertEqual(set(c2.keys()), {"enabled", "hit", "key"})
        self.assertIs(c2["enabled"], True)
        self.assertIs(c2["hit"], True)
        self.assertEqual(c2["key"], c1["key"])

        recs2 = self._read_invocations(inv_hit)
        self.assertEqual([r["kind"] for r in recs2], ["build", "bench"])
        self.assertEqual(recs2[0]["role"], "L_inst")

    def test_3_error_paths_prebuild_failed_bench_failed_and_nsys_missing_rep(self) -> None:
        ws = self._root / "ws_errors"
        shutil.copytree(self._case_dir / "workspace", ws)
        pairs = _make_pairs((self._root, "<TMP>"), (REPO_ROOT, "<REPO>"))

        # (3a) Non-zero build_cmd exit -> BLOCKED with pre-build failed literal
        inv_b = self._root / "inv_err_build.jsonl"
        out_b = self._root / "err_prebuild_out"
        prev_b = self._run_with_isolated_env(
            self._root / "cache_err_build",
            inv_b,
            {"P1C_BUILD_FAIL_L0_RC": "7"},
        )
        try:
            rep_b = _REAL_RUN_MEGAKERNEL_AUDIT(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=True,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_b,
            )
        finally:
            self._restore_env(prev_b)

        norm_b = _norm_paths(rep_b, pairs)
        self.assertEqual(norm_b["status"], "BLOCKED")
        self.assertEqual(norm_b["mode"], "shadow_build_run")
        self.assertIs(norm_b["seal_verified"], False)
        self.assertIsNone(norm_b["seal_path"])
        self.assertEqual(
            norm_b["structural_blockers"],
            ["SHADOW_RUN_FAILED: RuntimeError: pre-build failed: {'L0': 7, 'L_inst': 0} (see <TMP>/err_prebuild_out/build_*.log)"],
        )
        recs_b = self._read_invocations(inv_b)
        self.assertEqual([r["kind"] for r in recs_b], ["build", "build"])

        # (3b) Non-zero bench_cmd exit -> BLOCKED with bench_cmd exited literal
        inv_m = self._root / "inv_err_bench.jsonl"
        out_m = self._root / "err_bench_out"
        prev_m = self._run_with_isolated_env(
            self._root / "cache_err_bench",
            inv_m,
            {"P1C_BENCH_FAIL_RC": "9"},
        )
        try:
            rep_m = _REAL_RUN_MEGAKERNEL_AUDIT(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=True,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_m,
            )
        finally:
            self._restore_env(prev_m)

        norm_m = _norm_paths(rep_m, pairs)
        self.assertEqual(norm_m["status"], "BLOCKED")
        self.assertEqual(norm_m["mode"], "shadow_build_run")
        self.assertIs(norm_m["seal_verified"], False)
        self.assertEqual(
            norm_m["structural_blockers"],
            ["SHADOW_RUN_FAILED: RuntimeError: bench_cmd exited 9 (see <TMP>/err_bench_out/bench.log)"],
        )
        recs_m = self._read_invocations(inv_m)
        self.assertEqual(len(recs_m), 3)
        self.assertEqual(recs_m[2]["kind"], "bench")

        # (3c) nsys profile exits 0 without writing .nsys-rep -> NSYS_CAPTURE_FAILED: nsys profile exited 0 literal
        inv_n = self._root / "inv_err_nsys.jsonl"
        out_n = self._root / "n"
        prev_n = self._run_with_isolated_env(
            self._root / "cache_err_nsys",
            inv_n,
            {"P1C_NSYS_OMIT_REP": "1"},
        )
        try:
            rep_n = _REAL_RUN_MEGAKERNEL_AUDIT(
                ws,
                entry_source="matmul-pingpong-v1.cu",
                build_cmd=list(self._build_cmd),
                bench_cmd=list(self._bench_cmd),
                capture_nsys=True,
                card="h20",
                arch="sm_90a",
                total_sms=78,
                out_dir=out_n,
            )
        finally:
            self._restore_env(prev_n)

        norm_n = _norm_paths(rep_n, pairs)
        self.assertEqual(norm_n["status"], "FAIL")
        self.assertEqual(norm_n["mode"], "shadow_build_run")
        self.assertEqual(
            norm_n["structural_blockers"],
            [
                "NSYS_CAPTURE_FAILED: nsys profile exited 0 (see <TMP>/n/nsys_profile.log)",
                "LOW_MATH_WAIT_NO_LATENCY_HIDING",
                "OVERVIEW_PNG_FAILED: ValueError: MISSING_HARDWARE_UTILIZATION_CURVES: Panel (A) requires same-run hardware profiler utilization curves (200 ",
                "MISSING_OVERVIEW_PNG: render_overview_png=True requires a valid non-empty overview.png",
            ],
        )
        fb_n = _norm_paths(json.loads((out_n / "feedback.json").read_text(encoding="utf-8")), pairs)
        self.assertEqual(
            fb_n["structural_blockers"],
            [
                "NSYS_CAPTURE_FAILED: nsys profile exited 0 (see <TMP>/n/nsys_profile.log)",
                "LOW_MATH_WAIT_NO_LATENCY_HIDING",
            ],
        )
        recs_n = self._read_invocations(inv_n)
        self.assertEqual([r["kind"] for r in recs_n[2:]], ["bench", "nsys_profile", "bench"])


if __name__ == "__main__":
    unittest.main()

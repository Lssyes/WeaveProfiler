#!/usr/bin/env python3
"""Characterization tests pinning the AKA plugin adapter entrypoint against golden (P1c R22).

Replaces ``weaveprof.megakernel.run_megakernel_audit`` in the test process with a subprocess
forwarder invoking ``[PY, "-B", "<repo>/weaveprof/aka_adapter.py", "megakernel-audit"]`` (per
``plugin.json:13-15``) over JSON stdin/stdout, runs ``tests.golden.test_golden_audit.run_case``
on all 3 golden captures, and verifies zero diff against ``expected.json``.

Run:  python -B -m unittest tests/characterization/test_p1c_entry_aka.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO_ROOT)]:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure golden update switches are cleared BEFORE importing the golden module (N4),
# and import only the module object so unittest does not re-collect GoldenAuditTest (N1).
os.environ.pop("WEAVEPROF_GOLDEN_UPDATE", None)
os.environ.pop("WEAVEPROF_METRICS_UPDATE", None)
golden = importlib.import_module("tests.golden.test_golden_audit")

_ENV_KEYS = (
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
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
)

_EXPECTED_STATUS_BY_CASE = {
    "h20_pingpong_v1": "PASS",
    "h20_v3_ws_tma_wgmma": "FAIL",
    "h20_v6_persistent": "FAIL",
}


def _jsonable(val: Any) -> Any:
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonable(x) for x in val]
    return val


class AkaEntrypointGoldenTest(unittest.TestCase):
    PID_RECORDS: List[Tuple[int, str, int | None]] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        cls._saved_tempdir = tempfile.tempdir
        cls._root = Path(tempfile.mkdtemp(prefix="wp-p1c-r22-aka-")).resolve()
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
        ):
            os.environ.pop(clear_k, None)
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "weaveprof-golden-test-key"
        os.environ["PYTHONPATH"] = str(REPO_ROOT)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        cls._env = dict(os.environ)
        cls._cwd = cls._root / "cwd"
        cls._cwd.mkdir(parents=True, exist_ok=True)

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

    def test_plugin_json_megakernel_audit_command_template(self) -> None:
        plugin_doc = json.loads((REPO_ROOT / "plugin.json").read_text(encoding="utf-8"))
        tool_spec = plugin_doc["tools"]["megakernel-audit"]
        self.assertEqual(
            tool_spec["command"],
            ["{python}", "-B", "{plugin_root}/weaveprof/aka_adapter.py", "megakernel-audit"],
        )

    def _run_golden_case_via_aka(self, case_name: str) -> None:
        case_dir = REPO_ROOT / "tests" / "golden" / "cases" / case_name
        tmp_dir = self._root / f"case_{case_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        adapter_script = REPO_ROOT / "weaveprof" / "aka_adapter.py"
        calls_seen: List[int] = []

        def _aka_forwarder(workspace: str | Path, **kwargs: Any) -> Dict[str, Any]:
            req = {"workspace": str(workspace), **{k: _jsonable(v) for k, v in kwargs.items()}}
            case_env = {
                **self._env,
                "TMPDIR": str(tmp_dir),
                "TEMP": str(tmp_dir),
                "TMP": str(tmp_dir),
            }
            proc = subprocess.Popen(
                [sys.executable, "-B", str(adapter_script), "megakernel-audit"],
                cwd=str(self._cwd),
                env=case_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(json.dumps(req, ensure_ascii=False), timeout=180.0)
            poll_rc = proc.poll()
            self.__class__.PID_RECORDS.append((proc.pid, f"aka:{case_name}", poll_rc))
            self.assertIsNotNone(poll_rc)
            self.assertEqual(proc.returncode, 0, f"aka_adapter failed ({proc.returncode}): {stderr}")
            calls_seen.append(proc.pid)
            return json.loads(stdout)

        with mock.patch("weaveprof.megakernel.run_megakernel_audit", side_effect=_aka_forwarder):
            obs = golden.run_case(case_dir, tmp_dir)

        self.assertEqual(len(calls_seen), 1)
        self.assertIs(obs["png"]["checked"], True)
        self.assertIs(obs["png"]["valid"], True)
        self.assertEqual(obs["report"]["status"], _EXPECTED_STATUS_BY_CASE[case_name])

        expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
        diff = golden._first_diff(expected, json.loads(golden._canon(obs)))
        self.assertIsNone(diff, f"AKA golden mismatch for {case_name}: {diff}")

    def test_aka_megakernel_audit_golden_h20_pingpong_v1(self) -> None:
        self._run_golden_case_via_aka("h20_pingpong_v1")

    def test_aka_megakernel_audit_golden_h20_v3_ws_tma_wgmma(self) -> None:
        self._run_golden_case_via_aka("h20_v3_ws_tma_wgmma")

    def test_aka_megakernel_audit_golden_h20_v6_persistent(self) -> None:
        self._run_golden_case_via_aka("h20_v6_persistent")


if __name__ == "__main__":
    unittest.main()

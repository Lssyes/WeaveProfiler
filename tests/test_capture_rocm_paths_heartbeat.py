#!/usr/bin/env python3
"""Offline regressions for ROCm payload paths and heartbeat terminal states.

The capture tests stop at the ``run_capture`` boundary: no collector or GPU
workload is launched.  The heartbeat tests use a fake Popen so they exercise the real
process-polling code without launching a workload or touching a GPU.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = ROOT / "3_capture"
for path in (ROOT, CAPTURE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import capture_rocm  # noqa: E402
import run as runner  # noqa: E402


class CaptureReached(Exception):
    """Sentinel raised by the fake capture runner after recording argv."""


class CaptureRocmPayloadPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-rocm-paths-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    @staticmethod
    def _collector_selection() -> dict:
        return {
            "family": capture_rocm.ROCPROF_SYS_FAMILY,
            "path": "/fake/rocprof-sys-run",
            "version": "fake-version",
            "fallback": {"used": False},
            "selection_reason": "offline path regression",
            "capabilities": {"sampling": {"mode": "event-trace"}},
            "hard_failures": [],
            "capability_probe": {"hard_failures": []},
        }

    @staticmethod
    def _idle_gpu_report() -> dict:
        return {
            "amd_smi_available": True,
            "target_indices": [0],
            "memory_threshold_mib": 1024,
            "utilization_threshold_pct": 10,
            "gpus": [{
                "index": 0,
                "memory_used_mib": 0,
                "utilization_pct": 0,
                "over_memory": False,
                "over_utilization": False,
            }],
            "foreign_processes": [],
            "shared_system_pids": [],
            "busy": False,
            "reasons": [],
        }

    def _run_capture_to_boundary(self, case: str, payload: list[str]) -> tuple[list[str], Path]:
        project = self.root / case
        project.mkdir(parents=True)
        executable = None
        for index, token in enumerate(payload):
            if token.startswith("-"):
                continue
            path = project / token
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("offline payload input\n", encoding="utf-8")
            if index == 0:
                executable = path
                path.chmod(0o755)
        self.assertIsNotNone(executable, "test payload must contain an executable")

        out_dir = project / "runs" / "capture"
        records: list[dict] = []

        def fake_run_capture(cmd, cwd, log_path, timeout, env):
            records.append({
                "cmd": list(cmd),
                "cwd": Path(cwd),
                "log_path": Path(log_path),
                "timeout": timeout,
                "env": dict(env),
            })
            Path(log_path).write_text("offline collector boundary\n", encoding="utf-8")
            raise CaptureReached

        argv = [
            "--root", str(project),
            "--out-dir", str(out_dir),
            "--amd-smi", str(project / "amd-smi"),
            "--payload", *payload,
        ]
        with mock.patch.object(
                capture_rocm, "select_collector",
                return_value=self._collector_selection()), \
             mock.patch.object(
                capture_rocm, "inspect_gpus",
                return_value=self._idle_gpu_report()), \
             mock.patch.object(
                capture_rocm, "run_capture", side_effect=fake_run_capture):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(CaptureReached):
                    capture_rocm.main(argv)

        self.assertEqual(1, len(records), "capture runner should be reached exactly once")
        return records[0]["cmd"], records[0]["cwd"]

    def test_payload_paths_with_slash_are_root_relative(self) -> None:
        project = self.root / "with-slash"
        payload = ["bin/runner", "inputs/data.bin", "plain_input.bin", "--flag"]
        cmd, cwd = self._run_capture_to_boundary("with-slash", payload)
        expected = [
            (project / "bin" / "runner").resolve(),
            (project / "inputs" / "data.bin").resolve(),
            (project / "plain_input.bin").resolve(),
            "--flag",
        ]
        self.assertEqual([str(path) for path in expected],
                         cmd[-len(expected):],
                         "relative payload paths must be resolved against --root")
        self.assertEqual(cwd, (project / "runs" / "capture").resolve(),
                         "collector must still execute in the capture output directory")

    def test_payload_paths_without_slash_are_root_relative(self) -> None:
        project = self.root / "without-slash"
        payload = ["runner", "inputs/data.bin", "plain_input.bin", "--flag"]
        cmd, cwd = self._run_capture_to_boundary("without-slash", payload)
        expected = [
            (project / "runner").resolve(),
            (project / "inputs" / "data.bin").resolve(),
            (project / "plain_input.bin").resolve(),
            "--flag",
        ]
        self.assertEqual([str(path) for path in expected],
                         cmd[-len(expected):],
                         "bare local executable and path arguments must be resolved "
                         "against --root")
        self.assertEqual(cwd, (project / "runs" / "capture").resolve(),
                         "collector must still execute in the capture output directory")


class FakeProcess:
    """Minimal Popen surface used by ``_run_observed``."""

    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.pid = 99_999_999
        self.stdout = io.StringIO("")

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        assert self.returncode is not None
        return self.returncode


class RunObservedHeartbeatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-heartbeat-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.heartbeat = self.root / "heartbeat.json"

    def _run(self, returncode: int | None, *,
             deadline: float | None = None) -> tuple[int, dict]:
        process = FakeProcess(returncode)
        with mock.patch.object(
                runner.subprocess, "Popen", return_value=process) as popen:
            rc = runner._run_observed(
                ["fake-collector", "--payload", "fake-app"],
                cwd=self.root,
                deadline=deadline,
                heartbeat=self.heartbeat,
                card="mi308x",
                phase="capture",
            )
        popen.assert_called_once()
        return rc, json.loads(self.heartbeat.read_text(encoding="utf-8"))

    def test_nonzero_exit_writes_fail_not_running(self) -> None:
        rc, heartbeat = self._run(7)
        self.assertEqual(7, rc)
        self.assertEqual("FAIL", heartbeat["status"])
        self.assertNotEqual("RUNNING", heartbeat["status"])
        self.assertEqual(7, heartbeat["exit_code"])

    def test_deadline_writes_deadline_exceeded(self) -> None:
        rc, heartbeat = self._run(None, deadline=time.time() - 1.0)
        self.assertEqual(124, rc)
        self.assertEqual("DEADLINE_EXCEEDED", heartbeat["status"])
        self.assertNotEqual("RUNNING", heartbeat["status"])
        self.assertEqual(124, heartbeat["exit_code"])

    def test_success_writes_pass(self) -> None:
        rc, heartbeat = self._run(0)
        self.assertEqual(0, rc)
        self.assertEqual("PASS", heartbeat["status"])
        self.assertEqual(0, heartbeat["exit_code"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

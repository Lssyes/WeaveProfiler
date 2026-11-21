#!/usr/bin/env python3
"""Strict NSYS launch-count and custom metric-config contracts."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "weaveprof/pipeline/capture" / "capture_nsys.py"


def _load_capture():
    spec = importlib.util.spec_from_file_location("capture_nsys_contract", CAPTURE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE_MODULE = _load_capture()


class CaptureProcessTreeTest(unittest.TestCase):
    def test_timeout_kills_nsys_descendants_and_returns_124(self) -> None:
        temp = tempfile.TemporaryDirectory(
            prefix="weaveprofiler-capture-timeout-", dir=str(ROOT / "tests"))
        self.addCleanup(temp.cleanup)
        work = Path(temp.name)
        pid_file = work / "grandchild.pid"
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "open(sys.argv[1], 'w').write(str(child.pid))\n"
            "time.sleep(30)\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            code, log_text = CAPTURE_MODULE.run_capture(
                [sys.executable, "-B", "-c", script, str(pid_file)],
                work, work / "capture.log", 1.0)
        self.assertEqual(124, code)
        self.assertIn("TIMEOUT", log_text)
        self.assertTrue(pid_file.is_file())
        grandchild = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and Path("/proc/{}".format(grandchild)).exists():
            time.sleep(0.05)
        self.assertFalse(Path("/proc/{}".format(grandchild)).exists())


class KernelProbeStrictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,
                end INTEGER,
                deviceId INTEGER,
                globalPid INTEGER,
                demangledName INTEGER
            );
            INSERT INTO StringIds(id, value) VALUES (1, 'gemm_wmma');
            """
        )
        self.addCleanup(self.conn.close)

    def add_launch(self, device: int, process: int, index: int,
                   *, complete: bool = True) -> None:
        start = 1_000 + index * 1_000
        end = start + 500 if complete else None
        self.conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, 1)",
            (start, end, device, process),
        )

    def probe(self, devices: Sequence[int], processes: int,
              expected: Optional[int] = None) -> dict:
        tables = [row[0] for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        return CAPTURE_MODULE.kernel_probe_strict(
            self.conn, tables, "gemm_wmma", list(devices), processes, expected)

    def test_default_is_one_complete_launch_per_device_process_pair(self) -> None:
        for index, device in enumerate((0, 1)):
            self.add_launch(device, 700, index)
        result = self.probe([0, 1], 1)
        self.assertTrue(result["valid"], result["hard_failures"])
        self.assertEqual(2, result["expected_total_launch_count"])
        self.assertEqual(1, result["expected_launch_count_per_pair"])

    def test_explicit_repeated_launches_are_distributed_evenly(self) -> None:
        for index in range(8):
            self.add_launch(0, 700, index)
        result = self.probe([0], 1, expected=8)
        self.assertTrue(result["valid"], result["hard_failures"])
        self.assertEqual(8, result["expected_launch_count_per_pair"])
        self.assertEqual(8, result["match_rows"])

    def test_repeated_launches_across_two_gpus_are_distributed_evenly(self) -> None:
        for index in range(8):
            self.add_launch(index % 2, 700 + index // 4, index)
        result = self.probe([0, 1], 2, expected=8)
        self.assertTrue(result["valid"], result["hard_failures"])
        self.assertEqual(4, result["expected_launch_count_per_device"])
        self.assertEqual({"0": 4, "1": 4}, result["launch_count_by_device"])

    def test_non_divisible_launch_count_fails(self) -> None:
        for index in range(7):
            self.add_launch(index % 2, 700 + index // 4, index)
        result = self.probe([0, 1], 2, expected=7)
        self.assertFalse(result["valid"])
        failures = "\n".join(result["hard_failures"])
        self.assertIn("cannot be distributed evenly across expected devices", failures)
        self.assertIn("cannot be distributed evenly across expected device/process pairs", failures)

    def test_missing_and_extra_launches_fail(self) -> None:
        for index in range(7):
            self.add_launch(0, 700, index)
        missing = self.probe([0], 1, expected=8)
        self.assertFalse(missing["valid"])
        self.assertIn("launch count mismatch", "\n".join(missing["hard_failures"]))

        self.add_launch(0, 700, 7)
        self.add_launch(0, 700, 8)
        extra = self.probe([0], 1, expected=8)
        self.assertFalse(extra["valid"])
        self.assertIn("launch count mismatch", "\n".join(extra["hard_failures"]))

    def test_incomplete_launch_fails_even_when_row_count_matches(self) -> None:
        self.add_launch(0, 700, 0, complete=False)
        result = self.probe([0], 1)
        self.assertFalse(result["valid"])
        self.assertEqual(1, len(result["incomplete_launches"]))
        self.assertIn("incomplete launch row", "\n".join(result["hard_failures"]))

    def test_missing_process_fails_even_when_total_rows_match(self) -> None:
        for index in range(4):
            self.add_launch(0, 700, index)
        result = self.probe([0], 2, expected=4)
        self.assertFalse(result["valid"])
        failures = "\n".join(result["hard_failures"])
        self.assertIn("process count mismatch", failures)
        self.assertIn("per-device/process count", failures)

    def test_launch_on_an_unexpected_device_fails(self) -> None:
        for index in range(7):
            self.add_launch(0, 700, index)
        self.add_launch(1, 700, 7)
        result = self.probe([0], 1, expected=8)
        self.assertFalse(result["valid"])
        self.assertIn("device 0", "\n".join(result["hard_failures"]))

    def test_cuda_visible_logical_device_zero_is_mapped_to_physical_device(self) -> None:
        self.add_launch(0, 700, 0)
        result = self.probe([4], 1)
        self.assertTrue(result["valid"], result["hard_failures"])
        self.assertEqual("cuda-visible-logical", result["kernel_device_id_space"])
        self.assertEqual({"0": 4}, result["kernel_device_to_physical"])


class MetricsConfigContractTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-metric-config-", dir=str(ROOT / "tests"))
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    @staticmethod
    def config(alias: str) -> str:
        return """version: 5
metricSets:
- alias: {alias}
  name: Test Metrics
  chips:
  - GH100
  metrics:
  - name: SMs Active
    id: sm__cycles_active
  - name: Tensor Active
    id: sm__pipe_tensor_cycles_active_realtime
""".format(alias=alias)

    def test_file_stem_matching_alias_passes_preflight(self) -> None:
        path = self.root / "gemm_h20.config"
        path.write_text(self.config("gemm_h20"), encoding="utf-8")
        result = CAPTURE_MODULE.inspect_metrics_config(str(path))
        self.assertTrue(result["valid"], result["hard_failures"])
        self.assertTrue(result["alias_matches_file_stem"])
        self.assertEqual("gemm_h20", result["selected_alias"])
        self.assertEqual(2, result["raw_counter_count"])

    def test_file_stem_alias_mismatch_fails_before_nsys(self) -> None:
        path = self.root / "renamed.config"
        path.write_text(self.config("gemm_h20"), encoding="utf-8")
        result = CAPTURE_MODULE.inspect_metrics_config(str(path))
        self.assertFalse(result["valid"])
        self.assertFalse(result["alias_matches_file_stem"])
        self.assertIn("filename stem must match", "\n".join(result["hard_failures"]))


class ValidateExistingLaunchContractTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(
            prefix="weaveprofiler-validate-existing-", dir=str(ROOT / "tests"))
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sqlite = self.root / "capture.sqlite"
        self.sqlite.write_bytes(b"")

    def _validate(self, manifest: dict) -> tuple[int, mock.Mock, dict]:
        manifest_path = self.root / "capture_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = self.root / "strengthened.json"
        args = CAPTURE_MODULE.parse_args([
            "--validate-existing", str(manifest_path),
            "--validation-output", str(output),
        ])
        evidence = {
            "failures": [],
            "warnings": [],
            "sqlite": {"path": str(self.sqlite)},
            "verdict": "PASS",
        }
        with mock.patch.object(
                CAPTURE_MODULE, "verify_evidence", return_value=evidence) as verify:
            rc = CAPTURE_MODULE.validate_existing_capture(args)
        return rc, verify, json.loads(output.read_text(encoding="utf-8"))

    def _base_manifest(self) -> dict:
        return {
            "evidence": {"sqlite": {"path": str(self.sqlite)}},
            "cuda_visible_devices": {"harness_physical_indices": [4]},
            "num_processes": 1,
            "frequency_hz": 200_000,
            "capture_range": "cudaProfilerApi",
            "capture_range_end": "stop",
            "flush_on_cudaprofiler_stop": True,
            "trace_fork_before_exec": True,
            "kernel_name_contains": "gemm_wmma",
        }

    def test_existing_manifest_explicit_launch_count_reaches_evidence_verifier(self) -> None:
        manifest = self._base_manifest()
        manifest["expected_launch_count"] = 8
        rc, verify, strengthened = self._validate(manifest)
        self.assertEqual(0, rc)
        self.assertEqual(8, verify.call_args.kwargs["expected_launch_count"])
        self.assertEqual(8, strengthened["expected_launch_count"])

    def test_old_manifest_recovers_launch_count_from_recorded_command(self) -> None:
        manifest = self._base_manifest()
        manifest["payload_command"] = [
            "python3", "gemm", "--expected-launch-count", "6"]
        rc, verify, strengthened = self._validate(manifest)
        self.assertEqual(0, rc)
        self.assertEqual(6, verify.call_args.kwargs["expected_launch_count"])
        self.assertEqual(6, strengthened["expected_launch_count"])


if __name__ == "__main__":
    unittest.main()

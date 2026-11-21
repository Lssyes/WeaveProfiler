#!/usr/bin/env python3
"""Offline tests for pipeline-run scoped GPU lease reuse.

These tests execute the real run_pipeline orchestration and the real whitelist
selection/evidence code.  Only subprocess execution and the nvidia-smi snapshot
are replaced, so no GPU, nsys, or rocprof installation is required.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_measurement as runner


GPU_STAGES = ("correctness", "overhead", "sw_capture", "hw_capture")


class GpuLeaseReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_obj = tempfile.TemporaryDirectory(prefix="weaveprofiler_gpu_lease_")
        self.addCleanup(self.tmp_obj.cleanup)
        self.root = Path(self.tmp_obj.name).resolve()
        self.query_lock = threading.Lock()
        self.query_calls = 0
        self.stage_outputs: list[dict] = []

    def _query(self, timeout=30.0):
        with self.query_lock:
            self.query_calls += 1
        return [
            {"index": index, "memory_used_mib": 0.0, "utilization_percent": 0.0}
            for index in range(4)
        ]

    @staticmethod
    def _flag(argv: list[str], name: str):
        for index, token in enumerate(argv):
            if token == name and index + 1 < len(argv):
                return argv[index + 1]
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
        return None

    def _fake_run_command(self, command, log_path):
        argv = list(command["argv"])
        env = dict(command.get("env") or {})
        started = runner._now()
        output = env.get("WEAVEPROFILER_TEST_OUTPUT")
        stage = env.get("WEAVEPROFILER_TEST_STAGE")
        if stage and output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stage:" + stage, encoding="utf-8")
            with self.query_lock:
                self.stage_outputs.append({
                    "stage": stage,
                    "visible": env.get("CUDA_VISIBLE_DEVICES"),
                    "started_at": started,
                })
        elif any(token.endswith("validate_preflight.py") for token in argv):
            contract = Path(self._flag(argv, "--contract")).resolve()
            out = Path(self._flag(argv, "--out")).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "status": "PASS",
                "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
            }), encoding="utf-8")
        elif any(token.endswith("validate_all.py") for token in argv):
            out = Path(self._flag(argv, "--out")).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        elif any(token.endswith("finalize_delivery.py") for token in argv):
            contract = Path(self._flag(argv, "--contract")).resolve()
            all_gates = Path(self._flag(argv, "--all-gates")).resolve()
            preflight = Path(self._flag(argv, "--preflight")).resolve()
            receipt = Path(self._flag(argv, "--receipt")).resolve()
            trace = next(Path(token) for token in argv if token.endswith("candidate.perfetto.json.gz"))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({
                "status": "PASS",
                "contract_sha256": digest(contract),
                "all_gates_sha256": digest(all_gates),
                "trace_sha256": digest(trace),
                "preflight_path": str(preflight),
                "preflight_sha256": digest(preflight),
            }), encoding="utf-8")
        else:
            raise AssertionError("unexpected command: {}".format(argv))
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("offline stub\n", encoding="utf-8")
        return {
            "argv": argv,
            "cwd": str(command["cwd"]),
            "env_overrides": dict(command.get("env_overrides") or {}),
            "started_at": started,
            "ended_at": runner._now(),
            "duration_seconds": 0.001,
            "exit_code": 0,
            "timed_out": False,
            "timeout_sec": command.get("timeout_sec"),
            "log": str(log_path),
        }

    def _make_pipeline(self, name: str, *, opt_out: bool = False,
                       parallel: bool = False) -> tuple[Path, Path, Path]:
        run_dir = self.root / name
        run_dir.mkdir(parents=True)
        candidate = run_dir / "candidate.perfetto.json.gz"
        receipt = run_dir / "delivery" / "delivery_receipt.json"
        stages = []
        for stage in runner.BUNDLE_STAGES["sw+hw"]:
            output = candidate if stage == "perfetto" else run_dir / (stage + ".out")
            spec = {
                "name": stage,
                "kind": runner.STAGE_KINDS[stage],
                "argv": [sys.executable, str(self.root / "stage_stub.py"), stage],
                "cwd": str(run_dir),
                "outputs": [str(output)],
                "env": {
                    "WEAVEPROFILER_TEST_STAGE": stage,
                    "WEAVEPROFILER_TEST_OUTPUT": str(output),
                },
            }
            if stage in GPU_STAGES:
                spec["gpus_required"] = 1
            stages.append(spec)
        execution = {
            "candidate_trace": str(candidate),
            "allowed_gpu_ids": [0, 1, 2, 3],
            "gpu_idle_policy": {
                "utilization_max_percent": 50,
                "poll_interval_sec": 0.01,
                "timeout_sec": 2,
            },
            "stages": stages,
            "finalize": {
                "args": [str(self.root / "project"), str(candidate),
                         "--receipt", str(receipt), "--information-bundle", "sw+hw"]
            },
        }
        if opt_out:
            execution["reuse_gpu_lease"] = False
        if parallel:
            execution["parallel_groups"] = [["correctness", "overhead"]]
        contract = {
            "lock": {"state": "frozen", "allow_in_place_edit": False},
            "capture": {"information_bundle": "sw+hw"},
            "execution": execution,
        }
        contract_path = run_dir / "CONTRACT.lock.json"
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        trace_processor = self.root / "trace_processor_shell"
        if not trace_processor.exists():
            trace_processor.write_text("offline", encoding="utf-8")
        return contract_path, run_dir, trace_processor

    def _run(self, contract: Path, run_dir: Path, trace_processor: Path, *, resume: bool = False):
        with mock.patch.object(runner, "_query_gpus_nvidia_smi", side_effect=self._query), \
             mock.patch.object(runner, "_run_command", side_effect=self._fake_run_command):
            return runner.run_pipeline(contract, run_dir, trace_processor,
                                       resume=resume, script_root=self.root)

    def test_sequential_gpu_stages_reuse_one_run_scoped_lease(self) -> None:
        contract, run_dir, processor = self._make_pipeline("reuse")
        state = self._run(contract, run_dir, processor)

        self.assertEqual(state["status"], "PASS")
        self.assertEqual(self.query_calls, 1, "only the first serial GPU stage may probe")
        rows = [state["stages"][stage] for stage in GPU_STAGES]
        self.assertEqual([row["bound_gpu"] for row in rows], [0, 0, 0, 0])
        lease_ids = {row["gpu_idle"]["lease"]["id"] for row in rows}
        self.assertEqual(len(lease_ids), 1)
        self.assertFalse(rows[0]["gpu_idle"]["lease"]["reused"])
        self.assertTrue(all(row["gpu_idle"]["lease"]["reused"] for row in rows[1:]))
        self.assertEqual([row["gpu_idle"]["lease"]["query_count"] for row in rows], [1, 0, 0, 0])
        self.assertEqual(rows[0]["gpu_idle"]["lease"]["acquired_by"], "correctness")
        self.assertEqual(state["gpu_lease"]["query_count"], 1)
        self.assertEqual(state["gpu_lease"]["acquisition_count"], 1)
        self.assertEqual(state["gpu_lease"]["reused_by"], ["overhead", "sw_capture", "hw_capture"])
        self.assertEqual([entry["visible"] for entry in self.stage_outputs if entry["stage"] in GPU_STAGES],
                         ["0", "0", "0", "0"])

        for index, stage in enumerate(GPU_STAGES, start=1):
            row = state["stages"][stage]
            self.assertIsInstance(row.get("gpu_idle"), dict)
            self.assertEqual(row["gpu_idle"]["chosen_gpu"], 0)
            self.assertEqual(row["gpu_idle"]["stage"], "{:02d}_{}".format(index, stage))
            self.assertTrue(Path(row["gpu_idle"]["log"]).is_file())
            evidence = run_dir / "validation" / "gpu_idle" / (row["gpu_idle"]["stage"] + ".json")
            self.assertTrue(evidence.is_file(), "missing per-stage gpu_idle evidence {}".format(evidence))
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertTrue(payload["idle"])
            self.assertEqual(payload["chosen_gpu"], 0)
            self.assertEqual(payload["stage"], row["gpu_idle"]["stage"])
        self.assertFalse(state["gpu_lease"]["held"], "CPU perfetto stage must close the GPU serial lane")

        timing = json.loads((run_dir / "TIMING.json").read_text(encoding="utf-8"))
        waits = {phase["name"]: phase["gpu_wait_seconds"] for phase in timing["phases"]}
        self.assertGreaterEqual(waits["correctness"], 0)
        self.assertEqual([waits[stage] for stage in GPU_STAGES[1:]], [0.0, 0.0, 0.0])

    def test_explicit_opt_out_keeps_per_stage_waits(self) -> None:
        contract, run_dir, processor = self._make_pipeline("opt_out", opt_out=True)
        state = self._run(contract, run_dir, processor)

        self.assertEqual(state["status"], "PASS")
        self.assertEqual(self.query_calls, len(GPU_STAGES))
        self.assertFalse(state["gpu_lease"]["enabled"])
        for stage in GPU_STAGES:
            lease = state["stages"][stage]["gpu_idle"]["lease"]
            self.assertFalse(lease["reused"])
            self.assertEqual(lease["scope"], "stage")
            self.assertEqual(lease["query_count"], 1)

    def test_new_pipeline_run_does_not_reuse_the_previous_lease(self) -> None:
        first_contract, first_run, first_processor = self._make_pipeline("run_one")
        first = self._run(first_contract, first_run, first_processor)
        second_contract, second_run, second_processor = self._make_pipeline("run_two")
        second = self._run(second_contract, second_run, second_processor)

        self.assertEqual(self.query_calls, 2)
        self.assertEqual(first["gpu_lease"]["scope"], "pipeline_run")
        self.assertEqual(second["gpu_lease"]["scope"], "pipeline_run")
        self.assertNotEqual(first["gpu_lease"]["id"], second["gpu_lease"]["id"])
        self.assertEqual(second["stages"]["correctness"]["gpu_idle"]["lease"]["acquired_by"], "correctness")
        self.assertEqual(second["stages"]["overhead"]["gpu_idle"]["lease"]["query_count"], 0)

    def test_resume_records_a_fresh_lease_for_the_first_stage_that_runs(self) -> None:
        contract, run_dir, processor = self._make_pipeline("resume")
        first = self._run(contract, run_dir, processor)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(self.query_calls, 1)

        # correctness is reusable, but overhead and its successors must rerun.
        (run_dir / "overhead.out").write_text("tampered", encoding="utf-8")
        self._run(contract, run_dir, processor, resume=True)
        state = runner._load_json(run_dir / "PIPELINE_STATE.json")

        self.assertEqual(state["stages"]["correctness"]["status"], "REUSED")
        self.assertEqual(self.query_calls, 2)
        overhead = state["stages"]["overhead"]["gpu_idle"]
        self.assertEqual(overhead["lease"]["acquired_by"], "overhead")
        self.assertFalse(overhead["lease"]["reused"])
        self.assertEqual(overhead["lease"]["query_count"], 1)
        self.assertEqual(state["stages"]["sw_capture"]["gpu_idle"]["lease"]["acquired_by"], "overhead")
        self.assertTrue(state["stages"]["sw_capture"]["gpu_idle"]["lease"]["reused"])
        evidence = run_dir / "validation" / "gpu_idle" / "02_overhead.json"
        self.assertTrue(evidence.is_file())
        self.assertEqual(json.loads(evidence.read_text(encoding="utf-8"))["lease"]["acquired_by"], "overhead")

    def test_active_parallel_group_keeps_cards_disjoint_and_disables_reuse(self) -> None:
        contract, run_dir, processor = self._make_pipeline("parallel", parallel=True)
        self._run(contract, run_dir, processor)
        state = runner._load_json(run_dir / "PIPELINE_STATE.json")

        self.assertEqual(state["status"], "PASS")
        correctness = set(state["stages"]["correctness"]["bound_gpus"])
        overhead = set(state["stages"]["overhead"]["bound_gpus"])
        self.assertEqual(len(correctness), 1)
        self.assertEqual(len(overhead), 1)
        self.assertFalse(correctness & overhead, "parallel stages must not share a card")
        self.assertTrue((correctness | overhead) <= {0, 1, 2, 3})
        self.assertFalse(state["gpu_lease"]["enabled"])
        self.assertIn("parallel_groups", state["gpu_lease"]["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

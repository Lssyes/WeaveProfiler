#!/usr/bin/env python3
"""Offline unit contracts for the packaged GEMM hardware/software probe."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "3_capture" / "gemm_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("gemm_probe_unit", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE_MODULE = _load_probe()


class ShapeAndCorrectnessTest(unittest.TestCase):
    def test_shape_requires_tile_aligned_positive_dimensions(self) -> None:
        self.assertEqual((2048, 2048, 4096), PROBE_MODULE.parse_shape("2048x2048x4096"))
        self.assertEqual((256, 512, 32), PROBE_MODULE.parse_shape("256 X 512 x 32"))
        for text in ("2048,2048,4096", "64x64x0", "60x64x16", "64x64x15"):
            with self.subTest(text=text):
                with self.assertRaises(PROBE_MODULE.ProbeError):
                    PROBE_MODULE.parse_shape(text)

    def test_correctness_requires_every_output_element_and_zero_failures(self) -> None:
        good = {
            "checked_elements": 64 * 64,
            "mismatches": 0,
            "nonfinite": 0,
            "max_abs": 0.01,
            "rel_l2": 1e-6,
            "median_us": 10.0,
        }
        checked = PROBE_MODULE.check_correctness_payload(good, (64, 64, 16), "good")
        self.assertEqual(64 * 64, checked["checked_elements"])
        self.assertEqual("recorder_detached", checked["measurement_scope"])

        for field, value in (("checked_elements", 1), ("mismatches", 1), ("nonfinite", 1)):
            bad = dict(good)
            bad[field] = value
            with self.subTest(field=field):
                with self.assertRaises(PROBE_MODULE.ProbeError):
                    PROBE_MODULE.check_correctness_payload(bad, (64, 64, 16), "bad")


class GpuSelectionTest(unittest.TestCase):
    GPU_TEXT = "\n".join((
        "0, NVIDIA H20, GPU-0, 80, 97871, 0",
        "1, NVIDIA H20, GPU-1, 4, 97871, 0",
        "2, NVIDIA H20, GPU-2, 8, 97871, 0",
    ))

    def test_auto_selects_the_idle_gpu_with_least_memory_in_use(self) -> None:
        inventory = PROBE_MODULE.parse_nvidia_smi_inventory(self.GPU_TEXT)
        selected = PROBE_MODULE.choose_idle_gpu(inventory, "auto")
        self.assertEqual(1, selected["index"])
        self.assertEqual("auto", selected["selection"])
        self.assertTrue(selected["idle"])

    def test_foreign_process_makes_gpu_busy_and_is_never_selected(self) -> None:
        apps = "GPU-0, 4242, foreign-job\n"
        inventory = PROBE_MODULE.parse_nvidia_smi_inventory(self.GPU_TEXT, apps)
        selected = PROBE_MODULE.choose_idle_gpu(inventory, "auto")
        self.assertEqual(1, selected["index"])
        with self.assertRaisesRegex(PROBE_MODULE.ProbeError, "not idle"):
            PROBE_MODULE.choose_idle_gpu(inventory, "0")

    def test_all_busy_gpus_fail_without_eviction(self) -> None:
        apps = "GPU-0, 4242, foreign-a\nGPU-1, 4243, foreign-b\nGPU-2, 4244, foreign-c\n"
        inventory = PROBE_MODULE.parse_nvidia_smi_inventory(self.GPU_TEXT, apps)
        with self.assertRaisesRegex(PROBE_MODULE.ProbeError, "no idle GPU"):
            PROBE_MODULE.choose_idle_gpu(inventory, "auto")

    def test_resource_limits_also_mark_a_gpu_non_idle(self) -> None:
        inventory = PROBE_MODULE.parse_nvidia_smi_inventory(self.GPU_TEXT)
        selected = PROBE_MODULE.choose_idle_gpu(
            inventory, "auto", memory_limit_mib=4.0)
        self.assertEqual(1, selected["index"])
        with self.assertRaisesRegex(PROBE_MODULE.ProbeError, "not idle"):
            PROBE_MODULE.choose_idle_gpu(inventory, "2", memory_limit_mib=4.0)

    def test_auto_never_uses_a_different_gpu_model_for_h20(self) -> None:
        mixed = "\n".join((
            "0, NVIDIA H100, GPU-h100, 4, 97871, 0",
            "1, NVIDIA H200, GPU-h200, 6, 97871, 0",
            "2, NVIDIA H20, GPU-h20, 8, 97871, 0",
        ))
        inventory = PROBE_MODULE.parse_nvidia_smi_inventory(mixed)
        selected = PROBE_MODULE.choose_idle_gpu(
            inventory, "auto", expected_card="h20")
        self.assertEqual(2, selected["index"])
        self.assertTrue(selected["model_matches"])
        with self.assertRaisesRegex(PROBE_MODULE.ProbeError, "expected card h20"):
            PROBE_MODULE.choose_idle_gpu(
                inventory, "0", expected_card="h20")


class BuildEvidenceTest(unittest.TestCase):
    def test_ptxas_parser_extracts_registers_and_zero_spills(self) -> None:
        text = """ptxas info    : Compiling entry function '_gemm_wmma' for 'sm_90a'
ptxas info    : Function properties for _gemm_wmma
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 78 registers, 32768 bytes smem
ptxas info    : Compiling entry function '_init_half' for 'sm_90a'
ptxas info    : Function properties for _init_half
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 0 bytes smem
"""
        result = PROBE_MODULE.parse_ptxas_log(text)
        self.assertEqual(78, result["registers_per_thread"])
        self.assertEqual(0, result["spill_stores_bytes"])
        self.assertEqual(0, result["spill_loads_bytes"])
        self.assertTrue(result["has_register_count"])
        self.assertTrue(result["has_spill_count"])
        self.assertIn("gemm_wmma", result["matched_kernel"])

    def test_ptxas_parser_fails_for_a_different_only_kernel(self) -> None:
        text = """ptxas info    : Compiling entry function '_other' for 'sm_90a'
ptxas info    : Used 10 registers
"""
        result = PROBE_MODULE.parse_ptxas_log(text)
        self.assertFalse(result["has_register_count"])
        self.assertFalse(result["has_spill_count"])

    def test_deadline_bound_cannot_exceed_remaining_budget(self) -> None:
        deadline = time.monotonic() + 2.0
        self.assertLessEqual(PROBE_MODULE.bounded_timeout(deadline, 30.0), 2.0)
        self.assertEqual(0.5, PROBE_MODULE.bounded_timeout(None, 0.5))
        with self.assertRaises(PROBE_MODULE.ProbeError):
            PROBE_MODULE.bounded_timeout(time.monotonic() - 1.0, 30.0)

    def test_register_caps_start_at_the_observed_kernel_footprint(self) -> None:
        self.assertEqual(80, PROBE_MODULE.register_cap_candidates(80)[0])
        self.assertEqual(32, PROBE_MODULE.register_cap_candidates(10)[0])
        self.assertEqual([250, 255], PROBE_MODULE.register_cap_candidates(250))
        for seed in (32, 78, 80, 255):
            caps = PROBE_MODULE.register_cap_candidates(seed)
            self.assertEqual(caps, sorted(set(caps)))
            self.assertEqual(255, caps[-1])

    def test_subprocess_timeout_kills_descendant_processes(self) -> None:
        temp = tempfile.TemporaryDirectory(
            prefix="weaveprofiler-gemm-timeout-", dir=str(ROOT / "tests"))
        self.addCleanup(temp.cleanup)
        work = Path(temp.name)
        pid_file = work / "grandchild.pid"
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "open(sys.argv[1], 'w').write(str(child.pid))\n"
            "time.sleep(30)\n"
        )
        result = PROBE_MODULE.run_subprocess(
            [sys.executable, "-B", "-c", script, str(pid_file)],
            cwd=work, env=None, timeout=1.0, log_path=work / "timeout.log")
        self.assertEqual(124, result["exit_code"])
        self.assertTrue(result["timed_out"])
        self.assertTrue(pid_file.is_file())
        grandchild = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and Path("/proc/{}".format(grandchild)).exists():
            time.sleep(0.05)
        self.assertFalse(Path("/proc/{}".format(grandchild)).exists())


class TraceImportContractTest(unittest.TestCase):
    def test_forbidden_stat_names_are_a_sql_string_list(self) -> None:
        query = PROBE_MODULE.trace_import_query()
        for name in PROBE_MODULE.FORBIDDEN_IMPORT_STATS:
            self.assertIn("'{}'".format(name), query)
        self.assertIn("name IN ('json_parser_failure'", query)
        self.assertNotIn("IN (json_parser_failure", query)


class AlignmentInfoTest(unittest.TestCase):
    def test_alignment_records_variant_evidence_without_key_error(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="weaveprofiler-gemm-align-", dir=str(ROOT / "tests"))
        self.addCleanup(temp.cleanup)
        out_dir = Path(temp.name) / "alignment"
        out_dir.mkdir()
        build = {
            "cache_key": "cache-key",
            "variants": {
                name: {
                    "registers_per_thread": 70 + index,
                    "spill_stores_bytes": 0,
                    "spill_loads_bytes": 0,
                }
                for index, name in enumerate(PROBE_MODULE.VARIANT_ORDER)
            },
        }
        benchmark = {"medians_us": {"off:detached": 1.0}, "overhead_pct": {"full:attached": 0.1}}
        software_trace = Path(temp.name) / "software.perfetto.json.gz"
        software_summary = Path(temp.name) / "software.summary.json"
        software_trace.write_bytes(b"decoded-trace")
        software_summary.write_text("{}", encoding="utf-8")
        saw_info = {}
        saw_command = {}

        def fake_run(command, **_kwargs):
            info_path = Path(command[command.index("--info-json") + 1])
            saw_info.update(json.loads(info_path.read_text(encoding="utf-8")))
            saw_command["argv"] = list(command)
            merged = out_dir / "gemm_sw_hw.perfetto.json.gz"
            merged.write_bytes(b"merged-trace")
            summary = out_dir / "gemm_sw_hw.perfetto.summary.json"
            summary.write_text(json.dumps({
                "align": {
                    "ok": True,
                    "tail_ok": True,
                    "metrics_ok": True,
                    "kernel_us": 100.0,
                    "sw_span_us": 100.1,
                }
            }), encoding="utf-8")
            return {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "log_path": str(out_dir / "align.log"),
            }

        with mock.patch.object(PROBE_MODULE, "run_subprocess", side_effect=fake_run):
            result = PROBE_MODULE.align_software_hardware(
                {"dump": "dump.bin", "sqlite": "capture.sqlite"},
                (64, 64, 16), build, benchmark, out_dir,
                software={"trace": str(software_trace), "summary": str(software_summary)})

        self.assertEqual("PASS", result["status"])
        self.assertTrue(result["software_decode_reused"])
        self.assertAlmostEqual(0.1, result["alignment_error_pct"])
        self.assertEqual(str(software_trace),
                         saw_command["argv"][saw_command["argv"].index("--decoded-trace") + 1])
        self.assertEqual(str(software_summary),
                         saw_command["argv"][saw_command["argv"].index("--decoded-summary") + 1])
        variants = saw_info["build"]["variants"]
        self.assertEqual(70, variants["off"]["registers_per_thread"])
        self.assertEqual(72, variants["full"]["registers_per_thread"])
        self.assertFalse(saw_info["delivery_evidence"])


if __name__ == "__main__":
    unittest.main()

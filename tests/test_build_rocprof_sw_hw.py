#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "weaveprof/pipeline/capture"
if str(CAPTURE) not in sys.path:
    sys.path.insert(0, str(CAPTURE))

import build_rocprof_sw_hw as builder


NAME = "trace__mfma_single__card-mi308x__bs1_sq1__fp16__grid1_warp64_ph1__sw.perfetto.json.gz"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _results_document(kernel_name: str = "mfma_kernel(float*, int)") -> dict:
    handles = {
        "GPU_UTIL": 11,
        "SIMD_UTILIZATION": 12,
        "TCC_EA0_RDREQ_DRAM_sum": 13,
        "TCC_EA0_WRREQ_DRAM_sum": 14,
        "MfmaUtil": 15,
    }
    start = 1_000_000
    return {
        "rocprofiler-sdk-tool": [{
            "agents": [{
                "id": {"handle": 7}, "name": "gfx942",
                "product_name": "AMD Instinct MI308X",
                "cu_count": 80, "wave_front_size": 64, "simd_count": 320,
                "max_engine_clk_fcompute": 1850,
            }],
            "kernel_symbols": [{
                "kernel_id": 3,
                "demangled_kernel_name": kernel_name,
            }],
            "counters": [{
                "id": {"handle": handle}, "agent_id": {"handle": 7},
                "name": name, "expression": "fixture",
            } for name, handle in handles.items()],
            "callback_records": {"counter_collection": [{
                "dispatch_data": {
                    "start_timestamp": start,
                    "end_timestamp": start + 10_000,
                    "dispatch_info": {
                        "kernel_id": 3, "agent_id": {"handle": 7},
                        "grid_size": {"x": 1}, "workgroup_size": {"x": 64},
                        "correlation_id": {"internal": 1},
                    },
                },
                "records": [{
                    "counter_id": {"handle": handles[name]}, "value": value,
                } for name, value in (
                    ("GPU_UTIL", 100.0),
                    ("SIMD_UTILIZATION", 0.10),
                    ("TCC_EA0_RDREQ_DRAM_sum", 100.0),
                    ("TCC_EA0_WRREQ_DRAM_sum", 100.0),
                    ("MfmaUtil", 42.0),
                )],
            }]},
        }],
    }


def _remove_counter_record(document: dict, name: str) -> None:
    tool = document["rocprofiler-sdk-tool"][0]
    handle = next(item["id"]["handle"] for item in tool["counters"]
                  if item["name"] == name)
    records = tool["callback_records"]["counter_collection"][0]["records"]
    records[:] = [item for item in records
                  if item["counter_id"]["handle"] != handle]


class RocprofSwHwBuilderTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="weaveprofiler-rocprof-sw-hw-")
        self.root = Path(self.td.name)
        self.results = self.root / "results.json"
        _write_json(self.results, _results_document())
        self.sw = self.root / NAME
        trace = {
            "traceEvents": [
                {"ph": "M", "name": "thread_name", "pid": 9, "tid": 0,
                 "args": {"name": "mma"}},
                {"ph": "X", "name": "mma", "pid": 9, "tid": 0,
                 "ts": 1.0, "dur": 2.0,
                 "args": {"semantic_id": "mma", "display_name": "mma"}},
            ],
            "otherData": {"target": {"card": "mi308x", "platform": "amd", "arch": "gfx942"}},
        }
        with gzip.open(self.sw, "wt", encoding="utf-8") as handle:
            json.dump(trace, handle)

    def tearDown(self):
        self.td.cleanup()

    def test_extract_and_merge_keep_matrix_core_separate_from_simd(self):
        hw = builder.extract_results(self.results, "mfma_kernel", card="mi308x", arch="gfx942")
        evidence = hw["utilization_evidence"]
        compute = evidence["compute_util"]
        self.assertEqual(compute["unit"], "percent_of_peak")
        self.assertEqual(compute["samples"], [[1_000_000, 42.0]])
        self.assertEqual(compute["source"], "rocprofv3:MfmaUtil")
        self.assertEqual(compute["native_metric"], "MfmaUtil")
        self.assertEqual(compute["compute_pipe"], "matrix_core")
        self.assertEqual(compute["timeline_label"], "compute_util [matrix_core]")
        tensor = evidence["tensor_util"]
        self.assertEqual(tensor["samples"], [[1_000_000, 42.0]])
        self.assertEqual(tensor["source"], "rocprofv3:MfmaUtil")
        self.assertEqual(tensor["native_metric"], "MfmaUtil")
        self.assertEqual(tensor["compute_pipe"], "matrix_core")
        self.assertTrue(tensor["auxiliary"])
        simd = evidence["simd_util"]
        self.assertEqual(simd["samples"], [[1_000_000, 10.0]])
        self.assertEqual(simd["source"], "rocprofv3:SIMD_UTILIZATION*100")
        self.assertEqual(simd["native_metric"], "SIMD_UTILIZATION")
        self.assertEqual(simd["compute_pipe"], "simd")
        self.assertTrue(simd["auxiliary"])
        self.assertEqual(evidence["sm_active"]["samples"], [[1_000_000, 100.0]])
        self.assertEqual(evidence["communication_util"]["evidence_status"], "unavailable")
        self.assertEqual(hw["counters"][0]["name"], "GPU_UTIL")
        mfma_counter = next(item for item in hw["counters"] if item["name"] == "MfmaUtil")
        self.assertEqual(mfma_counter["unit"], "percent")
        self.assertTrue(hw["compute_routing"]["simd_fallback_for_tensor_forbidden"])

        hw_path = self.root / "hw.json"
        _write_json(hw_path, hw)
        out, report = builder.merge_sw_hw(self.sw, hw_path, self.root / "out")
        self.assertTrue(out.name.endswith("__sw_hw.perfetto.json.gz"))
        self.assertIn("card-mi308x", out.name)
        self.assertEqual(report["software_x_events"], 1)
        with gzip.open(out, "rt", encoding="utf-8") as handle:
            trace = json.load(handle)
        counters = [event for event in trace["traceEvents"] if event.get("ph") == "C"]
        self.assertEqual({event["name"] for event in counters}, {
            "compute_util [matrix_core]", "tensor_util [matrix_core]",
            "simd_util [SIMD/CU]", "memory_util", "sm_active",
        })
        for event in counters:
            self.assertEqual(set(event["args"]), {"value"})
            self.assertIsInstance(event["args"]["value"], (int, float))
        compute_values = [event["args"]["value"] for event in counters
                          if event["name"] == "compute_util [matrix_core]"]
        tensor_values = [event["args"]["value"] for event in counters
                         if event["name"] == "tensor_util [matrix_core]"]
        simd_values = [event["args"]["value"] for event in counters
                       if event["name"] == "simd_util [SIMD/CU]"]
        self.assertEqual(compute_values, [42.0, 42.0])
        self.assertEqual(tensor_values, [42.0, 42.0])
        self.assertEqual(simd_values, [10.0, 10.0])

        merge = trace["otherData"]["hw_merge"]
        metrics = {item["semantic_id"]: item for item in merge["metrics"]}
        self.assertEqual(metrics["compute_util"]["source"], "rocprofv3:MfmaUtil")
        self.assertEqual(metrics["compute_util"]["native_metric"], "MfmaUtil")
        self.assertEqual(metrics["compute_util"]["compute_pipe"], "matrix_core")
        self.assertEqual(metrics["compute_util"]["timeline_label"],
                         "compute_util [matrix_core]")
        auxiliary = {item["semantic_id"]: item for item in merge["auxiliary_metrics"]}
        self.assertEqual(auxiliary["tensor_util"]["source"], "rocprofv3:MfmaUtil")
        self.assertEqual(auxiliary["tensor_util"]["native_metric"], "MfmaUtil")
        self.assertEqual(auxiliary["simd_util"]["native_metric"], "SIMD_UTILIZATION")
        self.assertEqual(merge["evidence_rejections"], [])
        self.assertIsNone(merge["display_policy"]["gpu_busy_tensor_mapping"])

    def test_missing_mfma_is_unavailable_but_simd_remains_available(self):
        doc = _results_document()
        _remove_counter_record(doc, "MfmaUtil")
        _write_json(self.results, doc)
        hw = builder.extract_results(self.results, "mfma_kernel")
        compute = hw["utilization_evidence"]["compute_util"]
        tensor = hw["utilization_evidence"]["tensor_util"]
        simd = hw["utilization_evidence"]["simd_util"]
        self.assertEqual(compute["evidence_status"], "unavailable")
        self.assertIsNone(compute["value"])
        self.assertNotIn("samples", compute)
        self.assertEqual(tensor["evidence_status"], "unavailable")
        self.assertIsNone(tensor["value"])
        self.assertNotIn("samples", tensor)
        self.assertEqual(simd["samples"], [[1_000_000, 10.0]])
        self.assertNotEqual(compute.get("value"), 0)
        self.assertNotEqual(tensor.get("value"), 0)

    def test_missing_simd_keeps_matrix_core_evidence(self):
        doc = _results_document()
        _remove_counter_record(doc, "SIMD_UTILIZATION")
        _write_json(self.results, doc)
        hw = builder.extract_results(self.results, "mfma_kernel")
        self.assertEqual(hw["utilization_evidence"]["compute_util"]["samples"],
                         [[1_000_000, 42.0]])
        self.assertEqual(hw["utilization_evidence"]["tensor_util"]["samples"],
                         [[1_000_000, 42.0]])
        simd = hw["utilization_evidence"]["simd_util"]
        self.assertEqual(simd["evidence_status"], "unavailable")
        self.assertIsNone(simd["value"])
        self.assertNotIn("samples", simd)

    def test_non_matrix_kernel_does_not_use_mfma_or_simd_as_tensor(self):
        doc = _results_document("elementwise_kernel(float*, int)")
        _write_json(self.results, doc)
        hw = builder.extract_results(self.results, "elementwise_kernel")
        compute = hw["utilization_evidence"]["compute_util"]
        tensor = hw["utilization_evidence"]["tensor_util"]
        self.assertEqual(compute["evidence_status"], "unavailable")
        self.assertIsNone(compute["value"])
        self.assertIn("MFMA/GEMM", compute["reason"])
        self.assertEqual(tensor["evidence_status"], "unavailable")
        self.assertIsNone(tensor["value"])
        self.assertEqual(hw["utilization_evidence"]["simd_util"]["samples"],
                         [[1_000_000, 10.0]])

    def test_merge_rejects_simd_relabeled_as_compute(self):
        hw = builder.extract_results(self.results, "mfma_kernel")
        ev = hw["utilization_evidence"]["compute_util"]
        ev["source"] = "rocprofv3:SIMD_UTILIZATION*100"
        ev["native_metric"] = "SIMD_UTILIZATION"
        ev["compute_pipe"] = "simd"
        hw_path = self.root / "hw.json"
        _write_json(hw_path, hw)
        out, report = builder.merge_sw_hw(self.sw, hw_path, self.root / "out")
        self.assertTrue(report["evidence_rejections"])
        with gzip.open(out, "rt", encoding="utf-8") as handle:
            trace = json.load(handle)
        metrics = {item["semantic_id"]: item
                   for item in trace["otherData"]["hw_merge"]["metrics"]}
        self.assertEqual(metrics["compute_util"]["evidence_status"], "unavailable")
        self.assertIsNone(metrics["compute_util"]["value"])
        names = {event["name"] for event in trace["traceEvents"]
                 if event.get("ph") == "C"}
        self.assertNotIn("compute_util [matrix_core]", names)
        self.assertIn("tensor_util [matrix_core]", names)
        auxiliary = {item["semantic_id"]: item
                     for item in trace["otherData"]["hw_merge"]["auxiliary_metrics"]}
        self.assertEqual(auxiliary["tensor_util"]["source"], "rocprofv3:MfmaUtil")

    def test_mfma_util_unit_is_percent(self):
        self.assertEqual(builder._counter_unit("MfmaUtil"), "percent")
        self.assertEqual(builder._counter_unit("SIMD_UTILIZATION"), "fraction")

    def test_rocblas_tensile_kernel_is_matrix_core_route(self):
        kernel = ("Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_"
                  "MT256x256x32_MI32x32x1_SN_LDSB1")
        _write_json(self.results, _results_document(kernel))
        hw = builder.extract_results(self.results, "Cijk_Ailk_Bljk_HHS")
        compute = hw["utilization_evidence"]["compute_util"]
        tensor = hw["utilization_evidence"]["tensor_util"]
        self.assertEqual(compute["samples"], [[1_000_000, 42.0]])
        self.assertEqual(tensor["samples"], [[1_000_000, 42.0]])
        self.assertEqual(compute["source"], "rocprofv3:MfmaUtil")
        self.assertIn("cijk_", compute["route_basis"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

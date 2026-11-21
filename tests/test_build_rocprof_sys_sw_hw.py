#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "weaveprof/pipeline/capture"
if str(CAPTURE) not in sys.path:
    sys.path.insert(0, str(CAPTURE))

import build_rocprof_sys_sw_hw as overlay


CANONICAL = ("trace__gemm_custom__card-mi308x__bs2048_sq2048__fp16__"
             "grid256_warp8_ph1__sw_hw.perfetto.json.gz")
WRONG_CARD = CANONICAL.replace("card-mi308x", "card-b300")


def _counter_rows(ts: int, *, mfma: float = 400.0, gui: float = 10.0,
                  simd: float = 0.75, gpu: float = 100.0,
                  read: float = 10.0, write: float = 5.0):
    values = {
        "GPU SQ_VALU_MFMA_BUSY_CYCLES [0]": mfma,
        "GPU GRBM_GUI_ACTIVE [0]": gui,
        "GPU SIMD_UTILIZATION [0]": simd,
        "GPU GPU_UTIL [0]": gpu,
        "GPU TCC_EA0_RDREQ_DRAM [0]": read,
        "GPU TCC_EA0_WRREQ_DRAM [0]": write,
    }
    return [
        {"track_name": track, "ts": str(ts), "value": str(value)}
        for track, value in values.items()
    ]


def _doc_with_two_dispatches():
    kernels = [
        {"ts": "5000", "dur": "1000", "name": "gemm_kernel_v1(float*, int)"},
        {"ts": "7000", "dur": "2000", "name": "gemm_kernel_v1(float*, int)"},
    ]
    counters = _counter_rows(5000) + _counter_rows(7000, mfma=160, gui=10)

    def fake_query(_tps, _proto, sql):
        return kernels if "FROM slice" in sql else counters

    with mock.patch.object(overlay, "_query_rows", side_effect=fake_query):
        return overlay.extract_dispatches(
            Path("fixture-tps"), Path("fixture.proto"), "gemm_kernel",
            hbm_peak_gbps=1000.0, hbm_request_bytes=64,
        )


def _trace_with_two_gemm_spans():
    return {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 9, "tid": 0,
             "args": {"name": "software"}},
            {"ph": "X", "name": "gemm", "pid": 9, "tid": 0,
             "ts": 10.0, "dur": 3.0,
             "args": {"semantic_id": "gemm", "display_name": "gemm"}},
            {"ph": "X", "name": "gemm", "pid": 9, "tid": 0,
             "ts": 20.0, "dur": 4.0,
             "args": {"semantic_id": "gemm", "display_name": "gemm"}},
        ],
        "otherData": {"target": {"card": "mi308x", "platform": "amd",
                                   "arch": "gfx942"}},
    }


class RocprofSysHardwareTests(unittest.TestCase):
    def test_same_timestamp_pairing_and_mfma_formula(self):
        kernels = [{"ts": "5000", "dur": "1000", "name": "gemm_kernel"}]
        # The nearby sample must lose to the exact same-timestamp sample.
        counters = _counter_rows(4998, mfma=99999.0) + _counter_rows(5000)

        def fake_query(_tps, _proto, sql):
            return kernels if "FROM slice" in sql else counters

        with mock.patch.object(overlay, "_query_rows", side_effect=fake_query):
            document = overlay.extract_dispatches(
                Path("fixture-tps"), Path("fixture.proto"), "gemm_kernel",
                cu_count=80, simd_per_cu=4, xcc_count=4,
                hbm_peak_gbps=1000.0, hbm_request_bytes=64,
            )
        dispatch = document["dispatches"][0]
        self.assertEqual(dispatch["counter_sample_ts_ns"], 5000)
        self.assertEqual(dispatch["counter_sample_offset_ns"], 0)
        # 100*400/(10*(80*4/4)) == 50 percent.
        self.assertAlmostEqual(dispatch["metrics"]["compute_util"], 50.0)
        self.assertAlmostEqual(dispatch["metrics"]["tensor_util"], 50.0)
        self.assertAlmostEqual(dispatch["metrics"]["simd_util"], 75.0)
        self.assertAlmostEqual(dispatch["metrics"]["sm_active"], 100.0)
        self.assertTrue(document["alignment"]["same_run"])
        self.assertTrue(document["alignment"]["same_dispatch"])

    def test_matrix_core_and_simd_are_separate_stable_series(self):
        document = _doc_with_two_dispatches()
        by_id = {item["semantic_id"]: item for item in document["series"]}
        self.assertEqual(by_id["compute_util"]["timeline_label"],
                         "compute_util [matrix_core]")
        self.assertEqual(by_id["tensor_util"]["timeline_label"],
                         "tensor_util [matrix_core]")
        self.assertEqual(by_id["simd_util"]["timeline_label"],
                         "simd_util [SIMD/CU]")
        self.assertEqual(by_id["compute_util"]["compute_pipe"], "matrix_core")
        self.assertEqual(by_id["simd_util"]["compute_pipe"], "simd")
        self.assertEqual(by_id["compute_util"]["point_count"], 2)
        self.assertAlmostEqual(by_id["compute_util"]["median"], 35.0)
        self.assertEqual(by_id["simd_util"]["evidence_status"],
                         "measured_or_derived")

    def test_missing_mfma_is_unavailable_not_zero(self):
        kernels = [{"ts": "5000", "dur": "1000", "name": "gemm_kernel"}]
        counters = _counter_rows(5000)
        counters = [row for row in counters
                    if "SQ_VALU_MFMA_BUSY_CYCLES" not in row["track_name"]]

        def fake_query(_tps, _proto, sql):
            return kernels if "FROM slice" in sql else counters

        with mock.patch.object(overlay, "_query_rows", side_effect=fake_query):
            document = overlay.extract_dispatches(
                Path("fixture-tps"), Path("fixture.proto"), "gemm_kernel",
                hbm_peak_gbps=1000.0,
            )
        dispatch = document["dispatches"][0]
        self.assertIsNone(dispatch["metrics"]["compute_util"])
        self.assertIsNone(dispatch["metrics"]["tensor_util"])
        self.assertFalse(document["availability"]["compute_util"])
        self.assertFalse(document["availability"]["tensor_util"])
        by_id = {item["semantic_id"]: item for item in document["series"]}
        self.assertEqual(by_id["compute_util"]["evidence_status"], "unavailable")
        self.assertEqual(by_id["compute_util"]["points"], [])
        self.assertIn("MFMA", by_id["compute_util"]["reason"])
        self.assertFalse(document["policy"]["missing_evidence_zero_filled"])

    def test_communication_is_unavailable_and_no_amd_smi_promotion(self):
        document = _doc_with_two_dispatches()
        self.assertFalse(document["availability"]["communication_util"])
        self.assertFalse(document["policy"]["amd_smi_used"])
        self.assertFalse(document["policy"]["gpu_busy_used_as_compute"])
        self.assertFalse(document["policy"]["simd_used_as_tensor"])
        self.assertNotIn("gpu_busy", {item["semantic_id"]
                                      for item in document["series"]})
        self.assertTrue(all("amd-smi" not in track
                            for track in document["native_tracks"]))
        communication = next(item for item in document["series"]
                             if item["semantic_id"] == "communication_util")
        self.assertEqual(communication["evidence_status"], "unavailable")
        self.assertEqual(communication["points"], [])
        self.assertIn("no same-dispatch XGMI/RDMA/fabric counter",
                      communication["reason"])

    def test_direct_proto_build_emits_perfetto_and_metadata(self):
        document = _doc_with_two_dispatches()
        with tempfile.TemporaryDirectory(prefix="weaveprofiler-sys-build-") as td:
            out = Path(td) / CANONICAL
            path, report = overlay.build_from_document(document, out)
            self.assertEqual(path, out)
            self.assertTrue(out.is_file())
            self.assertEqual(report["software_x_events"], 2)
            self.assertGreater(report["counter_events"], 0)
            with gzip.open(out, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)
            names = {event.get("name") for event in trace["traceEvents"]}
            self.assertIn("compute_util [matrix_core]", names)
            self.assertIn("simd_util [SIMD/CU]", names)
            self.assertIn("memory_util [HBM request estimate]", names)
            self.assertNotIn("gpu_busy [amd-smi coarse]", names)
            self.assertEqual(trace["otherData"]["target"]["card"], "mi308x")
            self.assertFalse(trace["otherData"]["hw_merge"]["policy"]["amd_smi_used"])
            self.assertEqual(
                trace["otherData"]["hw_merge"]["alignment"]["method"],
                "same-run kernel slice ts paired with exact/near-exact raw counter sample timestamp",
            )

    def test_report_is_generated_from_final_written_trace_not_stale_inputs(self):
        document = _doc_with_two_dispatches()
        stale_row = {
            "semantic_id": "compute_util",
            "dispatch_ordinal": 1,
            "hardware_start_ns": 5000,
            "hardware_dur_ns": 480001,
            "value": 15.224599,
        }
        document["hardware_rows"] = [stale_row]
        document["alignment"]["dispatch_rows"] = [stale_row]
        document["dispatches"][0]["dur_ns"] = 244720
        for metric in document["series"]:
            for point in metric["points"]:
                if point[0] == 5000:
                    point[1] = 244720
                    if metric["semantic_id"] == "compute_util":
                        point[2] = 29.557751

        with tempfile.TemporaryDirectory(prefix="weaveprofiler-sys-report-") as td:
            out = Path(td) / CANONICAL
            _, report = overlay.build_from_document(document, out)
            written_sha = hashlib.sha256(out.read_bytes()).hexdigest()
            written_bytes = out.stat().st_size
            with gzip.open(out, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)

        written_rows = trace["otherData"]["hardware_rows"]
        written_alignment = trace["otherData"]["hw_merge"]["alignment"]
        compute_row = next(
            row for row in written_rows
            if row["semantic_id"] == "compute_util" and row["dispatch_ordinal"] == 1
        )
        self.assertEqual(compute_row["hardware_dur_ns"], 244720)
        self.assertEqual(compute_row["value"], 29.557751)
        self.assertEqual(report["report_source"], "final_written_trace")
        self.assertEqual(report["trace_sha256"], written_sha)
        self.assertEqual(report["bytes"], written_bytes)
        self.assertEqual(report["trace_events"], len(trace["traceEvents"]))
        self.assertEqual(report["x_events"], sum(1 for event in trace["traceEvents"] if event.get("ph") == "X"))
        self.assertEqual(report["hardware_rows"], written_rows)
        self.assertEqual(report["hardware_row_count"], len(written_rows))
        self.assertEqual(report["alignment"], written_alignment["dispatch_rows"])
        self.assertEqual(report["alignment_quality"], written_alignment["quality"])
        self.assertEqual(
            report["source_trace"],
            trace["otherData"]["hw_merge"]["source_trace"],
        )
        self.assertEqual(
            report["alignment_quality"]["status"],
            "aligned_within_tolerance",
        )
        self.assertTrue(report["alignment_quality"]["alignment_claim"])
        self.assertTrue(report["alignment_quality"]["per_dispatch_aggregate"])
        self.assertFalse(report["alignment_quality"]["intra_kernel_curve"])

    def test_canonical_filename_rejects_wrong_card(self):
        document = _doc_with_two_dispatches()
        with tempfile.TemporaryDirectory(prefix="weaveprofiler-sys-card-") as td:
            with self.assertRaisesRegex(ValueError, "card-mi308x"):
                overlay.build_from_document(document, Path(td) / WRONG_CARD)

    def test_existing_software_trace_merges_by_dispatch_ordinal(self):
        document = _doc_with_two_dispatches()
        with tempfile.TemporaryDirectory(prefix="weaveprofiler-sys-merge-") as td:
            root = Path(td)
            sw = root / "software.perfetto.json.gz"
            with gzip.open(sw, "wt", encoding="utf-8") as handle:
                json.dump(_trace_with_two_gemm_spans(), handle)
            out, report = overlay.merge(sw, document, root / CANONICAL)
            self.assertTrue(out.is_file())
            self.assertEqual(report["software_x_events"], 2)
            self.assertEqual(len(report["alignment"]), 2)
            with gzip.open(out, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)
            names = [event.get("name") for event in trace["traceEvents"]]
            self.assertGreater(names.count("compute_util [matrix_core]"), 0)
            self.assertGreater(names.count("simd_util [SIMD/CU]"), 0)
            alignment = trace["otherData"]["hw_merge"]["alignment"]
            self.assertEqual(alignment["mode"],
                             "existing_software_trace_ordinal_mapping")
            self.assertEqual([row["dispatch_ordinal"] for row in alignment["dispatch_rows"]],
                             [1, 2])

    def test_alignment_quality_records_duration_ratio_mismatch_without_claim(self):
        document = _doc_with_two_dispatches()
        with tempfile.TemporaryDirectory(prefix="weaveprofiler-sys-ratio-") as td:
            root = Path(td)
            sw = root / "software.perfetto.json.gz"
            with gzip.open(sw, "wt", encoding="utf-8") as handle:
                json.dump(_trace_with_two_gemm_spans(), handle)
            out, report = overlay.merge(sw, document, root / CANONICAL)
            with gzip.open(out, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)

        quality = report["alignment_quality"]
        self.assertEqual(quality["status"], "duration_ratio_outside_tolerance")
        self.assertFalse(quality["alignment_claim"])
        self.assertEqual(quality["duration_ratio_outside_tolerance_count"], 2)
        ratio_evidence = [
            item for item in quality["evidence"]
            if item["kind"] == "duration_ratio_outside_tolerance"
        ]
        self.assertEqual(len(ratio_evidence), 2)
        self.assertEqual(ratio_evidence[0]["dispatch_ordinal"], 1)
        self.assertEqual(ratio_evidence[0]["software_dur_us"], 3.0)
        self.assertEqual(ratio_evidence[0]["hardware_dur_ns"], 1000)
        self.assertLess(ratio_evidence[0]["duration_ratio_hw_over_sw"], 1.0)
        self.assertFalse(report["alignment"][0]["alignment_claim"])
        self.assertTrue(report["alignment"][0]["per_dispatch_aggregate"])
        self.assertFalse(report["alignment"][0]["intra_kernel_curve"])
        self.assertEqual(
            quality,
            trace["otherData"]["hw_merge"]["alignment"]["quality"],
        )
        self.assertEqual(
            report["hardware_rows"],
            trace["otherData"]["hardware_rows"],
        )

    def test_merge_preserves_perwave_single_dispatch_and_adds_alignment_tracks(self):
        kernels = [{"ts": "5000", "dur": "5000", "name": "gemm_kernel"}]
        counters = _counter_rows(5000)

        def fake_query(_tps, _proto, sql):
            return kernels if "FROM slice" in sql else counters

        with mock.patch.object(overlay, "_query_rows", side_effect=fake_query):
            document = overlay.extract_dispatches(
                Path("fixture-tps"), Path("fixture.proto"), "gemm_kernel",
                hbm_peak_gbps=1000.0,
            )
        identities = [
            {"cta": 0, "warp": 0, "sm": 0, "xcc": 0, "se": 0, "cu": 0, "simd": 0},
            {"cta": 0, "warp": 1, "sm": 0, "xcc": 0, "se": 0, "cu": 0, "simd": 1},
        ]
        trace = {
            "traceEvents": [
                {"ph": "M", "name": "process_name", "pid": 1, "tid": 0,
                 "args": {"name": "software"}},
                {"ph": "X", "name": "load", "cat": "load", "ts": 10.0,
                 "dur": 1.0, "pid": 5, "tid": 0,
                 "args": {"semantic_id": "load", "display_name": "load", **identities[0]}},
                {"ph": "X", "name": "mma", "cat": "mma", "ts": 11.0,
                 "dur": 2.0, "pid": 5, "tid": 0,
                 "args": {"semantic_id": "mma", "display_name": "mma", **identities[0]}},
                {"ph": "X", "name": "load", "cat": "load", "ts": 12.0,
                 "dur": 1.0, "pid": 6, "tid": 1,
                 "args": {"semantic_id": "load", "display_name": "load", **identities[1]}},
                {"ph": "X", "name": "store", "cat": "store", "ts": 13.0,
                 "dur": 1.0, "pid": 6, "tid": 1,
                 "args": {"semantic_id": "store", "display_name": "store", **identities[1]}},
            ],
            "otherData": {
                "tool": "weaveprofiler",
                "observation_start_ns": 9000,
                "observation_end_ns": 14100,
                "observation_start_us": 9.0,
                "observation_end_us": 14.1,
                "observation_span_us": 5.1,
                "semantic_span_us": 4.0,
            },
        }
        with tempfile.TemporaryDirectory(prefix="weaveprofiler-sys-perwave-") as td:
            root = Path(td)
            sw = root / "perwave.perfetto.json.gz"
            with gzip.open(sw, "wt", encoding="utf-8") as handle:
                json.dump(trace, handle)
            workload = {"card_file_token": "card-mi308x", "language": "triton"}
            correctness = {"full_output_checked": True}
            benchmark = {"steady_state_ratio": 1.5}
            out, report = overlay.merge(
                sw, document, root / CANONICAL,
                workload=workload, correctness=correctness, benchmark=benchmark,
            )
            self.assertEqual(report["mode"],
                             "existing_perwave_trace_single_dispatch_start_alignment")
            self.assertEqual(report["software_x_events_preserved"], 4)
            self.assertEqual(report["added_alignment_x_events"], 2)
            self.assertEqual(report["software_semantic_counts"],
                             {"load": 2, "mma": 1, "store": 1})
            self.assertEqual(report["physical_wave_count"], 2)
            self.assertFalse(report["alignment"][0]["clock_mapping"] == "same absolute clock")
            self.assertEqual(report["alignment"][0]["software_start_us"], 9.0)
            self.assertEqual(report["alignment"][0]["software_dur_us"], 5.1)
            self.assertEqual(report["alignment"][0]["software_semantic_start_us"], 10.0)
            self.assertEqual(report["alignment"][0]["software_semantic_end_us"], 14.0)
            self.assertEqual(report["alignment"][0]["source_clock_offset_ns"], -4000)
            self.assertTrue(report["alignment"][0]["start_placement_is_integration_anchor"])
            self.assertFalse(report["alignment"][0]["start_placement_measured"])
            with gzip.open(out, "rt", encoding="utf-8") as handle:
                merged = json.load(handle)
            self.assertEqual(merged["otherData"]["tool"], "weaveprofiler")
            self.assertEqual(merged["otherData"]["workload"], workload)
            self.assertEqual(merged["otherData"]["correctness"], correctness)
            self.assertEqual(merged["otherData"]["benchmark"], benchmark)
            names = [event.get("name") for event in merged["traceEvents"]]
            self.assertEqual(names.count("gemm"), 1)
            self.assertEqual(names.count("gemm [hardware dispatch aggregate]"), 1)
            self.assertIn("compute_util [matrix_core]", names)
            self.assertIn("simd_util [SIMD/CU]", names)
            alignment = merged["otherData"]["hw_merge"]["alignment"]
            self.assertEqual(alignment["mode"],
                             "existing_perwave_trace_single_dispatch_start_alignment")
            self.assertEqual(alignment["counter_time_resolution"],
                             "per_dispatch_aggregate")
            self.assertFalse(alignment["absolute_clock_equality_claimed"])
            self.assertEqual(
                merged["otherData"]["software_semantics"]["preserved_event_count"], 4)

    def test_merge_fails_closed_on_dispatch_count_mismatch(self):
        document = _doc_with_two_dispatches()
        trace = _trace_with_two_gemm_spans()
        trace["traceEvents"] = trace["traceEvents"][:-1]
        with tempfile.TemporaryDirectory(prefix="weaveprofiler-sys-mismatch-") as td:
            root = Path(td)
            sw = root / "software.perfetto.json.gz"
            with gzip.open(sw, "wt", encoding="utf-8") as handle:
                json.dump(trace, handle)
            with self.assertRaisesRegex(ValueError, "dispatch count"):
                overlay.merge(sw, document, root / CANONICAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)

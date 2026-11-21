#!/usr/bin/env python3
"""Regression tests for the native Perfetto protobuf exporter."""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import shutil
import site
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "weaveprof/pipeline/capture"
if str(CAPTURE) not in sys.path:
    sys.path.insert(0, str(CAPTURE))

# run_all.sh uses -I, which intentionally omits the user site directory. The
# Perfetto protobuf wheel is installed there in the supported Python 3.10 env.
PERFETTO_USER_SITE = Path(site.getusersitepackages())
if PERFETTO_USER_SITE.is_dir() and str(PERFETTO_USER_SITE) not in sys.path:
    sys.path.append(str(PERFETTO_USER_SITE))

import export_perfetto_native as native  # noqa: E402
from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as pb  # noqa: E402


WAVE_COUNT = 2048
TRACE_PROCESSOR = Path(
    os.environ.get("TRACE_PROCESSOR_SHELL") or shutil.which("trace_processor_shell")
    or "/tmp/weaveprofiler_tps_probe/v47"
)
CANONICAL_OUTPUT = (
    "trace__synthetic_waves__card-h20__bs1_sq1__fp16__"
    "grid256_warp8_ph1__sw_hw.perfetto-trace.gz"
)
HARDWARE_ROWS = (
    ("compute_util", "compute_util [matrix_core]", 42.0),
    ("simd_util", "simd_util [SIMD/CU]", 37.5),
    ("memory_util", "memory_util [HBM request estimate]", 55.0),
    ("sm_active", "sm_active [GPU active]", 75.0),
)


def _document(*, card: str | None = None, card_section: str = "meta",
              wave_count: int = WAVE_COUNT, hardware: bool = True) -> dict:
    events = []
    base_us = 1000.0
    for index in range(wave_count):
        cta = index // 8
        warp = index % 8
        for offset, semantic_id in enumerate(("load", "mma", "store")):
            events.append({
                "name": "display {}".format(semantic_id.upper()),
                "ph": "X",
                "pid": 7,
                "tid": index,
                "ts": base_us + offset * 2.0,
                "dur": 2.0,
                "args": {
                    "semantic_id": semantic_id,
                    "display_name": semantic_id,
                    "original_label": "display {}".format(semantic_id.upper()),
                    "cta": cta,
                    "warp": warp,
                    "wave_slot": index,
                    "sm": index % 8,
                },
            })
    for ts, dur in ((base_us, 3.0), (base_us + 3.0, 3.0)):
        events.append({
            "name": "gemm",
            "ph": "X",
            "pid": 7,
            "tid": 0,
            "ts": ts,
            "dur": dur,
            "args": {"semantic_id": "gemm", "display_name": "gemm"},
        })
    other_data = {"platform": "nv", "arch": "sm_90"}
    if card is not None:
        other_data[card_section] = {"card": card}
    if hardware:
        other_data["hardware_rows"] = [{
            "semantic_id": semantic_id,
            "timeline_label": label,
            "value": value,
            "hardware_start_placement_us": base_us,
            "hardware_dur_us": 6.0,
            "dispatch_ordinal": 0,
            "counter_time_resolution": "per_dispatch_aggregate",
            "intra_kernel_curve": False,
        } for semantic_id, label, value in HARDWARE_ROWS]
    return {"traceEvents": events, "otherData": other_data}


def _write_document(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")


class PerfettoNativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="weaveprofiler-native-export-")
        cls.root = Path(cls.temp_dir.name)
        cls.input_path = cls.root / "synthetic-waves.perfetto.json"
        _write_document(cls.input_path, _document(card="h20"))
        cls.out_path = cls.root / CANONICAL_OUTPUT
        cls.report = native.export(cls.input_path, cls.out_path)
        with gzip.open(cls.out_path, "rb") as handle:
            cls.payload = handle.read()
        cls.trace = pb.Trace()
        cls.trace.ParseFromString(cls.payload)
        cls.packets = list(cls.trace.packet)
        cls.descriptors = [
            packet.track_descriptor for packet in cls.packets
            if packet.HasField("track_descriptor")
        ]

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_report_gzip_and_canonical_wave_contract(self):
        self.assertEqual(self.report["schema"], "weaveprofiler-perfetto-native-export/v1")
        self.assertEqual(self.report["format"], "perfetto-native-protobuf")
        self.assertTrue(self.report["compressed"])
        self.assertEqual(self.report["bytes"], self.out_path.stat().st_size)
        self.assertEqual(self.report["semantic_slices"], WAVE_COUNT * 3)
        self.assertEqual(self.report["semantic_counts"], {
            "load": WAVE_COUNT,
            "mma": WAVE_COUNT,
            "store": WAVE_COUNT,
        })
        self.assertEqual(self.report["physical_waves"], WAVE_COUNT)
        self.assertEqual(self.report["wave_tracks"], WAVE_COUNT)
        self.assertEqual(self.report["gemm_envelopes"], 1)
        self.assertEqual(self.report["source_gemm_spans"], 2)
        self.assertEqual(self.report["hardware_tracks"], len(HARDWARE_ROWS))
        self.assertEqual(
            [row["semantic_id"] for row in self.report["hardware"]],
            [semantic_id for semantic_id, _label, _value in HARDWARE_ROWS],
        )
        self.assertTrue(self.payload)

        wave_uuids = {
            descriptor.uuid for descriptor in self.descriptors
            if descriptor.name.startswith("wave ")
        }
        self.assertEqual(len(wave_uuids), WAVE_COUNT)
        canonical_names = Counter(
            packet.track_event.name for packet in self.packets
            if packet.HasField("track_event")
            and packet.track_event.type == pb.TrackEvent.TYPE_SLICE_BEGIN
            and packet.track_event.track_uuid in wave_uuids
        )
        self.assertEqual(canonical_names, Counter({
            "load": WAVE_COUNT,
            "mma": WAVE_COUNT,
            "store": WAVE_COUNT,
        }))

    def test_protobuf_descriptors_counter_tracks_and_unavailable_communication(self):
        self.assertGreaterEqual(len(self.descriptors), WAVE_COUNT)
        wave_descriptors = [
            descriptor for descriptor in self.descriptors
            if descriptor.name.startswith("wave ")
        ]
        self.assertEqual(len(wave_descriptors), WAVE_COUNT)
        self.assertEqual(len({descriptor.uuid for descriptor in wave_descriptors}), WAVE_COUNT)
        self.assertEqual(len({descriptor.name for descriptor in wave_descriptors}), WAVE_COUNT)

        counter_names = {
            descriptor.name for descriptor in self.descriptors
            if descriptor.HasField("counter")
        }
        self.assertTrue(counter_names)
        for _semantic_id, label, _value in HARDWARE_ROWS:
            self.assertIn(label, counter_names)
        self.assertIn(
            "software_load_active_waves [derived from intervals]",
            counter_names,
        )

        communication = next(
            descriptor for descriptor in self.descriptors
            if descriptor.name == "communication_util [interconnect]"
        )
        self.assertTrue(communication.HasField("counter"))
        unavailable = [
            packet.track_event for packet in self.packets
            if packet.HasField("track_event")
            and packet.track_event.track_uuid == communication.uuid
            and packet.track_event.type == pb.TrackEvent.TYPE_INSTANT
        ]
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0].name, "unavailable")

    def test_trace_processor_imports_trace_and_finds_slices(self):
        if not TRACE_PROCESSOR.is_file() or not os.access(TRACE_PROCESSOR, os.X_OK):
            self.skipTest("Trace Processor probe is unavailable: {}".format(TRACE_PROCESSOR))
        raw_trace = self.root / "synthetic-waves.perfetto-trace.pb"
        raw_trace.write_bytes(self.payload)
        query = "SELECT count(*) AS slice_count FROM slice"
        completed = subprocess.run(
            [str(TRACE_PROCESSOR), "-Q", query, str(raw_trace)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, combined[-4000:])
        lowered = combined.lower()
        for marker in ("import error", "trace import failed", "failed to import"):
            self.assertNotIn(marker, lowered)

        rows = list(csv.DictReader(io.StringIO(completed.stdout)))
        self.assertTrue(rows, "Trace Processor returned no SQL rows:\n" + combined[-4000:])
        count_key = "slice_count" if "slice_count" in rows[0] else next(iter(rows[0]))
        self.assertGreater(int(rows[0][count_key]), 0)


class PerfettoNativeCardContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="weaveprofiler-native-card-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _export(self, document: dict, filename: str) -> tuple[Path, dict]:
        trace_path = self.root / "input.perfetto.json"
        _write_document(trace_path, document)
        out_path = self.root / filename
        return out_path, native.export(trace_path, out_path)

    def test_meta_and_target_cards_require_matching_filename_token(self):
        for section in ("meta", "target"):
            for card in native.KNOWN_CARDS:
                with self.subTest(section=section, card=card):
                    out_path, _report = self._export(
                        _document(card=card, card_section=section,
                                  wave_count=1, hardware=False),
                        "trace__unit__card-{}__.perfetto-trace.gz".format(card),
                    )
                    self.assertTrue(out_path.is_file())

    def test_known_card_rejects_missing_or_mismatched_filename_token(self):
        document = _document(card="h20", wave_count=1, hardware=False)
        missing = self.root / "trace__unit__bs1.perfetto-trace.gz"
        with self.assertRaisesRegex(ValueError, "card-h20"):
            trace_path = self.root / "input.perfetto.json"
            _write_document(trace_path, document)
            native.export(trace_path, missing)
        self.assertFalse(missing.exists())

        wrong = self.root / "trace__unit__card-b300__.perfetto-trace.gz"
        with self.assertRaisesRegex(ValueError, "card-h20"):
            native.export(self.root / "input.perfetto.json", wrong)
        self.assertFalse(wrong.exists())

    def test_unknown_card_does_not_guess_or_rename_output(self):
        for section in ("meta", "target"):
            with self.subTest(section=section):
                output_name = "trace__unit__unknown-card__.perfetto-trace.gz"
                out_path, report = self._export(
                    _document(card="h100", card_section=section,
                              wave_count=1, hardware=False),
                    output_name,
                )
                self.assertEqual(out_path.name, output_name)
                self.assertEqual(Path(report["out"]).name, output_name)
                self.assertNotIn("card-h20", out_path.name)
                self.assertNotIn("card-b300", out_path.name)
                self.assertNotIn("card-mi308x", out_path.name)

    def test_conflicting_known_card_metadata_is_rejected(self):
        document = _document(card="h20", wave_count=1, hardware=False)
        document["otherData"]["target"] = {"card": "b300"}
        trace_path = self.root / "input.perfetto.json"
        _write_document(trace_path, document)
        with self.assertRaisesRegex(ValueError, "conflicting embedded card metadata"):
            native.export(trace_path, self.root / "trace__unit__card-h20__.perfetto-trace.gz")


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Pure-CPU host characterization tests for B3-G1 ROCm trace builders CLI.

Covers:
1. ``weaveprof/pipeline/capture/build_rocprof_sw_hw.py`` CLI ``main(argv)`` across
   ``extract``, ``merge``, and ``all`` subcommands, plus invalid-input ``rc=2``
   and ``[fatal]`` error diagnostics (and suspected bugs Z1 and Z2).
2. ``weaveprof/pipeline/capture/build_rocprof_sys_sw_hw.py`` ``_query_rows`` and
   CLI ``main(argv)`` (both direct-proto ``v2`` build and existing-SW ``v3`` merge,
   plus invalid-input ``rc=2`` and ``[fatal]`` diagnostics), using an executable
   fake ``trace_processor`` script in a temporary directory prepended to ``PATH``.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Process-wide zero-external-network guard (installed BEFORE weaveprof imports)
# ---------------------------------------------------------------------------
_ORIG_CONNECT = socket.socket.connect
_ORIG_CONNECT_EX = socket.socket.connect_ex
_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_GETADDRINFO = socket.getaddrinfo
_NON_LOOPBACK_ATTEMPTS: list[tuple[str, Any]] = []
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})


def _extract_host(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0]).strip()
    return str(address).strip()


def _guard_connect(self: socket.socket, address: Any) -> Any:
    host = _extract_host(address)
    if host not in _LOOPBACK_HOSTS:
        _NON_LOOPBACK_ATTEMPTS.append(("connect", address))
        raise OSError(f"External network blocked by B3 characterization guard: {address!r}")
    return _ORIG_CONNECT(self, address)


def _guard_connect_ex(self: socket.socket, address: Any) -> int:
    host = _extract_host(address)
    if host not in _LOOPBACK_HOSTS:
        _NON_LOOPBACK_ATTEMPTS.append(("connect_ex", address))
        raise OSError(f"External network blocked by B3 characterization guard: {address!r}")
    return _ORIG_CONNECT_EX(self, address)


def _guard_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
    host = _extract_host(address)
    if host not in _LOOPBACK_HOSTS:
        _NON_LOOPBACK_ATTEMPTS.append(("create_connection", address))
        raise OSError(f"External network blocked by B3 characterization guard: {address!r}")
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


def _guard_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
    host_str = "" if host is None else str(host).strip()
    if host_str not in _LOOPBACK_HOSTS:
        _NON_LOOPBACK_ATTEMPTS.append(("getaddrinfo", (host, port)))
        raise socket.gaierror(f"External DNS blocked by B3 characterization guard: {host!r}")
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


socket.socket.connect = _guard_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guard_connect_ex  # type: ignore[assignment]
socket.create_connection = _guard_create_connection  # type: ignore[assignment]
socket.getaddrinfo = _guard_getaddrinfo  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = ROOT / "weaveprof" / "pipeline" / "capture"
if str(CAPTURE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPTURE_DIR))

import build_rocprof_sw_hw as v3_builder  # noqa: E402
import build_rocprof_sys_sw_hw as sys_builder  # noqa: E402

CANONICAL_SW_NAME = (
    "trace__mfma_single__card-mi308x__bs1_sq1__fp16__grid1_warp64_ph1__sw.perfetto.json.gz"
)
CANONICAL_SW_HW_NAME = (
    "trace__mfma_single__card-mi308x__bs1_sq1__fp16__grid1_warp64_ph1__sw_hw.perfetto.json.gz"
)
CANONICAL_SYS_SW_HW_NAME = (
    "trace__gemm_custom__card-mi308x__bs2048_sq2048__fp16__grid256_warp8_ph1__sw_hw.perfetto.json.gz"
)
WRONG_CARD_SYS_SW_HW_NAME = (
    "trace__gemm_custom__card-h20__bs2048_sq2048__fp16__grid256_warp8_ph1__sw_hw.perfetto.json.gz"
)


def _run_cli(main_fn: Any, argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        rc = main_fn(argv)
    return rc, stdout_buf.getvalue(), stderr_buf.getvalue()


def _make_rocprofv3_results_doc(
    *,
    kernel_name: str = "mfma_gemm_kernel(float*, int)",
    include_second_dispatch: bool = False,
) -> dict[str, Any]:
    handles = {
        "GPU_UTIL": 11,
        "SIMD_UTILIZATION": 12,
        "TCC_EA0_RDREQ_DRAM_sum": 13,
        "TCC_EA0_WRREQ_DRAM_sum": 14,
        "MfmaUtil": 15,
    }
    dispatches = [
        {
            "dispatch_data": {
                "start_timestamp": 1_000_000,
                "end_timestamp": 1_010_000,
                "correlation_id": {"internal": 1},
                "dispatch_info": {
                    "kernel_id": 3,
                    "agent_id": {"handle": 7},
                    "grid_size": {"x": 4},
                    "workgroup_size": {"x": 64},
                },
            },
            "records": [
                {"counter_id": {"handle": 11}, "value": 80.0},
                {"counter_id": {"handle": 12}, "value": 0.25},
                {"counter_id": {"handle": 13}, "value": 60000.0},
                {"counter_id": {"handle": 14}, "value": 40000.0},
                {"counter_id": {"handle": 15}, "value": 60.0},
            ],
        }
    ]
    if include_second_dispatch:
        dispatches.append(
            {
                "dispatch_data": {
                    "start_timestamp": 1_020_000,
                    "end_timestamp": 1_030_000,
                    "correlation_id": {"internal": 2},
                    "dispatch_info": {
                        "kernel_id": 3,
                        "agent_id": {"handle": 7},
                        "grid_size": {"x": 4},
                        "workgroup_size": {"x": 64},
                    },
                },
                "records": [
                    {"counter_id": {"handle": 11}, "value": 70.0},
                    {"counter_id": {"handle": 15}, "value": 50.0},
                ],
            }
        )
    return {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {
                        "id": {"handle": 7},
                        "name": "gfx942",
                        "product_name": "AMD Instinct MI308X",
                        "cu_count": 80,
                        "wave_front_size": 64,
                        "simd_count": 320,
                        "max_engine_clk_fcompute": 1850,
                    }
                ],
                "kernel_symbols": [
                    {
                        "kernel_id": 3,
                        "demangled_kernel_name": kernel_name,
                    }
                ],
                "counters": [
                    {
                        "id": {"handle": h},
                        "agent_id": {"handle": 7},
                        "name": name,
                        "expression": "fixture_expr",
                    }
                    for name, h in handles.items()
                ],
                "callback_records": {"counter_collection": dispatches},
            }
        ]
    }


def _write_sw_trace_gz(path: Path) -> None:
    trace = {
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 0, "args": {"name": "mma"}},
            {
                "ph": "X",
                "name": "load",
                "pid": 0,
                "tid": 0,
                "ts": 10.0,
                "dur": 4.0,
                "args": {"semantic_id": "load", "display_name": "load"},
            },
            {
                "ph": "X",
                "name": "mma",
                "pid": 0,
                "tid": 0,
                "ts": 14.0,
                "dur": 6.0,
                "args": {"semantic_id": "mma", "display_name": "mma"},
            },
        ],
        "otherData": {"target": {"card": "mi308x", "platform": "amd", "arch": "gfx942"}},
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)


class _IsolatedCaptureTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = dict(os.environ)
        self.addCleanup(self._restore_env)
        self._tmp = tempfile.TemporaryDirectory(prefix="wp_b3_g1_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)

    def tearDown(self) -> None:
        self.assertEqual(_NON_LOOPBACK_ATTEMPTS, [])

    @classmethod
    def tearDownClass(cls) -> None:
        if _NON_LOOPBACK_ATTEMPTS:
            raise AssertionError(f"Non-loopback network attempts detected: {_NON_LOOPBACK_ATTEMPTS}")


class RocprofV3BuilderCliCharacterizationTests(_IsolatedCaptureTestBase):
    """Characterize ``build_rocprof_sw_hw.main(argv)`` across extract / merge / all."""

    def test_extract_cli_json_report_written_document_and_suspected_bug_Z1(self) -> None:
        results_path = self.root / "results.json"
        results_path.write_text(json.dumps(_make_rocprofv3_results_doc()), encoding="utf-8")
        hw_out = self.root / "sub" / "hw_metrics.json"

        rc, stdout_text, stderr_text = _run_cli(
            v3_builder.main,
            [
                "extract",
                "--results",
                str(results_path),
                "--kernel-name-contains",
                "mfma_gemm",
                "--out",
                str(hw_out),
                "--hbm-peak-gbps",
                "3200.0",
                "--dram-request-bytes",
                "64.0",
            ],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(stderr_text, "")
        self.assertTrue(hw_out.is_file())

        cli_report = json.loads(stdout_text)
        self.assertEqual(
            set(cli_report.keys()),
            {"out", "bytes", "launches", "counters", "metrics"},
        )
        self.assertEqual(Path(cli_report["out"]).resolve(), hw_out.resolve())
        self.assertEqual(cli_report["bytes"], hw_out.stat().st_size)
        self.assertEqual(cli_report["launches"], 1)
        self.assertEqual(
            cli_report["counters"],
            [
                "GPU_UTIL",
                "MfmaUtil",
                "SIMD_UTILIZATION",
                "TCC_EA0_RDREQ_DRAM_sum",
                "TCC_EA0_WRREQ_DRAM_sum",
            ],
        )
        # Suspected bug Z1: extract_results stores per-dispatch 'samples' without a scalar 'value'
        # key on derived entries, so extract CLI stdout prints value=None even when status='derived'.
        self.assertEqual(
            cli_report["metrics"],
            {
                "communication_util": {"value": None, "status": "unavailable"},
                "compute_util": {"value": None, "status": "derived"},
                "memory_util": {"value": None, "status": "derived"},
                "simd_util": {"value": None, "status": "derived"},
                "sm_active": {"value": None, "status": "derived"},
                "tensor_util": {"value": None, "status": "derived"},
            },
        )

        written_doc = json.loads(hw_out.read_text(encoding="utf-8"))
        self.assertEqual(written_doc["schema"], "gpu-hw-metrics/v1")
        self.assertEqual(written_doc["policy"], "weaveprofiler-rocprof-dispatch-v1")
        self.assertEqual(written_doc["platform"], "amd")
        self.assertEqual(written_doc["target_card"], "mi308x")
        self.assertEqual(written_doc["target_arch"], "gfx942")
        self.assertEqual(written_doc["device"], 0)
        self.assertEqual(written_doc["agent_handle"], 7)
        self.assertEqual(written_doc["hardware_info"]["hbm_peak_gbps"], 3200.0)
        self.assertEqual(written_doc["hardware_info"]["hbm_peak_source"], "cli")
        self.assertEqual(written_doc["hardware_info"]["dram_request_bytes_assumption"], 64.0)
        self.assertEqual(
            written_doc["compute_routing"]["matrix_core_route_tokens"],
            ["gemm", "mfma"],
        )

        ev = written_doc["utilization_evidence"]
        self.assertEqual(ev["compute_util"]["samples"], [[1_000_000, 60.0]])
        self.assertEqual(ev["tensor_util"]["samples"], [[1_000_000, 60.0]])
        self.assertEqual(ev["simd_util"]["samples"], [[1_000_000, 25.0]])
        self.assertEqual(ev["sm_active"]["samples"], [[1_000_000, 80.0]])
        # Hand-calculated: (60000 + 40000) * 64 B / 10e-6 s / 3200e9 B/s * 100 = 20.0%
        self.assertEqual(ev["memory_util"]["samples"], [[1_000_000, 20.0]])
        self.assertEqual(ev["memory_util"]["details"][0]["bytes_estimate"], 6_400_000.0)
        self.assertEqual(ev["memory_util"]["details"][0]["memory_util"], 20.0)
        self.assertEqual(ev["communication_util"]["evidence_status"], "unavailable")
        self.assertIsNone(ev["communication_util"]["value"])

    def test_merge_and_all_cli_reports_and_canonical_perfetto_gz_contents(self) -> None:
        results_path = self.root / "results.json"
        results_path.write_text(json.dumps(_make_rocprofv3_results_doc()), encoding="utf-8")
        sw_path = self.root / CANONICAL_SW_NAME
        _write_sw_trace_gz(sw_path)
        hw_path = self.root / "hw_metrics.json"

        rc_ext, _, err_ext = _run_cli(
            v3_builder.main,
            [
                "extract",
                "--results",
                str(results_path),
                "--kernel-name-contains",
                "mfma_gemm",
                "--out",
                str(hw_path),
                "--hbm-peak-gbps",
                "3200.0",
                "--dram-request-bytes",
                "64.0",
            ],
        )
        self.assertEqual(rc_ext, 0)
        self.assertEqual(err_ext, "")

        merge_dir = self.root / "merge_out"
        merge_dir.mkdir()
        rc_merge, merge_stdout, merge_stderr = _run_cli(
            v3_builder.main,
            [
                "merge",
                "--sw-trace",
                str(sw_path),
                "--hw-metrics",
                str(hw_path),
                "--out",
                str(merge_dir),
            ],
        )
        self.assertEqual(rc_merge, 0)
        self.assertEqual(merge_stderr, "")
        merge_report = json.loads(merge_stdout)

        all_dir = self.root / "all_out"
        rc_all, all_stdout, all_stderr = _run_cli(
            v3_builder.main,
            [
                "all",
                "--results",
                str(results_path),
                "--sw-trace",
                str(sw_path),
                "--kernel-name-contains",
                "mfma_gemm",
                "--out-dir",
                str(all_dir),
                "--hbm-peak-gbps",
                "3200.0",
                "--dram-request-bytes",
                "64.0",
            ],
        )
        self.assertEqual(rc_all, 0)
        self.assertEqual(all_stderr, "")
        all_report = json.loads(all_stdout)

        self.assertEqual(Path(all_report["hw_metrics"]).name, "hw_metrics.json")
        self.assertTrue((all_dir / "hw_metrics.json").is_file())

        for mode_name, rep, expected_dir in (
            ("merge", merge_report, merge_dir),
            ("all", all_report, all_dir),
        ):
            out_gz = Path(rep["out"])
            self.assertEqual(out_gz.parent.resolve(), expected_dir.resolve(), msg=mode_name)
            self.assertEqual(out_gz.name, CANONICAL_SW_HW_NAME, msg=mode_name)
            self.assertTrue(out_gz.is_file(), msg=mode_name)
            self.assertEqual(rep["bytes"], out_gz.stat().st_size, msg=mode_name)
            self.assertEqual(rep["schema"], "rocprof-sw-hw-build/v1", msg=mode_name)
            self.assertEqual(rep["selected_launch_ordinal"], 0, msg=mode_name)
            self.assertEqual(rep["software_window_ns"], [10_000, 20_000], msg=mode_name)
            self.assertEqual(rep["hardware_window_ns"], [1_000_000, 1_010_000], msg=mode_name)
            self.assertEqual(rep["software_x_events"], 2, msg=mode_name)
            self.assertEqual(rep["curve_events"], 10, msg=mode_name)
            self.assertEqual(rep["evidence_rejections"], [], msg=mode_name)
            self.assertEqual(
                rep["timeline_rebase"],
                {
                    "schema": "timeline-rebase/v1",
                    "applied": False,
                    "method": (
                        "single global translation of every numeric ts; no clamp/drop/rescale; "
                        "serialization normalized to 1e-9 us"
                    ),
                    "reason": (
                        "HW PM samples may legally precede entry_release software zero; "
                        "Perfetto requires every displayed timestamp >= 0"
                    ),
                    "offset_us": 0.0,
                    "pre_min_ts_us": 10.0,
                    "post_min_ts_us": 10.0,
                    "shifted_event_count": 0,
                    "relative_timing_preserved": True,
                    "duration_fields_unchanged": True,
                    "post_negative_count": 0,
                    "post_invalid_count": 0,
                },
                msg=mode_name,
            )
            self.assertEqual(
                rep["final_timestamp_report"],
                {
                    "timed_event_count": 12,
                    "min_ts_us": 10.0,
                    "max_ts_us": 20.0,
                    "invalid_count": 0,
                    "negative_count": 0,
                    "invalid_samples": [],
                    "negative_samples": [],
                },
                msg=mode_name,
            )
            by_metric = {m["semantic_id"]: m for m in rep["metrics"]}
            self.assertEqual(by_metric["compute_util"]["value"], 60.0)
            self.assertEqual(by_metric["compute_util"]["display_samples"], 2)
            self.assertTrue(by_metric["compute_util"]["endpoint_duplicated_for_flat_aggregate"])
            self.assertEqual(by_metric["memory_util"]["value"], 20.0)
            self.assertEqual(by_metric["sm_active"]["value"], 80.0)
            self.assertIsNone(by_metric["communication_util"]["value"])
            self.assertEqual(by_metric["communication_util"]["evidence_status"], "unavailable")
            self.assertEqual(by_metric["communication_util"]["display_samples"], 0)

            by_aux = {m["semantic_id"]: m for m in rep["auxiliary_metrics"]}
            self.assertEqual(by_aux["tensor_util"]["value"], 60.0)
            self.assertEqual(by_aux["simd_util"]["value"], 25.0)

            with gzip.open(out_gz, "rt", encoding="utf-8") as handle:
                merged_trace = json.load(handle)
            events = merged_trace["traceEvents"]
            self.assertEqual(len(events), 22, msg=mode_name)
            self.assertEqual(sum(1 for e in events if e.get("ph") == "X"), 2)
            self.assertEqual(sum(1 for e in events if e.get("ph") == "M"), 10)
            self.assertEqual(sum(1 for e in events if e.get("ph") == "C"), 10)

            canonical_events_sha = hashlib.sha256(
                json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                canonical_events_sha,
                "deb8e58d5f2b0bf7b23a4b3da1b0ae083532475d79867dccdbfb6cfb802be90d",
                msg=mode_name,
            )

            hw_merge = merged_trace["otherData"]["hw_merge"]
            self.assertEqual(hw_merge["schema"], "rocprof-dispatch-merge/v1")
            self.assertEqual(hw_merge["pid_shift_applied"], 1)
            self.assertEqual(hw_merge["hw_pid"], 0)
            self.assertTrue(hw_merge["r6_hw_pid_lt_all_software"])
            self.assertEqual(hw_merge["alignment"]["scale"], 1.0)
            self.assertEqual(
                hw_merge["display_policy"]["single_evidence_sample_endpoint_duplicated"],
                5,
            )

            c_points: dict[str, list[tuple[float, float, int]]] = {}
            for ev_item in events:
                if ev_item.get("ph") == "C":
                    c_points.setdefault(ev_item["name"], []).append(
                        (ev_item["ts"], ev_item["args"]["value"], ev_item["tid"])
                    )
            self.assertEqual(
                c_points,
                {
                    "compute_util [matrix_core]": [(10.0, 60.0, 0), (20.0, 60.0, 0)],
                    "memory_util": [(10.0, 20.0, 0), (20.0, 20.0, 0)],
                    "sm_active": [(10.0, 80.0, 0), (20.0, 80.0, 0)],
                    "tensor_util [matrix_core]": [(10.0, 60.0, 1), (20.0, 60.0, 1)],
                    "simd_util [SIMD/CU]": [(10.0, 25.0, 2), (20.0, 25.0, 2)],
                },
            )

    def test_invalid_inputs_return_rc2_with_exact_fatal_messages_and_suspected_bug_Z2(
        self,
    ) -> None:
        results_path = self.root / "results.json"
        results_path.write_text(json.dumps(_make_rocprofv3_results_doc()), encoding="utf-8")
        sw_path = self.root / CANONICAL_SW_NAME
        _write_sw_trace_gz(sw_path)
        hw_path = self.root / "hw_metrics.json"

        # 1. Missing --results file in extract -> wrapped into ValueError -> rc=2
        rc1, out1, err1 = _run_cli(
            v3_builder.main,
            [
                "extract",
                "--results",
                str(self.root / "does_not_exist.json"),
                "--kernel-name-contains",
                "mfma_gemm",
                "--out",
                str(hw_path),
            ],
        )
        self.assertEqual(rc1, 2)
        self.assertEqual(out1, "")
        self.assertTrue(
            err1.startswith("[fatal] 无法读取 rocprof results JSON:"),
            msg=err1,
        )

        # 2. Non-matching kernel needle in extract -> rc=2 + exact fatal text
        rc2, out2, err2 = _run_cli(
            v3_builder.main,
            [
                "extract",
                "--results",
                str(results_path),
                "--kernel-name-contains",
                "nonexistent_kernel",
                "--out",
                str(hw_path),
            ],
        )
        self.assertEqual(rc2, 2)
        self.assertEqual(out2, "")
        self.assertEqual(
            err2,
            "[fatal] rocprof results 中没有匹配 kernel 'nonexistent_kernel' 的 counter_collection dispatch\n",
        )

        # 3. Missing requested --counter in extract -> rc=2 + exact fatal text
        rc3, out3, err3 = _run_cli(
            v3_builder.main,
            [
                "extract",
                "--results",
                str(results_path),
                "--kernel-name-contains",
                "mfma_gemm",
                "--out",
                str(hw_path),
                "--counter",
                "SQ_MISSING_COUNTER",
            ],
        )
        self.assertEqual(rc3, 2)
        self.assertEqual(out3, "")
        self.assertEqual(err3, "[fatal] 请求的 PMC 没有采到: ['SQ_MISSING_COUNTER']\n")

        # Prepare valid hw_metrics.json for merge error checks
        rc_ok, _, _ = _run_cli(
            v3_builder.main,
            [
                "extract",
                "--results",
                str(results_path),
                "--kernel-name-contains",
                "mfma_gemm",
                "--out",
                str(hw_path),
            ],
        )
        self.assertEqual(rc_ok, 0)

        # 4. Out-of-range --launch-ordinal in merge -> rc=2
        rc4, out4, err4 = _run_cli(
            v3_builder.main,
            [
                "merge",
                "--sw-trace",
                str(sw_path),
                "--hw-metrics",
                str(hw_path),
                "--out",
                str(self.root / "bad_ord_out"),
                "--launch-ordinal",
                "5",
            ],
        )
        self.assertEqual(rc4, 2)
        self.assertEqual(out4, "")
        self.assertEqual(err4, "[fatal] --launch-ordinal 5 越界（共 1 个）\n")

        # 5. Non-AMD hardware document in merge -> rc=2
        nv_hw_path = self.root / "nv_hw.json"
        nv_doc = json.loads(hw_path.read_text(encoding="utf-8"))
        nv_doc["platform"] = "nv"
        nv_doc["target_card"] = "h20"
        nv_doc["target_arch"] = "sm_90"
        nv_doc["arch"] = "sm_90"
        nv_hw_path.write_text(json.dumps(nv_doc), encoding="utf-8")
        rc5, out5, err5 = _run_cli(
            v3_builder.main,
            [
                "merge",
                "--sw-trace",
                str(sw_path),
                "--hw-metrics",
                str(nv_hw_path),
                "--out",
                str(self.root / "nv_out"),
            ],
        )
        self.assertEqual(rc5, 2)
        self.assertEqual(out5, "")
        self.assertEqual(err5, "[fatal] 本合并器只接受 AMD/gfx942 dispatch-aggregate 证据\n")

        # 6. Ambiguous multi-dispatch results in 'all' without --launch-ordinal -> rc=2
        multi_results = self.root / "multi_results.json"
        multi_results.write_text(
            json.dumps(_make_rocprofv3_results_doc(include_second_dispatch=True)),
            encoding="utf-8",
        )
        rc6, out6, err6 = _run_cli(
            v3_builder.main,
            [
                "all",
                "--results",
                str(multi_results),
                "--sw-trace",
                str(sw_path),
                "--kernel-name-contains",
                "mfma_gemm",
                "--out-dir",
                str(self.root / "multi_out"),
            ],
        )
        self.assertEqual(rc6, 2)
        self.assertEqual(out6, "")
        self.assertEqual(err6, "[fatal] 匹配到 2 个 dispatch；请用 --launch-ordinal 明确选择\n")

        # Suspected bug Z2: main() only catches ValueError, so missing --sw-trace in merge
        # raises uncaught FileNotFoundError instead of returning rc=2.
        with self.assertRaises(FileNotFoundError):
            _run_cli(
                v3_builder.main,
                [
                    "merge",
                    "--sw-trace",
                    str(self.root / "missing_sw.perfetto.json.gz"),
                    "--hw-metrics",
                    str(hw_path),
                    "--out",
                    str(self.root / "z2_out"),
                ],
            )


class RocprofSysBuilderCliCharacterizationTests(_IsolatedCaptureTestBase):
    """Characterize ``build_rocprof_sys_sw_hw._query_rows`` and CLI ``main(argv)``."""

    def setUp(self) -> None:
        super().setUp()
        self.bin_dir = self.root / "fake_bin"
        self.bin_dir.mkdir()
        self.guard_dir = self.root / "subproc_guard"
        self.guard_dir.mkdir()
        self.guard_marker = self.root / "subproc_guard_loaded.txt"
        self.guard_violations = self.root / "subproc_net_violations.log"
        self.tps_calls_log = self.root / "tps_calls.jsonl"

        (self.guard_dir / "sitecustomize.py").write_text(
            textwrap.dedent(
                f"""\
                import os, pathlib, socket
                pathlib.Path({str(self.guard_marker)!r}).write_text("loaded\\n", encoding="utf-8")
                _LOOPBACK = {{"127.0.0.1", "::1", "localhost", ""}}
                _orig_connect = socket.socket.connect
                _orig_gai = socket.getaddrinfo
                def _log_block(kind, target):
                    with open({str(self.guard_violations)!r}, "a", encoding="utf-8") as fh:
                        fh.write(f"{{kind}}:{{target!r}}\\n")
                    raise OSError(f"Subprocess network blocked: {{kind}} {{target!r}}")
                def _c(self, addr):
                    h = str(addr[0]).strip() if isinstance(addr, tuple) and addr else str(addr).strip()
                    if h not in _LOOPBACK:
                        _log_block("connect", addr)
                    return _orig_connect(self, addr)
                def _g(host, port, *a, **kw):
                    hs = "" if host is None else str(host).strip()
                    if hs not in _LOOPBACK:
                        _log_block("getaddrinfo", (host, port))
                    return _orig_gai(host, port, *a, **kw)
                socket.socket.connect = _c
                socket.getaddrinfo = _g
                """
            ),
            encoding="utf-8",
        )

        self.fake_tps = self.bin_dir / "trace_processor"
        self._write_fake_trace_processor(fail_rc=0)

        existing_py_path = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{self.guard_dir}{os.pathsep}{existing_py_path}"
            if existing_py_path
            else str(self.guard_dir)
        )
        os.environ["PATH"] = f"{self.bin_dir}{os.pathsep}/usr/bin:/bin"

    def tearDown(self) -> None:
        if self.guard_violations.exists():
            self.assertEqual(self.guard_violations.read_text(encoding="utf-8"), "")
        super().tearDown()

    def _write_fake_trace_processor(self, *, fail_rc: int = 0) -> None:
        script = textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, sys
            args = sys.argv[1:]
            with open({str(self.tps_calls_log)!r}, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(args) + "\\n")
            if {fail_rc} != 0:
                sys.stderr.write("SQL syntax error near FROM\\n")
                sys.exit({fail_rc})
            if len(args) != 3 or args[0] != "-Q":
                sys.stderr.write(f"unexpected args: {{args!r}}\\n")
                sys.exit(64)
            sql = args[1]
            sys.stdout.write("Loading trace: 35.00%\\n")
            sys.stdout.write("Loading trace: 100.00%\\n")
            if "FROM slice" in sql:
                sys.stdout.write("ts,dur,name\\n")
                sys.stdout.write("1000,500,mfma_kernel_warmup(float*)\\n")
                sys.stdout.write("5000,2000,mfma_kernel_main(float*, int)\\n")
                sys.stdout.write("8000,0,mfma_kernel_zero_dur(float*)\\n")
            elif "FROM counter" in sql:
                sys.stdout.write("track_name,ts,value\\n")
                # Farther sample at 4997 (|4997-5000|=3) must lose to 5002 (|5002-5000|=2)
                sys.stdout.write("GPU SQ_VALU_MFMA_BUSY_CYCLES [0],4997,800\\n")
                sys.stdout.write("GPU GRBM_GUI_ACTIVE [0],4997,10\\n")
                sys.stdout.write("GPU SQ_VALU_MFMA_BUSY_CYCLES [0],5002,320\\n")
                sys.stdout.write("GPU GRBM_GUI_ACTIVE [0],5002,10\\n")
                sys.stdout.write("GPU SIMD_UTILIZATION [0],5002,0.35\\n")
                sys.stdout.write("GPU GPU_UTIL [0],5002,85\\n")
                sys.stdout.write("GPU TCC_EA0_RDREQ_DRAM [0],5002,30000\\n")
                sys.stdout.write("GPU TCC_EA0_WRREQ_DRAM [0],5002,20000\\n")
                sys.stdout.write("GPU UNTRACKED_METRIC [0],5002,999\\n")
            else:
                sys.stderr.write(f"unrecognized SQL: {{sql!r}}\\n")
                sys.exit(65)
            """
        )
        self.fake_tps.write_text(script, encoding="utf-8")
        self.fake_tps.chmod(0o755)

    def _assert_which_trace_processor_is_fake(self) -> str:
        resolved = shutil.which("trace_processor")
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(Path(resolved).resolve(), self.fake_tps.resolve())
        return resolved

    def test_query_rows_filters_loading_banner_quotes_sql_and_raises_on_nonzero_rc(
        self,
    ) -> None:
        tps_bin = self._assert_which_trace_processor_is_fake()
        proto_path = self.root / "sample.proto"
        proto_path.write_bytes(b"fake-perfetto-proto")

        rows = sys_builder._query_rows(
            Path(tps_bin),
            proto_path,
            "SELECT ts, dur, name FROM slice WHERE name LIKE '%mfma_kernel%'   ",
        )
        self.assertTrue(self.guard_marker.is_file())
        self.assertEqual(len(rows), 3)
        self.assertEqual(sys_builder._sql_quote("a'b'c"), "a''b''c")

        logged_calls = [
            json.loads(line)
            for line in self.tps_calls_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(logged_calls), 1)
        self.assertEqual(logged_calls[0][0], "-Q")
        self.assertTrue(logged_calls[0][1].endswith(";\n") or logged_calls[0][1].endswith("\n"))
        self.assertEqual(Path(logged_calls[0][2]).resolve(), proto_path.resolve())

        self._write_fake_trace_processor(fail_rc=7)
        with self.assertRaises(ValueError) as ctx:
            sys_builder._query_rows(Path(tps_bin), proto_path, "SELECT * FROM slice;")
        self.assertIn("trace_processor failed rc=7:", str(ctx.exception))
        self.assertIn("SQL syntax error near FROM", str(ctx.exception))

    def test_cli_main_direct_proto_build_and_existing_sw_merge_with_fake_trace_processor(
        self,
    ) -> None:
        tps_bin = self._assert_which_trace_processor_is_fake()
        proto_path = self.root / "capture.proto"
        proto_path.write_bytes(b"fake-rocprof-sys-proto")
        extracted_path = self.root / "extracted" / "dispatches.json"
        workload_path = self.root / "workload.json"
        workload_path.write_text(json.dumps({"m": 2048, "n": 2048, "k": 2048}), encoding="utf-8")
        direct_out = self.root / CANONICAL_SYS_SW_HW_NAME

        # 1. Direct proto build (no --trace -> schema rocprof-sys-sw-hw-build/v2)
        rc_v2, out_v2, err_v2 = _run_cli(
            sys_builder.main,
            [
                "--rocprof-sys-proto",
                str(proto_path),
                "--trace-processor",
                tps_bin,
                "--out",
                str(direct_out),
                "--extracted-json",
                str(extracted_path),
                "--workload-json",
                str(workload_path),
                "--kernel-name-contains",
                "mfma_kernel",
                "--skip-first",
                "1",
                "--hbm-peak-gbps",
                "3200.0",
                "--hbm-request-bytes",
                "64",
            ],
        )
        self.assertEqual(rc_v2, 0)
        self.assertEqual(err_v2, "")
        self.assertTrue(self.guard_marker.is_file())
        self.assertTrue(direct_out.is_file())
        self.assertTrue(extracted_path.is_file())

        extracted_doc = json.loads(extracted_path.read_text(encoding="utf-8"))
        self.assertEqual(extracted_doc["schema"], "rocprof-sys-sw-hw/v2")
        self.assertEqual(extracted_doc["ignored_counter_rows"], 1)
        self.assertEqual(extracted_doc["kernel_selection"]["matching_slice_count"], 2)
        self.assertEqual(extracted_doc["kernel_selection"]["selected_slice_count"], 1)
        self.assertEqual(
            extracted_doc["kernel_selection"]["method"],
            "all_matches_after_skip_first",
        )
        disp = extracted_doc["dispatches"][0]
        self.assertEqual(disp["start_ns"], 5000)
        self.assertEqual(disp["dur_ns"], 2000)
        self.assertEqual(disp["counter_sample_ts_ns"], 5002)
        self.assertEqual(disp["counter_sample_offset_ns"], 2)
        # Hand-calculated derivations from raw CSV:
        # compute/tensor = 100 * 320 / (10 * (80*4/4)) = 40.0
        # simd = 0.35 * 100 = 35.0
        # sm_active = 85.0
        # mem_read = 100 * (30000*64*1e9/2000) / 3200e9 = 30.0
        # mem_write = 100 * (20000*64*1e9/2000) / 3200e9 = 20.0
        # mem_total = 30.0 + 20.0 = 50.0
        self.assertEqual(
            disp["metrics"],
            {
                "compute_util": 40.0,
                "tensor_util": 40.0,
                "simd_util": 35.0,
                "communication_util": None,
                "memory_util": 50.0,
                "memory_read_util": 30.0,
                "memory_write_util": 20.0,
                "sm_active": 85.0,
            },
        )

        rep_v2 = json.loads(out_v2)
        self.assertEqual(rep_v2["schema"], "rocprof-sys-sw-hw-build/v2")
        self.assertEqual(rep_v2["report_source"], "final_written_trace")
        self.assertEqual(rep_v2["mode"], "direct_proto_dispatch_timestamp_alignment")
        self.assertEqual(rep_v2["software_x_events"], 1)
        self.assertEqual(rep_v2["x_events"], 2)
        self.assertEqual(rep_v2["counter_events"], 21)
        self.assertEqual(rep_v2["hardware_row_count"], 7)
        self.assertEqual(
            rep_v2["trace_sha256"],
            hashlib.sha256(direct_out.read_bytes()).hexdigest(),
        )
        self.assertEqual(rep_v2["bytes"], direct_out.stat().st_size)
        self.assertEqual(rep_v2["alignment_quality"]["status"], "aligned_within_tolerance")
        self.assertTrue(rep_v2["alignment_quality"]["alignment_claim"])

        with gzip.open(direct_out, "rt", encoding="utf-8") as handle:
            direct_trace = json.load(handle)
        self.assertEqual(direct_trace["otherData"]["workload"], {"m": 2048, "n": 2048, "k": 2048})
        self.assertEqual(len(direct_trace["traceEvents"]), rep_v2["trace_events"])

        # 2. Existing SW trace merge (--trace supplied -> schema rocprof-sys-sw-hw-build/v3)
        # Software gemm dur=4.0 us vs hardware dur=2000 ns (2.0 us) -> ratio 0.5 (outside 1.0+-0.20)
        sw_trace_path = self.root / "existing_sw.perfetto.json.gz"
        with gzip.open(sw_trace_path, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "ph": "M",
                            "name": "process_sort_index",
                            "pid": 5,
                            "tid": 0,
                            "args": {"sort_index": 5},
                        },
                        {
                            "ph": "X",
                            "name": "gemm",
                            "pid": 5,
                            "tid": 0,
                            "ts": 20.0,
                            "dur": 4.0,
                            "args": {"semantic_id": "gemm", "display_name": "gemm"},
                        },
                    ],
                    "otherData": {
                        "target": {"card": "mi308x", "platform": "amd", "arch": "gfx942"}
                    },
                },
                handle,
            )
        merged_out = self.root / "merged_v3" / CANONICAL_SYS_SW_HW_NAME
        rc_v3, out_v3, err_v3 = _run_cli(
            sys_builder.main,
            [
                "--rocprof-sys-proto",
                str(proto_path),
                "--trace-processor",
                tps_bin,
                "--trace",
                str(sw_trace_path),
                "--out",
                str(merged_out),
                "--kernel-name-contains",
                "mfma_kernel",
                "--skip-first",
                "1",
                "--hbm-peak-gbps",
                "3200.0",
                "--hbm-request-bytes",
                "64",
            ],
        )
        self.assertEqual(rc_v3, 0)
        self.assertEqual(err_v3, "")
        rep_v3 = json.loads(out_v3)
        self.assertEqual(rep_v3["schema"], "rocprof-sys-sw-hw-build/v3")
        self.assertEqual(rep_v3["mode"], "existing_software_trace_ordinal_mapping")
        self.assertEqual(rep_v3["software_x_events"], 1)
        self.assertEqual(rep_v3["x_events"], 1)
        self.assertEqual(rep_v3["counter_events"], 21)
        self.assertEqual(rep_v3["hardware_row_count"], 7)
        self.assertEqual(
            rep_v3["alignment_quality"]["status"],
            "duration_ratio_outside_tolerance",
        )
        self.assertFalse(rep_v3["alignment_quality"]["alignment_claim"])
        self.assertEqual(rep_v3["alignment_quality"]["duration_ratio_outside_tolerance_count"], 1)
        self.assertEqual(rep_v3["alignment_quality"]["duration_ratio_hw_over_sw_median"], 0.5)
        self.assertEqual(
            rep_v3["trace_sha256"],
            hashlib.sha256(merged_out.read_bytes()).hexdigest(),
        )
        with gzip.open(merged_out, "rt", encoding="utf-8") as handle:
            merged_v3_trace = json.load(handle)
        sw_x = [e for e in merged_v3_trace["traceEvents"] if e.get("ph") == "X"]
        self.assertEqual(len(sw_x), 1)
        self.assertEqual(sw_x[0]["pid"], 105)

    def test_cli_main_invalid_inputs_return_rc2_with_exact_fatal_messages(self) -> None:
        tps_bin = self._assert_which_trace_processor_is_fake()
        proto_path = self.root / "capture.proto"
        proto_path.write_bytes(b"fake-rocprof-sys-proto")

        # 1. Empty --kernel-name-contains -> rc=2
        rc1, out1, err1 = _run_cli(
            sys_builder.main,
            [
                "--rocprof-sys-proto",
                str(proto_path),
                "--trace-processor",
                tps_bin,
                "--out",
                str(self.root / "err1.perfetto.json.gz"),
                "--kernel-name-contains",
                "   ",
            ],
        )
        self.assertEqual(rc1, 2)
        self.assertEqual(out1, "")
        self.assertEqual(err1, "[fatal] --kernel-name-contains must not be empty\n")

        # 2. Canonical output name with non-mi308x card -> rc=2
        rc2, out2, err2 = _run_cli(
            sys_builder.main,
            [
                "--rocprof-sys-proto",
                str(proto_path),
                "--trace-processor",
                tps_bin,
                "--out",
                str(self.root / WRONG_CARD_SYS_SW_HW_NAME),
                "--kernel-name-contains",
                "mfma_kernel",
                "--skip-first",
                "1",
            ],
        )
        self.assertEqual(rc2, 2)
        self.assertEqual(out2, "")
        self.assertEqual(
            err2,
            "[fatal] output must use card-mi308x and __sw_hw.perfetto.json.gz\n",
        )

        # 3. Missing --trace file -> caught by except (OSError, ValueError, JSONDecodeError) -> rc=2
        rc3, out3, err3 = _run_cli(
            sys_builder.main,
            [
                "--rocprof-sys-proto",
                str(proto_path),
                "--trace-processor",
                tps_bin,
                "--trace",
                str(self.root / "no_such_sw.perfetto.json.gz"),
                "--out",
                str(self.root / "err3.perfetto.json.gz"),
                "--kernel-name-contains",
                "mfma_kernel",
                "--skip-first",
                "1",
            ],
        )
        self.assertEqual(rc3, 2)
        self.assertEqual(out3, "")
        self.assertTrue(err3.startswith("[fatal] "), msg=err3)

        # 4. Non-zero trace_processor exit -> rc=2
        self._write_fake_trace_processor(fail_rc=9)
        rc4, out4, err4 = _run_cli(
            sys_builder.main,
            [
                "--rocprof-sys-proto",
                str(proto_path),
                "--trace-processor",
                tps_bin,
                "--out",
                str(self.root / "err4.perfetto.json.gz"),
                "--kernel-name-contains",
                "mfma_kernel",
            ],
        )
        self.assertEqual(rc4, 2)
        self.assertEqual(out4, "")
        self.assertTrue(
            err4.startswith("[fatal] trace_processor failed rc=9:"),
            msg=err4,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

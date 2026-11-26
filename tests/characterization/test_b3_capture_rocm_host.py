#!/usr/bin/env python3
"""Pure-CPU host characterization tests for B3-G2 ``capture_rocm.py``.

Covers:
1. ``inspect_gpus`` & ``shared_system_pids``: 8-GPU fake ``amd-smi`` where all
   8 GPUs share a co-resident host system daemon PID (exempted) and GPU 3 holds
   a foreign process PID (flagged busy without killing), plus suspected bug R1.
2. ``sampler_stats``: hand-calculated median/min/max/mean interval and measured
   Hz from synthetic ``amdsmi_samples.jsonl`` timestamps and error rows.
3. ``counter_probe_json``: missing/empty/partial PMC failure modes and suspected
   bug R2 (handle-only ``rocprofv3`` records yielding empty ``counter_names``).
4. ``--validate-existing``: ``rc=0`` PASS (including ``--validation-output``)
   and ``rc=4`` FAIL paths on synthetic existing capture directories, plus
   suspected bug R3.
5. ``--selftest`` and ``--dry-run``: intercepted via fake ``rocprof-sys``,
   ``rocprofv3``, and ``amd-smi`` scripts on ``PATH``, verifying zero GPU touch
   and zero capture directory/file creation.
"""
from __future__ import annotations

import contextlib
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

import capture_rocm  # noqa: E402

# Hardcoded CLI help flag literals (never read from tested module constants per Req #8)
_FAKE_ROCPROF_SYS_HELP = "\n".join(
    (
        "Usage: rocprof-sys [options] -- <app>",
        "  --trace",
        "  --use-rocm",
        "  --rocm-domains",
        "  --rocm-events",
        "  --output",
        "  --perfetto-backend",
        "  --trace-buffer-size",
        "  --perfetto-shmem-size-hint-kb",
        "  --help",
        "  --version",
    )
)
_FAKE_ROCPROFV3_HELP = "\n".join(
    (
        "Usage: rocprofv3 [options] -- <app>",
        "  --kernel-trace",
        "  --hip-trace",
        "  --memory-copy-trace",
        "  --output-format",
        "  -d, --output-directory",
        "  -o, --output-file",
        "  --pmc",
        "  --log-level",
        "  --stats",
        "  --kernel-include-regex",
        "  --perfetto-buffer-size",
        "  --help",
    )
)

_SYSTEM_DAEMON_PID = 424_242
_FOREIGN_JOB_PID = 999_991


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        rc = capture_rocm.main(argv)
    return rc, stdout_buf.getvalue(), stderr_buf.getvalue()


def _write_sampler_fixture(path: Path) -> None:
    # 5 valid samples -> 4 intervals: 0.25, 0.25, 0.25, 0.75
    # sorted deltas = [0.25, 0.25, 0.25, 0.75]
    # median = 0.25, min = 0.25, max = 0.75, mean = 1.5 / 4 = 0.375
    # measured_hz = 1.0 / 0.25 = 4.0
    # plus 1 failed sample (metric=None) at t_mono=100.125 and 1 blank line.
    rows = [
        {"t_mono": 100.00, "t_utc": "2026-11-25T00:00:00Z", "gpu": 0, "metric": {"index": 0, "utilization_pct": 12.0}},
        {"t_mono": 100.125, "t_utc": "2026-11-25T00:00:00Z", "gpu": 0, "metric": None},
        {"t_mono": 100.25, "t_utc": "2026-11-25T00:00:00Z", "gpu": 0, "metric": {"index": 0, "utilization_pct": 40.0}},
        {"t_mono": 100.50, "t_utc": "2026-11-25T00:00:00Z", "gpu": 0, "metric": {"index": 0, "utilization_pct": 55.0}},
        {"t_mono": 100.75, "t_utc": "2026-11-25T00:00:01Z", "gpu": 0, "metric": {"index": 0, "utilization_pct": 60.0}},
        {"t_mono": 101.50, "t_utc": "2026-11-25T00:00:01Z", "gpu": 0, "metric": {"index": 0, "utilization_pct": 30.0}},
    ]
    text = "\n".join(json.dumps(r) for r in rows[:3]) + "\n\n" + "\n".join(
        json.dumps(r) for r in rows[3:]
    ) + "\n"
    path.write_text(text, encoding="utf-8")


class CaptureRocmHostCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = dict(os.environ)
        self.addCleanup(self._restore_env)
        self._tmp = tempfile.TemporaryDirectory(prefix="wp_b3_g2_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self.bin_dir = self.root / "fake_bin"
        self.bin_dir.mkdir()
        self.guard_dir = self.root / "subproc_guard"
        self.guard_dir.mkdir()
        self.guard_marker = self.root / "subproc_guard_loaded.txt"
        self.guard_violations = self.root / "subproc_net_violations.log"

        self.amdsmi_log = self.root / "amdsmi_calls.jsonl"
        self.rocprofsys_log = self.root / "rocprofsys_calls.jsonl"
        self.rocprofv3_log = self.root / "rocprofv3_calls.jsonl"

        (self.guard_dir / "sitecustomize.py").write_text(
            textwrap.dedent(
                f"""\
                import pathlib, socket
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

        self.fake_amdsmi = self._write_fake_amdsmi(self.bin_dir / "amd-smi")
        self.fake_rocprofsys = self._write_fake_collector(
            self.bin_dir / "rocprof-sys",
            version_line="rocprof-sys 1.3.0",
            help_text=_FAKE_ROCPROF_SYS_HELP,
            call_log=self.rocprofsys_log,
        )
        self.fake_rocprofv3 = self._write_fake_collector(
            self.bin_dir / "rocprofv3",
            version_line="ROCProfiler-SDK v7.2.0",
            help_text=_FAKE_ROCPROFV3_HELP,
            call_log=self.rocprofv3_log,
        )

        for env_key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(env_key, None)

        existing_py_path = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{self.guard_dir}{os.pathsep}{existing_py_path}"
            if existing_py_path
            else str(self.guard_dir)
        )
        os.environ["PATH"] = f"{self.bin_dir}{os.pathsep}/usr/bin:/bin"

        # Guarantee chosen PIDs never collide with current test runner or parent PID
        self.assertNotIn(_SYSTEM_DAEMON_PID, {os.getpid(), os.getppid()})
        self.assertNotIn(_FOREIGN_JOB_PID, {os.getpid(), os.getppid()})

    def _restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)

    def tearDown(self) -> None:
        if self.guard_violations.exists():
            self.assertEqual(self.guard_violations.read_text(encoding="utf-8"), "")
        self.assertEqual(_NON_LOOPBACK_ATTEMPTS, [])

    @classmethod
    def tearDownClass(cls) -> None:
        if _NON_LOOPBACK_ATTEMPTS:
            raise AssertionError(f"Non-loopback network attempts detected: {_NON_LOOPBACK_ATTEMPTS}")

    def _write_fake_amdsmi(self, path: Path) -> Path:
        script = textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, sys
            args = sys.argv[1:]
            with open({str(self.amdsmi_log)!r}, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(args) + "\\n")
            if args == ["list", "--json"]:
                out = [
                    {{"gpu": idx, "uuid": f"GPU-MI308X-{{idx}}", "bdf": f"0000:0{{idx + 1}}:00.0"}}
                    for idx in range(8)
                ]
                print(json.dumps(out))
                sys.exit(0)
            if args == ["process", "--json"]:
                rows = []
                for idx in range(8):
                    plist = [
                        {{
                            "process_info": {{
                                "pid": {_SYSTEM_DAEMON_PID},
                                "name": "kfd_sys_daemon",
                                "memory_usage": {{"vram_mem": {{"value": 64, "unit": "MB"}}}},
                            }}
                        }}
                    ]
                    if idx == 3:
                        plist.append(
                            {{
                                "process_info": {{
                                    "pid": {_FOREIGN_JOB_PID},
                                    "name": "foreign_train_job",
                                    "memory_usage": {{"vram_mem": {{"value": 8192, "unit": "MB"}}}},
                                }}
                            }}
                        )
                    rows.append({{"gpu": idx, "process_list": plist}})
                print(json.dumps(rows))
                sys.exit(0)
            if len(args) == 4 and args[0] == "metric" and args[1] == "-g" and args[3] == "--json":
                idx = int(args[2])
                if idx == 0:
                    gfx_act = {{"value": 0, "unit": "%"}}
                elif idx == 1:
                    gfx_act = {{"value": 4, "unit": "%"}}
                elif idx == 3:
                    gfx_act = {{"value": 5, "unit": "%"}}
                else:
                    gfx_act = "N/A"
                out = [
                    {{
                        "gpu": idx,
                        "mem_usage": {{
                            "used_vram": {{"value": 128, "unit": "MB"}},
                            "total_vram": {{"value": 196608, "unit": "MB"}},
                        }},
                        "usage": {{"gfx_activity": gfx_act}},
                    }}
                ]
                print(json.dumps(out))
                sys.exit(0)
            sys.stderr.write(f"unexpected amd-smi args: {{args!r}}\\n")
            sys.exit(2)
            """
        )
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
        return path

    def _write_fake_collector(
        self,
        path: Path,
        *,
        version_line: str,
        help_text: str,
        call_log: Path,
    ) -> Path:
        script = textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, sys
            args = sys.argv[1:]
            with open({str(call_log)!r}, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(args) + "\\n")
            if args == ["--version"]:
                print({version_line!r})
                sys.exit(0)
            if args == ["--help"]:
                print({help_text!r})
                sys.exit(0)
            sys.stderr.write(f"collector execution blocked in host-only test: {{args!r}}\\n")
            sys.exit(99)
            """
        )
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
        return path

    def _assert_which_binaries_are_fakes(self) -> tuple[str, str, str]:
        amdsmi = shutil.which("amd-smi")
        rocprofsys = shutil.which("rocprof-sys")
        rocprofv3 = shutil.which("rocprofv3")
        self.assertIsNotNone(amdsmi)
        self.assertIsNotNone(rocprofsys)
        self.assertIsNotNone(rocprofv3)
        assert amdsmi is not None and rocprofsys is not None and rocprofv3 is not None
        self.assertEqual(Path(amdsmi).resolve(), self.fake_amdsmi.resolve())
        self.assertEqual(Path(rocprofsys).resolve(), self.fake_rocprofsys.resolve())
        self.assertEqual(Path(rocprofv3).resolve(), self.fake_rocprofv3.resolve())
        self.assertIsNone(shutil.which("rocprof-sys-run"))
        self.assertIsNone(shutil.which("rocprof-sys-record"))
        return amdsmi, rocprofsys, rocprofv3

    def _make_existing_capture_dir(self, name: str = "existing_cap") -> Path:
        cap_dir = self.root / name
        cap_dir.mkdir(parents=True, exist_ok=True)
        pftrace_path = cap_dir / "capture_results.pftrace"
        pftrace_path.write_bytes(b"PERFETTO_FAKE_TRACE_BYTES")
        results_path = cap_dir / "capture_results.json"
        results_path.write_text(
            json.dumps(
                {
                    "rocprofiler-sdk-tool": [
                        {
                            "kernel_symbols": [
                                {
                                    "kernel_id": 7,
                                    "formatted_kernel_name": "mfma_gemm_kernel(float*, int)",
                                }
                            ],
                            "buffer_records": {
                                "kernel_dispatch": [
                                    {
                                        "dispatch_info": {
                                            "kernel_id": 7,
                                            "agent_id": {"handle": 100},
                                        },
                                        "start_timestamp": 10_000,
                                        "end_timestamp": 25_000,
                                    }
                                ]
                            },
                            "callback_records": {
                                "counter_collection": [
                                    {
                                        "counters": [
                                            {"name": "GRBM_GUI_ACTIVE", "value": 500},
                                            {"name": "SQ_INSTS_MFMA", "value": 120},
                                        ]
                                    }
                                ]
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        _write_sampler_fixture(cap_dir / "amdsmi_samples.jsonl")
        (cap_dir / "capture_rocprof.log").write_text(
            "$ rocprofv3 --kernel-trace --hip-trace -o capture -- ./app\n"
            "E20261125 00:00:00.000000 Opened result file capture_results.pftrace\n"
            "W20261125 00:00:01.000000 warning: minor timestamp skew observed\n",
            encoding="utf-8",
        )
        (cap_dir / "capture_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "rocprof-hw-capture/v1",
                    "collector": {"family": "rocprofv3"},
                    "kernel_name_contains": "mfma_gemm",
                    "capture_contract": {
                        "hw_requested": True,
                        "pmc": ["GRBM_GUI_ACTIVE", "SQ_INSTS_MFMA"],
                        "amdsmi_sample": {"requested_interval_s": 0.25},
                    },
                    "artifacts": [],
                }
            ),
            encoding="utf-8",
        )
        return cap_dir

    def test_inspect_gpus_exempts_8gpu_shared_system_pids_flags_foreign_pid_and_suspected_bug_R1(
        self,
    ) -> None:
        amdsmi, _, _ = self._assert_which_binaries_are_fakes()

        proc_rows = capture_rocm.query_processes(amdsmi)
        self.assertTrue(self.guard_marker.is_file())
        self.assertIsNotNone(proc_rows)
        assert proc_rows is not None
        self.assertEqual(len(proc_rows), 8)
        self.assertEqual(capture_rocm.shared_system_pids(proc_rows), [_SYSTEM_DAEMON_PID])
        # Single-GPU list never treats a single GPU's PID as multi-card shared system PID
        self.assertEqual(capture_rocm.shared_system_pids([proc_rows[0]]), [])

        # Idle GPUs [0, 1, 2]: system PID 424242 is on all 8 GPUs -> exempted -> busy=False
        idle_rep = capture_rocm.inspect_gpus(
            amdsmi,
            [0, 1, 2],
            mem_threshold_mib=1024.0,
            util_threshold_pct=10.0,
        )
        self.assertTrue(idle_rep["amd_smi_available"])
        self.assertEqual(idle_rep["shared_system_pids"], [_SYSTEM_DAEMON_PID])
        self.assertEqual(idle_rep["foreign_processes"], [])
        self.assertFalse(idle_rep["busy"])
        self.assertEqual(idle_rep["reasons"], [])
        self.assertEqual(len(idle_rep["gpus"]), 3)
        # Suspected bug R1: GPU 0 reports gfx_activity={"value": 0, "unit": "%"}, which
        # _metric_value() parses as 0.0, and `0.0 or _metric_value(usage.get('gfx'))`
        # turns falsy 0.0 into None! GPU 1 (4%) is preserved as 4.0; GPU 2 ("N/A") is None.
        self.assertIsNone(idle_rep["gpus"][0]["utilization_pct"])
        self.assertEqual(idle_rep["gpus"][1]["utilization_pct"], 4.0)
        self.assertIsNone(idle_rep["gpus"][2]["utilization_pct"])
        self.assertEqual(idle_rep["gpus"][0]["memory_used_mib"], 128.0)
        self.assertFalse(idle_rep["gpus"][0]["over_memory"])
        self.assertFalse(idle_rep["gpus"][1]["over_utilization"])

        # Target includes GPU 3 -> foreign PID 999991 triggers busy=True while 424242 stays exempted
        busy_rep = capture_rocm.inspect_gpus(
            amdsmi,
            [0, 3],
            mem_threshold_mib=1024.0,
            util_threshold_pct=10.0,
        )
        self.assertTrue(busy_rep["amd_smi_available"])
        self.assertEqual(busy_rep["shared_system_pids"], [_SYSTEM_DAEMON_PID])
        self.assertTrue(busy_rep["busy"])
        self.assertEqual(
            busy_rep["foreign_processes"],
            [
                {
                    "pid": _FOREIGN_JOB_PID,
                    "name": "foreign_train_job",
                    "vram_bytes": 8192.0,
                    "index": 3,
                }
            ],
        )
        self.assertEqual(
            busy_rep["reasons"],
            [
                f"GPU 3 上存在其他进程 pid={_FOREIGN_JOB_PID} "
                "name=foreign_train_job（不会 kill，请等待空闲）"
            ],
        )

        # Missing GPU index -> busy=True with explicit reason
        oob_rep = capture_rocm.inspect_gpus(amdsmi, [8], 1024.0, 10.0)
        self.assertTrue(oob_rep["busy"])
        self.assertEqual(oob_rep["reasons"], ["GPU 8 不存在"])

        # Unavailable amd-smi binary -> busy=True fail-closed
        missing_smi_rep = capture_rocm.inspect_gpus(
            str(self.root / "missing-amd-smi"),
            [0],
            1024.0,
            10.0,
        )
        self.assertFalse(missing_smi_rep["amd_smi_available"])
        self.assertTrue(missing_smi_rep["busy"])
        self.assertEqual(missing_smi_rep["reasons"], ["amd-smi 不可用，无法确认 GPU 空闲"])

    def test_sampler_stats_hand_calculated_frequency_errors_and_edge_cases(self) -> None:
        self._assert_which_binaries_are_fakes()

        jsonl_path = self.root / "amdsmi_samples.jsonl"
        _write_sampler_fixture(jsonl_path)
        stats = capture_rocm.sampler_stats(jsonl_path, requested_interval=0.25)
        self.assertEqual(
            stats,
            {
                "path": str(jsonl_path),
                "available": True,
                "requested_interval_s": 0.25,
                "sample_count": 5,
                "error_count": 1,
                "interval_s": {
                    "median": 0.25,
                    "min": 0.25,
                    "max": 0.75,
                    "mean": 0.375,
                },
                "measured_hz": 4.0,
                "window_s": [100.0, 101.5],
            },
        )

        missing_path = self.root / "no_samples.jsonl"
        missing_stats = capture_rocm.sampler_stats(missing_path, requested_interval=0.5)
        self.assertEqual(
            missing_stats,
            {
                "path": str(missing_path),
                "available": False,
                "requested_interval_s": 0.5,
                "sample_count": 0,
                "error_count": 0,
                "reason": "采样文件未生成",
            },
        )

        single_path = self.root / "single_sample.jsonl"
        single_path.write_text(
            json.dumps({"t_mono": 50.0, "gpu": 0, "metric": {"index": 0}}) + "\n",
            encoding="utf-8",
        )
        single_stats = capture_rocm.sampler_stats(single_path, requested_interval=0.25)
        self.assertEqual(single_stats["sample_count"], 1)
        self.assertEqual(single_stats["error_count"], 0)
        self.assertEqual(single_stats["window_s"], [50.0, 50.0])
        self.assertNotIn("measured_hz", single_stats)
        self.assertNotIn("interval_s", single_stats)

        bad_path = self.root / "corrupt.jsonl"
        bad_path.write_text("{not-valid-json\n", encoding="utf-8")
        bad_stats = capture_rocm.sampler_stats(bad_path, requested_interval=0.25)
        self.assertTrue(bad_stats["available"])
        self.assertEqual(bad_stats["sample_count"], 0)
        self.assertTrue(bad_stats["reason"].startswith("采样文件解析失败:"))

    def test_counter_probe_json_missing_counter_failure_and_suspected_bug_R2(self) -> None:
        self._assert_which_binaries_are_fakes()

        # 1. Missing callback_records.counter_collection
        no_coll = capture_rocm.counter_probe_json({}, ["GRBM_GUI_ACTIVE"])
        self.assertFalse(no_coll["available"])
        self.assertEqual(
            no_coll["hard_failures"],
            ["rocprof json 缺 callback_records.counter_collection（--pmc 未采到）"],
        )
        self.assertNotIn("valid", no_coll)

        # 2. Empty counter_collection list -> valid=False
        empty_coll = capture_rocm.counter_probe_json(
            {"callback_records": {"counter_collection": []}},
            ["GRBM_GUI_ACTIVE"],
        )
        self.assertTrue(empty_coll["available"])
        self.assertFalse(empty_coll["valid"])
        self.assertEqual(empty_coll["record_count"], 0)
        self.assertEqual(empty_coll["samples_by_counter"], {})
        self.assertEqual(
            empty_coll["hard_failures"],
            ["counter_collection 为空：PMC 一组都没采到"],
        )

        # 3. Missing requested counter when named counters exist -> valid=False
        missing_one = capture_rocm.counter_probe_json(
            {
                "callback_records": {
                    "counter_collection": [
                        {"counters": [{"name": "GRBM_GUI_ACTIVE", "value": 100}]}
                    ]
                }
            },
            ["GRBM_GUI_ACTIVE", "SQ_INSTS_MFMA", "TCC_MISS"],
        )
        self.assertTrue(missing_one["available"])
        self.assertFalse(missing_one["valid"])
        self.assertEqual(missing_one["record_count"], 1)
        self.assertEqual(missing_one["samples_by_counter"], {"GRBM_GUI_ACTIVE": 1})
        self.assertEqual(
            missing_one["hard_failures"],
            ["请求的 PMC 无样本: ['SQ_INSTS_MFMA', 'TCC_MISS']"],
        )

        # 4. All requested counters present -> valid=True
        all_ok = capture_rocm.counter_probe_json(
            {
                "callback_records": {
                    "counter_collection": [
                        {
                            "counters": [
                                {"name": "GRBM_GUI_ACTIVE", "value": 100},
                                {"counter_name": "SQ_INSTS_MFMA_XCC0", "value": 50},
                            ]
                        },
                        {"records": [{"name": "GRBM_GUI_ACTIVE", "value": 110}]},
                    ]
                }
            },
            ["GRBM_GUI_ACTIVE", "SQ_INSTS_MFMA"],
        )
        self.assertTrue(all_ok["valid"])
        self.assertEqual(all_ok["hard_failures"], [])
        self.assertEqual(
            all_ok["samples_by_counter"],
            {"GRBM_GUI_ACTIVE": 2, "SQ_INSTS_MFMA_XCC0": 1},
        )

        # Suspected bug R2: when counter_collection is non-empty but records only carry
        # rocprofv3's handle-based items {"counter_id": {"handle": 11}, "value": 42.0}
        # without inline 'name'/'counter_name', counter_names is empty, and
        # `if counter_names and zero:` skips the missing-PMC failure check (valid=True)!
        handle_only = capture_rocm.counter_probe_json(
            {
                "callback_records": {
                    "counter_collection": [
                        {
                            "dispatch_data": {"start_timestamp": 100, "end_timestamp": 200},
                            "records": [{"counter_id": {"handle": 11}, "value": 42.0}],
                        }
                    ]
                }
            },
            ["GRBM_GUI_ACTIVE", "MfmaUtil"],
        )
        self.assertTrue(handle_only["available"])
        self.assertEqual(handle_only["record_count"], 1)
        self.assertEqual(handle_only["samples_by_counter"], {})
        self.assertEqual(handle_only["hard_failures"], [])
        self.assertTrue(handle_only["valid"])

    def test_validate_existing_rc0_pass_rc4_failures_and_suspected_bug_R3(self) -> None:
        self._assert_which_binaries_are_fakes()

        cap_dir = self._make_existing_capture_dir("cap_pass")
        rc_pass, out_pass, err_pass = _run_main(["--validate-existing", str(cap_dir)])
        self.assertEqual(rc_pass, 0)
        self.assertEqual(err_pass, "")
        # --validate-existing is purely read-only and must not invoke amd-smi or rocprof*
        self.assertFalse(self.amdsmi_log.exists())
        self.assertFalse(self.rocprofsys_log.exists())
        self.assertFalse(self.rocprofv3_log.exists())

        doc_pass = json.loads(out_pass)
        self.assertEqual(doc_pass["verdict"], "PASS")
        self.assertEqual(
            Path(doc_pass["validation_source_manifest"]).resolve(),
            (cap_dir / "capture_manifest.json").resolve(),
        )
        ev = doc_pass["evidence"]
        self.assertEqual(ev["verdict"], "PASS")
        self.assertEqual(ev["failures"], [])
        self.assertEqual(ev["warnings"], [])
        self.assertEqual(ev["pftrace"]["bytes"], 25)
        self.assertEqual(ev["kernel"]["match_rows"], 1)
        self.assertEqual(ev["kernel"]["complete_launches"], 1)
        self.assertTrue(ev["counters"]["valid"])
        self.assertEqual(
            ev["counters"]["samples_by_counter"],
            {"GRBM_GUI_ACTIVE": 1, "SQ_INSTS_MFMA": 1},
        )
        self.assertEqual(ev["amdsmi_sampler"]["measured_hz"], 4.0)
        self.assertEqual(ev["amdsmi_sampler"]["sample_count"], 5)
        self.assertEqual(ev["amdsmi_sampler"]["error_count"], 1)
        # Benign "Opened result file" E-prefixed glog line is ignored; warning line is counted
        self.assertEqual(ev["log_findings"]["rocm_error"]["count"], 0)
        self.assertEqual(ev["log_findings"]["warning"]["count"], 1)

        # --validation-output writes strengthened manifest JSON to disk
        val_out = self.root / "strengthened" / "validated_manifest.json"
        rc_out, out_msg, err_msg = _run_main(
            [
                "--validate-existing",
                str(cap_dir / "capture_manifest.json"),
                "--validation-output",
                str(val_out),
            ]
        )
        self.assertEqual(rc_out, 0)
        self.assertEqual(err_msg, "")
        self.assertEqual(out_msg, f"[validate-existing] manifest {val_out}\n")
        self.assertEqual(json.loads(val_out.read_text(encoding="utf-8"))["verdict"], "PASS")

        # Suspected bug R3: when hw_requested=True and amdsmi_samples.jsonl is completely absent,
        # validate_existing_capture passes sampler=None so verify_evidence emits NO warning,
        # whereas an empty 0-line amdsmi_samples.jsonl emits the 0-sample warning.
        cap_no_sampler = self._make_existing_capture_dir("cap_no_sampler")
        (cap_no_sampler / "amdsmi_samples.jsonl").unlink()
        rc_ns, out_ns, _ = _run_main(["--validate-existing", str(cap_no_sampler)])
        self.assertEqual(rc_ns, 0)
        doc_ns = json.loads(out_ns)
        self.assertNotIn("amdsmi_sampler", doc_ns["evidence"])
        self.assertEqual(doc_ns["evidence"]["warnings"], [])

        cap_empty_sampler = self._make_existing_capture_dir("cap_empty_sampler")
        (cap_empty_sampler / "amdsmi_samples.jsonl").write_text("", encoding="utf-8")
        rc_es, out_es, _ = _run_main(["--validate-existing", str(cap_empty_sampler)])
        self.assertEqual(rc_es, 0)
        doc_es = json.loads(out_es)
        self.assertEqual(
            doc_es["evidence"]["warnings"],
            ["amd-smi 粗采样 0 个样本（hw bundle 少一条腿）"],
        )

        # rc=4 Failure Path A: non-existent manifest path
        rc_missing, out_missing, err_missing = _run_main(
            ["--validate-existing", str(self.root / "no_such_dir")]
        )
        self.assertEqual(rc_missing, 4)
        self.assertEqual(out_missing, "")
        self.assertTrue(
            err_missing.startswith("[validate-existing] ERROR 找不到已有 manifest:"),
            msg=err_missing,
        )

        # rc=4 Failure Path B: missing requested PMC counter + blocking HSA_STATUS_ERROR + empty pftrace
        cap_fail = self._make_existing_capture_dir("cap_fail")
        (cap_fail / "capture_results.pftrace").write_bytes(b"")
        (cap_fail / "capture_rocprof.log").write_text(
            "HSA_STATUS_ERROR_OUT_OF_RESOURCES: queue creation failed\n",
            encoding="utf-8",
        )
        fail_manifest = json.loads((cap_fail / "capture_manifest.json").read_text(encoding="utf-8"))
        fail_manifest["capture_contract"]["pmc"] = [
            "GRBM_GUI_ACTIVE",
            "SQ_INSTS_MFMA",
            "SQ_WAVES",
        ]
        (cap_fail / "capture_manifest.json").write_text(
            json.dumps(fail_manifest),
            encoding="utf-8",
        )
        rc_fail, out_fail, err_fail = _run_main(["--validate-existing", str(cap_fail)])
        self.assertEqual(rc_fail, 4)
        self.assertEqual(err_fail, "")
        doc_fail = json.loads(out_fail)
        self.assertEqual(doc_fail["verdict"], "FAIL")
        self.assertEqual(doc_fail["evidence"]["verdict"], "FAIL")
        self.assertEqual(
            doc_fail["evidence"]["failures"],
            [
                "collector 日志出现 hsa_error 相关行 1 条: "
                "['HSA_STATUS_ERROR_OUT_OF_RESOURCES: queue creation failed']",
                "pftrace 为空",
                "请求的 PMC 无样本: ['SQ_WAVES']",
            ],
        )

    def test_selftest_and_dry_run_use_fake_path_binaries_without_touching_gpu_or_writing_files(
        self,
    ) -> None:
        self._assert_which_binaries_are_fakes()

        # 1. --selftest (default auto -> selects fake rocprof-sys on PATH, never calls amd-smi)
        rc_st_sys, out_st_sys, err_st_sys = _run_main(["--selftest"])
        self.assertEqual(rc_st_sys, 0)
        self.assertEqual(err_st_sys, "")
        self.assertTrue(self.guard_marker.is_file())
        self.assertFalse(self.amdsmi_log.exists())
        st_sys = json.loads(out_st_sys)
        self.assertEqual(st_sys["collector_selection"]["family"], "rocprof-sys")
        self.assertEqual(
            Path(st_sys["collector_selection"]["path"]).resolve(),
            self.fake_rocprofsys.resolve(),
        )
        self.assertEqual(st_sys["collector_selection"]["version"], "rocprof-sys 1.3.0")
        self.assertEqual(st_sys["flag_selftest"]["collector"], "rocprof-sys")
        self.assertEqual(st_sys["hard_failures"], [])

        # 2. --selftest --collector rocprofv3 -> probes fake rocprofv3 --help & --version
        rc_st_v3, out_st_v3, err_st_v3 = _run_main(["--selftest", "--collector", "rocprofv3"])
        self.assertEqual(rc_st_v3, 0)
        self.assertEqual(err_st_v3, "")
        self.assertFalse(self.amdsmi_log.exists())
        st_v3 = json.loads(out_st_v3)
        self.assertEqual(st_v3["collector_selection"]["family"], "rocprofv3")
        self.assertTrue(st_v3["flag_selftest"]["available"])
        self.assertEqual(st_v3["flag_selftest"]["flags_missing"], [])
        self.assertEqual(st_v3["flag_selftest"]["version"], "ROCProfiler-SDK v7.2.0")
        self.assertEqual(st_v3["hard_failures"], [])

        # 3. --selftest with a rocprofv3 missing '--pmc' -> returns rc=4
        broken_v3 = self._write_fake_collector(
            self.root / "broken_rocprofv3",
            version_line="ROCProfiler-SDK v0.1",
            help_text=_FAKE_ROCPROFV3_HELP.replace("  --pmc\n", ""),
            call_log=self.rocprofv3_log,
        )
        rc_st_bad, out_st_bad, _ = _run_main(
            ["--selftest", "--collector", "rocprofv3", "--rocprof", str(broken_v3)]
        )
        self.assertEqual(rc_st_bad, 4)
        st_bad = json.loads(out_st_bad)
        self.assertEqual(st_bad["flag_selftest"]["flags_missing"], ["--pmc"])
        self.assertEqual(
            st_bad["hard_failures"],
            ["rocprofv3 --help 缺少本模块用到的 flag: ['--pmc']"],
        )

        # 4. --dry-run on idle GPUs (via HIP_VISIBLE_DEVICES="0,1" env var):
        # verifies plan output, payload path resolution, amd-smi idle check, rc=0, and NO out_dir created.
        proj_root = self.root / "proj"
        (proj_root / "bin").mkdir(parents=True)
        payload_marker = self.root / "payload_ran.txt"
        bench_app = proj_root / "bin" / "bench_app"
        bench_app.write_text(
            f"#!{sys.executable}\n"
            f"import pathlib; pathlib.Path({str(payload_marker)!r}).write_text('ran')\n",
            encoding="utf-8",
        )
        bench_app.chmod(0o755)
        contract_path = proj_root / "CONTRACT.lock.json"
        contract_path.write_text(
            json.dumps({"platform": {"persistent": False}}),
            encoding="utf-8",
        )
        dry_out = proj_root / "runs" / "dry_capture"
        os.environ["HIP_VISIBLE_DEVICES"] = "0,1"

        rc_dry, out_dry, err_dry = _run_main(
            [
                "--root",
                str(proj_root),
                "--out-dir",
                str(dry_out),
                "--contract",
                str(contract_path),
                "--pmc",
                "GRBM_GUI_ACTIVE",
                "SQ_INSTS_MFMA",
                "--kernel-name-contains",
                "mfma_gemm",
                "--dry-run",
                "--payload",
                "./bin/bench_app",
                "--iters",
                "5",
            ]
        )
        self.assertEqual(rc_dry, 0)
        self.assertEqual(err_dry, "")
        self.assertFalse(dry_out.exists())
        self.assertFalse(payload_marker.exists())
        self.assertIn("[plan] collector      rocprofv3\n", out_dry)
        self.assertIn("[plan] bundle         sw+hw\n", out_dry)
        self.assertIn("[plan] payload paths  1 项已按 root 解析为绝对路径\n", out_dry)
        self.assertIn(
            "[plan] HIP_VISIBLE_DEVICES='0,1' selector=hip-visible-index harness 物理卡=[0, 1]\n",
            out_dry,
        )
        self.assertIn(
            f"[gpu] amd_smi_available=True busy=False shared_system_pids=[{_SYSTEM_DAEMON_PID}]\n",
            out_dry,
        )
        self.assertIn(
            "[dry-run] 仅打印计划，未创建目录、未启动 collector、未写任何文件\n",
            out_dry,
        )

        # 5. --dry-run targeting busy GPU 3 (foreign process PID 999991):
        # returns rc=2 without --force, and rc=0 with --force while still creating zero files.
        os.environ["HIP_VISIBLE_DEVICES"] = "3"
        rc_busy, out_busy, err_busy = _run_main(
            [
                "--root",
                str(proj_root),
                "--out-dir",
                str(dry_out),
                "--dry-run",
                "--payload",
                "./bin/bench_app",
            ]
        )
        self.assertEqual(rc_busy, 2)
        self.assertEqual(err_busy, "")
        self.assertFalse(dry_out.exists())
        self.assertFalse(payload_marker.exists())
        self.assertIn(
            "[gpu] ERROR 目标 GPU 被占用或状态未知（绝不 kill 任何进程）。等空闲后重跑，或显式加 --force。\n",
            out_busy,
        )

        rc_force, out_force, err_force = _run_main(
            [
                "--root",
                str(proj_root),
                "--out-dir",
                str(dry_out),
                "--dry-run",
                "--force",
                "--payload",
                "./bin/bench_app",
            ]
        )
        self.assertEqual(rc_force, 0)
        self.assertEqual(err_force, "")
        self.assertFalse(dry_out.exists())
        self.assertFalse(payload_marker.exists())
        self.assertIn("[gpu] WARNING 目标 GPU 有占用，--force 已指定，继续执行\n", out_force)
        self.assertIn(
            "[dry-run] 仅打印计划，未创建目录、未启动 collector、未写任何文件\n",
            out_force,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end characterization of `wp_decode -> hardware_window -> align` on golden `h20_pingpong_v1`."""
from __future__ import annotations

from collections import Counter
import ipaddress
import json
import lzma
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# In-process zero-external-network socket guard (installed BEFORE importing weaveprof)
# ---------------------------------------------------------------------------
_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_GETADDRINFO = socket.getaddrinfo
_NON_LOOPBACK_ATTEMPTS: list[str] = []


def _is_loopback_host(host: object) -> bool:
    return isinstance(host, str) and host.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip_literal(host: object) -> bool:
    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host.strip())
        return True
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: object) -> object:
    if isinstance(address, tuple) and address and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"connect:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if isinstance(address, tuple) and address and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"connect_ex:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT_EX(self, address)


def _guarded_create_connection(address: tuple[str, int], *args: object, **kwargs: object) -> socket.socket:
    host = address[0] if isinstance(address, tuple) and address else ""
    if not _is_loopback_host(host):
        _NON_LOOPBACK_ATTEMPTS.append(f"create_connection:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> list[tuple]:
    if not (_is_loopback_host(host) or _is_ip_literal(host)):
        _NON_LOOPBACK_ATTEMPTS.append(f"getaddrinfo:{host!r}:{port!r}")
        raise OSError(f"blocked by test guard: DNS lookup for {host!r}")
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


socket.socket.connect = _guarded_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment]
socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof.megakernel import load_wp_decode  # noqa: E402
from weaveprof.pipeline.validate.tools import fi_merge_perfetto as merge  # noqa: E402

FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
GOLDEN_CASE_DIR = ROOT / "tests" / "golden" / "cases" / "h20_pingpong_v1"

EXPECTED_H20_PINGPONG_CURVE_NAMES = {
    "Active SM Residency (%) [78 SMs]",
    "Active SMs (%) [NSYS PMU]",
    "Compute Warps in Flight [Avg Warps per Cycle]",
    "Compute Warps in Flight [Avg]",
    "Compute Warps in Flight [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
    "GPC Clock Frequency [MHz]",
    "GR Active [Throughput %]",
    "Memory Util — HBM Read+Write (%) [NSYS PMU]",
    "NVLink RX Requests Protocol Data [Throughput %]",
    "NVLink RX Requests User Data [Throughput %]",
    "NVLink RX Responses Protocol Data [Throughput %]",
    "NVLink RX Responses User Data [Throughput %]",
    "NVLink RX User Data Total (%) [NSYS PMU]",
    "NVLink TX Requests Protocol Data [Throughput %]",
    "NVLink TX Requests User Data [Throughput %]",
    "NVLink TX Responses Protocol Data [Throughput %]",
    "NVLink TX Responses User Data [Throughput %]",
    "NVLink TX User Data Total (%) [NSYS PMU]",
    "PCIe RX Throughput [Throughput %]",
    "PCIe TX Throughput [Throughput %]",
    "Pixel Warps in Flight [Avg Warps per Cycle]",
    "Pixel Warps in Flight [Avg]",
    "Pixel Warps in Flight [Throughput %]",
    "SM Issue [Throughput %]",
    "SMs Active [Throughput %]",
    "SYS Clock Frequency [MHz]",
    "Unallocated Warps in Active SMs [Avg Warps per Cycle]",
    "Unallocated Warps in Active SMs [Avg]",
    "Unallocated Warps in Active SMs [Throughput %]",
    "Vertex/Tess/Geometry Warps in Flight [Avg Warps per Cycle]",
    "Vertex/Tess/Geometry Warps in Flight [Avg]",
    "Vertex/Tess/Geometry Warps in Flight [Throughput %]",
    "tensor_active [Tensor Active x4 (H20 die-normalization corrected)]",
}


class WpProfileLoopE2ECharacterizationTests(unittest.TestCase):
    _ENV_SAVE_KEYS = (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
        "WEAVEPROF_SESSIONS_DIR",
        "WEAVEPROF_CACHE",
        "WEAVEPROF_LOCK_DIR",
        "WEAVEPROF_WORKDIR",
        "TMPDIR",
        "WEAVEPROF_SUPERVISOR_HMAC_KEY",
        "WEAVEPROF_SUPERVISOR_KEY_FILE",
        "QWEN_PROJECT_DIR",
        "QWEN_SESSION_ID",
        "WEAVEPROF_LOCK_PROVIDER",
        "WEAVEPROF_KEEP_SHADOW",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        _NON_LOOPBACK_ATTEMPTS.clear()
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}
        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_r14_e2e_iso_")).resolve()
        for sub in ("home", "xdg_cache", "xdg_config", "mpl", "sessions", "cache", "locks", "workdir", "tmp", "cwd"):
            (cls._iso_root / sub).mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(cls._iso_root / "home")
        os.environ["XDG_CACHE_HOME"] = str(cls._iso_root / "xdg_cache")
        os.environ["XDG_CONFIG_HOME"] = str(cls._iso_root / "xdg_config")
        os.environ["MPLCONFIGDIR"] = str(cls._iso_root / "mpl")
        os.environ["WEAVEPROF_SESSIONS_DIR"] = str(cls._iso_root / "sessions")
        os.environ["WEAVEPROF_CACHE"] = str(cls._iso_root / "cache")
        os.environ["WEAVEPROF_LOCK_DIR"] = str(cls._iso_root / "locks")
        os.environ["WEAVEPROF_WORKDIR"] = str(cls._iso_root / "workdir")
        os.environ["TMPDIR"] = str(cls._iso_root / "tmp")
        os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = FIXED_HMAC_KEY
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        for k in (
            "WEAVEPROF_SUPERVISOR_KEY_FILE",
            "QWEN_PROJECT_DIR",
            "QWEN_SESSION_ID",
            "WEAVEPROF_LOCK_PROVIDER",
            "WEAVEPROF_KEEP_SHADOW",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        ):
            os.environ.pop(k, None)
        tempfile.tempdir = str(cls._iso_root / "tmp")
        os.chdir(cls._iso_root / "cwd")

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._saved_cwd)
        tempfile.tempdir = cls._saved_tempdir
        for k, val in cls._saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        shutil.rmtree(cls._iso_root, ignore_errors=True)
        if _NON_LOOPBACK_ATTEMPTS:
            raise AssertionError(f"Unexpected non-loopback network attempts: {_NON_LOOPBACK_ATTEMPTS}")

    def test_h20_pingpong_v1_decode_hardware_window_and_align_end_to_end(self) -> None:
        """Follow `backend.py:459-460, 527-528` call order on read-only golden `h20_pingpong_v1` capture."""
        work_dir = Path(tempfile.mkdtemp(prefix="h20_pingpong_e2e_", dir=str(self._iso_root / "tmp"))).resolve()
        (work_dir / "capture.sqlite").write_bytes(lzma.decompress((GOLDEN_CASE_DIR / "nsys.sqlite.xz").read_bytes()))
        (work_dir / "capture.fi02.bin").write_bytes(
            lzma.decompress((GOLDEN_CASE_DIR / "capture.fi02.bin.xz").read_bytes())
        )

        runtime = json.loads((GOLDEN_CASE_DIR / "runtime.json").read_text(encoding="utf-8"))
        wp_dec = load_wp_decode()
        trace, summary = wp_dec.decode(
            wp_dec.read_dump(str(work_dir / "capture.fi02.bin")),
            wp_dec.load_names(str(GOLDEN_CASE_DIR / "names.json")),
        )

        # Hand calculation of `offset_ns`:
        #   summary["observation_start_ns"] = 1795465733448828928
        #   summary["observation_end_ns"]   = 1795465733449895392 (span = 1_066_464 ns)
        #   window start = 40098789, end = 41166728 (dur = 1_067_939 ns >= span)
        #   lower = 40098789 - 1795465733448828928 = -1795465733408730139
        #   upper = 41166728 - 1795465733449895392 = -1795465733408728664
        #   (lower + upper) // 2 = -3590931466817458803 // 2 = -1795465733408729402
        window = merge.hardware_window(work_dir, runtime)
        aligned = merge.align(trace, summary, window, runtime)

        al = aligned["otherData"]["alignment"]
        self.assertEqual(al["offset_ns"], -1795465733408729402)
        self.assertEqual(al["kernel_start_ns"], 40098789)
        self.assertEqual(al["kernel_end_ns"], 41166728)
        self.assertEqual(al["kernel_duration_us"], 1067.939)
        self.assertEqual(al["first_open_ns"], 1795465733448828928)
        self.assertEqual(al["hardware_samples"], 6390)
        self.assertEqual(al["physical_units_observed"], 78)
        self.assertEqual(al["ctas_observed"], 78)
        self.assertEqual(al["warps_per_cta"], 12)
        self.assertEqual(al["cross_warp_spans"], 262144)
        self.assertIs(al["has_compute_curve"], True)
        self.assertIs(al["has_memory_curve"], True)
        self.assertIs(al["has_sm_curve"], True)
        self.assertEqual(
            al["tensor_active_scale"],
            {
                "scale": 4.0,
                "applied": True,
                "card_factor": 4.0,
                "source": "nsys TARGET_INFO_GPU name 'NVIDIA H20'",
                "raw_mean_pct": 22.376,
                "raw_max_pct": 25.0,
                "raw_ceiling_pct": 29.0,
                "reason": "nsys normalizes Tensor Active to the full GH100 die; H20 Tensor throughput per SM is 1/4 of it, so values are multiplied by 4",
            },
        )

        c_names = {ev["name"] for ev in aligned["traceEvents"] if ev.get("ph") == "C"}
        self.assertEqual(c_names, EXPECTED_H20_PINGPONG_CURVE_NAMES)
        self.assertFalse(any("[stage-calibrated]" in n for n in c_names))

        ph_counts = dict(Counter(ev.get("ph") for ev in aligned["traceEvents"]))
        self.assertEqual(ph_counts, {"M": 2030, "X": 865193, "C": 7457})


if __name__ == "__main__":
    unittest.main()

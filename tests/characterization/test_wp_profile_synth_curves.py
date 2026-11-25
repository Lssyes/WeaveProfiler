#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for A4 synthesized hardware curves, H20 x4 ceiling boundary, and `viz._classify_hw_metric`."""
from __future__ import annotations

import ipaddress
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

from weaveprof.pipeline.validate.tools import fi_merge_perfetto as merge  # noqa: E402
from weaveprof.viz import _classify_hw_metric  # noqa: E402

FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _counter_series(aligned: dict, curve_name: str) -> list[tuple[int, float]]:
    return [
        (int(ev["ts_ns"]), float(ev["args"]["value"]))
        for ev in aligned["traceEvents"]
        if ev.get("ph") == "C" and ev.get("name") == curve_name
    ]


class WpProfileSynthCurvesCharacterizationTests(unittest.TestCase):
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
        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_r14_synth_iso_")).resolve()
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

    def test_suspected_bug_A4_nvidia_synthesizes_stage_calibrated_curves_without_pmu(self) -> None:
        """这是疑似 bug：论文的说法是没有同次运行的硬件计数器就不出图，见 `交付/疑似bug清单.md` 的 A4。这里只钉现状，修复时要同时改这里的期望值。"""
        # Hand calculation:
        #   window: start=10_000, end=14_000 -> dur_ns=4_000, bin_ns=2_000 -> 2 bins [0..2000], [2000..4000]
        #   origin=10_000, obs_end=14_000 -> offset_ns = 0
        #   load stage [10_000, 12_000] -> mem_exp = [1.0, 0.0], widths = [2000, 2000], tot_exp = 2000.0
        #   mma  stage [12_000, 14_000] -> mma_exp = [0.0, 1.0], widths = [2000, 2000], tot_exp = 2000.0
        #   Compute (`:479-493`): workload_flops=800_000.0, peak_tflops=1.0
        #     mean_pct = 100.0 * (800_000 / 4_000 / 1_000) / 1.0 = 20.0
        #     bin 0 (10_000): 0.0 * 20.0 * 4_000 / 2_000 = 0.0
        #     bin 1 (12_000): 1.0 * 20.0 * 4_000 / 2_000 = 40.0
        #     end   (14_000): 0.0
        #   Memory (`:495-509`): workload_bytes=300_000.0, peak_bw_gbps=200.0
        #     mean_bw_pct = 100.0 * (300_000 / 4_000) / 200.0 = 37.5
        #     bin 0 (10_000): 1.0 * 37.5 * 4_000 / 2_000 = 75.0
        #     bin 1 (12_000): 0.0
        #     end   (14_000): 0.0
        trace = {
            "traceEvents": [
                {"ph": "X", "ts": 0.0, "dur": 2.0, "name": "load", "args": {"cta": 0, "warp": 0, "semantic_id": "load"}},
                {"ph": "X", "ts": 2.0, "dur": 2.0, "name": "mma", "args": {"cta": 0, "warp": 0, "semantic_id": "mma"}},
            ],
            "otherData": {},
        }
        summary = {
            "observation_start_ns": 10_000,
            "observation_end_ns": 14_000,
            "first_open_ns": 10_000,
            "wpc": 1,
            "platform": "nvidia",
            "cta_sm": {"0": 0},
            "lane_bounds_ns": {"0": [10_000, 14_000, 0]},
        }
        runtime = {
            "total_hw_units": 10,
            "workload_flops": 800_000.0,
            "peak_tflops": 1.0,
            "workload_bytes": 300_000.0,
            "peak_bw_gbps": 200.0,
        }
        aligned = merge.align(trace, summary, (10_000, 14_000, "gemm_no_pmu", []), runtime)
        al = aligned["otherData"]["alignment"]
        self.assertEqual(al["hardware_samples"], 0)
        self.assertIs(al["has_compute_curve"], True)
        self.assertIs(al["has_memory_curve"], True)
        self.assertIs(al["has_sm_curve"], True)

        comp_name = "Compute Util — Tensor Core (%) [stage-calibrated]"
        mem_name = "Memory Util — HBM Read+Write (%) [stage-calibrated]"
        self.assertEqual(_counter_series(aligned, comp_name), [(10_000, 0.0), (12_000, 40.0), (14_000, 0.0)])
        self.assertEqual(_counter_series(aligned, mem_name), [(10_000, 75.0), (12_000, 0.0), (14_000, 0.0)])
        self.assertEqual(_classify_hw_metric(comp_name), "tensor_active")
        self.assertEqual(_classify_hw_metric(mem_name), "dram_bw_direct")

    def test_suspected_bug_A4_amd_synthesizes_calibrated_curves_even_when_real_pmc_present(self) -> None:
        """这是疑似 bug：论文的说法是没有同次运行的硬件计数器就不出图，见 `交付/疑似bug清单.md` 的 A4。这里只钉现状，修复时要同时改这里的期望值。"""
        # Phenomenon (a) (`fi_merge_perfetto.py:479,495`): condition is `amd or not has_compute_hw` / `amd or not has_mem_hw`.
        # On AMD (`platform == "amd"`), even when `samples` ALREADY provide real ROCm PMC compute + memory counters,
        # `align` still emits `Wave MFMA Stage Activity (%) [calibrated]` and `Wave HBM Load/Store Activity (%) [calibrated]`.
        # On NVIDIA (`platform == "nvidia"`), real PMU samples set `has_compute_hw=True` and `has_mem_hw=True`, suppressing `[stage-calibrated]`.
        trace = {
            "traceEvents": [
                {"ph": "X", "ts": 0.0, "dur": 2.0, "name": "load", "args": {"cta": 0, "warp": 0, "semantic_id": "load"}},
                {"ph": "X", "ts": 2.0, "dur": 2.0, "name": "mma", "args": {"cta": 0, "warp": 0, "semantic_id": "mma"}},
            ],
            "otherData": {},
        }
        amd_summary = {
            "observation_start_ns": 10_000,
            "observation_end_ns": 14_000,
            "first_open_ns": 10_000,
            "wpc": 1,
            "platform": "amd",
            "cta_sm": {"0": 0},
            "cta_hw": {"0": [0, 1, 2, 0]},
            "lane_bounds_ns": {"0": [10_000, 14_000, 0]},
        }
        runtime = {
            "total_hw_units": 80,
            "workload_flops": 1_200_000.0,
            "peak_tflops": 1.0,
            "workload_bytes": 200_000.0,
            "peak_bw_gbps": 200.0,
        }
        amd_samples = [
            (10_000, "Compute Util — Matrix Core MFMA (%) [ROCm PMC]", 55.0),
            (14_000, "Compute Util — Matrix Core MFMA (%) [ROCm PMC]", 55.0),
            (10_000, "Memory Util — HBM3 Read+Write (%) [ROCm PMC]", 42.0),
            (14_000, "Memory Util — HBM3 Read+Write (%) [ROCm PMC]", 42.0),
        ]
        # Hand calculation on AMD:
        #   mean_pct = 100 * (1_200_000 / 4_000 / 1_000) / 1.0 = 30.0 -> bin 1 (12_000) = 30.0 * 4000 / 2000 = 60.0
        #   mean_bw_pct = 100 * (200_000 / 4_000) / 200.0 = 25.0 -> bin 0 (10_000) = 25.0 * 4000 / 2000 = 50.0
        aligned_amd = merge.align(trace, amd_summary, (10_000, 14_000, "amd_gemm", amd_samples), runtime)
        amd_comp = "Wave MFMA Stage Activity (%) [calibrated]"
        amd_mem = "Wave HBM Load/Store Activity (%) [calibrated]"
        self.assertEqual(_counter_series(aligned_amd, amd_comp), [(10_000, 0.0), (12_000, 60.0), (14_000, 0.0)])
        self.assertEqual(_counter_series(aligned_amd, amd_mem), [(10_000, 50.0), (12_000, 0.0), (14_000, 0.0)])
        self.assertEqual(_classify_hw_metric(amd_comp), "tensor_active")
        self.assertEqual(_classify_hw_metric(amd_mem), "dram_bw_direct")

        # Contrast: on NVIDIA with real PMU samples, `[stage-calibrated]` curves are NOT emitted
        nv_summary = dict(amd_summary, platform="nvidia")
        nv_summary.pop("cta_hw", None)
        nv_samples = [
            (10_000, "Tensor Active [Throughput %]", 20.0),
            (10_000, "DRAM Read Bandwidth [Throughput %]", 35.0),
        ]
        aligned_nv = merge.align(trace, nv_summary, (10_000, 14_000, "nv_gemm", nv_samples), runtime)
        nv_c_names = {ev["name"] for ev in aligned_nv["traceEvents"] if ev.get("ph") == "C"}
        self.assertNotIn("Compute Util — Tensor Core (%) [stage-calibrated]", nv_c_names)
        self.assertNotIn("Memory Util — HBM Read+Write (%) [stage-calibrated]", nv_c_names)

    def test_suspected_bug_A4_active_sm_residency_sets_has_sm_curve_and_viz_classifications(self) -> None:
        """这是疑似 bug：论文的说法是没有同次运行的硬件计数器就不出图，见 `交付/疑似bug清单.md` 的 A4。这里只钉现状，修复时要同时改这里的期望值。"""
        # Phenomenon (b) / A6 (`fi_merge_perfetto.py:453-468, 525` & `viz.py:258-296`):
        # Even with zero hardware samples and empty `runtime`, `align` emits `"Active SM Residency (%) [78 SMs]"`
        # (NVIDIA) / `"Active CU Residency (%) [80 CUs]"` (AMD) from software `lane_bounds_ns` and sets `has_sm_curve=True`.
        # Meanwhile `viz._classify_hw_metric` returns `None` for `"Active SM Residency (%) [78 SMs]"` (no `"sm active"`),
        # but returns `"sm_active"` for `"Active CU Residency (%) [80 CUs]"` (`"active" in low and "cu" in low` at `:264`).
        base_summary = {
            "observation_start_ns": 10_000,
            "observation_end_ns": 12_000,
            "first_open_ns": 10_000,
            "wpc": 1,
            "cta_sm": {"0": 0},
            "lane_bounds_ns": {"0": [10_000, 12_000, 0]},
        }
        nv_aligned = merge.align({"traceEvents": []}, dict(base_summary, platform="nvidia"), (10_000, 12_000, "k_nv", []), {})
        self.assertEqual(nv_aligned["otherData"]["alignment"]["hardware_samples"], 0)
        self.assertIs(nv_aligned["otherData"]["alignment"]["has_compute_curve"], False)
        self.assertIs(nv_aligned["otherData"]["alignment"]["has_memory_curve"], False)
        self.assertIs(nv_aligned["otherData"]["alignment"]["has_sm_curve"], True)
        nv_res_name = "Active SM Residency (%) [78 SMs]"
        # 1 active CTA / 78 SMs * 100 = 100.0 / 78 = 1.2820512820512822
        nv_pts = _counter_series(nv_aligned, nv_res_name)
        self.assertEqual(len(nv_pts), 2)
        self.assertEqual(nv_pts[0][0], 10_000)
        self.assertAlmostEqual(nv_pts[0][1], 100.0 / 78.0, places=12)
        self.assertEqual(nv_pts[1], (12_000, 0.0))
        self.assertIsNone(_classify_hw_metric(nv_res_name))

        amd_aligned = merge.align({"traceEvents": []}, dict(base_summary, platform="amd"), (10_000, 12_000, "k_amd", []), {})
        self.assertIs(amd_aligned["otherData"]["alignment"]["has_sm_curve"], True)
        amd_res_name = "Active CU Residency (%) [80 CUs]"
        # 1 active CU / 80 CUs * 100 = 1.25 (exact binary float)
        self.assertEqual(_counter_series(amd_aligned, amd_res_name), [(10_000, 1.25), (12_000, 0.0)])
        self.assertEqual(_classify_hw_metric(amd_res_name), "sm_active")

    def test_resolve_h20_tensor_scale_ceiling_29_pct_boundary_and_align_emission(self) -> None:
        # 1) Exactly at the 29.0% ceiling -> `applied == True`, `scale == 4.0`
        at_ceil = merge._resolve_h20_tensor_scale([20.0, 29.0], "NVIDIA H20", None)
        self.assertEqual(
            at_ceil,
            {
                "scale": 4.0,
                "applied": True,
                "card_factor": 4.0,
                "source": "nsys TARGET_INFO_GPU name 'NVIDIA H20'",
                "raw_mean_pct": 24.5,
                "raw_max_pct": 29.0,
                "raw_ceiling_pct": 29.0,
                "reason": "nsys normalizes Tensor Active to the full GH100 die; H20 Tensor throughput per SM is 1/4 of it, so values are multiplied by 4",
            },
        )

        # 2) Just above the 29.0% ceiling (`29.01%`) -> `applied == False`, `scale == 1.0`, left raw
        over_ceil = merge._resolve_h20_tensor_scale([20.0, 29.01], "NVIDIA H20", None)
        self.assertEqual(
            over_ceil,
            {
                "scale": 1.0,
                "applied": False,
                "card_factor": 4.0,
                "source": "nsys TARGET_INFO_GPU name 'NVIDIA H20'",
                "raw_mean_pct": 24.505,
                "raw_max_pct": 29.01,
                "raw_ceiling_pct": 29.0,
                "reason": "raw max 29.01% exceeds the 29% die-normalization ceiling; left raw",
            },
        )

        # 3) Empty samples, non-H20 card, and `card="h20"` fallback without `gpu_name`
        no_samp = merge._resolve_h20_tensor_scale([], "NVIDIA H20", None)
        self.assertIs(no_samp["applied"], False)
        self.assertEqual(no_samp["reason"], "no Tensor Active samples")

        non_h20 = merge._resolve_h20_tensor_scale([22.0], "NVIDIA B300", None)
        self.assertIs(non_h20["applied"], False)
        self.assertEqual(non_h20["card_factor"], 1.0)
        self.assertEqual(non_h20["raw_ceiling_pct"], 101.0)
        self.assertEqual(non_h20["reason"], "card is not die-normalized (no correction needed)")

        card_fallback = merge._resolve_h20_tensor_scale([15.0], None, "h20")
        self.assertIs(card_fallback["applied"], True)
        self.assertEqual(card_fallback["scale"], 4.0)
        self.assertEqual(card_fallback["source"], "card='h20' (no GPU name in the nsys report)")

        # 4) Verify `align` curve suppression/scaling at `29.0%` vs unscaled emission at `29.01%`, plus `_classify_hw_metric`
        summary = {"observation_start_ns": 10_000, "observation_end_ns": 12_000, "first_open_ns": 10_000}
        win_at = merge._HwWindow(
            (10_000, 12_000, "k_h20", [(10_000, "Tensor Active [Throughput %]", 20.0), (11_000, "Tensor Active [Throughput %]", 29.0)])
        )
        win_at.gpu_name = "NVIDIA H20"
        al_at = merge.align({"traceEvents": []}, summary, win_at, {})
        self.assertEqual(_counter_series(al_at, "Tensor Active [Throughput %]"), [])
        scaled_name = "tensor_active [Tensor Active x4 (H20 die-normalization corrected)]"
        # 20.0 * 4.0 = 80.0; min(100.0, 29.0 * 4.0 = 116.0) = 100.0
        self.assertEqual(_counter_series(al_at, scaled_name), [(10_000, 80.0), (11_000, 100.0)])
        self.assertEqual(_classify_hw_metric(scaled_name), "tensor_active")

        win_over = merge._HwWindow(
            (10_000, 12_000, "k_h20", [(10_000, "Tensor Active [Throughput %]", 20.0), (11_000, "Tensor Active [Throughput %]", 29.01)])
        )
        win_over.gpu_name = "NVIDIA H20"
        al_over = merge.align({"traceEvents": []}, summary, win_over, {})
        self.assertEqual(_counter_series(al_over, "Tensor Active [Throughput %]"), [(10_000, 20.0), (11_000, 29.01)])
        self.assertEqual(
            _counter_series(al_over, "Compute Util — Tensor Core (%) [NSYS PMU]"),
            [(10_000, 20.0), (11_000, 29.01)],
        )

        # Pin remaining `_classify_hw_metric` branches (`viz.py:258-296`), including word-order mismatch on `fi_merge_perfetto.py:449`
        self.assertEqual(_classify_hw_metric("Inter-Node RDMA RoCE (%)"), "inter_rdma")
        self.assertIsNone(_classify_hw_metric("Active SMs (%) [NSYS PMU]"))
        self.assertEqual(_classify_hw_metric("SMs Active [Throughput %]"), "sm_active")
        self.assertEqual(_classify_hw_metric("SM Issue [Throughput %]"), "sm_issue")
        self.assertEqual(_classify_hw_metric("DRAM Read Bandwidth [Throughput %]"), "dram_read")
        self.assertEqual(_classify_hw_metric("DRAM Write Bandwidth [Throughput %]"), "dram_write")
        self.assertIsNone(_classify_hw_metric("NVLink TX Requests Protocol Data [Throughput %]"))
        self.assertEqual(_classify_hw_metric("NVLink RX Requests User Data [Throughput %]"), "nvlink_rx_req")
        self.assertEqual(_classify_hw_metric("NVLink TX Responses User Data [Throughput %]"), "nvlink_tx_rsp")
        self.assertEqual(_classify_hw_metric("NVLink TX User Data Total (%) [NSYS PMU]"), "nvlink_tx_total")
        self.assertEqual(_classify_hw_metric("NVLink RX User Data Total (%) [NSYS PMU]"), "nvlink_rx_total")
        self.assertIsNone(_classify_hw_metric("GPC Clock Frequency [MHz]"))


if __name__ == "__main__":
    unittest.main()

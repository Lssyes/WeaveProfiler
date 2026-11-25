#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Characterization tests for `fi_merge_perfetto`: constant-offset alignment, `hardware_window`, `_interval_union`, `_bin_exposure`."""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
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

FIXED_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _software_events_by_name_and_kind(aligned: dict) -> list[tuple[str, str, str, int, float]]:
    """Extract emitted software events `(ph, kind, name, ts_ns, ts)` sorted by `(ts_ns, ph, name)`."""
    out: list[tuple[str, str, str, int, float]] = []
    for ev in aligned["traceEvents"]:
        pid = ev.get("pid", 0)
        ph = ev.get("ph")
        if pid >= 100000 and ph in ("X", "i"):
            kind = str((ev.get("args") or {}).get("kind", ""))
            out.append((str(ph), kind, str(ev["name"]), int(ev["ts_ns"]), float(ev["ts"])))
    out.sort(key=lambda item: (item[3], item[0], item[1], item[2]))
    return out


class WpProfileAlignCharacterizationTests(unittest.TestCase):
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
        "TRACE_PROCESSOR_SHELL",
        "PYTHONDONTWRITEBYTECODE",
    )

    @classmethod
    def setUpClass(cls) -> None:
        _NON_LOOPBACK_ATTEMPTS.clear()
        cls._saved_cwd = os.getcwd()
        cls._saved_tempdir = tempfile.tempdir
        cls._saved_env = {k: os.environ.get(k) for k in cls._ENV_SAVE_KEYS}
        cls._iso_root = Path(tempfile.mkdtemp(prefix="wp_r14_align_iso_")).resolve()
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
            "TRACE_PROCESSOR_SHELL",
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

    def _new_case_dir(self, tag: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(self._iso_root / "tmp"))).resolve()

    def test_align_exact_fit_lower_equals_upper(self) -> None:
        """Case 1: software span == hardware window duration (`lower == upper`), exact fit."""
        # Hand calculation:
        #   window: start = 10_000, end = 16_000 -> dur_ns = 6_000
        #   summary: origin = 50_000, obs_end = 56_000 (span = 6_000), first_open_ns = 50_500
        #   lower = 10_000 - 50_000 = -40_000
        #   upper = 16_000 - 56_000 = -40_000
        #   lower <= upper (-40_000 <= -40_000) -> offset_ns = (-40_000 + -40_000) // 2 = -40_000
        #   (Notice first_open_ns anchor would have been 10_000 - 50_500 = -40_500.)
        #   Constant shift on every software event (+ offset_ns = -40_000):
        #     residency lane 0 (50_000..56_000): 50_000 - 40_000 = 10_000 (10.0 us)
        #     stage "load"     (50_000 + 500):   50_500 - 40_000 = 10_500 (10.5 us)
        #     residency lane 1 (51_000..55_000): 51_000 - 40_000 = 11_000 (11.0 us)
        #     stage "mma"      (50_000 + 2500):  52_500 - 40_000 = 12_500 (12.5 us)
        #     mark  "sync_pt"  (50_000 + 3000):  53_000 - 40_000 = 13_000 (13.0 us)
        trace = {
            "traceEvents": [
                {"ph": "X", "ts": 0.5, "dur": 2.0, "name": "load", "args": {"cta": 0, "warp": 0, "semantic_id": "load"}},
                {"ph": "X", "ts": 2.5, "dur": 1.5, "name": "mma", "args": {"cta": 0, "warp": 1, "semantic_id": "mma"}},
                {"ph": "i", "ts": 3.0, "name": "sync_pt", "args": {"cta": 0, "warp": 0}},
            ],
            "otherData": {"roles": {"0": "dma", "1": "math"}},
        }
        summary = {
            "observation_start_ns": 50_000,
            "observation_end_ns": 56_000,
            "first_open_ns": 50_500,
            "wpc": 2,
            "cta_sm": {"0": 7},
            "lane_bounds_ns": {
                "0": [50_000, 56_000, 0],
                "1": [51_000, 55_000, 1],
            },
        }
        window = (10_000, 16_000, "k_exact", [])
        aligned = merge.align(trace, summary, window, {})
        al = aligned["otherData"]["alignment"]
        self.assertEqual(al["kernel_start_ns"], 10_000)
        self.assertEqual(al["kernel_end_ns"], 16_000)
        self.assertEqual(al["kernel_duration_us"], 6.0)
        self.assertEqual(al["offset_ns"], -40_000)
        self.assertEqual(al["ctas_observed"], 1)
        self.assertEqual(al["physical_units_observed"], 1)
        self.assertEqual(
            _software_events_by_name_and_kind(aligned),
            [
                ("X", "residency", "Warp residency", 10_000, 10.0),
                ("X", "stage", "load", 10_500, 10.5),
                ("X", "residency", "Warp residency", 11_000, 11.0),
                ("X", "stage", "mma", 12_500, 12.5),
                ("i", "mark", "sync_pt", 13_000, 13.0),
            ],
        )

    def test_align_slack_odd_positive_sum_floor_div(self) -> None:
        """Case 2a: span fits with slack and `lower + upper` is odd positive (`23_005 // 2 == 11_502`)."""
        # Hand calculation:
        #   window: start = 20_000, end = 28_005 -> dur_ns = 8_005
        #   summary: origin = 10_000, obs_end = 15_000 (span = 5_000), first_open_ns = 10_200
        #   lower = 20_000 - 10_000 = 10_000
        #   upper = 28_005 - 15_000 = 13_005
        #   lower <= upper; lower + upper = 23_005 (odd positive) -> 23_005 // 2 = 11_502
        #   Shifted software events (+ 11_502):
        #     residency lane 0 (10_000..15_000): 10_000 + 11_502 = 21_502 (21.502 us)
        #     stage "load"     (10_000 + 1_000): 11_000 + 11_502 = 22_502 (22.502 us)
        #     mark  "m_pos"    (10_000 + 4_000): 14_000 + 11_502 = 25_502 (25.502 us)
        trace = {
            "traceEvents": [
                {"ph": "X", "ts": 1.0, "dur": 2.0, "name": "load", "args": {"cta": 0, "warp": 0}},
                {"ph": "i", "ts": 4.0, "name": "m_pos", "args": {"cta": 0, "warp": 0}},
            ],
            "otherData": {},
        }
        summary = {
            "observation_start_ns": 10_000,
            "observation_end_ns": 15_000,
            "first_open_ns": 10_200,
            "wpc": 1,
            "cta_sm": {"0": 3},
            "lane_bounds_ns": {"0": [10_000, 15_000, 0]},
        }
        aligned = merge.align(trace, summary, (20_000, 28_005, "k_pos_odd", []), {})
        self.assertEqual(aligned["otherData"]["alignment"]["offset_ns"], 11_502)
        self.assertEqual(
            _software_events_by_name_and_kind(aligned),
            [
                ("X", "residency", "Warp residency", 21_502, 21.502),
                ("X", "stage", "load", 22_502, 22.502),
                ("i", "mark", "m_pos", 25_502, 25.502),
            ],
        )

    def test_align_slack_odd_negative_sum_floor_div(self) -> None:
        """Case 2b: span fits with slack and `lower + upper` is odd negative (`-36_995 // 2 == -18_498`)."""
        # Hand calculation:
        #   window: start = 10_000, end = 18_005 -> dur_ns = 8_005
        #   summary: origin = 30_000, obs_end = 35_000 (span = 5_000), first_open_ns = 30_400
        #   lower = 10_000 - 30_000 = -20_000
        #   upper = 18_005 - 35_000 = -16_995
        #   lower <= upper; lower + upper = -36_995 (odd negative)
        #   Python floor division `//`: -36_995 // 2 = -18_498 (truncation toward 0 would give -18_497!)
        #   Shifted software events (- 18_498):
        #     residency lane 0 (30_000..35_000): 30_000 - 18_498 = 11_502 (11.502 us)
        #     stage "load"     (30_000 + 1_000): 31_000 - 18_498 = 12_502 (12.502 us)
        #     stage "store"    (30_000 + 3_500): 33_500 - 18_498 = 15_002 (15.002 us)
        trace = {
            "traceEvents": [
                {"ph": "X", "ts": 1.0, "dur": 1.5, "name": "load", "args": {"cta": 0, "warp": 0}},
                {"ph": "X", "ts": 3.5, "dur": 1.0, "name": "store", "args": {"cta": 0, "warp": 0}},
            ],
            "otherData": {},
        }
        summary = {
            "observation_start_ns": 30_000,
            "observation_end_ns": 35_000,
            "first_open_ns": 30_400,
            "wpc": 1,
            "cta_sm": {"0": 3},
            "lane_bounds_ns": {"0": [30_000, 35_000, 0]},
        }
        aligned = merge.align(trace, summary, (10_000, 18_005, "k_neg_odd", []), {})
        self.assertEqual(aligned["otherData"]["alignment"]["offset_ns"], -18_498)
        self.assertEqual(
            _software_events_by_name_and_kind(aligned),
            [
                ("X", "residency", "Warp residency", 11_502, 11.502),
                ("X", "stage", "load", 12_502, 12.502),
                ("X", "stage", "store", 15_002, 15.002),
            ],
        )

    def test_align_overflow_by_1ns_anchors_first_open_ns(self) -> None:
        """Case 3: software span is 1 ns longer than window (`lower == upper + 1`), anchoring `first_open_ns`."""
        # Hand calculation:
        #   window: start = 10_000, end = 16_000 -> dur_ns = 6_000
        #   summary: origin = 50_000, obs_end = 56_001 (span = 6_001 = dur_ns + 1), first_open_ns = 50_300
        #   lower = 10_000 - 50_000 = -40_000
        #   upper = 16_000 - 56_001 = -40_001  (lower == upper + 1 -> lower <= upper is False!)
        #   Anchor branch: offset_ns = start - first_open_ns = 10_000 - 50_300 = -40_300
        #   (Midpoint branch would have produced (-40_000 + -40_001) // 2 = -40_001.)
        #   Shifted software events (- 40_300):
        #     residency lane 0 (50_300..56_001): 50_300 - 40_300 = 10_000 (10.0 us)
        #     stage "mma"      (50_000 + 1_300): 51_300 - 40_300 = 11_000 (11.0 us)
        #     mark  "tail_mk"  (50_000 + 6_000): 56_000 - 40_300 = 15_700 (15.7 us)
        trace = {
            "traceEvents": [
                {"ph": "X", "ts": 1.3, "dur": 3.0, "name": "mma", "args": {"cta": 0, "warp": 0}},
                {"ph": "i", "ts": 6.0, "name": "tail_mk", "args": {"cta": 0, "warp": 0}},
            ],
            "otherData": {},
        }
        summary = {
            "observation_start_ns": 50_000,
            "observation_end_ns": 56_001,
            "first_open_ns": 50_300,
            "wpc": 1,
            "cta_sm": {"0": 2},
            "lane_bounds_ns": {"0": [50_300, 56_001, 0]},
        }
        aligned = merge.align(trace, summary, (10_000, 16_000, "k_over_1ns", []), {})
        self.assertEqual(aligned["otherData"]["alignment"]["offset_ns"], -40_300)
        self.assertEqual(
            _software_events_by_name_and_kind(aligned),
            [
                ("X", "residency", "Warp residency", 10_000, 10.0),
                ("X", "stage", "mma", 11_000, 11.0),
                ("i", "mark", "tail_mk", 15_700, 15.7),
            ],
        )

    def test_align_missing_observation_end_ns_and_no_lanes_fallback(self) -> None:
        """Case 4: missing `observation_end_ns` defaults to `origin` (`:229`), including empty-lanes fallback (`:351-356`)."""
        # Hand calculation:
        #   window: start = 10_000, end = 14_000 -> dur_ns = 4_000
        #   summary: origin = 40_000, no `observation_end_ns` -> obs_end = 40_000
        #   lower = 10_000 - 40_000 = -30_000
        #   upper = 14_000 - 40_000 = -26_000
        #   offset_ns = (-30_000 + -26_000) // 2 = -28_000
        #   Shifted events (- 28_000):
        #     ts=0.5 us (500 ns) -> 40_000 + 500 - 28_000 = 12_500 (12.5 us)
        #     ts=1.5 us (1500 ns) -> 40_000 + 1_500 - 28_000 = 13_500 (13.5 us)
        trace = {
            "traceEvents": [
                {"ph": "X", "pid": 1, "tid": 1, "ts": 0.5, "dur": 0.5, "name": "load", "args": {"cta": 0, "warp": 0}},
                {"ph": "i", "pid": 1, "tid": 1, "ts": 1.5, "name": "done", "args": {"cta": 0, "warp": 0}},
            ],
            "otherData": {},
        }
        summary_with_lanes = {
            "observation_start_ns": 40_000,
            "first_open_ns": 40_100,
            "wpc": 1,
            "cta_sm": {"0": 0},
            "lane_bounds_ns": {"0": [40_000, 41_500, 0]},
        }
        aligned_lanes = merge.align(trace, summary_with_lanes, (10_000, 14_000, "k_no_obsend", []), {})
        self.assertEqual(aligned_lanes["otherData"]["alignment"]["offset_ns"], -28_000)
        self.assertEqual(
            _software_events_by_name_and_kind(aligned_lanes),
            [
                ("X", "residency", "Warp residency", 12_000, 12.0),
                ("X", "stage", "load", 12_500, 12.5),
                ("i", "mark", "done", 13_500, 13.5),
            ],
        )

        # Fallback branch (`:351-356`) when `cta_sm` / `lane_bounds_ns` are empty
        summary_no_lanes = {
            "observation_start_ns": 40_000,
            "first_open_ns": 40_100,
        }
        aligned_fallback = merge.align(trace, summary_no_lanes, (10_000, 14_000, "k_no_lanes", []), {})
        self.assertEqual(aligned_fallback["otherData"]["alignment"]["offset_ns"], -28_000)
        sw_fallback = [(e["name"], e["ts_ns"], e["ts"]) for e in aligned_fallback["traceEvents"] if e.get("pid") == 1]
        self.assertEqual(sw_fallback, [("load", 12_500, 12.5), ("done", 13_500, 13.5)])

    def test_align_zero_and_negative_window_duration_raise_value_error(self) -> None:
        """Case 5: zero or negative hardware window length raises `ValueError('Invalid hardware dispatch duration')`."""
        summary = {"observation_start_ns": 1_000, "observation_end_ns": 2_000, "first_open_ns": 1_000}
        for bad_window in ((10_000, 10_000, "k_zero", []), (10_000, 9_999, "k_neg", [])):
            with self.assertRaises(ValueError) as ctx:
                merge.align({"traceEvents": []}, summary, bad_window, {})
            self.assertEqual(str(ctx.exception), "Invalid hardware dispatch duration")

    def test_align_lane_bounds_clamped_to_window_and_residency_curve_endpoints(self) -> None:
        """Case 6: `lane_bounds_ns` exceeding `[start, end]` are clamped to `[0, dur_ns]` (`:248-252`)."""
        # Hand calculation:
        #   window: start = 10_000, end = 14_000 -> dur_ns = 4_000, bin_ns = 2_000 (2 bins + trailing end=0.0)
        #   summary: origin = 50_000, obs_end = 56_000 (span = 6_000 > 4_000), first_open_ns = 50_500
        #   lower = -40_000, upper = -42_000 (lower > upper) -> offset_ns = 10_000 - 50_500 = -40_500
        #   CTA 0 (SM 1) lane "0": raw [50_000, 52_500] -> shifted [9_500, 12_000] -> rel [-500, 2_000] -> clamped [0, 2_000]
        #   CTA 1 (SM 2) lane "1": raw [52_500, 56_000] -> shifted [12_000, 15_500] -> rel [2_000, 5_500] -> clamped [2_000, 4_000]
        #   CTA 2 (SM 2) lane "2": raw [54_700, 55_300] -> shifted [14_200, 14_800] -> rel [4_200, 4_800]:
        #     `b_rel = max(0, 4_200) = 4_200`, `e_rel = min(4_000, 4_800) = 4_000`.
        #     Because CTA 1's `e_rel` was clamped to `4_000 <= 4_200`, greedy slot allocator (`:263`) assigns CTA 2
        #     to `resident_slot == 0` (`tid == 1`), and `_interval_union` drops `(4_200, 4_000)` (`b <= a`).
        #   Physical SM residency curve (`total_hw_units = 4`):
        #     bin 0 (ts_ns=10_000): active_union exposure = 1.0 CTA -> 100.0 * 1.0 / 4 = 25.0
        #     bin 1 (ts_ns=12_000): active_union exposure = 1.0 CTA -> 100.0 * 1.0 / 4 = 25.0
        #     end   (ts_ns=14_000): trailing zero point -> 0.0
        trace = {
            "traceEvents": [
                {"ph": "X", "ts": 1.0, "dur": 1.0, "name": "load", "args": {"cta": 0, "warp": 0}},
            ],
            "otherData": {},
        }
        summary = {
            "observation_start_ns": 50_000,
            "observation_end_ns": 56_000,
            "first_open_ns": 50_500,
            "wpc": 1,
            "cta_sm": {"0": 1, "1": 2, "2": 2},
            "lane_bounds_ns": {
                "0": [50_000, 52_500, 0],
                "1": [52_500, 56_000, 0],
                "2": [54_700, 55_300, 0],
            },
        }
        aligned = merge.align(trace, summary, (10_000, 14_000, "k_clamp", []), {"total_hw_units": 4})
        al = aligned["otherData"]["alignment"]
        self.assertEqual(al["offset_ns"], -40_500)
        self.assertEqual(al["ctas_observed"], 3)
        self.assertEqual(al["physical_units_observed"], 2)
        self.assertEqual(
            _software_events_by_name_and_kind(aligned),
            [
                ("X", "residency", "Warp residency", 9_500, 9.5),
                ("X", "stage", "load", 10_500, 10.5),
                ("X", "residency", "Warp residency", 12_000, 12.0),
                ("X", "residency", "Warp residency", 14_200, 14.2),
            ],
        )
        res_slots = [
            (int(ev["args"]["cta"]), int(ev["args"]["resident_slot"]), int(ev["tid"]))
            for ev in aligned["traceEvents"]
            if ev.get("ph") == "X" and (ev.get("args") or {}).get("kind") == "residency"
        ]
        self.assertEqual(res_slots, [(0, 0, 1), (1, 0, 1), (2, 0, 1)])
        res_curve = [
            (int(ev["ts_ns"]), float(ev["args"]["value"]))
            for ev in aligned["traceEvents"]
            if ev.get("ph") == "C" and ev.get("name") == "Active SM Residency (%) [4 SMs]"
        ]
        self.assertEqual(res_curve, [(10_000, 25.0), (12_000, 25.0), (14_000, 0.0)])
        self.assertEqual(res_curve[0], (10_000, 25.0))
        self.assertEqual(res_curve[-1], (14_000, 0.0))

        # Pre-window CTA span `[-500, -300]` (`[50_000, 50_200]` + offset `-40_500` = `[9_500, 9_700]` before `start=10_000`):
        # `:248-249` clamps `b_rel = max(0, -500) = 0` and `e_rel = min(100, max(0 + 1, -300)) = 1`, producing a 1-ns stub:
        # with `dur_ns = 100` (`widths = [100]`) and `total_hw_units = 4`, bin 0 = `100.0 * (1 / 100) / 4 = 0.25`.
        pre_summary = {
            "observation_start_ns": 50_000,
            "observation_end_ns": 51_000,
            "first_open_ns": 50_500,
            "wpc": 1,
            "cta_sm": {"0": 1},
            "lane_bounds_ns": {"0": [50_000, 50_200, 0]},
        }
        aligned_pre = merge.align({"traceEvents": []}, pre_summary, (10_000, 10_100, "k_pre", []), {"total_hw_units": 4})
        pre_curve = [
            (int(ev["ts_ns"]), float(ev["args"]["value"]))
            for ev in aligned_pre["traceEvents"]
            if ev.get("ph") == "C" and ev.get("name") == "Active SM Residency (%) [4 SMs]"
        ]
        self.assertEqual(pre_curve, [(10_000, 0.25), (10_100, 0.0)])

    def test_interval_union_and_bin_exposure_exact_literals(self) -> None:
        self.assertEqual(
            merge._interval_union([(50, 70), (10, 25), (20, 30), (30, 40), (35, 35), (60, 55)]),
            [(10, 40), (50, 70)],
        )
        # Hand calculation for `_bin_exposure([(-500, 4_500)], dur_ns=5_000, bin_ns=2_000)`:
        #   clamped interval = [0, 4_500]; n = ceil(5000 / 2000) = 3; widths = [2000, 2000, 1000]
        #   areas = [2000.0, 2000.0, 500.0] -> exposures = [1.0, 1.0, 0.5]
        exp, widths = merge._bin_exposure([(-500, 4_500), (4_800, 4_800)], 5_000, 2_000)
        self.assertEqual(widths, [2000, 2000, 1000])
        self.assertEqual(exp, [1.0, 1.0, 0.5])

    def test_hardware_window_nsys_pm_json_and_error_paths(self) -> None:
        case_dir = self._new_case_dir("hw_win")
        with self.assertRaises(ValueError) as empty_k:
            merge.hardware_window(case_dir, {"kernel_name": ""})
        self.assertEqual(str(empty_k.exception), "A hardware kernel name is required")

        with self.assertRaises(ValueError) as no_cap:
            merge.hardware_window(case_dir, {"kernel_name": "gemm_k"})
        self.assertEqual(
            str(no_cap.exception),
            "Missing hardware capture (capture.sqlite, pm.json, rocprof.json or rocprof-sys.pftrace)",
        )

        # 1) `capture.sqlite` (`_nsys`, `_select`, `_nsys_gpu_name`)
        sq_dir = case_dir / "nsys_case"
        sq_dir.mkdir()
        db = sqlite3.connect(str(sq_dir / "capture.sqlite"))
        try:
            db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            db.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER)"
            )
            db.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT)")
            db.execute("CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT)")
            db.execute("CREATE TABLE GPU_METRICS (timestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL)")
            db.execute("INSERT INTO StringIds VALUES (1, 'void gemm_k(float*)')")
            db.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H20')")
            db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (1, 10, 'Tensor Active [Throughput %]')")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000, 3000, 0, 1)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (5000, 9000, 0, 1)")
            db.execute("INSERT INTO GPU_METRICS VALUES (2000, 1, 10, 20.0)")
            db.execute("INSERT INTO GPU_METRICS VALUES (7000, 1, 10, 24.0)")
            db.commit()
        finally:
            db.close()

        with self.assertRaises(ValueError) as multi_err:
            merge.hardware_window(sq_dir, {"kernel_name": "gemm_k"})
        self.assertEqual(str(multi_err.exception), "Expected one matching hardware dispatch, found 2; specify capture_instance")

        win1 = merge.hardware_window(sq_dir, {"kernel_name": "gemm_k", "capture_instance": 1})
        self.assertEqual(tuple(win1), (5000, 9000, "void gemm_k(float*)", [(7000, "Tensor Active [Throughput %]", 24.0)]))
        self.assertEqual(win1.gpu_name, "NVIDIA H20")

        # 2) `pm.json` (`_pm_json`)
        # Hand calculation:
        #   start=2_000, dur=10_000 -> end=12_000
        #   mfma_pct = 100 * 40_000 / (1_000 * 80) = 50.0
        #   hbm_pct  = 100 * (460.5 + 460.5) * 64 / (10_000 * 3684) = 100 * 58_944 / 36_840_000 = 0.16
        #   SIMD_UTILIZATION = 0.75 (<= 1.5) -> 75.0; GPU_UTIL = 88.5
        pm_dir = case_dir / "pm_case"
        pm_dir.mkdir()
        (pm_dir / "pm.json").write_text(
            json.dumps(
                {
                    "dispatch": {"ts": 2_000, "dur": 10_000, "name": "amd_gemm"},
                    "raw": {
                        "GRBM_GUI_ACTIVE": 1_000.0,
                        "SQ_VALU_MFMA_BUSY_CYCLES": 40_000.0,
                        "TCC_EA0_RDREQ_DRAM": 460.5,
                        "TCC_EA0_WRREQ_DRAM": 460.5,
                        "SIMD_UTILIZATION": 0.75,
                        "GPU_UTIL": 88.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        pm_win = merge.hardware_window(pm_dir, {"kernel_name": "amd_gemm"})
        self.assertEqual(
            pm_win,
            (
                2_000,
                12_000,
                "amd_gemm",
                [
                    (2_000, "Compute Util — Matrix Core MFMA (%) [ROCm PMC]", 50.0),
                    (12_000, "Compute Util — Matrix Core MFMA (%) [ROCm PMC]", 50.0),
                    (2_000, "Memory Util — HBM3 Read+Write (%) [ROCm PMC]", 0.16),
                    (12_000, "Memory Util — HBM3 Read+Write (%) [ROCm PMC]", 0.16),
                    (2_000, "SIMD VALU Util (%) [ROCm PMC]", 75.0),
                    (12_000, "SIMD VALU Util (%) [ROCm PMC]", 75.0),
                    (2_000, "Active CUs / GPU Busy (%) [ROCm PMC]", 88.5),
                    (12_000, "Active CUs / GPU Busy (%) [ROCm PMC]", 88.5),
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()

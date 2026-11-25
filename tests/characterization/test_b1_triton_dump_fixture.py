# -*- coding: utf-8 -*-
"""Characterization tests for Tier B1-G4: real H20 Triton FILITE02 dumps -> wp_decode loop
and pure-CPU ``wp_triton_amd._timer_contract`` validation in a torch-free environment.

Independent Derivation of Expected Counts and Record Layout
-----------------------------------------------------------
The two read-only fixtures under ``tests/characterization/fixtures/b1_triton/{t0,t1}/``
were captured on an NVIDIA H20 GPU (``sm_90a``, commit ``70dd64dca40a8afcdd6220d6d1d6d3dac4f49caf``)
by running a minimal 1D-grid Triton GEMM kernel ``min_gemm_kernel``:

1. Grid and CTA / warp geometry:
   - Problem size: ``M = 128, N = 64, K = 512``;
     Tile size: ``BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32`` (``512 / 32 = 16`` K-loop steps).
   - 1D grid: ``GRID = (M // BLOCK_M,) = (2,)``, so ``N_CTAS = 2`` CTAs (``pid in {0, 1}``).
   - ``NUM_WARPS = 4`` warps per CTA (``4 * 32 = 128`` threads per CTA).
   - Recorder switches: ``WP_GRAN = 0`` (warp granularity: lane 0 of every warp writes its own slot)
     and ``WP_LEVEL = 2`` (all spans and instant marks enabled).
   - Total slots: ``n_slots = N_CTAS * NUM_WARPS = 2 * 4 = 8`` (``slot 0..3`` for CTA 0,
     ``slot 4..7`` for CTA 1), ``warps_per_cta = 4``.

2. Per-warp (per-slot) instrumentation call sequence inside ``min_gemm_kernel``:
   (a) ``wp_open(role=0)``:
       - ``WP_TIMER = 0`` (``t0``): writes 1 physical 16-byte record:
         ``OPEN | role(0)`` (``tag = 0x40000000``, ``aux = %smid``, ``t = %globaltimer`` ns).
       - ``WP_TIMER = 1`` (``t1``): writes 2 physical 16-byte records:
         1) ``OPEN | role(0)`` (``tag = 0x40000000``, ``aux = %smid``, ``t = %globaltimer`` ns),
         2) ``CLK`` open anchor (``tag = 0x10000000``, ``aux = 0``, ``t = %clock64`` cycles).
   (b) 7 body instrumentation calls (identical in ``t0`` and ``t1``, all passing ``aux = pid``):
       1) ``wp_b(EV_GEMM=10)``      -> ``tag = 0x0000000a`` (outer ``gemm`` span begin)
       2) ``wp_b(EV_MMA=3)``        -> ``tag = 0x00000003`` (nested ``mma`` span begin around K-loop)
       3) ``wp_e(EV_MMA=3)``        -> ``tag = 0x80000003`` (nested ``mma`` span end)
       4) ``wp_m(EV_EPILOGUE=5)``   -> ``tag = 0x20000005`` (instant mark after fp32->fp16 cast)
       5) ``wp_e(EV_GEMM=10)``      -> ``tag = 0x8000000a`` (outer ``gemm`` span end)
       6) ``wp_b(EV_STORE=6)``      -> ``tag = 0x00000006`` (sequential ``store`` span begin)
       7) ``wp_e(EV_STORE=6)``      -> ``tag = 0x80000006`` (sequential ``store`` span end)
   (c) ``wp_close()``:
       - ``WP_TIMER = 0`` (``t0``): writes 1 physical 16-byte record:
         ``CLOSE`` (``tag = 0x08000000``, ``aux = 0``, ``t = %globaltimer`` ns).
       - ``WP_TIMER = 1`` (``t1``): writes 2 physical 16-byte records:
         1) ``CLOSE`` (``tag = 0x08000000``, ``aux = 0``, ``t = %globaltimer`` ns),
         2) ``CLK`` close anchor (``tag = 0x10000000``, ``aux = 1``, ``t = %clock64`` cycles).

3. Derived physical vs decoded counts:
   - ``t0`` (``WP_TIMER = 0``): ``1 + 7 + 1 = 9`` physical records per slot across 8 slots
     (``72`` physical records = ``1,152`` payload bytes; minimum non-overflow ``cap = 9``).
   - ``t1`` (``WP_TIMER = 1``): ``2 + 7 + 2 = 11`` physical records per slot across 8 slots
     (``88`` physical records = ``1,408`` payload bytes; minimum non-overflow ``cap = 11`` because
     ``wp_close`` writes 2 records starting at ``n = 9``, requiring ``lane_full_at = cap - 1 > 9``).
   - The 2 extra physical records per slot in ``t1`` vs ``t0`` are at physical index ``1``
     (``tag = 0x10000000, aux = 0``, open ``CLK`` anchor) and physical index ``10``
     (``tag = 0x10000000, aux = 1``, close ``CLK`` anchor).
   - During ``read_dump()``, ``to_globaltimer()`` fits ``ns_per_clk = (g1 - g0) / (k1 - k0)`` per
     slot, converts the 7 body ``%clock64`` timestamps to ``%globaltimer`` ns, and strips both
     ``CLK`` anchor records, leaving ``9`` decoded records per slot (``72`` total), ``3`` spans per
     slot (``8 * 3 = 24`` total: ``8`` ``gemm``, ``8`` ``mma``, ``8`` ``store``), and ``1`` instant
     mark per slot (``8 * 1 = 8`` total: ``epilogue``).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import ipaddress
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weaveprof.megakernel import load_wp_decode  # noqa: E402

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "b1_triton"
T0_DIR = FIXTURE_ROOT / "t0"
T1_DIR = FIXTURE_ROOT / "t1"

EXPECTED_T0_BIN_SHA256 = "7b95fd2f372cd749d41b7ee293768933a7953f49a82cd54a8220f22216577b84"
EXPECTED_T1_BIN_SHA256 = "c030dbc8b20da3c47c02d118c86c1d32c9d67880f332d73ad674c5699dd0c119"
EXPECTED_NAMES_SHA256 = "f50ec571d98e928c65637bd4ad74813e1002f160fe6f5dcaf1d5746938a41825"

# Independently specified wire constants from the FILITE02 specification (not imported from SUT).
RAW_HEADER_STRUCT = struct.Struct("<6I")
RAW_RECORD_STRUCT = struct.Struct("<QII")
WIRE_TAG_END = 0x80000000
WIRE_TAG_OPEN = 0x40000000
WIRE_TAG_MARK = 0x20000000
WIRE_TAG_CLK = 0x10000000
WIRE_TAG_CLOSE = 0x08000000

EV_MMA_ID = 3
EV_EPILOGUE_ID = 5
EV_STORE_ID = 6
EV_GEMM_ID = 10


def _is_loopback_host(host: object) -> bool:
    if host in ("127.0.0.1", "::1", "localhost", b"127.0.0.1", b"::1", b"localhost", ""):
        return True
    if not isinstance(host, str):
        return False
    if host.startswith("127."):
        return True
    return host == "0.0.0.0"


def _unpack_raw_filite02(raw: bytes) -> dict:
    magic = raw[:8]
    n_slots, cap, warps_per_cta, overflow, timer, meta_len = RAW_HEADER_STRUCT.unpack_from(raw, 8)
    meta_off = 32
    meta = json.loads(raw[meta_off : meta_off + meta_len].decode("utf-8"))
    cnt_off = meta_off + meta_len
    cnt = list(struct.unpack_from(f"<{n_slots}I", raw, cnt_off))
    rec_off = cnt_off + 4 * n_slots
    slots = {}
    cursor = rec_off
    for s in range(n_slots):
        n = cnt[s]
        end = cursor + 16 * n
        slots[s] = list(RAW_RECORD_STRUCT.iter_unpack(raw[cursor:end]))
        cursor = end
    return {
        "magic": magic,
        "n_slots": n_slots,
        "cap": cap,
        "warps_per_cta": warps_per_cta,
        "overflow": overflow,
        "timer": timer,
        "meta_len": meta_len,
        "meta": meta,
        "cnt": cnt,
        "rec_off": rec_off,
        "end_off": cursor,
        "slots": slots,
    }


SUBPROCESS_SITECUSTOMIZE = """\
import socket

_orig_socket = socket.socket
_LoopbackAllowed = {"127.0.0.1", "::1", "localhost", ""}

class _GuardedSocket(_orig_socket):
    def connect(self, address):
        host = address[0] if isinstance(address, tuple) and address else ""
        if host not in _LoopbackAllowed and not (isinstance(host, str) and host.startswith("127.")):
            raise RuntimeError(f"External network blocked in test subprocess: {address!r}")
        return super().connect(address)

    def connect_ex(self, address):
        host = address[0] if isinstance(address, tuple) and address else ""
        if host not in _LoopbackAllowed and not (isinstance(host, str) and host.startswith("127.")):
            raise RuntimeError(f"External network blocked in test subprocess: {address!r}")
        return super().connect_ex(address)

socket.socket = _GuardedSocket

def _blocked_getaddrinfo(host, *args, **kwargs):
    if host not in _LoopbackAllowed and not (isinstance(host, str) and host.startswith("127.")):
        raise RuntimeError(f"External DNS lookup blocked in test subprocess: {host!r}")
    return _orig_getaddrinfo(host, *args, **kwargs)

_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = _blocked_getaddrinfo
"""


class TritonDumpFixtureDecodeTest(unittest.TestCase):
    """Pin real H20 Triton FILITE02 fixture decoding, negative mutations, and AMD timer contract."""

    def setUp(self) -> None:
        orig_socket = socket.socket
        orig_create_connection = socket.create_connection
        orig_getaddrinfo = socket.getaddrinfo

        class GuardedSocket(orig_socket):
            def connect(self, address):  # type: ignore[override]
                host = address[0] if isinstance(address, tuple) and address else ""
                if not _is_loopback_host(host):
                    raise RuntimeError(f"External network blocked: {address!r}")
                return super().connect(address)

            def connect_ex(self, address):  # type: ignore[override]
                host = address[0] if isinstance(address, tuple) and address else ""
                if not _is_loopback_host(host):
                    raise RuntimeError(f"External network blocked: {address!r}")
                return super().connect_ex(address)

        def guarded_create_connection(address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) and address else ""
            if not _is_loopback_host(host):
                raise RuntimeError(f"External create_connection blocked: {address!r}")
            return orig_create_connection(address, *args, **kwargs)

        def guarded_getaddrinfo(host, *args, **kwargs):
            if not _is_loopback_host(host):
                if isinstance(host, str):
                    addr = ipaddress.ip_address(host) if host.replace(".", "").isdigit() else None
                    if addr is None or not addr.is_loopback:
                        raise RuntimeError(f"External DNS lookup blocked: {host!r}")
                else:
                    raise RuntimeError(f"External DNS lookup blocked: {host!r}")
            return orig_getaddrinfo(host, *args, **kwargs)

        socket.socket = GuardedSocket  # type: ignore[assignment]
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
        self.addCleanup(setattr, socket, "socket", orig_socket)
        self.addCleanup(setattr, socket, "create_connection", orig_create_connection)
        self.addCleanup(setattr, socket, "getaddrinfo", orig_getaddrinfo)

        self.wp = load_wp_decode()

    def test_fixture_sha256_headers_slot_counts_and_t1_extra_clk_anchors(self) -> None:
        t0_bin = T0_DIR / "capture.fi02.bin"
        t0_names = T0_DIR / "names.json"
        t1_bin = T1_DIR / "capture.fi02.bin"
        t1_names = T1_DIR / "names.json"

        raw0 = t0_bin.read_bytes()
        raw1 = t1_bin.read_bytes()
        names0_bytes = t0_names.read_bytes()
        names1_bytes = t1_names.read_bytes()

        self.assertEqual(len(raw0), 1663)
        self.assertEqual(len(raw1), 1914)
        self.assertEqual(len(names0_bytes), 494)
        self.assertEqual(len(names1_bytes), 494)
        self.assertEqual(hashlib.sha256(raw0).hexdigest(), EXPECTED_T0_BIN_SHA256)
        self.assertEqual(hashlib.sha256(raw1).hexdigest(), EXPECTED_T1_BIN_SHA256)
        self.assertEqual(hashlib.sha256(names0_bytes).hexdigest(), EXPECTED_NAMES_SHA256)
        self.assertEqual(hashlib.sha256(names1_bytes).hexdigest(), EXPECTED_NAMES_SHA256)

        # Raw physical inspection of FILITE02 layout before decoder consumption.
        u0 = _unpack_raw_filite02(raw0)
        u1 = _unpack_raw_filite02(raw1)
        self.assertEqual(u0["magic"], b"FILITE02")
        self.assertEqual(u1["magic"], b"FILITE02")
        self.assertEqual(
            (u0["n_slots"], u0["cap"], u0["warps_per_cta"], u0["overflow"], u0["timer"], u0["meta_len"]),
            (8, 9, 4, 0, 0, 447),
        )
        self.assertEqual(
            (u1["n_slots"], u1["cap"], u1["warps_per_cta"], u1["overflow"], u1["timer"], u1["meta_len"]),
            (8, 11, 4, 0, 1, 442),
        )
        self.assertEqual(u0["cnt"], [9, 9, 9, 9, 9, 9, 9, 9])
        self.assertEqual(u1["cnt"], [11, 11, 11, 11, 11, 11, 11, 11])
        self.assertEqual(u0["rec_off"], 511)
        self.assertEqual(u0["end_off"], 1663)
        self.assertEqual(u1["rec_off"], 506)
        self.assertEqual(u1["end_off"], 1914)

        # Derived expected (tag, aux) sequence per slot for t0 (9 physical records) and t1 (11 physical records).
        for slot in range(8):
            cta = slot // 4
            expected_sm = 64 if cta == 0 else 66
            expected_9_tag_aux = [
                (WIRE_TAG_OPEN | 0, expected_sm),
                (EV_GEMM_ID, cta),
                (EV_MMA_ID, cta),
                (WIRE_TAG_END | EV_MMA_ID, cta),
                (WIRE_TAG_MARK | EV_EPILOGUE_ID, cta),
                (WIRE_TAG_END | EV_GEMM_ID, cta),
                (EV_STORE_ID, cta),
                (WIRE_TAG_END | EV_STORE_ID, cta),
                (WIRE_TAG_CLOSE, 0),
            ]
            recs0 = u0["slots"][slot]
            recs1 = u1["slots"][slot]
            self.assertEqual(len(recs0), 9)
            self.assertEqual(len(recs1), 11)
            self.assertEqual([(tag, aux) for _, tag, aux in recs0], expected_9_tag_aux)

            # Pin the exact 2 extra physical records in timer=1 vs timer=0:
            # index 1 is open CLK anchor (tag=0x10000000, aux=0) and index 10 is close CLK anchor (tag=0x10000000, aux=1).
            self.assertEqual((recs1[1][1], recs1[1][2]), (WIRE_TAG_CLK, 0))
            self.assertEqual((recs1[10][1], recs1[10][2]), (WIRE_TAG_CLK, 1))
            t1_without_clk_anchors = [r for idx, r in enumerate(recs1) if idx not in (1, 10)]
            self.assertEqual([(tag, aux) for _, tag, aux in t1_without_clk_anchors], expected_9_tag_aux)

            # Verify timestamp domains: t0 uses ~1.795e18 ns wall clock throughout;
            # t1 uses ~1.795e18 ns at OPEN/CLOSE (indices 0, 9) and ~5.41e14 clock64 cycles at indices 1..8 and 10.
            self.assertGreater(recs0[0][0], 1_700_000_000_000_000_000)
            self.assertGreater(recs1[0][0], 1_700_000_000_000_000_000)
            self.assertGreater(recs1[9][0], 1_700_000_000_000_000_000)
            for clk_idx in (1, 2, 3, 4, 5, 6, 7, 8, 10):
                self.assertLess(recs1[clk_idx][0], 1_000_000_000_000_000)
                self.assertGreater(recs1[clk_idx][0], 100_000_000_000_000)

        # Verify read_dump() header fields, anchor stripping, and linear clock64 conversion.
        d0 = self.wp.read_dump(t0_bin)
        d1 = self.wp.read_dump(t1_bin)
        self.assertEqual(d0["magic"], "FILITE02")
        self.assertEqual(d0["meta"]["fi_version"], "fast-instrument-triton")
        self.assertEqual((d0["n_slots"], d0["cap"], d0["wpc"], d0["overflow"], d0["timer"]), (8, 9, 4, 0, 0))
        self.assertEqual(d0["cnt"], [9, 9, 9, 9, 9, 9, 9, 9])
        self.assertIs(d0["amd"], False)
        self.assertEqual(d0["platform_source"], "meta.backend")
        self.assertEqual(d0["timer_hz"], 1000000000)
        self.assertIs(d0["timer_hz_applied"], False)
        self.assertEqual(
            d0["meta"]["timer_hz_ignored"],
            "NVIDIA 转储（platform=nvidia，backend=nvidia_ptx）：%globaltimer 记录值已是 ns，timer_hz=1000000000 不当换算系数用",
        )
        self.assertEqual(d0["clock_fit"], {})
        self.assertEqual(d0["gran"], 0)
        self.assertEqual([len(d0["slots"][s]) for s in range(8)], [9, 9, 9, 9, 9, 9, 9, 9])

        self.assertEqual(d1["magic"], "FILITE02")
        self.assertEqual(d1["meta"]["fi_version"], "fast-instrument-triton")
        self.assertEqual((d1["n_slots"], d1["cap"], d1["wpc"], d1["overflow"], d1["timer"]), (8, 11, 4, 0, 1))
        self.assertEqual(d1["cnt"], [11, 11, 11, 11, 11, 11, 11, 11])
        self.assertIs(d1["amd"], False)
        self.assertEqual(d1["platform_source"], "meta.backend")
        self.assertEqual(d1["timer_hz"], 0)
        self.assertIs(d1["timer_hz_applied"], False)
        self.assertEqual(d1["gran"], 0)
        self.assertEqual(
            d1["clock_fit"],
            {0: 1738.8, 1: 1737.9, 2: 1742.0, 3: 1724.4, 4: 1738.5, 5: 1720.0, 6: 1724.1, 7: 1724.1},
        )
        # After to_globaltimer() consumes the 2 CLK anchors per slot, all 8 slots have 9 decoded records.
        self.assertEqual([len(d1["slots"][s]) for s in range(8)], [9, 9, 9, 9, 9, 9, 9, 9])
        for slot in range(8):
            self.assertEqual([tag & WIRE_TAG_CLK for _, tag, _ in d1["slots"][slot]], [0] * 9)

        # Hand-calculated linear interpolation check on slot 0 of t1:
        # g0 = 1795625014331764352, k0 = 541939155112383, g1 = 1795625014331767488, k1 = 541939155117836
        # dg = 3136 ns, dk = 5453 cycles -> ns_per_clk = 3136 / 5453, MHz = round(1000 * 5453 / 3136, 1) = 1738.8
        raw_s0 = u1["slots"][0]
        self.assertEqual((raw_s0[0][0], raw_s0[1][0]), (1795625014331764352, 541939155112383))
        self.assertEqual((raw_s0[9][0], raw_s0[10][0]), (1795625014331767488, 541939155117836))
        expected_slot0_ns = [
            1795625014331764352,
            1795625014331764404,
            1795625014331764431,
            1795625014331767191,
            1795625014331767238,
            1795625014331767265,
            1795625014331767295,
            1795625014331767448,
            1795625014331767488,
        ]
        self.assertEqual([t for t, _, _ in d1["slots"][0]], expected_slot0_ns)

    def test_decode_summary_names_mapping_cli_write_gz_and_suspected_bug_T1(self) -> None:
        tr0, sm0 = self.wp.decode_file(T0_DIR / "capture.fi02.bin", T0_DIR / "names.json")
        tr1, sm1 = self.wp.decode_file(T1_DIR / "capture.fi02.bin", T1_DIR / "names.json")

        self.assertEqual(sm0["source_dump_sha256"], EXPECTED_T0_BIN_SHA256)
        self.assertEqual(sm1["source_dump_sha256"], EXPECTED_T1_BIN_SHA256)

        for sm in (sm0, sm1):
            self.assertEqual(sm["magic"], "FILITE02")
            self.assertEqual(sm["overflow"], 0)
            self.assertEqual(sm["errors"], {})
            self.assertEqual(sm["n_slots"], 8)
            self.assertEqual(sm["wpc"], 4)
            self.assertEqual(sm["n_ctas"], 2)
            self.assertEqual(sm["lanes"], 8)
            self.assertEqual(sm["lanes_with_sublanes"], 0)
            self.assertEqual(sm["max_sublanes"], 1)
            self.assertEqual(sm["records"], 72)
            self.assertEqual(sm["spans"], 24)
            self.assertEqual(sm["marks"], 8)
            self.assertEqual(sm["cta_sm"], {"0": 64, "1": 66})
            self.assertEqual(sm["cta_hw"], {})
            self.assertEqual(sm["platform"], "nvidia")
            self.assertEqual(sm["platform_source"], "meta.backend")
            self.assertEqual(sm["arch"], "sm_90a")
            self.assertEqual(sm["gran"], 0)
            self.assertEqual(sm["lane_kind"], "warp")
            self.assertEqual(sm["group"], "role")
            self.assertEqual(sm["pair"], "stack")
            self.assertIs(sm["semantics_ok"], True)
            self.assertEqual(sm["semantic_schema"], "weaveprofiler-semantic-contract/v1")
            self.assertEqual(sm["unnamed_event_ids"], [])
            self.assertEqual(sm["semantic_unknown_event_ids"], [])
            self.assertEqual(sm["semantic_ambiguous_event_ids"], [])
            self.assertEqual(sm["semantic_errors"], [])

        # Timer-specific summary fields and event statistics.
        self.assertEqual(sm0["timer"], "globaltimer")
        self.assertEqual(sm0["timer_hz"], 1000000000)
        self.assertIsNone(sm0["clock_mhz_per_lane"])
        self.assertEqual(sm0["timer_min_step_ns"], 32)
        self.assertEqual(
            sm0["events"],
            {
                "gemm": {"n": 8, "total_us": 22.912, "mean_us": 2.864, "p50_us": 2.88, "max_us": 2.88},
                "mma": {"n": 8, "total_us": 22.016, "mean_us": 2.752, "p50_us": 2.752, "max_us": 2.752},
                "store": {"n": 8, "total_us": 1.216, "mean_us": 0.152, "p50_us": 0.16, "max_us": 0.16},
            },
        )
        self.assertEqual(
            sm0["roles"],
            {"gemm_warp": {"ctas": 2, "end_min_us": 3.136, "end_median_us": 3.136, "end_max_us": 3.136, "tail_us": 0.0}},
        )

        self.assertEqual(sm1["timer"], "clock64+锚点")
        self.assertIsNone(sm1["timer_hz"])
        self.assertEqual(sm1["clock_mhz_per_lane"], {"min": 1720.0, "max": 1742.0})
        self.assertEqual(sm1["timer_min_step_ns"], 27)
        self.assertEqual(
            sm1["events"],
            {
                "gemm": {"n": 8, "total_us": 22.998, "mean_us": 2.875, "p50_us": 2.887, "max_us": 2.893},
                "mma": {"n": 8, "total_us": 22.168, "mean_us": 2.771, "p50_us": 2.781, "max_us": 2.789},
                "store": {"n": 8, "total_us": 1.288, "mean_us": 0.161, "p50_us": 0.166, "max_us": 0.167},
            },
        )
        self.assertEqual(
            sm1["roles"],
            {"gemm_warp": {"ctas": 2, "end_min_us": 3.168, "end_median_us": 3.168, "end_max_us": 3.168, "tail_us": 0.0}},
        )

        # Pin Suspected Bug T1 (fi_lite_decode.py:734):
        # observation_span_us computes `round(max(all_t) / 1000.0 - t0 / 1000.0, 3)` instead of
        # subtracting integer ns before `/ 1000.0`, losing float64 precision (~0.25 us ULP at ~1.8e15 us).
        self.assertEqual(sm0["observation_end_ns"] - sm0["observation_start_ns"], 3136)
        self.assertEqual(sm0["span_us"], 3.0)
        self.assertEqual((sm0["semantic_start_us"], sm0["semantic_end_us"], sm0["semantic_span_us"]), (0.064, 3.136, 3.072))
        self.assertEqual(sm1["observation_end_ns"] - sm1["observation_start_ns"], 3168)
        self.assertEqual(sm1["span_us"], 3.25)
        self.assertEqual((sm1["semantic_start_us"], sm1["semantic_end_us"], sm1["semantic_span_us"]), (0.043, 3.127, 3.084))

        # Pin event name / role / aux / semantic correspondence in Perfetto traceEvents.
        for tr in (tr0, tr1):
            self.assertEqual(tr["displayTimeUnit"], "ns")
            evs = tr["traceEvents"]
            self.assertEqual(len(evs), 50)
            x_evs = [e for e in evs if e["ph"] == "X"]
            i_evs = [e for e in evs if e["ph"] == "i"]
            m_evs = [e for e in evs if e["ph"] == "M"]
            self.assertEqual((len(x_evs), len(i_evs), len(m_evs)), (24, 8, 18))

            by_name_count = {}
            for e in x_evs:
                by_name_count[e["name"]] = by_name_count.get(e["name"], 0) + 1
                slot = e["args"]["wave_slot"]
                cta = slot // 4
                self.assertEqual(e["pid"], 0)
                self.assertEqual(e["tid"], 1 + slot * 64)
                self.assertEqual(e["args"]["role"], "gemm_warp")
                self.assertEqual(e["args"]["cta"], cta)
                self.assertEqual(e["args"]["cta_id"], cta)
                self.assertEqual(e["args"]["aux"], cta)
                self.assertEqual(e["args"]["sm"], 64 if cta == 0 else 66)
                self.assertEqual(e["args"]["semantic_source"], "explicit")
            self.assertEqual(by_name_count, {"gemm": 8, "mma": 8, "store": 8})

            for e in i_evs:
                cta = e["args"]["cta"]
                self.assertEqual(e["name"], "epilogue")
                self.assertEqual(e["cat"], "epilogue")
                self.assertEqual(e["args"]["event_id"], 5)
                self.assertEqual(e["args"]["semantic_id"], "epilogue")
                self.assertEqual(e["args"]["original_label"], "epilogue")
                self.assertEqual(e["args"]["role"], "gemm_warp")
                self.assertEqual(e["args"]["aux"], cta)

            proc_names = {e["pid"]: e["args"]["name"] for e in m_evs if e["name"] == "process_name"}
            thread_names = {e["tid"]: e["args"]["name"] for e in m_evs if e["name"] == "thread_name"}
            self.assertEqual(proc_names, {0: "gemm_warp"})
            self.assertEqual(
                thread_names,
                {
                    1: "CTA000 w00 (gemm_warp) SM64",
                    65: "CTA000 w01 (gemm_warp) SM64",
                    129: "CTA000 w02 (gemm_warp) SM64",
                    193: "CTA000 w03 (gemm_warp) SM64",
                    257: "CTA001 w00 (gemm_warp) SM66",
                    321: "CTA001 w01 (gemm_warp) SM66",
                    385: "CTA001 w02 (gemm_warp) SM66",
                    449: "CTA001 w03 (gemm_warp) SM66",
                },
            )

        # Also exercise CLI wp_decode.py with --write-gz and --summary in a guarded subprocess.
        wp_decode_script = REPO_ROOT / "weaveprof" / "pipeline" / "validate" / "tools" / "wp_decode.py"
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            guard_dir = td_path / "guard"
            guard_dir.mkdir()
            (guard_dir / "sitecustomize.py").write_text(SUBPROCESS_SITECUSTOMIZE, encoding="utf-8")
            out_gz = td_path / "t1.perfetto.json.gz"
            out_sum = td_path / "t1.summary.json"
            env = dict(os.environ)
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{guard_dir}{os.pathsep}{REPO_ROOT}" + (
                f"{os.pathsep}{existing_pp}" if existing_pp else ""
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(wp_decode_script),
                    str(T1_DIR / "capture.fi02.bin"),
                    "--names",
                    str(T1_DIR / "names.json"),
                    "-o",
                    str(out_gz),
                    "--write-gz",
                    "--summary",
                    str(out_sum),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("错误：无", proc.stdout)
            cli_trace = json.loads(gzip.decompress(out_gz.read_bytes()).decode("utf-8"))
            cli_summary = json.loads(out_sum.read_text(encoding="utf-8"))
            self.assertEqual(cli_trace["traceEvents"], tr1["traceEvents"])
            self.assertEqual(cli_summary["source_dump_sha256"], EXPECTED_T1_BIN_SHA256)
            self.assertEqual(cli_summary["names_sha256"], EXPECTED_NAMES_SHA256)
            self.assertEqual(cli_summary["records"], 72)
            self.assertEqual(cli_summary["spans"], 24)
            self.assertEqual(cli_summary["marks"], 8)

    def test_negative_truncated_bytes_anchor_removal_and_suspected_bug_T2(self) -> None:
        raw0 = (T0_DIR / "capture.fi02.bin").read_bytes()
        raw1 = (T1_DIR / "capture.fi02.bin").read_bytes()

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # 1) Truncate 10 bytes from tail of t0 (1663 -> 1653, mid-record in slot 7).
            p_tail10 = td_path / "t0_trunc_tail10.bin"
            p_tail10.write_bytes(raw0[:-10])
            with self.assertRaises(SystemExit) as cm_tail10:
                self.wp.read_dump(p_tail10)
            self.assertTrue(str(cm_tail10.exception).startswith(f"{p_tail10}: "))
            self.assertEqual(
                str(cm_tail10.exception).removeprefix(f"{p_tail10}: "),
                "文件长度与计数不符（1663 vs 1653），转储损坏",
            )

            # 2) Truncate 1 full 16-byte record from tail of t0 (1663 -> 1647).
            p_tail16 = td_path / "t0_trunc_tail16.bin"
            p_tail16.write_bytes(raw0[:-16])
            with self.assertRaises(SystemExit) as cm_tail16:
                self.wp.read_dump(p_tail16)
            self.assertEqual(
                str(cm_tail16.exception).removeprefix(f"{p_tail16}: "),
                "文件长度与计数不符（1663 vs 1647），转储损坏",
            )

            # 3) Truncate inside slot 0 of t0 (520 bytes; slot 0 expects end_off = 511 + 9*16 = 655).
            p_slot0 = td_path / "t0_trunc_slot0.bin"
            p_slot0.write_bytes(raw0[:520])
            with self.assertRaises(SystemExit) as cm_slot0:
                self.wp.read_dump(p_slot0)
            self.assertEqual(
                str(cm_slot0.exception).removeprefix(f"{p_slot0}: "),
                "文件长度与计数不符（655 vs 520），转储损坏",
            )

            # 4) Truncate 16 bytes from tail of t1 (1914 -> 1898, missing close CLK anchor in slot 7).
            p_t1_tail16 = td_path / "t1_trunc_tail16.bin"
            p_t1_tail16.write_bytes(raw1[:-16])
            with self.assertRaises(SystemExit) as cm_t1_tail16:
                self.wp.read_dump(p_t1_tail16)
            self.assertEqual(
                str(cm_t1_tail16.exception).removeprefix(f"{p_t1_tail16}: "),
                "文件长度与计数不符（1914 vs 1898），转储损坏",
            )

            # 5) Append 1 trailing garbage byte to t0 (1663 -> 1664): caught by post-loop off != len(buf).
            p_extra = td_path / "t0_extra_1b.bin"
            p_extra.write_bytes(raw0 + b"\x00")
            with self.assertRaises(SystemExit) as cm_extra:
                self.wp.read_dump(p_extra)
            self.assertEqual(
                str(cm_extra.exception).removeprefix(f"{p_extra}: "),
                "文件长度与计数不符（1663 vs 1664），转储损坏",
            )

            # 6) Valid byte length on t1 with slot 7's close CLK anchor dropped (cnt[7]: 11 -> 10, len 1898):
            # proves to_globaltimer() requires the close CLK anchor when meta has no clock_mhz.
            u1 = _unpack_raw_filite02(raw1)
            mut_no_close_clk = bytearray(raw1[:-16])
            struct.pack_into("<I", mut_no_close_clk, 32 + u1["meta_len"] + 7 * 4, 10)
            p_no_close = td_path / "t1_no_close_anchor.bin"
            p_no_close.write_bytes(bytes(mut_no_close_clk))
            with self.assertRaises(SystemExit) as cm_no_close:
                self.wp.read_dump(p_no_close)
            self.assertEqual(
                str(cm_no_close.exception),
                "slot 7: 缺关锚点（早退路径没调 FI_CLOSE？）且 meta 没给 clock_mhz/timer_hz",
            )

            # 7) Pin Suspected Bug T2 (fi_lite_decode.py:144-149):
            # truncating inside the 32-byte header or meta JSON raises raw struct.error / JSONDecodeError
            # instead of SystemExit("...转储损坏").
            p_hdr20 = td_path / "t0_trunc_hdr20.bin"
            p_hdr20.write_bytes(raw0[:20])
            with self.assertRaises(struct.error) as cm_hdr20:
                self.wp.read_dump(p_hdr20)
            self.assertEqual(
                str(cm_hdr20.exception),
                "unpack_from requires a buffer of at least 32 bytes for unpacking 24 bytes at offset 8 (actual buffer size is 20)",
            )

            p_meta50 = td_path / "t0_trunc_meta50.bin"
            p_meta50.write_bytes(raw0[:50])
            with self.assertRaises(json.JSONDecodeError) as cm_meta50:
                self.wp.read_dump(p_meta50)
            self.assertEqual(cm_meta50.exception.pos, 14)

    def test_negative_header_overflow_mutations_and_cli_fail_closed(self) -> None:
        raw0 = (T0_DIR / "capture.fi02.bin").read_bytes()
        raw1 = (T1_DIR / "capture.fi02.bin").read_bytes()
        names0 = self.wp.load_names(T0_DIR / "names.json")
        names1 = self.wp.load_names(T1_DIR / "names.json")

        # In FILITE02 (`magic[8] | <6I: n_slots, cap, warps_per_cta, overflow, timer, meta_len`),
        # `overflow` is the 4th uint32 at byte offset 8 + 3*4 = 20.
        cases = [
            (1, {"泳道写满（cap 给小了）": 1}),
            (2, {"slot 越界（warps_per_cta 或 CTA 数给小了）": 1}),
            (3, {"泳道写满（cap 给小了）": 1, "slot 越界（warps_per_cta 或 CTA 数给小了）": 1}),
            (4, {"unknown_overflow_bits": 1}),
        ]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for label, base_raw, names in (("t0", raw0, names0), ("t1", raw1, names1)):
                for ovf_bits, expected_errors in cases:
                    mut = bytearray(base_raw)
                    struct.pack_into("<I", mut, 20, ovf_bits)
                    mut_path = td_path / f"{label}_ovf_{ovf_bits}.bin"
                    mut_path.write_bytes(bytes(mut))

                    d = self.wp.read_dump(mut_path)
                    self.assertEqual(d["overflow"], ovf_bits)
                    tr, sm = self.wp.decode(d, names)
                    self.assertEqual(sm["overflow"], ovf_bits)
                    self.assertEqual(sm["errors"], expected_errors)
                    self.assertEqual(tr["otherData"]["errors"], expected_errors)

            # Verify CLI fails closed with exit code 2 and refuses to write trace when overflow=1 without --allow-errors.
            guard_dir = td_path / "guard"
            guard_dir.mkdir()
            (guard_dir / "sitecustomize.py").write_text(SUBPROCESS_SITECUSTOMIZE, encoding="utf-8")
            env = dict(os.environ)
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{guard_dir}{os.pathsep}{REPO_ROOT}" + (
                f"{os.pathsep}{existing_pp}" if existing_pp else ""
            )
            wp_decode_script = REPO_ROOT / "weaveprof" / "pipeline" / "validate" / "tools" / "wp_decode.py"
            blocked_gz = td_path / "blocked.perfetto.json.gz"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(wp_decode_script),
                    str(td_path / "t0_ovf_1.bin"),
                    "--names",
                    str(T0_DIR / "names.json"),
                    "-o",
                    str(blocked_gz),
                    "--write-gz",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("有结构错误，未写文件；确认后加 --allow-errors 强制输出", proc.stderr)
            self.assertFalse(blocked_gz.exists())

    def test_swapped_names_json_status_quo_and_suspected_bug_T4(self) -> None:
        t0_bin = T0_DIR / "capture.fi02.bin"
        t0_names = T0_DIR / "names.json"
        t1_bin = T1_DIR / "capture.fi02.bin"
        t1_names = T1_DIR / "names.json"

        # Status quo of swapping t0/names.json and t1/names.json:
        # Both files are emitted from identical write_names(...) arguments and carry no timer-dependent keys,
        # so their bytes are identical and cross-decoding produces byte-identical traces and summaries.
        self.assertEqual(t0_names.read_bytes(), t1_names.read_bytes())
        tr0_orig, sm0_orig = self.wp.decode(self.wp.read_dump(t0_bin), self.wp.load_names(t0_names))
        tr0_swap, sm0_swap = self.wp.decode(self.wp.read_dump(t0_bin), self.wp.load_names(t1_names))
        tr1_orig, sm1_orig = self.wp.decode(self.wp.read_dump(t1_bin), self.wp.load_names(t1_names))
        tr1_swap, sm1_swap = self.wp.decode(self.wp.read_dump(t1_bin), self.wp.load_names(t0_names))
        self.assertEqual(tr0_swap, tr0_orig)
        self.assertEqual(sm0_swap, sm0_orig)
        self.assertEqual(tr1_swap, tr1_orig)
        self.assertEqual(sm1_swap, sm1_orig)

        # Contrast with missing event definitions in names:
        # 1) Removing span event "10" (gemm) is flagged in unnamed_event_ids and errors.
        names_missing_span = json.loads(t0_names.read_text(encoding="utf-8"))
        del names_missing_span["events"]["10"]
        del names_missing_span["semantics"]["10"]
        _, sm_no_span = self.wp.decode(self.wp.read_dump(t0_bin), names_missing_span)
        self.assertEqual(sm_no_span["unnamed_event_ids"], [10])
        self.assertEqual(sm_no_span["errors"], {"names.json 缺事件名": 1})

        # 2) Pin Suspected Bug T4 (fi_lite_decode.py:690-691,725-727):
        # `missing = sorted(set(by_name) - set(ev_name))` only inspects span IDs (`by_name`) and ignores
        # instant mark IDs (`marks`), so removing mark event "5" (epilogue) emits "unknown #0"/"unknown #1"
        # instant events while leaving unnamed_event_ids == [] and errors == {}.
        names_missing_mark = json.loads(t0_names.read_text(encoding="utf-8"))
        del names_missing_mark["events"]["5"]
        del names_missing_mark["semantics"]["5"]
        tr_no_mark, sm_no_mark = self.wp.decode(self.wp.read_dump(t0_bin), names_missing_mark)
        self.assertEqual(sm_no_mark["unnamed_event_ids"], [])
        self.assertEqual(sm_no_mark["errors"], {})
        mark_evs = [e for e in tr_no_mark["traceEvents"] if e["ph"] == "i"]
        self.assertEqual(len(mark_evs), 8)
        self.assertEqual({e["name"] for e in mark_evs}, {"unknown #0", "unknown #1"})
        self.assertEqual({e["args"]["semantic_id"] for e in mark_evs}, {"unknown"})

    def test_wp_triton_amd_timer_contract_cpu_and_suspected_bug_T3(self) -> None:
        env_keys = ("WP_LEVEL", "FI_LEVEL", "WP_GRAN", "FI_GRAN", "WP_TIMER", "FI_TIMER")
        saved_env = {k: os.environ.get(k) for k in env_keys}

        def restore_env() -> None:
            for key, val in saved_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

        self.addCleanup(restore_env)
        for key in env_keys:
            os.environ.pop(key, None)

        amd_path = REPO_ROOT / "recorders" / "triton" / "wp_triton_amd.py"
        spec = importlib.util.spec_from_file_location("_wp_triton_amd_cpu_char", amd_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        amd_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(amd_mod)

        self.assertIsNone(amd_mod.torch)
        self.assertIsNone(amd_mod.triton)

        self.assertEqual(amd_mod._timer_contract(0), (0, "s_memrealtime"))
        self.assertEqual(amd_mod._timer_contract(1), (1, "s_memtime+anchors"))
        self.assertEqual(amd_mod._timer_contract("0"), (0, "s_memrealtime"))
        self.assertEqual(amd_mod._timer_contract("1"), (1, "s_memtime+anchors"))

        for bad_int in (-1, 2, 99):
            with self.assertRaises(ValueError) as cm_bad:
                amd_mod._timer_contract(bad_int)
            self.assertEqual(
                str(cm_bad.exception),
                "timer_mode 只能是 0(每记录 s_memrealtime) 或 1(s_memtime+锚点)",
            )

        with self.assertRaises(ValueError) as cm_str:
            amd_mod._timer_contract("bad")
        self.assertEqual(str(cm_str.exception), "invalid literal for int() with base 10: 'bad'")

        with self.assertRaises(TypeError):
            amd_mod._timer_contract(None)  # type: ignore[arg-type]

        # Pin Suspected Bug T3 (recorders/triton/wp_triton_amd.py:428):
        # `mode = int(timer_mode)` truncates non-integral floats toward zero before checking `mode in (0, 1)`.
        self.assertEqual(amd_mod._timer_contract(1.9), (1, "s_memtime+anchors"))  # type: ignore[arg-type]
        self.assertEqual(amd_mod._timer_contract(-0.9), (0, "s_memrealtime"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-CPU characterization tests for `recorders/triton/wp_triton.py` and `triton_ptx_async_hook` (B1-G1..G3)."""
from __future__ import annotations

import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
import unittest.mock

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# In-process zero-external-network socket guard (installed BEFORE imports)
# ---------------------------------------------------------------------------
_ORIG_SOCKET_CONNECT = socket.socket.connect
_ORIG_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIG_CREATE_CONNECTION = socket.create_connection
_ORIG_GETADDRINFO = socket.getaddrinfo
_NON_LOOPBACK_ATTEMPTS: list[str] = []


def _is_loopback_host(host: object) -> bool:
    if not isinstance(host, str):
        return False
    return host.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip_literal(host: object) -> bool:
    if not isinstance(host, str):
        return False
    try:
        ipaddress.ip_address(host.strip())
        return True
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: object) -> object:
    if isinstance(address, tuple) and len(address) >= 1:
        if not _is_loopback_host(address[0]):
            _NON_LOOPBACK_ATTEMPTS.append(f"connect:{address!r}")
            raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT(self, address)


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if isinstance(address, tuple) and len(address) >= 1:
        if not _is_loopback_host(address[0]):
            _NON_LOOPBACK_ATTEMPTS.append(f"connect_ex:{address!r}")
            raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_SOCKET_CONNECT_EX(self, address)


def _guarded_create_connection(address: tuple[object, ...], *args: object, **kwargs: object) -> socket.socket:
    if isinstance(address, tuple) and len(address) >= 1 and not _is_loopback_host(address[0]):
        _NON_LOOPBACK_ATTEMPTS.append(f"create_connection:{address!r}")
        raise ConnectionRefusedError(f"blocked by test guard: {address}")
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> object:
    if not _is_loopback_host(host) and not _is_ip_literal(host):
        _NON_LOOPBACK_ATTEMPTS.append(f"getaddrinfo:{host!r}:{port!r}")
        raise socket.gaierror(8, f"blocked external DNS lookup: {host}")
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)


socket.socket.connect = _guarded_connect  # type: ignore[assignment]
socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[assignment]
socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Clear recorder import-time switches BEFORE importing recorder modules
# ---------------------------------------------------------------------------
_RECORDER_ENV_KEYS = (
    "WP_LEVEL",
    "FI_LEVEL",
    "WP_GRAN",
    "FI_GRAN",
    "WP_TIMER",
    "FI_TIMER",
)
for _env_key in (*_RECORDER_ENV_KEYS, "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ.pop(_env_key, None)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recorders.triton import wp_triton  # noqa: E402
from weaveprof.pipeline.instrument import triton_ptx_async  # noqa: E402

_WP_TRITON_PATH = REPO_ROOT / "recorders" / "triton" / "wp_triton.py"
_LOAD_SEQ = 0

_ARCH_FALLBACK_MSG = (
    "本机拿不到 torch.cuda 设备能力（AttributeError: 'NoneType' object has no attribute 'cuda'）："
    "请用 FiCfg(arch='sm_90a') 或 dump(meta={'arch': ...}) 显式指定"
)

# Expected SHA-256 digests and record counts for each (gran, timer) pair at
# cap=64, warps=4, n_slots=32, threads=128 (open role=3, openby role=5).
_EXPECTED_PTX_BY_GT: dict[tuple[int, int], dict[str, object]] = {
    (0, 0): {
        "warps_per_cta_in_dump": 4,
        "records_per_open": 1,
        "timer_hz": 1_000_000_000,
        "open_3_sha": "efb654748469187a3ae3a20e7a6d4a8cb6523b8691cf7f99565f0ae74e42989d",
        "openby_5_sha": "d3fae31a6bad0fcd4033f80b5373554f8f39ce5ff39ec0f94159713d5379cb93",
        "open_nr": 1,
        "event_sha": "90d05a4669dc6f605ec650e0a042f33aa1e578f3540070426f74a4fd6d84830f",
        "event_nr": 1,
        "event_by_sha": "2d8b16005c01c5fd449411d5ca5356c28747126961888b2b3716a471bbc85a35",
        "event_by_nr": 1,
        "trans_sha": "ac660410ed406877709060ee5b00dd85c6fd61e6704bf90789fd5a4a5cc647ce",
        "trans_nr": 2,
        "close_sha": "fe1f960b07e57abc3027b7ce9b7ca335eb0ffa613974563164629880b35298a8",
        "close_nr": 1,
        "close_by_sha": "d1102969b3747c7f515f0e7a0c5b028396cb364bd3fb507d3c5bf2f947332ab5",
        "close_by_nr": 1,
    },
    (0, 1): {
        "warps_per_cta_in_dump": 4,
        "records_per_open": 2,
        "timer_hz": None,
        "open_3_sha": "7632bc146af925e2318a9c41ac8faf3a87eb6fa91b93480f6ade06a52d4ad5b5",
        "openby_5_sha": "94c9d75ac67552c77a7ee9d8eee9797150845fb4bc2ec4ddc83a6316b180a2d7",
        "open_nr": 2,
        "event_sha": "a8508be4480d08cf8313eb5868d33ff200a80cb6eb2c8db0b31eca715644d343",
        "event_nr": 1,
        "event_by_sha": "89f0b88f55a79188fd251f51ef48f489bca7d817082892561f1b906e2bef3bcb",
        "event_by_nr": 1,
        "trans_sha": "b757e90a020f0ff9536d8b9a60db7e1b85b7601064e7ba2c80cff227c3bb4849",
        "trans_nr": 2,
        "close_sha": "95c0ab21177ee70834c78211d2628c6ef86d37d1b443ae11f37dfd7fe0d5e8a2",
        "close_nr": 2,
        "close_by_sha": "b67ea2e9fdf356734a03888724d39ce8bd30f841519e61753150675d6ce319d4",
        "close_by_nr": 2,
    },
    (1, 0): {
        "warps_per_cta_in_dump": 1,
        "records_per_open": 1,
        "timer_hz": 1_000_000_000,
        "open_3_sha": "c1686196897dd3fa8871838b6d7c4a9967245b56b7c2c3b73b0e4dea66d4d039",
        "openby_5_sha": "7a2dd3206d8696f1b93785f3c858ce5ef1f551d8185c5552a2258a28173f1bb8",
        "open_nr": 1,
        "event_sha": "0e888ef08e68bd32496f53cd3adc29559b9694c9d6d9c289db74ffad7edd8450",
        "event_nr": 1,
        "event_by_sha": "878dc30dc8a7567858c656b239559d45dcb3e67d1dda427592319ef0b9132219",
        "event_by_nr": 1,
        "trans_sha": "e03a194ad5b4530692fddb58b5007b9b8c45d7e406f3e99aefeb165641a0cb16",
        "trans_nr": 2,
        "close_sha": "eae9f4b7db4fb454331f50d1af2f0c8faf6d1c8e4b03d381c9681551bfa9471f",
        "close_nr": 1,
        "close_by_sha": "4f73a085193f3da195b5b1b68e7023c2b0ed242d4b0f45de7f4395de3fa4ff5e",
        "close_by_nr": 1,
    },
    (1, 1): {
        "warps_per_cta_in_dump": 1,
        "records_per_open": 2,
        "timer_hz": None,
        "open_3_sha": "ec638c035ffbd82dc698ca7c55b631106a22d18b2e6ac8b8bb87434db0ded7a2",
        "openby_5_sha": "f03b32585b9889202017197a4c2e01f381d8e4208db0b1ef28026cbc91393732",
        "open_nr": 2,
        "event_sha": "7b5a8917e58c891832d3ee37fcd80e63027582dc9e06182ee76692e7a038a3c2",
        "event_nr": 1,
        "event_by_sha": "7c6274b00096655365ff6d0ade0f1bb108971d18ef179d8dda8f7f2495ec702b",
        "event_by_nr": 1,
        "trans_sha": "348f297fe45da432bb8014e61245ac1e95842f29ae0bff34d6380e332e23d6fa",
        "trans_nr": 2,
        "close_sha": "7e712b6941569b64d6ee2bd4b43ca273be8f75e85e376f85536b40ce049a2c8d",
        "close_nr": 2,
        "close_by_sha": "de097c971e383c0c7e90eacc91a4eb9eb08a7761085947fb53f58ecadb2617c2",
        "close_by_nr": 2,
    },
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_fresh_wp_triton(env_overrides: dict[str, str]) -> tuple[types.ModuleType, dict[str, str]]:
    """Load `wp_triton.py` under a unique module name with controlled env switches."""
    global _LOAD_SEQ
    _LOAD_SEQ += 1
    mod_name = f"_wp_triton_fresh_{_LOAD_SEQ}"
    saved_env = dict(os.environ)
    try:
        for k in _RECORDER_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in env_overrides.items():
            os.environ[k] = v
        spec = importlib.util.spec_from_file_location(mod_name, _WP_TRITON_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(mod_name, None)
        return mod, dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


class _FakeConstexpr:
    """Pure-Python `.value` wrapper simulating a `constexpr`-like object without `triton`."""

    def __init__(self, value: object) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"constexpr[{self.value}]"


class _FakeGridKernel:
    """Minimal subscriptable Triton kernel test double recording `(grid, args, kwargs)`."""

    def __init__(self, name: str, attr_tag: str) -> None:
        self.__name__ = name
        self.kernel_attr = attr_tag
        self.calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
        self._grid_launchers: dict[object, object] = {}

    def __getitem__(self, grid: object) -> object:
        if grid not in self._grid_launchers:
            def _launch(*args: object, **kwargs: object) -> None:
                self.calls.append((grid, args, dict(kwargs)))
            self._grid_launchers[grid] = _launch
        return self._grid_launchers[grid]


class _FakeNvBuf:
    """Minimal buffer test double holding distinct sentinels and a real `FiCfg`."""

    def __init__(self, rec: object, overflow: object, cfg: wp_triton.FiCfg) -> None:
        self.rec = rec
        self.overflow = overflow
        self.cfg = cfg


class _FakeConfigWithAllKwargs:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = dict(data)

    def all_kwargs(self) -> dict[str, object]:
        return dict(self._data)


class _FakeConfigKwargsOnly:
    def __init__(self, data: dict[str, object]) -> None:
        self.kwargs = dict(data)


class WpTritonCpuCharacterizationTests(unittest.TestCase):
    """Pure-CPU characterization tests for NVIDIA Triton recorder (`wp_triton.py`)."""

    def setUp(self) -> None:
        self._saved_env = dict(os.environ)
        for k in (*_RECORDER_ENV_KEYS, "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(k, None)
        wp_triton.fi_attach(None)
        _NON_LOOPBACK_ATTEMPTS.clear()

    def tearDown(self) -> None:
        wp_triton.fi_attach(None)
        os.environ.clear()
        os.environ.update(self._saved_env)
        self.assertEqual(_NON_LOOPBACK_ATTEMPTS, [])

    def test_ficfg_ptx_all_12_combinations_and_key_instructions(self) -> None:
        for gran in (0, 1):
            for timer in (0, 1):
                exp = _EXPECTED_PTX_BY_GT[(gran, timer)]
                for level in (0, 1, 2):
                    with self.subTest(gran=gran, timer=timer, level=level):
                        cfg = wp_triton.FiCfg(
                            cap=64,
                            warps=4,
                            n_slots=32,
                            gran=gran,
                            timer=timer,
                            level=level,
                            threads=128,
                        )
                        self.assertEqual(cfg.warps_per_cta_in_dump, exp["warps_per_cta_in_dump"])
                        self.assertEqual(cfg.records_per_open(), exp["records_per_open"])

                        # Constraints and register/record counts
                        self.assertEqual(cfg.open_cons(3), "=r,r,l,l,r,r")
                        self.assertEqual(cfg.openby_cons(5), "=r,r,l,l,r,r,r")
                        self.assertEqual(cfg.event_cons, "=r,r,l,l,r,r,r,r")
                        self.assertEqual(cfg.event_by_cons, "=r,r,l,l,r,r,r,r,r")
                        self.assertEqual(cfg.trans_cons, "=r,r,l,l,r,r,r,r,r")
                        self.assertEqual(cfg.close_cons, "=r,r,l,l,r,r")
                        self.assertEqual(cfg.close_by_cons, "=r,r,l,l,r,r,r")

                        self.assertEqual(cfg.open_nr, exp["open_nr"])
                        self.assertEqual(cfg.event_nr, exp["event_nr"])
                        self.assertEqual(cfg.event_by_nr, exp["event_by_nr"])
                        self.assertEqual(cfg.trans_nr, exp["trans_nr"])
                        self.assertEqual(cfg.close_nr, exp["close_nr"])
                        self.assertEqual(cfg.close_by_nr, exp["close_by_nr"])

                        open_ptx = cfg.open_asm(3)
                        openby_ptx = cfg.openby_asm(5)
                        event_ptx = cfg.event_asm
                        event_by_ptx = cfg.event_by_asm
                        trans_ptx = cfg.trans_asm
                        close_ptx = cfg.close_asm
                        close_by_ptx = cfg.close_by_asm

                        # Exact SHA-256 of every generated PTX block
                        self.assertEqual(_sha256_text(open_ptx), exp["open_3_sha"])
                        self.assertEqual(_sha256_text(openby_ptx), exp["openby_5_sha"])
                        self.assertEqual(_sha256_text(event_ptx), exp["event_sha"])
                        self.assertEqual(_sha256_text(event_by_ptx), exp["event_by_sha"])
                        self.assertEqual(_sha256_text(trans_ptx), exp["trans_sha"])
                        self.assertEqual(_sha256_text(close_ptx), exp["close_sha"])
                        self.assertEqual(_sha256_text(close_by_ptx), exp["close_by_sha"])

                        # Key PTX instructions: slot calculation & lane/thread predicates
                        if gran == 0:
                            self.assertIn("mul.lo.u32 slot, pid, 4;\n    add.u32 slot, slot, warp;", open_ptx)
                            self.assertIn("setp.eq.u32 p_rec, lane, 0;", open_ptx)
                            self.assertIn("setp.ne.u32 p_rec, $6, 0;", openby_ptx)
                            self.assertNotIn("p_user", openby_ptx)
                            self.assertIn("setp.ne.u32 p_rec, $8, 0;", event_by_ptx)
                            self.assertIn("setp.ne.u32 p_rec, $6, 0;", close_by_ptx)
                        else:
                            self.assertIn("mov.u32 slot, pid;", open_ptx)
                            self.assertIn("setp.eq.u32 p_rec, tid, 0;", open_ptx)
                            self.assertIn(
                                "setp.eq.u32 p_rec, tid, 0;\n    setp.ne.u32 p_user, $6, 0;\n    and.pred p_rec, p_rec, p_user;",
                                openby_ptx,
                            )
                            self.assertIn(
                                "setp.eq.u32 p_rec, tid, 0;\n    setp.ne.u32 p_user, $8, 0;\n    and.pred p_rec, p_rec, p_user;",
                                event_by_ptx,
                            )
                            self.assertIn(
                                "setp.eq.u32 p_rec, tid, 0;\n    setp.ne.u32 p_user, $6, 0;\n    and.pred p_rec, p_rec, p_user;",
                                close_by_ptx,
                            )

                        # Key PTX instructions: timer registers, store offsets, immediate tags & bounds
                        self.assertIn("mul.wide.u32 soff, slot, 1024;", open_ptx)
                        self.assertIn("setp.lt.u32 p_bounds, warp, 4;", open_ptx)
                        self.assertIn("setp.lt.u32 p_cap, slot, 32;", open_ptx)
                        self.assertIn("@p_store mov.u64 t0, %globaltimer;", open_ptx)
                        self.assertIn("@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, 1073741827, sm};", open_ptx)
                        self.assertIn("@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, 1073741829, sm};", openby_ptx)
                        self.assertIn("and.b32 tagv, $6, 0xF0000FFF;", event_ptx)
                        self.assertIn("@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, tagv, $7};", event_ptx)
                        self.assertIn("and.b32 tagv2, $7, 0xF0000FFF;", trans_ptx)
                        self.assertIn("@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, tagv, $8};", trans_ptx)
                        self.assertIn("@p_store st.global.cs.v4.u32 [addr+16], {tlo0, thi0, tagv2, $8};", trans_ptx)
                        self.assertIn("@p_store mov.u64 t0, %globaltimer;", close_ptx)
                        self.assertIn("@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, 134217728, 0};", close_ptx)

                        if timer == 0:
                            self.assertNotIn("%clock64", open_ptx)
                            self.assertNotIn("[addr+16]", open_ptx)
                            self.assertIn("setp.ge.u32 p_ovf, n, 64;", open_ptx)
                            self.assertIn("@p_store mov.u64 t0, %globaltimer;", event_ptx)
                            self.assertNotIn("%clock64", event_ptx)
                            self.assertIn("@p_store mov.u64 t0, %globaltimer;", trans_ptx)
                            self.assertNotIn("[addr+16]", close_ptx)
                        else:
                            self.assertIn("setp.lt.u32 p_cap, n, 63;", open_ptx)
                            self.assertIn("@p_store mov.u64 t1, %clock64;", open_ptx)
                            self.assertIn("@p_store st.global.cs.v4.u32 [addr+16], {tlo1, thi1, 268435456, 0};", open_ptx)
                            self.assertIn("setp.ge.u32 p_ovf, n, 63;", open_ptx)
                            self.assertIn("@p_store mov.u64 t0, %clock64;", event_ptx)
                            self.assertNotIn("%globaltimer", event_ptx)
                            self.assertIn("@p_store mov.u64 t0, %clock64;", trans_ptx)
                            self.assertIn("@p_store mov.u64 t1, %clock64;", close_ptx)
                            self.assertIn("@p_store st.global.cs.v4.u32 [addr+16], {tlo1, thi1, 268435456, 1};", close_ptx)

                        # Exact 17-field meta() dictionary
                        self.assertEqual(
                            cfg.meta(),
                            {
                                "fi_version": "fast-instrument-triton",
                                "language": "triton",
                                "triton": "unavailable",
                                "gran": gran,
                                "timer": timer,
                                "level": level,
                                "groups": 4294967295,
                                "arch": "sm_unknown",
                                "arch_source": _ARCH_FALLBACK_MSG,
                                "backend": "nvidia_ptx",
                                "backend_isa": "ptx",
                                "backend_validated": True,
                                "fence": 0,
                                "cap_per_lane": 64,
                                "n_slots": 32,
                                "warps_per_cta": exp["warps_per_cta_in_dump"],
                                "timer_hz": exp["timer_hz"],
                            },
                        )

        # Fence=1 emits membar.gl at entry and exit; meta(**extra) drops None values
        fenced = wp_triton.FiCfg(cap=32, warps=4, n_slots=8, threads=128, arch="sm_90a", fence=1)
        self.assertEqual(fenced.event_asm.count("membar.gl;"), 2)
        meta_extra = fenced.meta(kernel_name="matmul_k", omit_me=None)
        self.assertEqual(meta_extra["arch"], "sm_90a")
        self.assertEqual(meta_extra["arch_source"], "FiCfg.arch 显式给定")
        self.assertEqual(meta_extra["fence"], 1)
        self.assertEqual(meta_extra["kernel_name"], "matmul_k")
        self.assertNotIn("omit_me", meta_extra)

        with self.assertRaises(ValueError) as bad_kind:
            wp_triton._emit_asm(fenced, "unknown_kind")
        self.assertEqual(str(bad_kind.exception), "未知 kind=unknown_kind")

    def test_ficfg_invalid_arguments(self) -> None:
        invalid_cases: list[tuple[dict[str, object], str]] = [
            (
                {"cap": 16, "warps": 4, "n_slots": 8, "gran": 2, "threads": 128},
                "FI_GRAN 只能是 0(warp) 或 1(SM/CU)，收到 2",
            ),
            (
                {"cap": 16, "warps": 4, "n_slots": 8, "timer": 2, "threads": 128},
                "FI_TIMER 只能是 0(墙钟) 或 1(本地周期)，收到 2",
            ),
            (
                {"cap": 16, "warps": 4, "n_slots": 8, "level": 3, "threads": 128},
                "FI_LEVEL 只能是 0/1/2，收到 3",
            ),
            (
                {"cap": 16, "warps": 4, "n_slots": 8, "backend": "amd_gcn", "threads": 128},
                "backend 只能是 ['nvidia_ptx']，收到 amd_gcn",
            ),
            (
                {"cap": 16, "warps": 0, "n_slots": 8, "gran": 1, "threads": 128},
                "warps 必须为正整数，收到 0",
            ),
            (
                {"cap": 16, "warps": 4, "n_slots": 0, "threads": 128},
                "n_slots 必须落在 1..2**32-1，收到 0",
            ),
            (
                {"cap": 16, "warps": 4, "n_slots": 4294967296, "threads": 128},
                "n_slots 必须落在 1..2**32-1，收到 4294967296",
            ),
            (
                {"cap": 16, "warps": 3, "n_slots": 8, "threads": 96},
                "threads 必须是 2 的幂（= num_warps*32），收到 96",
            ),
            (
                {"cap": 16, "warps": 4, "n_slots": 8, "gran": 0, "threads": 64},
                "threads(64) 必须等于 warps*32(128)，否则每线程不是恰好执行一次 asm，记录会重复或丢失",
            ),
            (
                {"cap": 0, "warps": 4, "n_slots": 8, "threads": 128},
                "cap 非法：0（cap*16 必须放得进 32 位立即数）",
            ),
            (
                {"cap": 268435456, "warps": 4, "n_slots": 8, "threads": 128},
                "cap 非法：268435456（cap*16 必须放得进 32 位立即数）",
            ),
        ]
        for kwargs, expected_msg in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError) as ctx:
                    wp_triton.FiCfg(**kwargs)  # type: ignore[arg-type]
                self.assertEqual(str(ctx.exception), expected_msg)

    def test_record_gate_plan_and_status_boundaries(self) -> None:
        with self.assertRaises(ValueError) as bad_cap:
            wp_triton._record_gate_plan(0, 1)
        self.assertEqual(str(bad_cap.exception), "cap/n_records 必须为正整数，收到 0/1")

        with self.assertRaises(ValueError) as bad_nr:
            wp_triton._record_gate_plan(8, 0)
        self.assertEqual(str(bad_nr.exception), "cap/n_records 必须为正整数，收到 8/0")

        plan1 = wp_triton._record_gate_plan(8, 1)
        self.assertEqual(plan1.record_cap_limits, (8,))
        self.assertEqual(plan1.lane_full_at, 8)
        self.assertIs(plan1.record_fits(7, 0), True)
        self.assertIs(plan1.record_fits(8, 0), False)
        self.assertIs(plan1.record_fits(0, 1), False)
        self.assertIs(plan1.lane_overflows(7), False)
        self.assertIs(plan1.lane_overflows(8), True)

        plan2 = wp_triton._record_gate_plan(8, 2)
        self.assertEqual(plan2.record_cap_limits, (8, 7))
        self.assertEqual(plan2.lane_full_at, 7)
        self.assertIs(plan2.record_fits(7, 0), True)
        self.assertIs(plan2.record_fits(7, 1), False)
        self.assertIs(plan2.lane_overflows(6), False)
        self.assertIs(plan2.lane_overflows(7), True)

        # Single-record capacity: exact boundary n=7 (in) vs n=8 (overflow=1)
        self.assertEqual(
            wp_triton.record_gate_status(n=7, slot=15, warp=3, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": True, "slot_ok": True, "stores": (True,), "overflow": 0},
        )
        self.assertEqual(
            wp_triton.record_gate_status(n=8, slot=15, warp=3, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": True, "slot_ok": True, "stores": (False,), "overflow": 1},
        )

        # Two-record capacity: n=6 (both fit), n=7 (second overflows), n=8 (both overflow)
        self.assertEqual(
            wp_triton.record_gate_status(n=6, slot=0, warp=0, cap=8, n_slots=16, warps=4, n_records=2),
            {"warp_ok": True, "slot_ok": True, "stores": (True, True), "overflow": 0},
        )
        self.assertEqual(
            wp_triton.record_gate_status(n=7, slot=0, warp=0, cap=8, n_slots=16, warps=4, n_records=2),
            {"warp_ok": True, "slot_ok": True, "stores": (True, False), "overflow": 1},
        )
        self.assertEqual(
            wp_triton.record_gate_status(n=8, slot=0, warp=0, cap=8, n_slots=16, warps=4, n_records=2),
            {"warp_ok": True, "slot_ok": True, "stores": (False, False), "overflow": 1},
        )

        # Slot lower/upper boundaries: slot=0 & slot=15 (in), slot=-1 & slot=16 (OOB -> overflow=2)
        self.assertEqual(
            wp_triton.record_gate_status(n=0, slot=0, warp=0, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": True, "slot_ok": True, "stores": (True,), "overflow": 0},
        )
        self.assertEqual(
            wp_triton.record_gate_status(n=0, slot=-1, warp=0, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": True, "slot_ok": False, "stores": (False,), "overflow": 2},
        )
        self.assertEqual(
            wp_triton.record_gate_status(n=0, slot=16, warp=0, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": True, "slot_ok": False, "stores": (False,), "overflow": 2},
        )

        # Warp lower/upper boundaries: warp=3 (in), warp=-1 & warp=4 (OOB -> overflow=2)
        self.assertEqual(
            wp_triton.record_gate_status(n=0, slot=0, warp=3, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": True, "slot_ok": True, "stores": (True,), "overflow": 0},
        )
        self.assertEqual(
            wp_triton.record_gate_status(n=0, slot=0, warp=-1, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": False, "slot_ok": True, "stores": (False,), "overflow": 2},
        )
        self.assertEqual(
            wp_triton.record_gate_status(n=0, slot=0, warp=4, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": False, "slot_ok": True, "stores": (False,), "overflow": 2},
        )

        # When slot/warp is OOB and n >= cap, bounds_ok gates lane_full so overflow is 2 (not 3)
        self.assertEqual(
            wp_triton.record_gate_status(n=8, slot=16, warp=3, cap=8, n_slots=16, warps=4, n_records=1),
            {"warp_ok": True, "slot_ok": False, "stores": (False,), "overflow": 2},
        )

    def test_resolve_arch_cap_for_and_default_timer(self) -> None:
        # Three CPU branches of resolve_arch
        self.assertEqual(
            wp_triton.resolve_arch("nvidia_ptx", explicit="sm_90a"),
            ("sm_90a", "FiCfg.arch 显式给定"),
        )
        self.assertEqual(
            wp_triton.resolve_arch("custom_isa", explicit=None),
            ("custom_isa", "backend 名（非 NVIDIA PTX 后端直接用后端标识）"),
        )
        self.assertEqual(
            wp_triton.resolve_arch("nvidia_ptx", explicit=None),
            ("sm_unknown", _ARCH_FALLBACK_MSG),
        )

        # Hand-calculated cap_for expectations:
        # 1) records_per_tile=4, max_tiles_per_cta=10, default extra=8, default timer=0 -> 4*10 + 8 + 2 = 50
        self.assertEqual(wp_triton.cap_for(4, 10), 50)
        # 2) records_per_tile=4, max_tiles_per_cta=10, default extra=8, timer=1 -> 4*10 + 8 + 4 = 52
        self.assertEqual(wp_triton.cap_for(4, 10, timer=1), 52)
        # 3) records_per_tile=6, max_tiles_per_cta=8, extra=16, timer=0 -> 6*8 + 16 + 2 = 66
        self.assertEqual(wp_triton.cap_for(6, 8, extra=16, timer=0), 66)
        # 4) records_per_tile=3, max_tiles_per_cta=5, extra=0, timer=1 -> 3*5 + 0 + 4 = 19
        self.assertEqual(wp_triton.cap_for(3, 5, extra=0, timer=1), 19)

        # default_timer_for_device priority: explicit -> WP_TIMER -> FI_TIMER -> 0 (when torch is None)
        os.environ["WP_TIMER"] = "0"
        os.environ["FI_TIMER"] = "0"
        self.assertEqual(wp_triton.default_timer_for_device("cuda:0", explicit=1), 1)

        os.environ["WP_TIMER"] = "1"
        os.environ["FI_TIMER"] = "1"
        self.assertEqual(wp_triton.default_timer_for_device("cuda:0", explicit=0), 0)

        os.environ["WP_TIMER"] = "1"
        os.environ["FI_TIMER"] = "0"
        self.assertEqual(wp_triton.default_timer_for_device("cuda:0"), 1)

        os.environ["WP_TIMER"] = ""
        os.environ["FI_TIMER"] = "1"
        self.assertEqual(wp_triton.default_timer_for_device("cuda:0"), 1)

        os.environ.pop("WP_TIMER", None)
        os.environ.pop("FI_TIMER", None)
        self.assertEqual(wp_triton.default_timer_for_device("cuda:0"), 0)

    def test_import_time_env_mapping_validation_and_lazy_symbols(self) -> None:
        # 1) All 6 env vars cleared -> (2, 0, 0)
        mod_clean, env_clean = _load_fresh_wp_triton({})
        self.assertEqual((mod_clean.WP_LEVEL, mod_clean.WP_GRAN, mod_clean.WP_TIMER), (2, 0, 0))
        self.assertNotIn("FI_LEVEL", env_clean)
        self.assertNotIn("FI_GRAN", env_clean)
        self.assertNotIn("FI_TIMER", env_clean)
        self.assertIs(mod_clean.wp_available(), False)
        cfg_default = mod_clean.wp_cfg(cap=16, warps=4, n_slots=8, threads=128)
        self.assertEqual((cfg_default.level, cfg_default.gran, cfg_default.timer), (2, 0, 0))

        # 2) Only WP_TIMER="1" -> FI_TIMER mirrored to "1" and wp_cfg() defaults follow
        mod_t1, env_t1 = _load_fresh_wp_triton({"WP_TIMER": "1"})
        self.assertEqual(env_t1["FI_TIMER"], "1")
        self.assertEqual((mod_t1.WP_LEVEL, mod_t1.WP_GRAN, mod_t1.WP_TIMER), (2, 0, 1))
        cfg_t1 = mod_t1.wp_cfg(cap=16, warps=4, n_slots=8, threads=128)
        self.assertEqual((cfg_t1.level, cfg_t1.gran, cfg_t1.timer), (2, 0, 1))

        # 3) WP_LEVEL="1", WP_GRAN="1", WP_TIMER="1" -> wp_cfg() defaults follow all 3
        mod_all, env_all = _load_fresh_wp_triton({"WP_LEVEL": "1", "WP_GRAN": "1", "WP_TIMER": "1"})
        self.assertEqual(env_all["FI_LEVEL"], "1")
        self.assertEqual(env_all["FI_GRAN"], "1")
        self.assertEqual(env_all["FI_TIMER"], "1")
        self.assertEqual((mod_all.WP_LEVEL, mod_all.WP_GRAN, mod_all.WP_TIMER), (1, 1, 1))
        cfg_all = mod_all.wp_cfg(cap=16, warps=4, n_slots=4, threads=128)
        self.assertEqual((cfg_all.level, cfg_all.gran, cfg_all.timer), (1, 1, 1))
        cfg_override = mod_all.wp_cfg(cap=16, warps=4, n_slots=8, threads=128, level=0, gran=0, timer=0)
        self.assertEqual((cfg_override.level, cfg_override.gran, cfg_override.timer), (0, 0, 0))

        # 4) Conflicting WP_TIMER="1" and FI_TIMER="0" -> RuntimeError
        with self.assertRaises(RuntimeError) as conflict_ctx:
            _load_fresh_wp_triton({"WP_TIMER": "1", "FI_TIMER": "0"})
        self.assertEqual(
            str(conflict_ctx.exception),
            "WP_TIMER='1' conflicts with FI_TIMER='0'; set only one spelling",
        )

        # 5) Out-of-range WP_GRAN="2" and WP_LEVEL="5" -> ValueError
        with self.assertRaises(ValueError) as bad_gran_ctx:
            _load_fresh_wp_triton({"WP_GRAN": "2"})
        self.assertEqual(str(bad_gran_ctx.exception), "WP_GRAN must be one of (0, 1), got 2")

        with self.assertRaises(ValueError) as bad_level_ctx:
            _load_fresh_wp_triton({"WP_LEVEL": "5"})
        self.assertEqual(str(bad_level_ctx.exception), "WP_LEVEL must be one of (0, 1, 2), got 5")

        # 6) Lazy PEP 562 __getattr__ for device symbols vs unknown attributes
        with self.assertRaises(ImportError) as dev_sym_ctx:
            _ = wp_triton.fi_open
        self.assertIn(
            "wp_triton.fi_open needs Triton + torch (CUDA); import of triton failed:",
            str(dev_sym_ctx.exception),
        )
        with self.assertRaises(AttributeError) as attr_ctx:
            _ = wp_triton.no_such_symbol
        self.assertEqual(str(attr_ctx.exception), "module 'wp_triton' has no attribute 'no_such_symbol'")

    def test_write_names_normalization_collisions_and_constexpr_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_path = td_path / "sub" / "names.json"
            returned = wp_triton.write_names(
                out_path,
                events={1: "gmem_load", "2": "mma_compute"},
                roles={0: "producer", True: "consumer"},
                aux={False: "stage0", 3: "stage3"},
                group="sm",
                pair="aux",
                label_aux=False,
                nested=({10: ("alpha", [1, 2])}, {False: "flag0"}),
            )
            self.assertEqual(returned, str(out_path))
            raw_text = out_path.read_text(encoding="utf-8")
            self.assertTrue(raw_text.endswith("\n"))
            self.assertEqual(
                json.loads(raw_text),
                {
                    "events": {"1": "gmem_load", "2": "mma_compute"},
                    "roles": {"0": "producer", "1": "consumer"},
                    "aux": {"0": "stage0", "3": "stage3"},
                    "group": "sm",
                    "pair": "aux",
                    "label_aux": False,
                    "nested": [{"10": ["alpha", [1, 2]]}, {"0": "flag0"}],
                },
            )

            # Normalized key collisions raise ValueError
            with self.assertRaises(ValueError) as ev_col:
                wp_triton.write_names(td_path / "col1.json", {1: "a", "1": "b"})
            self.assertEqual(
                str(ev_col.exception),
                "write_names: events 的 key '1' 与 1 规范化后都是 '1'",
            )

            with self.assertRaises(ValueError) as role_col:
                wp_triton.write_names(td_path / "col2.json", {1: "a"}, roles={True: "r0", "1": "r1"})
            self.assertEqual(
                str(role_col.exception),
                "write_names: roles 的 key '1' 与 True 规范化后都是 '1'",
            )

            with self.assertRaises(ValueError) as extra_col:
                wp_triton.write_names(td_path / "col3.json", {1: "a"}, extra_map={2: "x", "2": "y"})
            self.assertEqual(
                str(extra_col.exception),
                "write_names: key '2' 与 2 规范化后都是 '2'",
            )

            with self.assertRaises(TypeError) as bad_type:
                wp_triton.write_names(td_path / "bad.json", [("1", "a")])  # type: ignore[arg-type]
            self.assertEqual(str(bad_type.exception), "write_names: events 必须是 dict，收到 list")

            # Pin BASE actual behavior on pure-Python `.value` wrapper when `tl is None`:
            # `_uw` only unwraps `isinstance(v, tl.constexpr)` when `tl is not None`, so a
            # pure-Python `.value` object stringifies via `str(key)` ("constexpr[7]") and fails
            # JSON serialization when used as a value (reported as suspected bug in checkpoint).
            constexpr_path = td_path / "constexpr_key.json"
            wp_triton.write_names(constexpr_path, {_FakeConstexpr(7): "tile_load"})
            self.assertEqual(
                json.loads(constexpr_path.read_text(encoding="utf-8")),
                {
                    "events": {"constexpr[7]": "tile_load"},
                    "roles": {},
                    "aux": {},
                    "group": "role",
                    "pair": "stack",
                    "label_aux": True,
                },
            )
            with self.assertRaises(TypeError):
                wp_triton.write_names(td_path / "constexpr_val.json", {1: _FakeConstexpr("tile_load")})

    def test_attach_and_proxy_routing_and_autotune_forwarding(self) -> None:
        l0_fn = _FakeGridKernel("nv_gemm_kernel", "L0_ONLY_ATTR")
        inst_fn = _FakeGridKernel("nv_gemm_kernel_inst", "INST_ONLY_ATTR")
        proxy = wp_triton._FiTritonProxy(l0_fn, inst_fn)
        self.assertEqual(proxy.__name__, "nv_gemm_kernel")

        sentinel_rec = object()
        sentinel_ovf = object()
        self.assertIsNot(sentinel_rec, sentinel_ovf)

        cfg_l0 = wp_triton.FiCfg(cap=16, warps=4, n_slots=8, level=0, threads=128)
        cfg_l2 = wp_triton.FiCfg(cap=16, warps=4, n_slots=8, level=2, threads=128)
        buf_l0 = _FakeNvBuf(sentinel_rec, sentinel_ovf, cfg_l0)
        buf_l2 = _FakeNvBuf(sentinel_rec, sentinel_ovf, cfg_l2)

        # 1) Detached (`fi_attach(None)`) -> routes __getitem__ and __getattr__ to l0_fn
        self.assertIsNone(wp_triton.fi_get_attached())
        self.assertIs(proxy[(4, 1)], l0_fn[(4, 1)])
        self.assertEqual(proxy.kernel_attr, "L0_ONLY_ATTR")

        # 2) Attached with `buf.cfg.level == 0` -> ALSO routes to l0_fn without touching inst_fn
        self.assertIs(wp_triton.fi_attach(buf_l0), buf_l0)  # type: ignore[arg-type]
        self.assertIs(wp_triton.fi_get_attached(), buf_l0)
        self.assertIs(proxy[(4, 1)], l0_fn[(4, 1)])
        self.assertEqual(proxy.kernel_attr, "L0_ONLY_ATTR")
        proxy[(4, 1)]("lhs", BLOCK_M=64)
        self.assertEqual(l0_fn.calls, [((4, 1), ("lhs",), {"BLOCK_M": 64})])
        self.assertEqual(inst_fn.calls, [])

        # 3) Attached with `buf.cfg.level == 2` -> injects 3 exact sidecar kwargs into inst_fn
        wp_triton.fi_attach(buf_l2)  # type: ignore[arg-type]
        self.assertEqual(proxy.kernel_attr, "INST_ONLY_ATTR")
        proxy[(8,)]("lhs", "rhs", BLOCK_M=128)
        self.assertEqual(len(l0_fn.calls), 1)
        self.assertEqual(len(inst_fn.calls), 1)
        grid_arg, pos_args, kw_args = inst_fn.calls[0]
        self.assertEqual(grid_arg, (8,))
        self.assertEqual(pos_args, ("lhs", "rhs"))
        self.assertEqual(set(kw_args.keys()), {"BLOCK_M", "_fi_rec_ptr", "_fi_ovf_ptr", "_FI_CFG"})
        self.assertEqual(kw_args["BLOCK_M"], 128)
        self.assertIs(kw_args["_fi_rec_ptr"], sentinel_rec)
        self.assertIs(kw_args["_fi_ovf_ptr"], sentinel_ovf)
        self.assertIs(kw_args["_FI_CFG"], cfg_l2)

        # 4) _find_autotuner + _autotune_kwargs branches:
        # 4a) Nested .fn chain + best_config.all_kwargs() + caller explicit override wins
        at_leaf = _FakeGridKernel("at_leaf", "L0_AT")
        at_leaf.configs = [_FakeConfigWithAllKwargs({"BLOCK_M": 64, "num_warps": 4})]  # type: ignore[attr-defined]
        at_leaf.best_config = _FakeConfigWithAllKwargs({"BLOCK_M": 256, "num_warps": 8})  # type: ignore[attr-defined]
        nested_l0 = types.SimpleNamespace(fn=types.SimpleNamespace(fn=at_leaf), __name__="wrapped_k")
        inst_a = _FakeGridKernel("inst_a", "INST_A")
        proxy_a = wp_triton._FiTritonProxy(nested_l0, inst_a)
        proxy_a[(2,)]("x", BLOCK_M=32)
        _, _, kw_a = inst_a.calls[0]
        self.assertEqual(kw_a["BLOCK_M"], 32)
        self.assertEqual(kw_a["num_warps"], 8)

        # 4b) best_config with `.kwargs` only (no `all_kwargs` method)
        l0_kw_only = _FakeGridKernel("l0_kw_only", "L0_KW")
        l0_kw_only.configs = [_FakeConfigKwargsOnly({"BLOCK_N": 32})]  # type: ignore[attr-defined]
        l0_kw_only.best_config = _FakeConfigKwargsOnly({"BLOCK_N": 128, "num_stages": 3})  # type: ignore[attr-defined]
        inst_b = _FakeGridKernel("inst_b", "INST_B")
        wp_triton._FiTritonProxy(l0_kw_only, inst_b)[(1,)]()
        _, _, kw_b = inst_b.calls[0]
        self.assertEqual(kw_b["BLOCK_N"], 128)
        self.assertEqual(kw_b["num_stages"], 3)

        # 4c) No best_config -> last entry in `cache`
        l0_cache = _FakeGridKernel("l0_cache", "L0_CACHE")
        l0_cache.configs = [_FakeConfigWithAllKwargs({"BLOCK_K": 16})]  # type: ignore[attr-defined]
        l0_cache.best_config = None  # type: ignore[attr-defined]
        l0_cache.cache = {  # type: ignore[attr-defined]
            "k1": _FakeConfigWithAllKwargs({"BLOCK_K": 32}),
            "k2": _FakeConfigWithAllKwargs({"BLOCK_K": 64}),
        }
        inst_c = _FakeGridKernel("inst_c", "INST_C")
        wp_triton._FiTritonProxy(l0_cache, inst_c)[(1,)]()
        self.assertEqual(inst_c.calls[0][2]["BLOCK_K"], 64)

        # 4d) No best_config and empty `cache` -> `configs[0]`
        l0_cfg0 = _FakeGridKernel("l0_cfg0", "L0_CFG0")
        l0_cfg0.configs = [  # type: ignore[attr-defined]
            _FakeConfigWithAllKwargs({"BLOCK_K": 16}),
            _FakeConfigWithAllKwargs({"BLOCK_K": 128}),
        ]
        l0_cfg0.best_config = None  # type: ignore[attr-defined]
        l0_cfg0.cache = {}  # type: ignore[attr-defined]
        inst_d = _FakeGridKernel("inst_d", "INST_D")
        wp_triton._FiTritonProxy(l0_cfg0, inst_d)[(1,)]()
        self.assertEqual(inst_d.calls[0][2]["BLOCK_K"], 16)

        # 4e) When `inst_fn` also carries `configs`, do NOT merge l0's autotuner kwargs
        inst_with_at = _FakeGridKernel("inst_with_at", "INST_AT")
        inst_with_at.configs = [_FakeConfigWithAllKwargs({"BLOCK_M": 512})]  # type: ignore[attr-defined]
        wp_triton._FiTritonProxy(at_leaf, inst_with_at)[(1,)](caller_flag=7)
        _, _, kw_e = inst_with_at.calls[0]
        self.assertEqual(
            set(kw_e.keys()),
            {"caller_flag", "_fi_rec_ptr", "_fi_ovf_ptr", "_FI_CFG"},
        )
        self.assertEqual(kw_e["caller_flag"], 7)


class TritonPtxAsyncHookCharacterizationTests(unittest.TestCase):
    """Pure-CPU characterization tests for `triton_ptx_async_hook` (`006.md` item 3 exception)."""

    def tearDown(self) -> None:
        self.assertEqual(_NON_LOOPBACK_ATTEMPTS, [])

    def test_hook_without_triton_raises_module_not_found(self) -> None:
        self.assertNotIn("triton", sys.modules)
        with self.assertRaises(ModuleNotFoundError) as ctx:
            with triton_ptx_async.triton_ptx_async_hook():
                pass
        self.assertIn("triton", str(ctx.exception))

    def test_hook_replaces_and_restores_make_cubin_on_normal_and_exception_exit(self) -> None:
        recorded_calls: list[tuple[object, object, object, object]] = []

        class FakeCUDABackend:
            def make_cubin(self, src: object, metadata: object, options: object, capability: object) -> bytes:
                recorded_calls.append((src, metadata, options, capability))
                return b"CUBIN_BYTES"

        orig_make_cubin = FakeCUDABackend.make_cubin

        mod_triton = types.ModuleType("triton")
        mod_backends = types.ModuleType("triton.backends")
        mod_nvidia = types.ModuleType("triton.backends.nvidia")
        mod_compiler = types.ModuleType("triton.backends.nvidia.compiler")
        mod_compiler.CUDABackend = FakeCUDABackend  # type: ignore[attr-defined]
        mod_triton.backends = mod_backends  # type: ignore[attr-defined]
        mod_backends.nvidia = mod_nvidia  # type: ignore[attr-defined]
        mod_nvidia.compiler = mod_compiler  # type: ignore[attr-defined]

        fake_modules = {
            "triton": mod_triton,
            "triton.backends": mod_backends,
            "triton.backends.nvidia": mod_nvidia,
            "triton.backends.nvidia.compiler": mod_compiler,
        }

        ptx_with_wp_open = "\n".join(
            [
                ".visible .entry demo_kernel() {",
                "\t.reg .u64 %rd<4>;",
                "\t.reg .u32 %r<4>;",
                "\t.reg .f32 %f<4>;",
                "\t// begin inline asm",
                "\tmov.u32 pid, %r1;",
                "\tsetp.lt.u32 p_bounds, warp, 4;",
                "\tsetp.lt.u32 p_cap, slot, 16;",
                "\tadd.u64 addr, %rd1, off;",
                "\tsetp.lt.u32 p_cap, n, 32;",
                "\t@p_store st.global.cs.v4.u32 [addr], {tlo0, thi0, 1073741824, sm};",
                "\t@p_ovf atom.global.or.b32 ovfold, [%rd2], 1;",
                "\t// end inline asm",
                "\tcp.async.commit_group;",
                "\tcp.async.wait_group 0;",
                "\tbar.sync 0;",
                "\tmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0, %f1, %f2, %f3}, {%r1, %r2}, {%r3}, {%f0, %f1, %f2, %f3};",
                "\tret;",
                "}",
            ]
        )
        plain_ptx = ".visible .entry plain_kernel() {\n\tret;\n}\n"

        with unittest.mock.patch.dict(sys.modules, fake_modules):
            # Recorder module was already imported before entering patch.dict and must not flip switches
            self.assertIs(wp_triton._TRITON_AVAILABLE, False)
            self.assertIsNone(wp_triton.triton)

            stats_out: dict[str, int] = {"stale_key": 999}
            with triton_ptx_async.triton_ptx_async_hook(stats_out=stats_out):
                self.assertIsNot(FakeCUDABackend.make_cubin, orig_make_cubin)

                backend = FakeCUDABackend()
                backend.make_cubin(ptx_with_wp_open, {"name": "demo"}, {"opt": 3}, 90)
                self.assertEqual(
                    stats_out,
                    {
                        "patched": 1,
                        "cp_async_commits": 1,
                        "cp_async_waits": 1,
                        "wgmma_commits": 0,
                        "wgmma_waits": 0,
                        "mma_clusters": 1,
                        "rebound_wp_blocks": 0,
                    },
                )
                self.assertEqual(len(recorded_calls), 1)
                patched_src, meta0, opt0, cap0 = recorded_calls[0]
                self.assertIsInstance(patched_src, str)
                self.assertIn("mov.u32\t%wp_cur, 1;", str(patched_src))
                self.assertIn("$L__WP_DRAIN_1:", str(patched_src))
                self.assertEqual((meta0, opt0, cap0), ({"name": "demo"}, {"opt": 3}, 90))

                # PTX without `, 1073741824,` is forwarded untouched and leaves stats_out unchanged
                stats_out["keep_marker"] = 7
                backend.make_cubin(plain_ptx, {"name": "plain"}, {"opt": 2}, 80)
                self.assertEqual(len(recorded_calls), 2)
                self.assertIs(recorded_calls[1][0], plain_ptx)
                self.assertEqual(stats_out["keep_marker"], 7)

            # Normal exit restores exact original function object (`is`)
            self.assertIs(FakeCUDABackend.make_cubin, orig_make_cubin)

            # Exception exit also restores exact original function object (`is`)
            with self.assertRaises(RuntimeError) as boom_ctx:
                with triton_ptx_async.triton_ptx_async_hook():
                    self.assertIsNot(FakeCUDABackend.make_cubin, orig_make_cubin)
                    raise RuntimeError("simulated compile failure")
            self.assertEqual(str(boom_ctx.exception), "simulated compile failure")
            self.assertIs(FakeCUDABackend.make_cubin, orig_make_cubin)

        self.assertNotIn("triton", sys.modules)


if __name__ == "__main__":
    unittest.main()

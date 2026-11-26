#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-CPU characterization tests for `recorders/triton/wp_triton_amd.py` (B1-G3, `_timer_contract`, `write_names`)."""
from __future__ import annotations

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
# Clear recorder import-time switches BEFORE importing `wp_triton_amd`
# ---------------------------------------------------------------------------
_RECORDER_ENV_KEYS = (
    "WP_LEVEL",
    "FI_LEVEL",
    "WP_GRAN",
    "FI_GRAN",
    "WP_TIMER",
    "FI_TIMER",
)
for _env_key in (
    *_RECORDER_ENV_KEYS,
    "FI_TIMER_HZ",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
):
    os.environ.pop(_env_key, None)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recorders.triton import wp_triton_amd  # noqa: E402

_WP_TRITON_AMD_PATH = REPO_ROOT / "recorders" / "triton" / "wp_triton_amd.py"
_LOAD_SEQ = 0


def _load_fresh_wp_triton_amd(env_overrides: dict[str, str]) -> tuple[types.ModuleType, dict[str, str]]:
    """Load `wp_triton_amd.py` under a unique module name with controlled env switches."""
    global _LOAD_SEQ
    _LOAD_SEQ += 1
    mod_name = f"_wp_triton_amd_fresh_{_LOAD_SEQ}"
    saved_env = dict(os.environ)
    try:
        for k in _RECORDER_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in env_overrides.items():
            os.environ[k] = v
        spec = importlib.util.spec_from_file_location(mod_name, _WP_TRITON_AMD_PATH)
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
    """Pure-Python `.value` wrapper simulating Triton `constexpr` without importing `triton`."""

    def __init__(self, value: object) -> None:
        self.value = value


class _FakeAmdGridKernel:
    """Minimal subscriptable AMD Triton kernel test double recording `(grid, args, kwargs)`."""

    def __init__(self, name: str, attr_tag: str) -> None:
        self.__name__ = name
        self.amd_attr = attr_tag
        self.calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
        self._grid_launchers: dict[object, object] = {}

    def __getitem__(self, grid: object) -> object:
        if grid not in self._grid_launchers:
            def _launch(*args: object, **kwargs: object) -> None:
                self.calls.append((grid, args, dict(kwargs)))
            self._grid_launchers[grid] = _launch
        return self._grid_launchers[grid]


class _FakeAmdBuf:
    """Minimal AMD buffer test double with 7 pairwise-distinct sentinel attributes."""

    def __init__(
        self,
        *,
        rec: object,
        overflow: object,
        gran: object,
        cap: object,
        wpc: object,
        threads: object,
        timer_mode: object,
        level: int = 0,
    ) -> None:
        self.rec = rec
        self.overflow = overflow
        self.gran = gran
        self.cap = cap
        self.wpc = wpc
        self.threads = threads
        self.timer_mode = timer_mode
        # Extra level fields verify AMD proxy ignores buffer level and hardcodes `_FI_LEVEL=2`
        self.level = level
        self.cfg = types.SimpleNamespace(level=level)


class _FakeConfigWithAllKwargs:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = dict(data)

    def all_kwargs(self) -> dict[str, object]:
        return dict(self._data)


class _FakeConfigKwargsOnly:
    def __init__(self, data: dict[str, object]) -> None:
        self.kwargs = dict(data)


class WpTritonAmdCpuCharacterizationTests(unittest.TestCase):
    """Pure-CPU characterization tests for AMD Triton recorder (`wp_triton_amd.py`)."""

    def setUp(self) -> None:
        self._saved_env = dict(os.environ)
        for k in (*_RECORDER_ENV_KEYS, "FI_TIMER_HZ", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            os.environ.pop(k, None)
        wp_triton_amd.fi_attach(None)
        _NON_LOOPBACK_ATTEMPTS.clear()

    def tearDown(self) -> None:
        wp_triton_amd.fi_attach(None)
        os.environ.clear()
        os.environ.update(self._saved_env)
        self.assertEqual(_NON_LOOPBACK_ATTEMPTS, [])

    def test_timer_contract_modes_and_invalid_values(self) -> None:
        self.assertEqual(wp_triton_amd._timer_contract(0), (0, "s_memrealtime"))
        self.assertEqual(wp_triton_amd._timer_contract(1), (1, "s_memtime+anchors"))
        self.assertEqual(wp_triton_amd._timer_contract("1"), (1, "s_memtime+anchors"))  # type: ignore[arg-type]

        for bad_mode in (2, -1, 99):
            with self.subTest(bad_mode=bad_mode):
                with self.assertRaises(ValueError) as ctx:
                    wp_triton_amd._timer_contract(bad_mode)
                self.assertEqual(
                    str(ctx.exception),
                    "timer_mode 只能是 0(每记录 s_memrealtime) 或 1(s_memtime+锚点)",
                )

    def test_attach_and_proxy_routing_and_8_injected_kwargs(self) -> None:
        l0_fn = _FakeAmdGridKernel("amd_mfma_kernel", "AMD_L0_ATTR")
        inst_fn = _FakeAmdGridKernel("amd_mfma_kernel_inst", "AMD_INST_ATTR")
        proxy = wp_triton_amd._FiTritonProxy(l0_fn, inst_fn)
        self.assertEqual(proxy.__name__, "amd_mfma_kernel")

        sent_rec = object()
        sent_ovf = object()
        sent_gran = object()
        sent_cap = object()
        sent_wpc = object()
        sent_threads = object()
        sent_timer = object()
        self.assertEqual(
            len({id(sent_rec), id(sent_ovf), id(sent_gran), id(sent_cap), id(sent_wpc), id(sent_threads), id(sent_timer)}),
            7,
        )

        # 1) Detached (`fi_attach(None)`) -> routes __getitem__ and __getattr__ to l0_fn
        self.assertIsNone(wp_triton_amd.fi_get_attached())
        self.assertIs(proxy[(2, 2)], l0_fn[(2, 2)])
        self.assertEqual(proxy.amd_attr, "AMD_L0_ATTR")
        proxy[(2, 2)]("a_ptr", BLOCK_M=64)
        self.assertEqual(l0_fn.calls, [((2, 2), ("a_ptr",), {"BLOCK_M": 64})])
        self.assertEqual(inst_fn.calls, [])

        # 2) Attached (`buf is not None`) -> AMD proxy ignores level==0, selects inst_fn, and
        # injects 8 exact sidecar kwargs with hardcoded `_FI_LEVEL=2` (suspected bug on BASE).
        buf = _FakeAmdBuf(
            rec=sent_rec,
            overflow=sent_ovf,
            gran=sent_gran,
            cap=sent_cap,
            wpc=sent_wpc,
            threads=sent_threads,
            timer_mode=sent_timer,
            level=0,
        )
        self.assertIs(wp_triton_amd.fi_attach(buf), buf)
        self.assertIs(wp_triton_amd.fi_get_attached(), buf)
        self.assertEqual(proxy.amd_attr, "AMD_INST_ATTR")

        proxy[(4, 1)]("a_ptr", "b_ptr", BLOCK_M=128)
        self.assertEqual(len(l0_fn.calls), 1)
        self.assertEqual(len(inst_fn.calls), 1)
        grid_arg, pos_args, kw_args = inst_fn.calls[0]
        self.assertEqual(grid_arg, (4, 1))
        self.assertEqual(pos_args, ("a_ptr", "b_ptr"))
        self.assertEqual(
            set(kw_args.keys()),
            {
                "BLOCK_M",
                "_fi_rec_ptr",
                "_fi_ovf_ptr",
                "_FI_LEVEL",
                "_FI_GRAN",
                "_FI_CAP",
                "_FI_WPC",
                "_FI_THREADS",
                "_FI_TIMER",
            },
        )
        self.assertEqual(kw_args["BLOCK_M"], 128)
        self.assertIs(kw_args["_fi_rec_ptr"], sent_rec)
        self.assertIs(kw_args["_fi_ovf_ptr"], sent_ovf)
        self.assertEqual(kw_args["_FI_LEVEL"], 2)
        self.assertIs(kw_args["_FI_GRAN"], sent_gran)
        self.assertIs(kw_args["_FI_CAP"], sent_cap)
        self.assertIs(kw_args["_FI_WPC"], sent_wpc)
        self.assertIs(kw_args["_FI_THREADS"], sent_threads)
        self.assertIs(kw_args["_FI_TIMER"], sent_timer)

    def test_amd_proxy_autotune_kwargs_branches(self) -> None:
        buf = _FakeAmdBuf(
            rec=object(),
            overflow=object(),
            gran=0,
            cap=64,
            wpc=64,
            threads=256,
            timer_mode=0,
        )
        wp_triton_amd.fi_attach(buf)

        # 1) best_config with `all_kwargs()` + caller explicit argument overrides best_config
        l0_all = _FakeAmdGridKernel("l0_all", "L0")
        l0_all.configs = [_FakeConfigWithAllKwargs({"BLOCK_M": 64, "waves_per_eu": 2})]  # type: ignore[attr-defined]
        l0_all.best_config = _FakeConfigWithAllKwargs({"BLOCK_M": 256, "waves_per_eu": 4})  # type: ignore[attr-defined]
        inst_all = _FakeAmdGridKernel("inst_all", "INST")
        wp_triton_amd._FiTritonProxy(l0_all, inst_all)[(1,)]("x", BLOCK_M=32)
        self.assertEqual(inst_all.calls[0][2]["BLOCK_M"], 32)
        self.assertEqual(inst_all.calls[0][2]["waves_per_eu"], 4)

        # 2) best_config with `.kwargs` only
        l0_kw = _FakeAmdGridKernel("l0_kw", "L0")
        l0_kw.configs = [_FakeConfigKwargsOnly({"BLOCK_N": 32})]  # type: ignore[attr-defined]
        l0_kw.best_config = _FakeConfigKwargsOnly({"BLOCK_N": 128, "num_stages": 2})  # type: ignore[attr-defined]
        inst_kw = _FakeAmdGridKernel("inst_kw", "INST")
        wp_triton_amd._FiTritonProxy(l0_kw, inst_kw)[(1,)]()
        self.assertEqual(inst_kw.calls[0][2]["BLOCK_N"], 128)
        self.assertEqual(inst_kw.calls[0][2]["num_stages"], 2)

        # 3) No best_config -> last entry in `cache`
        l0_cache = _FakeAmdGridKernel("l0_cache", "L0")
        l0_cache.configs = [_FakeConfigWithAllKwargs({"BLOCK_K": 16})]  # type: ignore[attr-defined]
        l0_cache.best_config = None  # type: ignore[attr-defined]
        l0_cache.cache = {  # type: ignore[attr-defined]
            "c0": _FakeConfigWithAllKwargs({"BLOCK_K": 32}),
            "c1": _FakeConfigWithAllKwargs({"BLOCK_K": 64}),
        }
        inst_cache = _FakeAmdGridKernel("inst_cache", "INST")
        wp_triton_amd._FiTritonProxy(l0_cache, inst_cache)[(1,)]()
        self.assertEqual(inst_cache.calls[0][2]["BLOCK_K"], 64)

        # 4) No best_config and empty `cache` -> `configs[0]`
        l0_cfg0 = _FakeAmdGridKernel("l0_cfg0", "L0")
        l0_cfg0.configs = [  # type: ignore[attr-defined]
            _FakeConfigWithAllKwargs({"BLOCK_K": 16}),
            _FakeConfigWithAllKwargs({"BLOCK_K": 128}),
        ]
        l0_cfg0.best_config = None  # type: ignore[attr-defined]
        l0_cfg0.cache = {}  # type: ignore[attr-defined]
        inst_cfg0 = _FakeAmdGridKernel("inst_cfg0", "INST")
        wp_triton_amd._FiTritonProxy(l0_cfg0, inst_cfg0)[(1,)]()
        self.assertEqual(inst_cfg0.calls[0][2]["BLOCK_K"], 16)

        # 5) `inst_fn` also carries `configs` -> do NOT merge `l0_fn`'s autotuner kwargs
        inst_with_at = _FakeAmdGridKernel("inst_with_at", "INST")
        inst_with_at.configs = [_FakeConfigWithAllKwargs({"BLOCK_M": 512})]  # type: ignore[attr-defined]
        wp_triton_amd._FiTritonProxy(l0_all, inst_with_at)[(1,)](caller_only=9)
        self.assertNotIn("waves_per_eu", inst_with_at.calls[0][2])
        self.assertEqual(inst_with_at.calls[0][2]["caller_only"], 9)

        # 6) Pin BASE difference vs NVIDIA: nested `.fn` wrapper around autotuner is NOT unwrapped
        wrapped_l0 = types.SimpleNamespace(fn=l0_all, __name__="wrapped_amd")
        inst_nested = _FakeAmdGridKernel("inst_nested", "INST")
        wp_triton_amd._FiTritonProxy(wrapped_l0, inst_nested)[(1,)]()
        self.assertNotIn("waves_per_eu", inst_nested.calls[0][2])

    def test_write_names_unpinned_branches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # 1) Pure-Python `.value` unwrapping via `_name_key` + omitted optional `aux`/`label_aux`
            p_default = td_path / "nested" / "names_default.json"
            ret_default = wp_triton_amd.write_names(
                p_default,
                events={_FakeConstexpr(1): "global_load", "2": "mfma_compute"},
                roles={_FakeConstexpr(0): "wave_worker"},
            )
            self.assertEqual(ret_default, str(p_default))
            self.assertEqual(
                json.loads(p_default.read_text(encoding="utf-8")),
                {
                    "events": {"1": "global_load", "2": "mfma_compute"},
                    "roles": {"0": "wave_worker"},
                    "group": "cu",
                    "pair": "stack",
                },
            )

            # 2) Explicit `aux`, `group="cta"`, `pair="aux"`, `label_aux=False`, and `**extra`
            p_full = td_path / "names_full.json"
            wp_triton_amd.write_names(
                p_full,
                events={3: "lds_write"},
                roles={1: "epilogue"},
                aux={_FakeConstexpr(0): "buf0", 1: "buf1"},
                group="cta",
                pair="aux",
                label_aux=False,
                schema_note="gfx942",
            )
            self.assertEqual(
                json.loads(p_full.read_text(encoding="utf-8")),
                {
                    "events": {"3": "lds_write"},
                    "roles": {"1": "epilogue"},
                    "aux": {"0": "buf0", "1": "buf1"},
                    "group": "cta",
                    "pair": "aux",
                    "label_aux": False,
                    "schema_note": "gfx942",
                },
            )

            # 3) `bool` keys (including wrapped `.value = True`) raise TypeError
            for bad_events in ({True: "ev"}, {_FakeConstexpr(False): "ev"}):
                with self.subTest(bad_events=bad_events):
                    with self.assertRaises(TypeError) as bool_ctx:
                        wp_triton_amd.write_names(td_path / "bool.json", bad_events)
                    self.assertEqual(str(bool_ctx.exception), "names.json keys must be integers, not bool")

            # 4) Invalid `group` and invalid `pair` raise ValueError
            with self.assertRaises(ValueError) as grp_ctx:
                wp_triton_amd.write_names(td_path / "grp.json", {1: "ev"}, group="warp")
            self.assertEqual(str(grp_ctx.exception), "group must be role/sm/cu/cta, got 'warp'")

            with self.assertRaises(ValueError) as pair_ctx:
                wp_triton_amd.write_names(td_path / "pair.json", {1: "ev"}, pair="fifo")
            self.assertEqual(str(pair_ctx.exception), "pair must be stack/aux, got 'fifo'")

            # 5) Pin BASE silent overwrite on colliding normalized keys (suspected bug vs NVIDIA)
            p_col = td_path / "collision.json"
            wp_triton_amd.write_names(p_col, {_FakeConstexpr(1): "first_label", "1": "second_label"})
            self.assertEqual(
                json.loads(p_col.read_text(encoding="utf-8"))["events"],
                {"1": "second_label"},
            )

    def test_import_time_env_and_missing_runtime_errors(self) -> None:
        # 1) Clean env -> (2, 0, 0)
        mod_clean, env_clean = _load_fresh_wp_triton_amd({})
        self.assertEqual((mod_clean.WP_LEVEL, mod_clean.WP_GRAN, mod_clean.WP_TIMER), (2, 0, 0))
        self.assertNotIn("FI_LEVEL", env_clean)
        self.assertNotIn("FI_GRAN", env_clean)
        self.assertNotIn("FI_TIMER", env_clean)
        self.assertIs(mod_clean.wp_available(), False)

        # 2) Only WP_TIMER="1" -> FI_TIMER mirrored to "1"
        mod_t1, env_t1 = _load_fresh_wp_triton_amd({"WP_TIMER": "1"})
        self.assertEqual(env_t1["FI_TIMER"], "1")
        self.assertEqual((mod_t1.WP_LEVEL, mod_t1.WP_GRAN, mod_t1.WP_TIMER), (2, 0, 1))

        # 3) All 3 switches set -> (0, 1, 1)
        mod_all, env_all = _load_fresh_wp_triton_amd({"WP_LEVEL": "0", "WP_GRAN": "1", "WP_TIMER": "1"})
        self.assertEqual((env_all["FI_LEVEL"], env_all["FI_GRAN"], env_all["FI_TIMER"]), ("0", "1", "1"))
        self.assertEqual((mod_all.WP_LEVEL, mod_all.WP_GRAN, mod_all.WP_TIMER), (0, 1, 1))

        # 4) Conflict WP_TIMER="1" + FI_TIMER="0" -> RuntimeError
        with self.assertRaises(RuntimeError) as conflict_ctx:
            _load_fresh_wp_triton_amd({"WP_TIMER": "1", "FI_TIMER": "0"})
        self.assertEqual(
            str(conflict_ctx.exception),
            "WP_TIMER='1' conflicts with FI_TIMER='0'; set only one spelling",
        )

        # 5) Out-of-range WP_GRAN="2" and WP_LEVEL="4" -> ValueError
        with self.assertRaises(ValueError) as gran_ctx:
            _load_fresh_wp_triton_amd({"WP_GRAN": "2"})
        self.assertEqual(str(gran_ctx.exception), "WP_GRAN must be one of (0, 1), got 2")

        with self.assertRaises(ValueError) as level_ctx:
            _load_fresh_wp_triton_amd({"WP_LEVEL": "4"})
        self.assertEqual(str(level_ctx.exception), "WP_LEVEL must be one of (0, 1, 2), got 4")

        # 6) Host-side no-torch/no-triton/no-ROCm failure contracts
        self.assertIs(wp_triton_amd.wp_available(), False)
        with self.assertRaises(NotImplementedError) as open_by_ctx:
            wp_triton_amd.wp_open_by()
        self.assertEqual(
            str(open_by_ctx.exception),
            "wp_open_by is not available on the AMD Triton backend: wp_open selects the recording wave "
            "from HW_ID (lowest SIMD/wave per CTA for WP_GRAN=1, lane 0 of every wave for WP_GRAN=0)",
        )

        with self.assertRaises(ImportError) as dev_ctx:
            _ = wp_triton_amd.fi_open
        self.assertEqual(
            str(dev_ctx.exception),
            "wp_triton_amd.fi_open needs Triton (ROCm) for device code (triton not importable)",
        )
        with self.assertRaises(AttributeError) as attr_ctx:
            _ = wp_triton_amd.no_such_symbol
        self.assertEqual(str(attr_ctx.exception), "module 'wp_triton_amd' has no attribute 'no_such_symbol'")

        for factory in (
            lambda: wp_triton_amd.FiBuffer(1, 8),
            lambda: wp_triton_amd.FiPackedBuffer(1, 8),
            lambda: wp_triton_amd.wp_buffer(1, 8),
        ):
            with self.assertRaises(RuntimeError) as buf_ctx:
                factory()
            self.assertEqual(str(buf_ctx.exception), "FiBuffer 需要 torch（ROCm 版）")

        with self.assertRaises(RuntimeError) as triton_ctx:
            wp_triton_amd._need_triton()
        self.assertEqual(str(triton_ctx.exception), "fi_* device helpers 需要 triton")

        os.environ["FI_TIMER_HZ"] = "100000000"
        self.assertEqual(wp_triton_amd._wall_clock_hz(), 100_000_000)
        os.environ["FI_TIMER_HZ"] = "0"
        with self.assertRaises(ValueError) as hz_ctx:
            wp_triton_amd._wall_clock_hz()
        self.assertEqual(str(hz_ctx.exception), "FI_TIMER_HZ 必须为正整数")


if __name__ == "__main__":
    unittest.main()

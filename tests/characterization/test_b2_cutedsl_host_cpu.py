#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-CPU characterization tests for `recorders/cutedsl/wp_cute.py` host-side behavior (R16 / B2-G2)."""
from __future__ import annotations

import importlib.util
import ipaddress
import json
import lzma
import os
from pathlib import Path
import socket
import struct
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
# Clear recorder import-time switches BEFORE loading `wp_cute.py`
# ---------------------------------------------------------------------------
_RECORDER_ENV_KEYS = (
    "WP_LEVEL",
    "FI_LEVEL",
    "WP_GRAN",
    "FI_GRAN",
    "WP_TIMER",
    "FI_TIMER",
)
_ISOLATED_EXTRA_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "WEAVEPROF_ALLOW_GZ",
)
for _k in (*_RECORDER_ENV_KEYS, *_ISOLATED_EXTRA_ENV_KEYS):
    os.environ.pop(_k, None)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_WP_CUTE_PATH = REPO_ROOT / "recorders" / "cutedsl" / "wp_cute.py"
_GOLDEN_PINGPONG_DIR = REPO_ROOT / "tests" / "golden" / "cases" / "h20_pingpong_v1"
_LOAD_SEQ = 0

# Child-process socket guard loaded automatically via `PYTHONPATH` (`sitecustomize.py`)
# when `open_perfetto` spawns `[sys.executable, "-B", wp_decode.py, ...]`.
_SITECUSTOMIZE_GUARD_SOURCE = """\
import ipaddress
import os
from pathlib import Path
import socket

_marker = os.environ.get("WP_TEST_SUBPROC_GUARD_MARKER")
if _marker:
    Path(_marker).write_text("loaded\\n", encoding="utf-8")

_log = os.environ.get("WP_TEST_SUBPROC_NET_LOG")
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo


def _record_violation(entry):
    if _log:
        with open(_log, "a", encoding="utf-8") as fh:
            fh.write(entry + "\\n")


def _is_loopback(h):
    return isinstance(h, str) and h.strip() in ("127.0.0.1", "::1", "localhost")


def _is_ip(h):
    if not isinstance(h, str):
        return False
    try:
        ipaddress.ip_address(h.strip())
        return True
    except ValueError:
        return False


def _g_connect(self, address):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        _record_violation(f"connect:{address!r}")
        raise ConnectionRefusedError(f"blocked by subprocess test guard: {address}")
    return _orig_connect(self, address)


def _g_connect_ex(self, address):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        _record_violation(f"connect_ex:{address!r}")
        raise ConnectionRefusedError(f"blocked by subprocess test guard: {address}")
    return _orig_connect_ex(self, address)


def _g_create_connection(address, *args, **kwargs):
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        _record_violation(f"create_connection:{address!r}")
        raise ConnectionRefusedError(f"blocked by subprocess test guard: {address}")
    return _orig_create_connection(address, *args, **kwargs)


def _g_getaddrinfo(host, port, *args, **kwargs):
    if not (_is_loopback(host) or _is_ip(host)):
        _record_violation(f"getaddrinfo:{host!r}:{port!r}")
        raise OSError(f"blocked by subprocess test guard: {host!r}")
    return _orig_getaddrinfo(host, port, *args, **kwargs)


socket.socket.connect = _g_connect
socket.socket.connect_ex = _g_connect_ex
socket.create_connection = _g_create_connection
socket.getaddrinfo = _g_getaddrinfo
"""


def _load_fresh_wp_cute(env_overrides: dict[str, str] | None = None) -> tuple[types.ModuleType, dict[str, str]]:
    """Load a fresh `wp_cute.py` module under a unique name with controlled env switches."""
    global _LOAD_SEQ
    _LOAD_SEQ += 1
    mod_name = f"_wp_cute_fresh_{_LOAD_SEQ}"
    saved_env = dict(os.environ)
    try:
        for k in _RECORDER_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in (env_overrides or {}).items():
            os.environ[k] = v
        spec = importlib.util.spec_from_file_location(mod_name, _WP_CUTE_PATH)
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


def _build_minimal_filite02_dump() -> bytes:
    """Construct a minimal 1-slot, 4-record valid FILITE02 binary dump (OPEN, B, E, CLOSE)."""
    meta_bytes = json.dumps(
        {
            "fi_version": "fast-instrument-cutedsl-v1",
            "language": "cutedsl",
            "gran": 0,
            "arch": "sm_90a",
            "backend": "nvidia_cutedsl",
            "platform": "nv",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    # Header: n_slots=1, cap=8, warps_per_cta=1, ovf=0, timer=0, meta_len
    header = struct.pack("<6I", 1, 8, 1, 0, 0, len(meta_bytes))
    counts = struct.pack("<1I", 4)
    records = b"".join(
        [
            struct.pack("<QII", 1000, 0x40000000, 3),  # OPEN role=0 on SM 3
            struct.pack("<QII", 1100, 1, 0),           # BEGIN ev=1 (load)
            struct.pack("<QII", 2100, 0x80000001, 0),  # END ev=1 (load)
            struct.pack("<QII", 2200, 0x08000000, 0),  # CLOSE
        ]
    )
    return b"FILITE02" + header + meta_bytes + counts + records


class _IndexableInt:
    """Pure-Python object implementing `__index__` without subclassing `int`."""

    def __init__(self, value: int) -> None:
        self._value = value

    def __index__(self) -> int:
        return self._value


class _FakeConstexpr:
    """Pure-Python `.value` wrapper simulating `Constexpr` without importing `cutlass`."""

    def __init__(self, value: object = None) -> None:
        self.value = value


class _FakeCuteEntry:
    """Callable kernel test double WITH `__name__` recording `(args, kwargs)`."""

    def __init__(self, name: str, tag_attr: str) -> None:
        self.__name__ = name
        self.tag_attr = tag_attr
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, dict(kwargs)))


class _NamelessCallable:
    """Callable test double WITHOUT instance `__name__` attribute."""

    def __init__(self, tag_attr: str) -> None:
        self.tag_attr = tag_attr
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, dict(kwargs)))


class _FakeCuteBuf:
    """Minimal buffer test double holding pairwise-distinct sentinels for `records_cute`, `overflow_cute`, `timer`."""

    def __init__(self, *, records_cute: object, overflow_cute: object, timer: object) -> None:
        self.records_cute = records_cute
        self.overflow_cute = overflow_cute
        self.timer = timer


class _FreshWpCuteTestCase(unittest.TestCase):
    """Base test case loading a fresh `wp_cute` module per test and resetting `fi_attach(None)`."""

    def setUp(self) -> None:
        self._saved_env = dict(os.environ)
        for k in (*_RECORDER_ENV_KEYS, *_ISOLATED_EXTRA_ENV_KEYS):
            os.environ.pop(k, None)
        _NON_LOOPBACK_ATTEMPTS.clear()
        self.mod, _ = _load_fresh_wp_cute({})

    def tearDown(self) -> None:
        self.mod.fi_attach(None)
        os.environ.clear()
        os.environ.update(self._saved_env)
        self.assertEqual(_NON_LOOPBACK_ATTEMPTS, [])


class WpCuteEnvAndAvailabilityTests(_FreshWpCuteTestCase):
    """Item 1: import-time env mapping/validation, `_TIMER_ENV_EXPLICIT`, `wp_available()`, and lazy `__getattr__`."""

    def test_clean_env_defaults_and_cutlass_unavailable_contract(self) -> None:
        mod, post_env = _load_fresh_wp_cute({})
        self.assertEqual((mod.WP_LEVEL, mod.WP_GRAN, mod.WP_TIMER), (2, 0, 0))
        self.assertEqual((mod.FI_LEVEL, mod.FI_GRAN, mod.FI_TIMER), (2, 0, 0))
        self.assertIs(mod._TIMER_ENV_EXPLICIT, False)
        self.assertNotIn("FI_LEVEL", post_env)
        self.assertNotIn("FI_GRAN", post_env)
        self.assertNotIn("FI_TIMER", post_env)

        self.assertIs(mod.wp_available(), False)
        self.assertIs(mod._CUTLASS_AVAILABLE, False)
        self.assertIsInstance(mod._IMPORT_ERROR, ModuleNotFoundError)
        self.assertIn("cutlass", str(mod._IMPORT_ERROR))
        self.assertEqual(str(mod._IMPORT_ERROR), "No module named 'cutlass'")

        for sym in ("FiCtx", "WpCtx", "fi_open", "wp_b", "WP_CLOSE"):
            with self.subTest(sym=sym):
                with self.assertRaises(ImportError) as imp_ctx:
                    getattr(mod, sym)
                self.assertEqual(
                    str(imp_ctx.exception),
                    f"wp_cute.{sym} needs the CUTLASS CuTeDSL runtime (pip install nvidia-cutlass-dsl); "
                    "import of cutlass failed: No module named 'cutlass'",
                )

        with self.assertRaises(AttributeError) as attr_ctx:
            _ = mod.unknown_host_symbol
        self.assertEqual(str(attr_ctx.exception), "module 'wp_cute' has no attribute 'unknown_host_symbol'")

    def test_env_mapping_explicit_flag_conflicts_and_range_validation(self) -> None:
        # 1) Only WP_TIMER="1" -> mirrors FI_TIMER="1" and sets _TIMER_ENV_EXPLICIT=True
        mod_t1, env_t1 = _load_fresh_wp_cute({"WP_TIMER": "1"})
        self.assertEqual(env_t1["FI_TIMER"], "1")
        self.assertIs(mod_t1._TIMER_ENV_EXPLICIT, True)
        self.assertEqual((mod_t1.WP_LEVEL, mod_t1.WP_GRAN, mod_t1.WP_TIMER), (2, 0, 1))
        self.assertEqual((mod_t1.FI_LEVEL, mod_t1.FI_GRAN, mod_t1.FI_TIMER), (2, 0, 1))

        # 2) Explicit WP_TIMER="0" also sets _TIMER_ENV_EXPLICIT=True while keeping timer=0
        mod_t0, env_t0 = _load_fresh_wp_cute({"WP_TIMER": "0"})
        self.assertEqual(env_t0["FI_TIMER"], "0")
        self.assertIs(mod_t0._TIMER_ENV_EXPLICIT, True)
        self.assertEqual(mod_t0.WP_TIMER, 0)

        # 3) All 3 canonical switches set -> mirrored to FI_*
        mod_all, env_all = _load_fresh_wp_cute({"WP_LEVEL": "0", "WP_GRAN": "1", "WP_TIMER": "1"})
        self.assertEqual((env_all["FI_LEVEL"], env_all["FI_GRAN"], env_all["FI_TIMER"]), ("0", "1", "1"))
        self.assertEqual((mod_all.WP_LEVEL, mod_all.WP_GRAN, mod_all.WP_TIMER), (0, 1, 1))
        self.assertEqual((mod_all.FI_LEVEL, mod_all.FI_GRAN, mod_all.FI_TIMER), (0, 1, 1))

        # 4) Identical canonical and legacy values do not conflict
        mod_same, _ = _load_fresh_wp_cute({"WP_TIMER": "1", "FI_TIMER": "1"})
        self.assertEqual(mod_same.WP_TIMER, 1)

        # 5) Conflicting canonical and legacy values raise RuntimeError
        for conflict_env, expected_msg in (
            (
                {"WP_TIMER": "1", "FI_TIMER": "0"},
                "WP_TIMER='1' conflicts with FI_TIMER='0'; set only one spelling",
            ),
            (
                {"WP_LEVEL": "1", "FI_LEVEL": "2"},
                "WP_LEVEL='1' conflicts with FI_LEVEL='2'; set only one spelling",
            ),
            (
                {"WP_GRAN": "1", "FI_GRAN": "0"},
                "WP_GRAN='1' conflicts with FI_GRAN='0'; set only one spelling",
            ),
        ):
            with self.subTest(conflict_env=conflict_env):
                with self.assertRaises(RuntimeError) as r_ctx:
                    _load_fresh_wp_cute(conflict_env)
                self.assertEqual(str(r_ctx.exception), expected_msg)

        # 6) Out-of-range values raise ValueError with exact messages (`wp_cute.py:61-66`)
        for bad_env, expected_val_msg in (
            ({"WP_LEVEL": "3"}, "WP_LEVEL must be 0, 1 or 2, got 3"),
            ({"FI_LEVEL": "-1"}, "WP_LEVEL must be 0, 1 or 2, got -1"),
            ({"WP_GRAN": "2"}, "WP_GRAN must be 0 or 1, got 2"),
            ({"WP_TIMER": "2"}, "WP_TIMER must be 0 or 1, got 2"),
        ):
            with self.subTest(bad_env=bad_env):
                with self.assertRaises(ValueError) as v_ctx:
                    _load_fresh_wp_cute(bad_env)
                self.assertEqual(str(v_ctx.exception), expected_val_msg)

        # 7) FiBuffer.__init__ pre-torch argument checks vs missing-torch error
        with self.assertRaises(ValueError) as arch_ctx:
            self.mod.FiBuffer(2, 4, 32, arch="gfx942")
        self.assertEqual(
            str(arch_ctx.exception),
            "CuTeDSL arch must be an NVIDIA sm_* target such as sm_90, sm_100a or sm_103a; got 'gfx942'",
        )
        with self.assertRaises(ValueError) as bk_ctx:
            self.mod.FiBuffer(2, 4, 32, arch="sm_90a", backend="rocm")
        self.assertEqual(
            str(bk_ctx.exception),
            "CuTeDSL backend must be NVIDIA (for example 'nvidia_cutedsl'); got 'rocm'",
        )
        with self.assertRaises(ValueError) as tm_ctx:
            self.mod.FiBuffer(2, 4, 32, arch="sm_90a", timer=True)
        self.assertEqual(str(tm_ctx.exception), "timer must be 0 or 1, got True")
        with self.assertRaises(ModuleNotFoundError) as torch_ctx:
            self.mod.FiBuffer(2, 4, 32, arch="sm_90a")
        self.assertIn("torch", str(torch_ctx.exception))


class WpCuteNamesCharacterizationTests(_FreshWpCuteTestCase):
    """Item 2: `_positive_int`, `_coerce_gran`, `_name_key`, `_name_mapping`, `_names_document`, `write_names`."""

    def test_positive_int_and_coerce_gran_valid_and_invalid(self) -> None:
        self.assertEqual(self.mod._positive_int("cap", 1), 1)
        self.assertEqual(self.mod._positive_int("cap", 64), 64)
        self.assertEqual(self.mod._positive_int("n_ctas", _IndexableInt(12)), 12)

        for bad_val, expected_msg in (
            (True, "cap must be a positive integer, got True"),
            (False, "cap must be a positive integer, got False"),
            ("16", "cap must be a positive integer, got '16'"),
            (2.5, "cap must be a positive integer, got 2.5"),
            (0, "cap must be positive, got 0"),
            (-4, "cap must be positive, got -4"),
        ):
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(ValueError) as ctx:
                    self.mod._positive_int("cap", bad_val)
                self.assertEqual(str(ctx.exception), expected_msg)

        self.assertEqual(self.mod._coerce_gran(0), 0)
        self.assertEqual(self.mod._coerce_gran(1), 1)
        self.assertEqual(self.mod._coerce_gran(_IndexableInt(1)), 1)

        for bad_gran, expected_gran_msg in (
            (
                "warp",
                "gran must be numeric 0 (warp) or 1 (CTA); string values such as 'warp'/'sm' are not accepted: got 'warp'",
            ),
            (
                "0",
                "gran must be numeric 0 (warp) or 1 (CTA); string values such as 'warp'/'sm' are not accepted: got '0'",
            ),
            (True, "gran must be 0 or 1, got True"),
            (False, "gran must be 0 or 1, got False"),
            (None, "gran must be 0 or 1, got None"),
            (-1, "gran must be 0 or 1, got -1"),
            (2, "gran must be 0 or 1, got 2"),
        ):
            with self.subTest(bad_gran=bad_gran):
                with self.assertRaises(ValueError) as g_ctx:
                    self.mod._coerce_gran(bad_gran)
                self.assertEqual(str(g_ctx.exception), expected_gran_msg)

    def test_name_key_name_mapping_names_document_and_write_names(self) -> None:
        # 1) _name_key valid cases (int, bool, stripped str, indexable, nested .value up to 8 hops)
        self.assertEqual(self.mod._name_key(7, "events"), "7")
        self.assertEqual(self.mod._name_key(True, "roles"), "1")
        self.assertEqual(self.mod._name_key(False, "roles"), "0")
        self.assertEqual(self.mod._name_key("  12  ", "events"), "12")
        self.assertEqual(self.mod._name_key(_IndexableInt(9), "events"), "9")

        nested_8: object = 42
        for _ in range(8):
            nested_8 = _FakeConstexpr(nested_8)
        self.assertEqual(self.mod._name_key(nested_8, "events"), "42")

        # 2) _name_key invalid cases (cycle / >8 hops, non-string/int, empty string)
        nested_10: object = 42
        for _ in range(10):
            nested_10 = _FakeConstexpr(nested_10)
        with self.assertRaises(TypeError) as depth_ctx:
            self.mod._name_key(nested_10, "events")
        self.assertEqual(str(depth_ctx.exception), "events key has a cyclic .value chain")

        cyc_a = _FakeConstexpr()
        cyc_b = _FakeConstexpr(cyc_a)
        cyc_a.value = cyc_b
        with self.assertRaises(TypeError) as cyc_ctx:
            self.mod._name_key(cyc_a, "roles")
        self.assertEqual(str(cyc_ctx.exception), "roles key has a cyclic .value chain")

        self_ref = _FakeConstexpr()
        self_ref.value = self_ref
        with self.assertRaises(TypeError) as sref_ctx:
            self.mod._name_key(self_ref, "aux")
        self.assertIn("aux key must be a string/integer or expose .value; got", str(sref_ctx.exception))

        with self.assertRaises(TypeError) as float_ctx:
            self.mod._name_key(3.5, "events")
        self.assertEqual(
            str(float_ctx.exception),
            "events key must be a string/integer or expose .value; got 3.5",
        )

        for empty_key in ("", "   ", _FakeConstexpr(" \t ")):
            with self.subTest(empty_key=empty_key):
                with self.assertRaises(ValueError) as emp_ctx:
                    self.mod._name_key(empty_key, "events")
                self.assertEqual(str(emp_ctx.exception), "events key must not be empty")

        # 3) _name_mapping valid, non-mapping TypeError, empty-value ValueError, and duplicate key ValueError
        self.assertEqual(self.mod._name_mapping(None, "aux"), {})
        self.assertEqual(
            self.mod._name_mapping({1: "  tma_load ", _FakeConstexpr(2): "wgmma_mma"}, "events"),
            {"1": "tma_load", "2": "wgmma_mma"},
        )
        with self.assertRaises(TypeError) as map_type_ctx:
            self.mod._name_mapping([("1", "load")], "events")
        self.assertEqual(str(map_type_ctx.exception), "events must be a mapping, got list")

        for bad_ev_val in ("", "   ", 123, None):
            with self.subTest(bad_ev_val=bad_ev_val):
                with self.assertRaises(ValueError) as bad_v_ctx:
                    self.mod._name_mapping({1: bad_ev_val}, "events")
                self.assertEqual(str(bad_v_ctx.exception), "events['1'] must be a non-empty string")

        # Normalized key collision raises ValueError (like NVIDIA wp_triton, unlike AMD silent overwrite)
        for dup_map, expected_dup_msg in (
            ({1: "first", "1": "second"}, "events contains duplicate normalized key '1'"),
            ({_FakeConstexpr(5): "first", " 5 ": "second"}, "events contains duplicate normalized key '5'"),
            ({True: "first", "1": "second"}, "events contains duplicate normalized key '1'"),
        ):
            with self.subTest(dup_map=dup_map):
                with self.assertRaises(ValueError) as dup_ctx:
                    self.mod._name_mapping(dup_map, "events")
                self.assertEqual(str(dup_ctx.exception), expected_dup_msg)

        # 4) _names_document: auto `label_aux = bool(aux_map)` vs explicit `label_aux`
        doc_no_aux = self.mod._names_document(roles={0: "producer"}, events={1: "load"})
        self.assertEqual(
            doc_no_aux,
            {
                "events": {"1": "load"},
                "roles": {"0": "producer"},
                "aux": {},
                "label_aux": False,
                "pair": "stack",
                "group": "role",
            },
        )
        doc_with_aux = self.mod._names_document(
            roles={0: "producer"},
            events={1: "load"},
            aux={_FakeConstexpr(0): "stage0"},
            group="sm",
            pair="aux",
        )
        self.assertEqual(
            doc_with_aux,
            {
                "events": {"1": "load"},
                "roles": {"0": "producer"},
                "aux": {"0": "stage0"},
                "label_aux": True,
                "pair": "aux",
                "group": "sm",
            },
        )

        # 5) Module-level `write_names` on temporary directory + collision error propagation
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            out_file = td_p / "nested" / "names.json"
            ret_path = self.mod.write_names(
                str(out_file),
                events={_FakeConstexpr(1): "tma_load", 3: "wgmma"},
                roles={0: "dma_warp", 1: "mma_warp"},
                aux={0: "pipe0"},
                group="sm",
                pair="stack",
                label_aux=False,
                kernel_tag="cute_pingpong",
            )
            self.assertEqual(ret_path, str(out_file))
            raw_text = out_file.read_text(encoding="utf-8")
            self.assertTrue(raw_text.endswith("\n"))
            self.assertEqual(
                json.loads(raw_text),
                {
                    "events": {"1": "tma_load", "3": "wgmma"},
                    "roles": {"0": "dma_warp", "1": "mma_warp"},
                    "aux": {"0": "pipe0"},
                    "label_aux": False,
                    "pair": "stack",
                    "group": "sm",
                    "kernel_tag": "cute_pingpong",
                },
            )

            with self.assertRaises(ValueError) as wn_dup:
                self.mod.write_names(str(td_p / "dup.json"), {_FakeConstexpr(2): "a", "2": "b"})
            self.assertEqual(str(wn_dup.exception), "events contains duplicate normalized key '2'")


class WpCuteAttachProxyAndDescriptorTests(_FreshWpCuteTestCase):
    """Item 3 (B2-G2): `fi_attach` sticky timer mutation, `_FiCuteProxy`, `_FiCuteDescriptor`, and aliases."""

    def test_fi_attach_updates_module_timers_and_detach_leaves_them_unrestored(self) -> None:
        self.assertIsNone(self.mod.fi_get_attached())
        self.assertEqual((self.mod.FI_TIMER, self.mod.WP_TIMER), (0, 0))

        sent_rec = object()
        sent_ovf = object()
        buf_t1 = _FakeCuteBuf(records_cute=sent_rec, overflow_cute=sent_ovf, timer=1)

        # Attaching buf with timer=1 mutates module-level FI_TIMER and WP_TIMER to 1
        self.assertIs(self.mod.fi_attach(buf_t1), buf_t1)
        self.assertIs(self.mod.fi_get_attached(), buf_t1)
        self.assertEqual((self.mod.FI_TIMER, self.mod.WP_TIMER), (1, 1))

        # Pin BASE suspected bug: `fi_attach(None)` detaches buffer (`None`) but does NOT
        # restore `FI_TIMER` / `WP_TIMER` back to their pre-attach values (`0`).
        self.assertIsNone(self.mod.fi_attach(None))
        self.assertIsNone(self.mod.fi_get_attached())
        self.assertEqual((self.mod.FI_TIMER, self.mod.WP_TIMER), (1, 1))

        # Attaching a buffer with timer=0 updates FI_TIMER and WP_TIMER to 0
        buf_t0 = _FakeCuteBuf(records_cute=sent_rec, overflow_cute=sent_ovf, timer=0)
        self.mod.fi_attach(buf_t0)
        self.assertEqual((self.mod.FI_TIMER, self.mod.WP_TIMER), (0, 0))

    def test_fi_cute_proxy_call_getattr_and_fallback_name(self) -> None:
        l0_fn = _FakeCuteEntry("cute_gemm_entry", "L0_ATTR_VAL")
        inst_fn = _FakeCuteEntry("cute_gemm_entry_inst", "INST_ATTR_VAL")
        proxy = self.mod._FiCuteProxy(l0_fn, inst_fn)
        self.assertEqual(proxy.__name__, "cute_gemm_entry")

        nameless_l0 = _NamelessCallable("NAMELESS_L0")
        nameless_inst = _NamelessCallable("NAMELESS_INST")
        nameless_proxy = self.mod._FiCuteProxy(nameless_l0, nameless_inst)
        self.assertEqual(nameless_proxy.__name__, "cute_entry")

        sent_rec = object()
        sent_ovf = object()
        self.assertIsNot(sent_rec, sent_ovf)
        buf = _FakeCuteBuf(records_cute=sent_rec, overflow_cute=sent_ovf, timer=1)

        # 1) Detached -> calls l0_fn without sidecar kwargs; __getattr__ reads from l0_fn
        self.assertEqual(proxy.tag_attr, "L0_ATTR_VAL")
        proxy("tensor_a", "tensor_b", m_tile=128)
        self.assertEqual(l0_fn.calls, [(("tensor_a", "tensor_b"), {"m_tile": 128})])
        self.assertEqual(inst_fn.calls, [])

        # 2) Attached -> calls inst_fn injecting exact `_fi_rec` and `_fi_ovf`; __getattr__ reads from inst_fn
        self.mod.fi_attach(buf)
        self.assertEqual(proxy.tag_attr, "INST_ATTR_VAL")
        proxy("tensor_a", "tensor_b", m_tile=256)
        self.assertEqual(len(l0_fn.calls), 1)
        self.assertEqual(len(inst_fn.calls), 1)
        pos_args, kw_args = inst_fn.calls[0]
        self.assertEqual(pos_args, ("tensor_a", "tensor_b"))
        self.assertEqual(set(kw_args.keys()), {"m_tile", "_fi_rec", "_fi_ovf"})
        self.assertEqual(kw_args["m_tile"], 256)
        self.assertIs(kw_args["_fi_rec"], sent_rec)
        self.assertIs(kw_args["_fi_ovf"], sent_ovf)

    def test_fi_cute_descriptor_four_binding_states_and_aliases(self) -> None:
        l0_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
        inst_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def raw_l0(self_obj: object, *args: object, **kwargs: object) -> None:
            l0_calls.append((self_obj, args, dict(kwargs)))

        def raw_inst(self_obj: object, *args: object, **kwargs: object) -> None:
            inst_calls.append((self_obj, args, dict(kwargs)))

        descriptor = self.mod._FiCuteDescriptor(raw_l0, raw_inst)
        self.assertEqual(descriptor.__name__, "raw_l0")
        nameless_desc = self.mod._FiCuteDescriptor(_NamelessCallable("L0"), _NamelessCallable("INST"))
        self.assertEqual(nameless_desc.__name__, "cute_method")

        class DemoCuteKernelOp:
            __call__ = descriptor

        op = DemoCuteKernelOp()
        sent_rec = object()
        sent_ovf = object()
        self.assertIsNot(sent_rec, sent_ovf)
        buf = _FakeCuteBuf(records_cute=sent_rec, overflow_cute=sent_ovf, timer=1)

        # Case 1: `instance is None` & detached -> returns raw `l0_fn`
        self.assertIsNone(self.mod.fi_get_attached())
        self.assertIs(DemoCuteKernelOp.__call__, raw_l0)

        # Case 2: `instance` present & detached -> returns bound `l0_fn` method
        bound_l0 = op.__call__
        self.assertIs(getattr(bound_l0, "__self__", None), op)
        self.assertIs(getattr(bound_l0, "__func__", None), raw_l0)
        op("lhs_t", stage=2)
        self.assertEqual(l0_calls, [(op, ("lhs_t",), {"stage": 2})])
        self.assertEqual(inst_calls, [])

        # Case 3: `instance is None` & attached -> returns raw `inst_fn`
        self.mod.fi_attach(buf)
        self.assertIs(DemoCuteKernelOp.__call__, raw_inst)

        # Case 4: `instance` present & attached -> returns `_FiCuteProxy` wrapping bound methods
        bound_proxy = op.__call__
        self.assertIsInstance(bound_proxy, self.mod._FiCuteProxy)
        self.assertIs(getattr(bound_proxy._l0_fn, "__self__", None), op)
        self.assertIs(getattr(bound_proxy._l0_fn, "__func__", None), raw_l0)
        self.assertIs(getattr(bound_proxy._inst_fn, "__self__", None), op)
        self.assertIs(getattr(bound_proxy._inst_fn, "__func__", None), raw_inst)

        op("lhs_t", "rhs_t", stage=4)
        self.assertEqual(len(l0_calls), 1)
        self.assertEqual(len(inst_calls), 1)
        bound_self, pos_args, kw_args = inst_calls[0]
        self.assertIs(bound_self, op)
        self.assertEqual(pos_args, ("lhs_t", "rhs_t"))
        self.assertEqual(set(kw_args.keys()), {"stage", "_fi_rec", "_fi_ovf"})
        self.assertEqual(kw_args["stage"], 4)
        self.assertIs(kw_args["_fi_rec"], sent_rec)
        self.assertIs(kw_args["_fi_ovf"], sent_ovf)

        # Verify all 7 canonical host aliases with `is`
        self.assertIs(self.mod.WpBuffer, self.mod.FiBuffer)
        self.assertIs(self.mod.wp_attach, self.mod.fi_attach)
        self.assertIs(self.mod.wp_get_attached, self.mod.fi_get_attached)
        self.assertIs(self.mod._WpCuteProxy, self.mod._FiCuteProxy)
        self.assertIs(self.mod.wp_wrap, self.mod._FiCuteProxy)
        self.assertIs(self.mod._WpCuteDescriptor, self.mod._FiCuteDescriptor)
        self.assertIs(self.mod.wp_descriptor, self.mod._FiCuteDescriptor)


class WpCuteOpenPerfettoCharacterizationTests(_FreshWpCuteTestCase):
    """Item 4: `open_perfetto` subprocess hand-off, summary paths, error handling, and `.json` directory bug."""

    def setUp(self) -> None:
        super().setUp()
        self._guard_td = tempfile.TemporaryDirectory()
        guard_root = Path(self._guard_td.name)
        (guard_root / "sitecustomize.py").write_text(_SITECUSTOMIZE_GUARD_SOURCE, encoding="utf-8")
        self._guard_marker = guard_root / "subproc_guard_loaded.txt"
        self._subproc_net_log = guard_root / "subproc_net_violations.log"
        prev_pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{guard_root}{os.pathsep}{prev_pp}" if prev_pp else str(guard_root)
        os.environ["WP_TEST_SUBPROC_GUARD_MARKER"] = str(self._guard_marker)
        os.environ["WP_TEST_SUBPROC_NET_LOG"] = str(self._subproc_net_log)

    def tearDown(self) -> None:
        try:
            if self._subproc_net_log.exists():
                self.assertEqual(self._subproc_net_log.read_text(encoding="utf-8"), "")
        finally:
            self._guard_td.cleanup()
            super().tearDown()

    def test_open_perfetto_golden_decode_summary_paths_and_errors(self) -> None:
        raw_dump = lzma.decompress((_GOLDEN_PINGPONG_DIR / "capture.fi02.bin.xz").read_bytes())
        names_path = str(_GOLDEN_PINGPONG_DIR / "names.json")

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            dump_p = td_p / "capture.fi02.bin"
            dump_p.write_bytes(raw_dump)

            # 1) Suspected bug on BASE: without `WEAVEPROF_ALLOW_GZ=1`, `wp_decode.py` rejects
            # `-o <out>.json` / `-o <out>.json.gz` under default PNG-only policy because
            # `open_perfetto` does not pass `--write-gz`.
            os.environ.pop("WEAVEPROF_ALLOW_GZ", None)
            with self.assertRaises(RuntimeError) as png_only_ctx:
                self.mod.open_perfetto(str(dump_p), names_path, str(td_p / "blocked.json"))
            self.assertIn("WeaveProfiler 默认仅交付 PNG", str(png_only_ctx.exception))
            self.assertTrue(self._guard_marker.is_file())

            # 2) With `WEAVEPROF_ALLOW_GZ=1`: decode golden `h20_pingpong_v1` with `.json` and `.json.gz`
            os.environ["WEAVEPROF_ALLOW_GZ"] = "1"
            out_json = td_p / "trace_plain.json"
            summary_json = self.mod.open_perfetto(str(dump_p), names_path, str(out_json))
            expected_sum_json = td_p / "trace_plain.summary.json"
            self.assertTrue(out_json.is_file())
            self.assertTrue(expected_sum_json.is_file())

            out_gz = td_p / "trace_comp.json.gz"
            summary_gz = self.mod.open_perfetto(str(dump_p), names_path, str(out_gz))
            expected_sum_gz = td_p / "trace_comp.summary.json"
            self.assertTrue(out_gz.is_file())
            self.assertTrue(expected_sum_gz.is_file())
            self.assertFalse((td_p / "trace_comp.gz.summary.json").exists())

            # Verify `open_perfetto` actually read the exact summary files above:
            # overwrite `trace_comp.summary.json`? No — compare against disk JSON and check key fields!
            self.assertEqual(summary_json, json.loads(expected_sum_json.read_text(encoding="utf-8")))
            self.assertEqual(summary_gz, json.loads(expected_sum_gz.read_text(encoding="utf-8")))
            self.assertEqual(summary_json, summary_gz)

            # Pin concrete literal fields of the golden summary
            self.assertEqual(summary_json["arch"], "sm_90a")
            self.assertEqual(summary_json["platform"], "nvidia")
            self.assertEqual(summary_json["platform_source"], "meta.arch")
            self.assertEqual(summary_json["timer"], "clock64+锚点")
            self.assertIsNone(summary_json["timer_hz"])
            self.assertEqual(summary_json["timer_min_step_ns"], 63)
            self.assertEqual(summary_json["records"], 1533776)
            self.assertEqual(summary_json["spans"], 864256)
            self.assertEqual(summary_json["marks"], 0)
            self.assertEqual(summary_json["lanes"], 936)
            self.assertEqual(summary_json["lanes_with_sublanes"], 624)
            self.assertEqual(summary_json["max_sublanes"], 9)
            self.assertEqual(summary_json["span_us"], 1066.5)
            self.assertEqual(summary_json["semantic_span_us"], 1065.59)
            self.assertEqual(summary_json["pair"], "stack")
            self.assertEqual(summary_json["errors"], {})
            self.assertEqual(
                summary_json["names_sha256"],
                "c7e93e2a06e78fdf17fb72a6cfea3693be907df178fa4b0d2fd2b442bcfcee2c",
            )
            self.assertEqual(
                summary_json["events"]["load"],
                {"n": 262144, "total_us": 1501563.164, "mean_us": 5.728, "p50_us": 5.512, "max_us": 10.056},
            )
            self.assertEqual(
                summary_json["events"]["mma"],
                {"n": 262144, "total_us": 31513.309, "mean_us": 0.12, "p50_us": 0.109, "max_us": 2.328},
            )
            self.assertEqual(
                summary_json["events"]["wait"],
                {"n": 65536, "total_us": 69929.076, "mean_us": 1.067, "p50_us": 1.047, "max_us": 4.103},
            )

            # 3) Missing `dump_path` -> raises `RuntimeError` containing decoder's `FileNotFoundError`
            missing_dump = td_p / "does_not_exist.fi02.bin"
            with self.assertRaises(RuntimeError) as miss_ctx:
                self.mod.open_perfetto(str(missing_dump), names_path, str(td_p / "err_out.json"))
            self.assertIn("FileNotFoundError", str(miss_ctx.exception))
            self.assertIn("does_not_exist.fi02.bin", str(miss_ctx.exception))

    def test_open_perfetto_directory_containing_dot_json_bug(self) -> None:
        os.environ["WEAVEPROF_ALLOW_GZ"] = "1"
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            mini_dump = td_p / "mini.fi02.bin"
            mini_dump.write_bytes(_build_minimal_filite02_dump())
            mini_names = td_p / "names.json"
            self.mod.write_names(str(mini_names), events={1: "load"}, roles={0: "producer"})

            # Pin BASE `.replace(".json", "")` bug (`wp_cute.py:714` & `fi_lite_decode.py:1362`):
            # When parent directory contains `.json` (e.g. `run.json.dir`), `.replace(".json", "")`
            # strips `.json` out of the directory component too (`run.dir`)!
            # Subcase A: sibling `run.dir` does not exist -> decoder crashes with FileNotFoundError
            parent_a = td_p / "case_a" / "run.json.dir"
            parent_a.mkdir(parents=True)
            with self.assertRaises(RuntimeError) as miss_dir_ctx:
                self.mod.open_perfetto(str(mini_dump), str(mini_names), str(parent_a / "trace.json"))
            self.assertIn("FileNotFoundError", str(miss_dir_ctx.exception))
            self.assertIn("run.dir", str(miss_dir_ctx.exception))

            # Subcase B: sibling `run.dir` DOES exist -> summary is misrouted into `run.dir/trace.summary.json`
            # instead of `run.json.dir/trace.summary.json`, and `open_perfetto` reads that wrong path!
            parent_b_json = td_p / "case_b" / "run.json.dir"
            parent_b_stripped = td_p / "case_b" / "run.dir"
            parent_b_json.mkdir(parents=True)
            parent_b_stripped.mkdir(parents=True)
            got_summary = self.mod.open_perfetto(
                str(mini_dump),
                str(mini_names),
                str(parent_b_json / "trace.json"),
            )
            self.assertTrue((parent_b_json / "trace.json").is_file())
            self.assertFalse((parent_b_json / "trace.summary.json").exists())
            self.assertTrue((parent_b_stripped / "trace.summary.json").is_file())
            self.assertEqual(got_summary["records"], 4)
            self.assertEqual(got_summary["spans"], 1)
            self.assertEqual(got_summary["arch"], "sm_90a")


if __name__ == "__main__":
    unittest.main()

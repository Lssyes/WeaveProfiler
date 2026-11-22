#!/usr/bin/env python3
"""Hermetic tests for the first-class `wp` recorder frontends and the megakernel audit engine.

Run:  python -B -m unittest tests/test_wp_megakernel.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weaveprof import megakernel as mk  # noqa: E402

WPD = mk.load_wp_decode()
KB = {s["semantic_id"]: int(s["event_id"]) for s in WPD._fi.load_kb()["semantics"] if s.get("event_id") is not None}
E_LOAD, E_MMA, E_EPI, E_BAR = KB["load"], KB["mma"], KB["epilogue"], KB["barrier"]
OPEN, CLOSE, END = WPD.OPEN, WPD.CLOSE, WPD.END
CUDA_DIR = ROOT / "recorders" / "cuda_cpp"
HIP_DIR = ROOT / "recorders" / "hip_cpp"
CUTE_DIR = ROOT / "recorders" / "cutedsl"
TRITON_DIR = ROOT / "recorders" / "triton"
N_SMS, N_LONG, N_SHORT, LONG_US, SHORT_US = 78, 50, 28, 374, 191
BASE_NS = 5_000_000

NAMES = {
    "events": {str(E_LOAD): "kv_topk_load", str(E_MMA): "sparse_mla_mma", str(E_EPI): "epilogue",
               str(E_BAR): "grid_sync"},
    "semantics": {str(E_LOAD): "load", str(E_MMA): "mma", str(E_EPI): "epilogue", str(E_BAR): "barrier"},
    "roles": {"0": "persistent_cta"},
}
CLEAN_SOURCE = (
    "// SparseMLA persistent decode megakernel (clean production source)\n"
    "__global__ void sparse_mla_decode(const float* q, const float* kv, float* o) {\n"
    "    for (int tile = blockIdx.x; tile < 1024; tile += gridDim.x) { /* topk gather + mma */ }\n"
    "}\n"
)
INSTRUMENTED_SOURCE = CLEAN_SOURCE.replace("{\n    for", "{\n    WP_OPEN(0);\n    for").replace(
    "\n}\n", "\n    WP_CLOSE();\n}\n").replace("// SparseMLA", '#include "wp_lite.cuh"\n// SparseMLA')


def _sm_of(cta: int) -> int:
    return (cta * 7) % N_SMS  # TopK/head modulo placement: CTA -> physical SM permutation


def sparse_mla_slots(*, stub: bool = False, barrier: bool = False):
    """78 persistent CTAs (gran=1): CTAs 0..49 run 374 us, CTAs 50..77 finish at 191 us."""
    slots = {}
    for cta in range(N_SMS):
        long = cta < N_LONG
        work_end = BASE_NS + (LONG_US if long else SHORT_US) * 1000
        recs = [(BASE_NS, OPEN | 0, _sm_of(cta)),
                (BASE_NS + 1_000, E_LOAD, cta), (BASE_NS + 40_000, END | E_LOAD, cta),
                (BASE_NS + 40_000, E_MMA, cta), (work_end - 2_000, END | E_MMA, cta)]
        if stub:  # empty 64 ns probe: recorder self-latency only
            recs += [(work_end - 1_500, E_EPI, cta), (work_end - 1_436, END | E_EPI, cta)]
        close = work_end
        if barrier and not long:  # early SM waits at grid_sync for the stragglers
            close = BASE_NS + LONG_US * 1000
            recs += [(work_end, E_BAR, 0), (close, END | E_BAR, 0)]
        recs.append((close, CLOSE, 0))
        slots[cta] = recs
    return slots


def write_dump(path: Path, *, magic: bytes = b"WPLITE02", **kw) -> Path:
    blob = WPD.build_dump_bytes(sparse_mla_slots(**kw), n_slots=N_SMS, cap=16, wpc=1, magic=magic,
                                meta={"arch": "sm_90", "gran": 1, "kernel": "sparse_mla_decode"})
    path.write_bytes(blob)
    return path


def runtime_metrics(**over):
    base = [374_000.0, 374_400.0, 373_900.0, 374_200.0, 374_100.0, 373_800.0]
    rm = {
        "warmup": 3,
        "timing_protocol": "ABBA",
        "baseline_ns": base,
        "instrumented_ns": [b * 1.019 for b in base],
        "output_match": True,
        "l0_stats": {"regs": 72, "spill_stores": 16, "spill_loads": 16, "stack_bytes": 16, "stl": 2, "ldl": 2},
        "inst_stats": {"regs": 72, "spill_stores": 16, "spill_loads": 16, "stack_bytes": 16, "stl": 3, "ldl": 3},
        "kernel_name": "sparse_mla",
    }
    rm.update(over)
    return rm


def make_nsys_sqlite(path: Path, kernel_us: float = LONG_US) -> Path:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER,"
        " correlationId INTEGER, demangledName INTEGER, shortName INTEGER);"
        "CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT);"
        "CREATE TABLE GPU_METRICS (typeId INTEGER, timestamp INTEGER, metricId INTEGER, value INTEGER);"
    )
    conn.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "void sparse_mla_decode<128>(...)"), (2, "rmsnorm")])
    t = 1_000_000_000
    launches = [(t, t + 900_000, 1), (t + 2_000_000, t + 2_000_000 + 50_000, 2)]
    for i in range(3):  # warm launches
        s = t + 3_000_000 + i * 1_000_000
        launches.append((s, s + int(kernel_us * 1000) + (i - 1) * 400, 1))
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, ?)",
                     [(a, b, k, n, n) for k, (a, b, n) in enumerate(launches)])
    metrics = [(0, "SMs Active [Throughput %]"), (1, "SM Issue [Throughput %]"),
               (2, "Tensor Active [Throughput %]"), (3, "DRAM Read Bandwidth [Throughput %]"),
               (4, "DRAM Write Bandwidth [Throughput %]"), (5, "PCIe RX Throughput [Throughput %]")]
    conn.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (7, ?, ?)", metrics)
    rows = []
    for ts in range(t, t + 7_000_000, 5_000):  # 200 kHz
        for mid, _ in metrics:
            rows.append((7, ts, mid, 20 + 10 * mid))
    conn.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev
    return mod


_ENV_KEYS = ("WEAVEPROF_SUPERVISOR_HMAC_KEY", "WEAVEPROF_LOCK_DIR", "WEAVEPROF_CACHE", "WEAVEPROF_LOCK_PROVIDER")
_SAVED_ENV = {}
_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.mkdtemp(prefix="wp-mk-env-")
    for k in _ENV_KEYS:
        _SAVED_ENV[k] = os.environ.get(k)
    os.environ["WEAVEPROF_SUPERVISOR_HMAC_KEY"] = "wp-megakernel-test-key"
    os.environ["WEAVEPROF_LOCK_DIR"] = str(Path(_TMP) / "locks")
    os.environ["WEAVEPROF_CACHE"] = str(Path(_TMP) / "cache")
    os.environ.pop("WEAVEPROF_LOCK_PROVIDER", None)


def tearDownModule():
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    shutil.rmtree(_TMP, ignore_errors=True)


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="wp-mk-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


class WpDecodeTest(_TmpCase):
    def test_wplite02_and_filite02_decode_identically(self):
        results = {}
        for magic in (b"WPLITE02", b"FILITE02"):
            p = write_dump(self.tmp / f"{magic.decode()}.bin", magic=magic)
            trace, summary = WPD.decode_file(p, NAMES)
            self.assertEqual(summary["magic"], magic.decode())
            self.assertEqual(summary["errors"], {})
            self.assertEqual(summary["overflow"], 0)
            self.assertEqual(summary["n_ctas"], N_SMS)
            self.assertEqual(summary["events"]["mma"]["n"], N_SMS)
            results[magic] = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertEqual(results[b"WPLITE02"], results[b"FILITE02"])

    def test_bad_magic_rejected_and_selftest(self):
        bad = self.tmp / "bad.bin"
        bad.write_bytes(b"XXLITE02" + bytes(24))
        with self.assertRaises(SystemExit):
            WPD.read_dump(str(bad))
        legacy = _load_module("fi_lite_decode_magic_probe", ROOT / "weaveprof/pipeline/validate" / "tools" / "fi_lite_decode.py")
        good = write_dump(self.tmp / "wp.bin")
        self.assertEqual(legacy.read_dump(str(good))["magic"], "WPLITE02")
        with mock.patch("sys.stdout", new=open(os.devnull, "w", encoding="utf-8")) as sink:
            self.assertTrue(WPD.selftest())
            sink.close()

    def test_canonical_api_surface(self):
        for name in ("read_dump", "decode", "write_json", "main", "selftest"):
            self.assertTrue(callable(getattr(WPD, name)), name)


class HeaderContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wp = (CUDA_DIR / "wp_lite.cuh").read_text(encoding="utf-8")
        cls.hip = (HIP_DIR / "wp_lite_hip.h").read_text(encoding="utf-8")

    def test_wp_build_switches_map_before_defaults(self):
        for wp_name, fi_name in (("WP_LEVEL", "FI_LEVEL"), ("WP_TIMER", "FI_TIMER"), ("WP_GRAN", "FI_GRAN"),
                                 ("WP_GROUPS", "FI_GROUPS")):
            mapping = f"#define {fi_name} {wp_name}"
            self.assertIn(mapping, self.wp)
            self.assertLess(self.wp.index(mapping), self.wp.index(f"#ifndef {fi_name}\n#define {fi_name}"))
        for name in ("WP_SMEM_ZREG", "WP_CTX_1R", "WP_CTX_1R_NOINLINE", "WP_DEFINE_HOST_API"):
            self.assertIn(name, self.wp)
        for wp_name, fi_name in (("WP_LEVEL", "FI_LEVEL"), ("WP_TIMER", "FI_TIMER"), ("WP_GRAN", "FI_GRAN")):
            self.assertIn(f"#define {fi_name} {wp_name}", self.hip)
        self.assertLess(self.hip.index("#define FI_TIMER WP_TIMER"), self.hip.index("#define FI_TIMER 1"))

    def test_full_wp_alias_surface(self):
        macros = ("WP_TAG_END", "WP_TAG_OPEN", "WP_TAG_MARK", "WP_TAG_CLK", "WP_TAG_CLOSE", "WP_TAG_ROLE_SWITCH",
                  "WP_CTX_PARAM", "WP_CTX_ARG", "WP_BG", "WP_EG", "WP_MG", "WP_BG_PRED", "WP_EG_PRED", "WP_MG_PRED",
                  "WP_TRANS_PRED", "WP_OPEN", "WP_CLOSE", "WP_B", "WP_E", "WP_M")
        for m in macros:
            self.assertRegex(self.wp, rf"#define {m}\b", m)
        for fn in ("wp_alloc", "wp_attach", "wp_reset", "wp_dump", "wp_free", "wp_host_init", "wp_host_dump"):
            self.assertRegex(self.wp, rf"static inline \w+ {fn}\(", fn)
            self.assertRegex(self.hip, rf"static inline \w+ {fn}\(", fn)
        for m in ("WP_CTX_PARAM", "WP_CTX_ARG"):
            self.assertRegex(self.hip, rf"#define {m}\b")
        for export in ("wp_init", "wp_reset", "wp_dump", "wp_free"):
            self.assertRegex(self.wp, rf'extern "C" \w+ {export}\(')

    def test_sync_cursor_handles_every_context_mode(self):
        body = self.wp
        self.assertNotRegex(body, r"#define WP_SYNC_CURSOR_(?:OUT|IN)\(\).*fi_ctx_\.p")
        sync_body = body[body.index("wp_sync_cursor_out_zreg"):]
        i_zreg = sync_body.index("wp_sync_cursor_out_zreg(FI_ZREG_SLOT())")
        i_1r = sync_body.index("#elif defined(WP_CTX_1R)")
        i_def = sync_body.index("#else", i_1r)
        self.assertLess(i_zreg, i_1r)
        self.assertIn("wp_sync_cursor_out_1r(fi_ctx_)", sync_body[i_1r:i_def])
        self.assertIn("WP_1R_SLOT_SHIFT", sync_body[i_1r:i_def])
        self.assertIn("wp_sync_cursor_out_ctx(fi_ctx_)", sync_body[i_def:])
        self.assertEqual(len(re.findall(r"#define WP_SYNC_CURSOR_OUT\(\)", body)), 4)
        self.assertEqual(len(re.findall(r"#define WP_SYNC_CURSOR_IN\(\)", body)), 4)
        self.assertEqual(len(re.findall(r"static inline \w+ wp_host_init\(", body)), 1)
        # SMEM_ZREG callees bind the shared context under the kernel's name (FI_ZREG_SLOT resolves in both).
        self.assertIn("#define FI_ZREG_PARAM_DECL FiSmemCtx* fi_sctx_", self.wp)
        self.assertIn("#define FI_ZREG_PARAM_DECL FiSmemCtx& fi_sctx_", self.wp)
        self.assertIn("#define FI_CTX_ARG , fi_sctx_", self.wp)


class FrontendFacadeTest(unittest.TestCase):
    def _env(self, **values):
        keys = ("WP_LEVEL", "WP_GRAN", "WP_TIMER", "FI_LEVEL", "FI_GRAN", "FI_TIMER")
        saved = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update({k: str(v) for k, v in values.items()})

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)

    def test_wp_cute_maps_env_and_degrades_without_cutlass(self):
        self._env(WP_LEVEL=1, WP_GRAN=1)
        with mock.patch.dict(sys.modules, {"cutlass": None}):
            mod = _load_module("wp_cute_under_test", CUTE_DIR / "wp_cute.py")
        self.assertEqual((os.environ["FI_LEVEL"], os.environ["FI_GRAN"]), ("1", "1"))
        self.assertEqual((mod.WP_LEVEL, mod.WP_GRAN), (1, 1))
        self.assertFalse(mod.wp_available())
        with self.assertRaises(ImportError):
            mod.wp_open  # noqa: B018
        with tempfile.TemporaryDirectory() as td:
            p = mod.write_names(os.path.join(td, "names.json"), {1: "load"}, roles={0: "producer"})
            self.assertEqual(json.loads(Path(p).read_text(encoding="utf-8"))["events"], {"1": "load"})
        for name in ("open_perfetto", "write_names"):
            self.assertTrue(callable(getattr(mod, name)))

    def test_wp_cute_exports_device_api_when_cutlass_present(self):
        self._env()
        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.Boolean = bool
        fake_cutlass.Int32 = int
        fake_cutlass.Int64 = int
        fake_cutlass.Constexpr = type("Constexpr", (), {"__class_getitem__": classmethod(lambda cls, _i: cls)})
        fake_cutlass.const_expr = lambda v: v
        fake_cute = types.ModuleType("cutlass.cute")
        fake_cute.Tensor = object
        fake_cute.jit = lambda fn: fn
        fake_cute.arch = types.SimpleNamespace()
        fake_cutlass.cute = fake_cute
        fake_mlir = types.ModuleType("cutlass._mlir")
        fake_dialects = types.ModuleType("cutlass._mlir.dialects")
        fake_nvvm = types.ModuleType("cutlass._mlir.dialects.nvvm")
        fake_dialects.nvvm = fake_nvvm
        fake_mlir.dialects = fake_dialects
        fake_dsl = types.ModuleType("cutlass.cutlass_dsl")
        fake_dsl.T = types.SimpleNamespace(i64=lambda: "i64", i32=lambda: "i32")
        fake_dsl.dsl_user_op = lambda fn: fn
        fake_mods = {
            "cutlass": fake_cutlass, "cutlass.cute": fake_cute,
            "cutlass._mlir": fake_mlir, "cutlass._mlir.dialects": fake_dialects,
            "cutlass._mlir.dialects.nvvm": fake_nvvm, "cutlass.cutlass_dsl": fake_dsl,
        }
        with mock.patch.dict(sys.modules, fake_mods):
            mod = _load_module("wp_cute_fake_under_test", CUTE_DIR / "wp_cute.py")
        self.assertTrue(mod.wp_available())
        for wp_name, fi_name in (("WpBuffer", "FiBuffer"), ("WpCtx", "FiCtx"), ("wp_open", "fi_open"),
                                 ("wp_open_by", "fi_open_by"), ("WP_B", "fi_b"), ("WP_CLOSE", "fi_close"),
                                 ("wp_wrap", "_FiCuteProxy"), ("wp_get_attached", "fi_get_attached")):
            self.assertIs(getattr(mod, wp_name), getattr(mod, fi_name))

    def test_wp_triton_facades(self):
        self._env(WP_LEVEL=1, WP_TIMER=1)
        with mock.patch.dict(sys.modules, {"triton": None, "triton.language": None}):
            nv = _load_module("wp_triton_under_test", TRITON_DIR / "wp_triton.py")
            amd = _load_module("wp_triton_amd_under_test", TRITON_DIR / "wp_triton_amd.py")
        self.assertEqual(os.environ["FI_TIMER"], "1")
        cfg = nv.wp_cfg(cap=64, warps=4, n_slots=8, threads=128)
        self.assertEqual((cfg.level, cfg.timer, cfg.gran), (1, 1, 0))
        with self.assertRaises(ImportError):
            nv.wp_open  # noqa: B018
        with self.assertRaises(NotImplementedError):
            amd.wp_open_by()
        with self.assertRaises(ImportError):
            amd.wp_b_16b  # noqa: B018
        with tempfile.TemporaryDirectory() as td:
            doc = json.loads(Path(amd.write_names(os.path.join(td, "n.json"), {2: "wait"})).read_text("utf-8"))
            self.assertEqual(doc["group"], "cu")


class SkewAnalysisTest(_TmpCase):
    def _decoded(self, **kw):
        return WPD.decode_file(write_dump(self.tmp / "d.bin", **kw), NAMES)

    def test_sparse_mla_78sm_straggler_partition(self):
        trace, summary = self._decoded()
        skew = mk.analyze_megakernel_skew(summary, trace, total_sms=78)
        self.assertEqual(skew["observed_sms"], 78)
        self.assertEqual(skew["split_mode"], "bimodal_gap")
        self.assertEqual(skew["stragglers"]["count"], 50)
        self.assertEqual(skew["stragglers"]["busy_us_median"], 374.0)
        self.assertEqual(skew["early_finishing"]["count"], 28)
        self.assertEqual(skew["early_finishing"]["busy_us_median"], 191.0)
        self.assertEqual(skew["early_finishing"]["sm_ids"], sorted(_sm_of(c) for c in range(N_LONG, N_SMS))[:24])
        dist = skew["sm_busy"]
        self.assertEqual((dist["min_us"], dist["median_us"], dist["max_us"], dist["spread_us"]), (191.0, 374.0, 374.0, 183.0))
        self.assertAlmostEqual(skew["idle_sm_fraction"], round(28 / 78, 4))
        self.assertAlmostEqual(skew["sm_spread_ratio"], round(183 / 374, 4))
        codes = [h["code"] for h in skew["hints"]]
        self.assertIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", codes)
        self.assertIn("50 SMs @ 374us vs 28 SMs @ 191us", skew["hints"][0]["evidence"])
        self.assertEqual({s["stage"] for s in skew["stages"]}, {"load", "mma"})
        self.assertIn("persistent_cta", skew["role_stats"])

    def test_barrier_starved_sms_subtract_grid_sync_wait(self):
        trace, summary = self._decoded(barrier=True)
        skew = mk.analyze_megakernel_skew(summary, trace, total_sms=78)
        self.assertEqual(skew["busy_basis"], "lane_lifetime_minus_wait")
        self.assertEqual((skew["stragglers"]["count"], skew["early_finishing"]["count"]), (50, 28))
        self.assertEqual(skew["early_finishing"]["busy_us_median"], 191.0)
        self.assertEqual(skew["early_finishing"]["barrier_starved"], 28)
        self.assertIn("GRID_SYNC_WAIT_DOMINANT", [h["code"] for h in skew["hints"]])

    def test_balanced_kernel_has_no_split_hint(self):
        slots = {c: [(BASE_NS, OPEN, c), (BASE_NS + 1000, E_MMA, 0), (BASE_NS + 300_000 + c * 10, END | E_MMA, 0),
                     (BASE_NS + 301_000 + c * 10, CLOSE, 0)] for c in range(N_SMS)}
        p = self.tmp / "bal.bin"
        p.write_bytes(WPD.build_dump_bytes(slots, n_slots=N_SMS, cap=8, wpc=1, meta={"arch": "sm_90", "gran": 1}))
        trace, summary = WPD.decode_file(p, NAMES)
        skew = mk.analyze_megakernel_skew(summary, trace, total_sms=78)
        self.assertNotIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", [h["code"] for h in skew["hints"]])
        self.assertLess(skew["sm_spread_ratio"], 0.01)


class NsysSqliteTest(_TmpCase):
    def test_fast_extract_warm_kernel_and_metrics(self):
        db = make_nsys_sqlite(self.tmp / "cap.sqlite")
        hw = mk.extract_nsys_sqlite_fast(db, "sparse_mla")
        k = hw["kernel"]
        self.assertEqual(k["launches"], 4)
        self.assertEqual(k["duration_us"], 374.0)
        self.assertIn("sparse_mla_decode", k["name"])
        self.assertEqual(set(hw["hw_series"]), {"tensor_active", "sm_active", "warp_issue", "dram_read", "dram_write"})
        self.assertEqual(hw["hw_series"]["tensor_active"]["val"][0], 40.0)
        self.assertGreaterEqual(hw["in_window_samples"], 74)
        self.assertEqual(hw["sample_freq_hz"], 200000.0)
        self.assertLess(hw["elapsed_ms"], 150.0)
        with self.assertRaises(ValueError):
            mk.extract_nsys_sqlite_fast(db, "no_such_kernel")

    def test_fast_extract_multi_typeid_restarted_timestamps(self):
        # Mirrors real nsys SQLite where a secondary typeId block (e.g., PCIe/NVLink) is appended with restarted timestamps.
        db = self.tmp / "multi_tid.sqlite"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER,"
            " correlationId INTEGER, demangledName INTEGER, shortName INTEGER);"
            "CREATE TABLE TARGET_INFO_GPU_METRICS (typeId INTEGER, metricId INTEGER, metricName TEXT);"
            "CREATE TABLE GPU_METRICS (typeId INTEGER, timestamp INTEGER, metricId INTEGER, value INTEGER);"
        )
        conn.execute("INSERT INTO StringIds VALUES (1, 'kernel_sparse_mla_decode')")
        # Kernel runs near the end of block 1 (t = 9_500_000..9_874_000 ns)
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (9500000, 9874000, 0, 1, 1, 1)")
        conn.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)", [
            (7, 3, "SMs Active [Throughput %]"),
            (7, 5, "Tensor Active [Throughput %]"),
            (99, 0, "PCIe RX Throughput [Throughput %]"),
        ])
        # Block 1 (>50k rows to exercise O(log N) segment binary search): typeId=7, ts=0..10_000_000
        rows = [(7, ts, mid, 65) for ts in range(0, 10_000_000, 400) for mid in (3, 5)]
        # Block 2 (appended with restarted timestamps 0..10_000_000): typeId=99
        rows.extend((99, ts, 0, 10) for ts in range(0, 10_000_000, 2_000))
        conn.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)", rows)
        conn.commit()
        conn.close()
        hw = mk.extract_nsys_sqlite_fast(db, "sparse_mla")
        self.assertGreaterEqual(hw["in_window_samples"], 900)
        self.assertEqual(set(hw["hw_series"]), {"sm_active", "tensor_active"})
        self.assertLess(hw["elapsed_ms"], 200.0)


class WpSoControllerTest(unittest.TestCase):
    def test_ctypes_plumbing_with_fake_library(self):
        calls = []

        class Fn:
            def __init__(self, name, rc=0):
                self.name, self.rc = name, rc

            def __call__(self, *args):
                calls.append((self.name, args))
                return self.rc

        lib = types.SimpleNamespace(wp_init=Fn("wp_init"), wp_reset=Fn("wp_reset"), wp_dump=Fn("wp_dump", 0),
                                    wp_free=Fn("wp_free", None))
        with tempfile.TemporaryDirectory() as td:
            with mk.WpSoController(None, lib=lib) as ctl:
                ctl.init(78, 1, 64)
                ctl.reset(types.SimpleNamespace(cuda_stream=1234))
                self.assertEqual(ctl.dump(Path(td) / "c.wp02.bin", {"arch": "sm_90"}), 0)
            self.assertEqual([c[0] for c in calls], ["wp_init", "wp_reset", "wp_dump", "wp_free"])
            self.assertEqual(calls[1][1], (1234,))
            lib.wp_dump = Fn("wp_dump", -3)
            ctl = mk.WpSoController(None, lib=lib).init(1, 1, 8)
            with self.assertRaisesRegex(RuntimeError, "fopen"):
                ctl.dump(Path(td) / "x.bin")
        with self.assertRaises(AttributeError):
            mk.WpSoController(None, lib=types.SimpleNamespace(wp_init=Fn("wp_init")))


class MegakernelAuditTest(_TmpCase):
    def _workspace(self):
        ws = Path(tempfile.mkdtemp(prefix="ws-", dir=self.tmp))
        (ws / "sparse_mla.cu").write_bytes(CLEAN_SOURCE.encode("utf-8"))
        return ws

    def _audit(self, name, *, dump_kw=None, rm=None, nsys=False, **kw):
        ws = self._workspace()
        dump = write_dump(self.tmp / f"{name}.wp02.bin", **(dump_kw or {}))
        kw.setdefault("render_overview_png", False)
        return mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
            runtime_metrics=rm if rm is not None else runtime_metrics(),
            nsys_sqlite=make_nsys_sqlite(self.tmp / f"{name}.sqlite") if nsys else None,
            card="h20", limit=0.10, out_dir=self.tmp / f"out-{name}", **kw)

    def test_a_clean_78sm_pass_sealed_fast(self):
        mk.load_wp_decode()
        t0 = time.perf_counter()
        rep = self._audit("pass", nsys=True)
        cpu_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(rep["status"], "PASS", rep)
        self.assertTrue(rep["seal_verified"], rep)
        self.assertLess(rep["wall_clock_ms"], 200.0, rep)
        self.assertLess(cpu_ms, 400.0)
        self.assertLess(len(json.dumps(rep, separators=(",", ":")).encode("utf-8")), 2048)
        self.assertEqual(rep["kernel_us"], 374.0)
        self.assertAlmostEqual(rep["overhead_pct"], 1.9, places=3)
        self.assertEqual(rep["megakernel_skew"]["stragglers"], [50, 374.0])
        self.assertEqual(rep["megakernel_skew"]["early"], [28, 191.0])
        self.assertIn("SPLIT_KV_OR_DYNAMIC_TILE_QUEUE", rep["megakernel_skew"]["hints"])
        self.assertTrue(all(rep["fidelity_gates"][g]["ok"] for g in ("compiler_spill", "output", "stub", "overhead")))
        self.assertEqual((rep["fidelity_gates"]["overflow"], rep["fidelity_gates"]["pairing_errors"]), (0, 0))
        self.assertFalse(any("FAILED" in b for b in rep["structural_blockers"]), rep["structural_blockers"])
        out = self.tmp / "out-pass"
        for f in ("feedback.json", "verify.json", "seal.json"):
            self.assertTrue((out / f).is_file(), f)
        self.assertFalse((out / "trace.perfetto.json.gz").exists())
        fb = json.loads((out / "feedback.json").read_text(encoding="utf-8"))
        self.assertEqual(fb["causal"]["schema"], "weaveprof.causal-report/v1")
        self.assertEqual(fb["causal"]["observed_sms"], 78)
        self.assertGreater(fb["causal"]["hw_summary"]["tensor_active_cal_mean_pct"], 0.0)  # nsys curves merged
        self.assertEqual(fb["nsys"]["kernel"]["duration_us"], 374.0)
        seal = json.loads((out / "seal.json").read_text(encoding="utf-8"))
        self.assertEqual(seal["seal_sha256"], rep["seal_sha256"])
        self.assertEqual(seal["checks"], {"assembly": True, "output": True, "events": True, "overhead": True})
        self.assertIsNotNone(seal["artifacts"]["dump"])
        verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        self.assertEqual(verify["status"], "PASS")
        # Verify default PNG-first delivery (no .gz by default) + opt-in R6 nsys top-pinned trace.gz
        rep_png = self._audit("pass-png", nsys=True, render_overview_png=True, write_perfetto_gz=True)
        self.assertEqual(rep_png["status"], "PASS", rep_png)
        out_png = self.tmp / "out-pass-png"
        import gzip as _gzip
        tr = json.loads(_gzip.decompress((out_png / "trace.perfetto.json.gz").read_bytes()).decode("utf-8"))
        tevs = tr["traceEvents"]
        self.assertEqual(tevs[0]["pid"], 1)
        self.assertEqual(tevs[0]["name"], "process_name")
        self.assertEqual(tevs[1]["pid"], 1)
        self.assertEqual(tevs[1]["name"], "process_sort_index")
        self.assertEqual(tevs[1]["args"]["sort_index"], -1000)
        hw_counters = [e for e in tevs if e.get("ph") == "C"]
        self.assertGreater(len(hw_counters), 50)
        self.assertTrue(all(e["pid"] == 1 and float(e["ts"]) >= 0.0 for e in hw_counters))
        hw_kernels = [e for e in tevs if e.get("cat") == "nsys_kernel"]
        self.assertEqual(len(hw_kernels), 1)
        self.assertEqual((hw_kernels[0]["pid"], hw_kernels[0]["ts"], hw_kernels[0]["dur"]), (1, 0.0, 374.0))
        sw_pids = {e["pid"] for e in tevs if isinstance(e.get("pid"), int) and e["pid"] != 1}
        self.assertTrue(sw_pids and all(p >= 100 for p in sw_pids), sw_pids)
        try:
            import matplotlib  # noqa: F401
            png = out_png / "overview.png"
            self.assertTrue(png.is_file(), rep_png)
            raw_png = png.read_bytes()
            self.assertTrue(raw_png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(raw_png), 20_000)
            self.assertIn("overview.png", rep_png["artifacts"]["files"])
        except ImportError:
            pass
        # Round-to-round diff against the sealed causal report.
        rep2 = self._audit("pass-r2", prev_causal_report=fb)
        self.assertEqual(rep2["status"], "PASS")
        self.assertIsNotNone(rep2["round_diff"])
        self.assertTrue(rep2["causal_opportunities"])

    def test_revise_probe_spec_preserves_dominant_semantic_stage(self):
        from weaveprof.contract import revise_probe_spec
        src = (
            "__global__ void k(float* a, float* b) {\n"
            "  for (int i = 0; i < 64; ++i) {\n"
            "    float v = a[i];\n"
            "    b[i] = v * v;\n"
            "  }\n"
            "}\n"
        )
        doc = {
            "roles": {"w": "all"},
            "probes": [
                {"id": "load", "begin": {"role": "w", "line": 3}, "end": {"role": "w", "line": 3}},
                {"id": "mma", "begin": {"role": "w", "line": 2}, "end": {"role": "w", "line": 5}},
            ],
        }
        fb = {"status": "FAIL", "fires": {"load": 64, "mma": 1}, "enclosing_loop": [2, 5]}
        revised = revise_probe_spec(doc, fb, src)
        self.assertEqual([p["id"] for p in revised["probes"]], ["mma"])
        # Verify weak-LLM tolerance: inline comments & trailing ';' in role expressions + numeric stage suffixes
        from weaveprof.contract import normalize_probe_spec
        norm = normalize_probe_spec(
            {
                "roles": {"dma": "warp_group_idx == 0 /* producer */;"},
                "probes": [
                    {"id": "dma_issue", "begin": {"role": "dma", "line": 3}, "end": {"role": "dma", "line": 3}},
                    {"id": "softmax0", "begin": {"role": "dma", "line": 4}, "end": {"role": "dma", "line": 4}},
                ],
            },
            src,
        )
        self.assertEqual(norm["warp_roles"][0]["predicate"], "warp_group_idx == 0")
        self.assertEqual([p["semantic_id"] for p in norm["probes"]], ["load", "softmax"])
        # Verify weak-LLM half-open touching boundary repair (a.end == b.begin -> trim a.end by 1)
        norm_ho = normalize_probe_spec(
            {
                "roles": {"cta": "all"},
                "probes": [
                    {"id": "load_w", "begin": {"role": "cta", "line": 2}, "end": {"role": "cta", "line": 4}},
                    {"id": "mma_wh", "begin": {"role": "cta", "line": 4}, "end": {"role": "cta", "line": 5}},
                ],
            },
            src,
        )
        self.assertEqual((norm_ho["probes"][0]["begin_line"], norm_ho["probes"][0]["end_line"]), (2, 3))
        self.assertEqual((norm_ho["probes"][1]["begin_line"], norm_ho["probes"][1]["end_line"]), (4, 5))

    def test_single_stage_loop_collapse_fails_gate(self):
        # 99.5% of active time collapsed into 1 single compute slab/lane MUST fail the event gate.
        e_comp = KB["compute"]
        slots = {
            c: [
                (BASE_NS, OPEN | 0, _sm_of(c)),
                (BASE_NS + 1_000, E_LOAD, c), (BASE_NS + 3_000, END | E_LOAD, c),
                (BASE_NS + 3_000, e_comp, c), (BASE_NS + 650_000, END | e_comp, c),
                (BASE_NS + 651_000, CLOSE, 0),
            ]
            for c in range(N_SMS)
        }
        dump = self.tmp / "collapsed.wp02.bin"
        dump.write_bytes(WPD.build_dump_bytes(slots, n_slots=N_SMS, cap=16, wpc=1,
                                              meta={"arch": "sm_90", "gran": 1, "kernel": "collapsed"}))
        names_c = {
            "events": {str(E_LOAD): "load", str(e_comp): "compute"},
            "semantics": {str(E_LOAD): "load", str(e_comp): "compute"},
            "roles": {"0": "persistent_cta"},
        }
        ws = self._workspace()
        rep = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=names_c,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-collapse",
        )
        self.assertEqual(rep["status"], "FAIL", rep)
        self.assertFalse(rep["fidelity_gates"]["granularity"]["ok"])
        self.assertEqual(rep["fidelity_gates"]["granularity"]["code"], "SINGLE_STAGE_LOOP_COLLAPSE")
        self.assertTrue(any("SINGLE_STAGE_LOOP_COLLAPSE" in b for b in rep["structural_blockers"]), rep)

    def test_partial_warp_filtering_fails_gate(self):
        # Recording only 3 representative warps (0, 4, 8) out of wpc=12 MUST fail WARP_COVERAGE_INCOMPLETE.
        wpc = 12
        n_ctas = 4
        slots = {}
        for cta in range(n_ctas):
            for w in (0, 4, 8):
                s = cta * wpc + w
                slots[s] = [
                    (BASE_NS, OPEN | (0 if w == 0 else 1), cta),
                    (BASE_NS + 1_000, E_LOAD, cta), (BASE_NS + 15_000, END | E_LOAD, cta),
                    (BASE_NS + 15_000, E_MMA, cta), (BASE_NS + 60_000, END | E_MMA, cta),
                    (BASE_NS + 61_000, CLOSE, 0),
                ]
        dump = self.tmp / "partial_warps.wp02.bin"
        dump.write_bytes(WPD.build_dump_bytes(slots, n_slots=n_ctas * wpc, cap=16, wpc=wpc,
                                              meta={"arch": "sm_90", "gran": 0, "kernel": "partial_warps"}))
        ws = self._workspace()
        rep = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-partial-warps",
            render_overview_png=False,
        )
        self.assertEqual(rep["status"], "FAIL", rep)
        self.assertFalse(rep["fidelity_gates"]["warps"]["ok"])
        self.assertEqual(rep["fidelity_gates"]["warps"]["code"], "WARP_COVERAGE_INCOMPLETE")
        self.assertTrue(any("WARP_COVERAGE_INCOMPLETE" in b for b in rep["structural_blockers"]), rep)

        # When ALL 12 warps open/close (even if warps 1..3 are idle OPEN+CLOSE-only producer warps), warp gate PASSES.
        for cta in range(n_ctas):
            for w in range(wpc):
                s = cta * wpc + w
                if s not in slots:
                    slots[s] = [(BASE_NS, OPEN | 0, cta), (BASE_NS + 500, CLOSE, 0)]
        dump_full = self.tmp / "all12_warps.wp02.bin"
        dump_full.write_bytes(WPD.build_dump_bytes(slots, n_slots=n_ctas * wpc, cap=16, wpc=wpc,
                                                   meta={"arch": "sm_90", "gran": 0, "kernel": "all12_warps"}))
        rep_ok = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump_full, names=NAMES,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-all12-warps",
            render_overview_png=False,
        )
        self.assertEqual(rep_ok["status"], "PASS", rep_ok)
        self.assertTrue(rep_ok["fidelity_gates"]["warps"]["ok"])

        # Regression: 9-warp persistent megakernel (~1,200 us) with 3 outer macro-blocks (25% / 5% / 70%)
        # and a resident hollow producer warp (w8 open 0..1200 us with only 1 tiny 5 us dot) MUST fail closed.
        wpc9 = 9
        coarse_names = {
            "events": {"4": "csa_attn", "5": "norm_router", "6": "moe_fused", "1": "dummy_init"},
            "semantics": {"4": "attention", "5": "compute", "6": "moe", "1": "load"},
            "roles": {"0": "compute_wg", "1": "producer_dma"},
        }
        slots9 = {}
        end_ns = BASE_NS + 1_200_000
        for cta in range(n_ctas):
            for w in range(8):
                slots9[cta * wpc9 + w] = [
                    (BASE_NS, OPEN | 0, cta),
                    (BASE_NS + 1_000, 4, 0), (BASE_NS + 300_000, END | 4, 0),
                    (BASE_NS + 300_000, 5, 0), (BASE_NS + 360_000, END | 5, 0),
                    (BASE_NS + 360_000, 6, 0), (end_ns - 1_000, END | 6, 0),
                    (end_ns, CLOSE, 0),
                ]
            slots9[cta * wpc9 + 8] = [
                (BASE_NS, OPEN | 1, cta),
                (BASE_NS + 1_000, 1, 0), (BASE_NS + 6_000, END | 1, 0),
                (end_ns, CLOSE, 0),
            ]
        dump_lazy = self.tmp / "lazy_9w.wp02.bin"
        dump_lazy.write_bytes(WPD.build_dump_bytes(slots9, n_slots=n_ctas * wpc9, cap=16, wpc=wpc9,
                                                   meta={"arch": "sm_90", "gran": 0, "kernel": "csa_moe_mk"}))
        rep_lazy = mk.run_megakernel_audit(
            ws, entry_source="sparse_mla.cu", dump_path=dump_lazy, names=coarse_names,
            runtime_metrics=runtime_metrics(), card="h20", limit=0.10, out_dir=self.tmp / "out-lazy-9w",
            render_overview_png=False,
        )
        self.assertEqual(rep_lazy["status"], "FAIL", rep_lazy)
        self.assertFalse(rep_lazy["fidelity_gates"]["granularity"]["ok"])
        self.assertEqual(rep_lazy["fidelity_gates"]["granularity"]["code"], "COARSE_PHASE_ONLY_COLLAPSE")
        self.assertFalse(rep_lazy["fidelity_gates"]["warps"]["ok"])
        self.assertEqual(rep_lazy["fidelity_gates"]["warps"]["code"], "HOLLOW_WARP_LANE")

    def test_b_empty_64ns_stub_probe_fails(self):
        rep = self._audit("stub", dump_kw={"stub": True})
        self.assertEqual(rep["status"], "FAIL", rep)
        self.assertFalse(rep["fidelity_gates"]["stub"]["ok"])
        self.assertIn("epilogue", rep["fidelity_gates"]["stub"]["stubs"])
        self.assertTrue(rep["fidelity_gates"]["compiler_spill"]["ok"])
        seal = json.loads(Path(rep["seal_path"]).read_text(encoding="utf-8"))
        self.assertEqual(seal["status"], "FAIL")
        self.assertFalse(seal["checks"]["events"])

    def test_c_hidden_register_spill_fails(self):
        rm = runtime_metrics(inst_stats={"regs": 72, "spill_stores": 760, "spill_loads": 760, "stack_bytes": 760,
                                         "stl": 134, "ldl": 120})
        rep = self._audit("spill", rm=rm)
        self.assertEqual(rep["status"], "FAIL", rep)
        gate = rep["fidelity_gates"]["compiler_spill"]
        self.assertFalse(gate["ok"])
        self.assertTrue(gate["reg_blind_spot"])
        self.assertIn("SASS_STL_EXPLOSION", gate["codes"])
        self.assertTrue(rep["fidelity_gates"]["stub"]["ok"])

    def test_policy_and_evidence_fail_closed(self):
        ws = self._workspace()
        dump = write_dump(self.tmp / "l.wp02.bin")
        blocked = mk.run_megakernel_audit(ws, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
                                          runtime_metrics=runtime_metrics(), limit=0.25, out_dir=self.tmp / "o1")
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIn("LIMIT_RELAXATION_BLOCKED", blocked["structural_blockers"][0])
        no_out = runtime_metrics()
        no_out.pop("output_match")
        rep = self._audit("noout", rm=no_out)
        self.assertEqual(rep["status"], "FAIL")
        self.assertEqual(rep["fidelity_gates"]["output"].get("code"), "OUTPUT_EVIDENCE_MISSING")
        slow = runtime_metrics(instrumented_ns=[b * 1.25 for b in runtime_metrics()["baseline_ns"]])
        self.assertEqual(self._audit("slow", rm=slow)["status"], "FAIL")
        fast_pert = runtime_metrics(instrumented_ns=[b * 0.55 for b in runtime_metrics()["baseline_ns"]])
        neg = self._audit("neg_pert", rm=fast_pert)
        self.assertEqual(neg["status"], "FAIL")
        self.assertFalse(neg["fidelity_gates"]["overhead"]["ok"])
        self.assertTrue(any("NEGATIVE_OVERHEAD_PERTURBATION" in b for b in neg["structural_blockers"]), neg)
        ws2 = self.tmp / "ws2"
        ws2.mkdir()
        (ws2 / "sparse_mla.cu").write_text(INSTRUMENTED_SOURCE, encoding="utf-8", newline="\n")
        dirty = mk.run_megakernel_audit(ws2, entry_source="sparse_mla.cu", dump_path=dump, names=NAMES,
                                        runtime_metrics=runtime_metrics(), out_dir=self.tmp / "o2")
        self.assertEqual(dirty["status"], "FAIL")
        self.assertTrue(any("PRODUCTION_SOURCE_INSTRUMENTED" in b for b in dirty["structural_blockers"]))

    def test_workflow_a_shadow_build_run_with_l0_cache(self):
        ws = self._workspace()
        fixtures = self.tmp / "fixtures"
        fixtures.mkdir()
        write_dump(fixtures / "capture.wp02.bin")
        (fixtures / "runtime.json").write_text(json.dumps(runtime_metrics()), encoding="utf-8", newline="\n")
        (fixtures / "names.json").write_text(json.dumps(NAMES), encoding="utf-8", newline="\n")
        (ws / "bench.py").write_text(
            "import os, shutil, sys\n"
            "inst = open('sparse_mla.cu', encoding='utf-8').read()\n"
            "l0 = open(os.path.join(os.environ['WEAVEPROF_L0_DIR'], 'sparse_mla.cu'), encoding='utf-8').read()\n"
            "assert 'WP_OPEN' in inst and 'WP_OPEN' not in l0, 'shadow trees not isolated'\n"
            "assert os.path.isfile(os.path.join(os.environ['WEAVEPROF_L0_DIR'], 'built.txt')), 'L0 build missing'\n"
            "for f in ('capture.wp02.bin', 'runtime.json', 'names.json'):\n"
            "    shutil.copy(os.path.join(sys.argv[1], f), os.path.join(os.environ['WEAVEPROF_OUT_DIR'], f))\n",
            encoding="utf-8", newline="\n")
        build = [sys.executable, "-B", "-c", "open('built.txt', 'w').write('ok')"]
        kw = dict(entry_source="sparse_mla.cu", bench_cmd=[sys.executable, "-B", "bench.py", str(fixtures)],
                  build_cmd=build, inst_files={"sparse_mla.cu": INSTRUMENTED_SOURCE}, card="h20")
        rep = mk.run_megakernel_audit(ws, out_dir=self.tmp / "wa1", **kw)
        self.assertEqual(rep["status"], "PASS", rep)
        self.assertEqual(rep["mode"], "shadow_build_run")
        self.assertTrue(rep["seal_verified"])
        self.assertEqual((ws / "sparse_mla.cu").read_text(encoding="utf-8"), CLEAN_SOURCE)
        fb1 = json.loads((self.tmp / "wa1" / "feedback.json").read_text(encoding="utf-8"))
        self.assertFalse(fb1["l0_cache"]["hit"])
        rep2 = mk.run_megakernel_audit(ws, out_dir=self.tmp / "wa2", **kw)
        self.assertEqual(rep2["status"], "PASS", rep2)
        fb2 = json.loads((self.tmp / "wa2" / "feedback.json").read_text(encoding="utf-8"))
        self.assertTrue(fb2["l0_cache"]["hit"])
        seal = json.loads(Path(rep["seal_path"]).read_text(encoding="utf-8"))
        self.assertEqual(seal["region_lock"]["label"], "megakernel_audit")
        self.assertIsNotNone(seal["artifacts"]["instrumented_source"])


class HygieneTest(unittest.TestCase):
    OWNED = (
        "recorders/cuda_cpp/wp_lite.cuh",
        "recorders/hip_cpp/wp_lite_hip.h", "recorders/cutedsl/wp_cute.py",
        "recorders/triton/wp_triton.py", "recorders/triton/wp_triton_amd.py",
        "weaveprof/pipeline/validate/tools/wp_decode.py", "weaveprof/pipeline/validate/tools/fi_lite_decode.py", "weaveprof/megakernel.py",
        "weaveprof/viz.py", "tests/test_wp_megakernel.py",
    )

    def test_lf_only_and_bytecode_guard(self):
        for rel in self.OWNED:
            raw = (ROOT / rel).read_bytes()
            self.assertNotIn(b"\r\n", raw, rel)
            if rel.endswith(".py"):
                self.assertIn(b"sys.dont_write_bytecode = True", raw, rel)


if __name__ == "__main__":
    unittest.main()

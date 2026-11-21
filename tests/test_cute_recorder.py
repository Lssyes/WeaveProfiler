#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior tests for the CuTeDSL recorder without requiring a CUDA toolkit.

The test installs a small CuTeDSL value/tensor runtime and executes the real
``fi_cute.py`` functions.  It therefore checks emitted record contents, writer
selection, bounds handling and FILITE02 serialization rather than grepping the
source text.  The fake runtime is a test double only; these results are offline
behavior tests, not a claim of H20/B300 hardware verification.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "2_instrument" / "recorders" / "cutedsl" / "fi_cute.py"


class Scalar:
    """Integer SSA value with the operators used by the recorder."""

    __slots__ = ("value",)

    def __init__(self, value=0):
        self.value = int(value)

    @staticmethod
    def _int(other):
        return int(other.value) if isinstance(other, Scalar) else int(other)

    def _binary(self, other, op):
        return type(self)(op(self.value, self._int(other)))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return type(self)(self._int(other) - self.value)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __lshift__(self, other):
        return self._binary(other, lambda a, b: a << b)

    def __rlshift__(self, other):
        return type(self)(self._int(other) << self.value)

    def __rshift__(self, other):
        return self._binary(other, lambda a, b: a >> b)

    def __and__(self, other):
        return self._binary(other, lambda a, b: a & b)

    __rand__ = __and__

    def __or__(self, other):
        return self._binary(other, lambda a, b: a | b)

    __ror__ = __or__

    def __xor__(self, other):
        return self._binary(other, lambda a, b: a ^ b)

    def __lt__(self, other):
        return self.value < self._int(other)

    def __le__(self, other):
        return self.value <= self._int(other)

    def __gt__(self, other):
        return self.value > self._int(other)

    def __ge__(self, other):
        return self.value >= self._int(other)

    def __eq__(self, other):
        return self.value == self._int(other)

    def __bool__(self):
        return bool(self.value)

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __repr__(self):
        return f"{type(self).__name__}({self.value})"


class Int32(Scalar):
    pass


class Int64(Scalar):
    pass


class Boolean(Scalar):
    pass


class Constexpr:
    def __class_getitem__(cls, _item):
        return cls


class ConstexprKey:
    """Minimal stand-in for Triton/CuTe constexpr objects used as dict keys."""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, ConstexprKey) and self.value == other.value


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.write_count = 0
        self.data = self._zeros(self.shape)

    @classmethod
    def _zeros(cls, shape):
        if len(shape) == 1:
            return [0 for _ in range(shape[0])]
        return [cls._zeros(shape[1:]) for _ in range(shape[0])]

    @staticmethod
    def _idx(value):
        return int(value.value) if isinstance(value, Scalar) else int(value)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            return self.data[self._idx(index)]
        value = self.data
        for part in index:
            value = value[self._idx(part)]
        return value

    def __setitem__(self, index, value):
        parts = index if isinstance(index, tuple) else (index,)
        parts = tuple(self._idx(part) for part in parts)
        target = self.data
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = int(value.value) if isinstance(value, Scalar) else int(value)
        self.write_count += 1

    @property
    def iterator(self):
        return self


class FakeArch:
    def __init__(self):
        self.reset()

    def reset(self):
        self._block_idx = (0, 0, 0)
        self._grid_dim = (1, 1, 1)
        self._thread_idx = (0, 0, 0)
        self._block_dim = (1, 1, 1)
        self._lane_idx = 0
        self._warp_idx = 0
        self._smid = 11
        self._timer = 1_000
        self.overflow_writes = 0

    def block_idx(self):
        return tuple(Int32(x) for x in self._block_idx)

    def grid_dim(self):
        return tuple(Int32(x) for x in self._grid_dim)

    def thread_idx(self):
        return tuple(Int32(x) for x in self._thread_idx)

    def block_dim(self):
        return tuple(Int32(x) for x in self._block_dim)

    def lane_idx(self):
        return Int32(self._lane_idx)

    def warp_idx(self):
        return Int32(self._warp_idx)

    def smid(self):
        return Int32(self._smid)

    def globaltimer(self):
        value = self._timer
        self._timer += 10
        return value

    def atomic_or(self, tensor, value):
        tensor.data[0] = int(tensor.data[0]) | int(value.value)
        self.overflow_writes += 1


class WrappedDLPackTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)

    def mark_layout_dynamic(self):
        return self


def _install_fake_cutlass(arch):
    cutlass = types.ModuleType("cutlass")
    cutlass.Boolean = Boolean
    cutlass.Int32 = Int32
    cutlass.Int64 = Int64
    cutlass.Constexpr = Constexpr
    cutlass.const_expr = lambda value: value

    cute = types.ModuleType("cutlass.cute")
    cute.Tensor = FakeTensor
    cute.jit = lambda fn: fn
    cute.arch = types.SimpleNamespace(
        block_idx=arch.block_idx,
        grid_dim=arch.grid_dim,
        thread_idx=arch.thread_idx,
        block_dim=arch.block_dim,
        lane_idx=arch.lane_idx,
        warp_idx=arch.warp_idx,
        atomic_or=arch.atomic_or,
    )
    cute.runtime = types.ModuleType("cutlass.cute.runtime")
    cute.runtime.from_dlpack = lambda tensor, assumed_align=None: WrappedDLPackTensor(tensor)
    cutlass.cute = cute

    mlir = types.ModuleType("cutlass._mlir")
    dialects = types.ModuleType("cutlass._mlir.dialects")
    nvvm = types.ModuleType("cutlass._mlir.dialects.nvvm")
    nvvm.read_ptx_sreg_globaltimer = lambda dtype, loc=None, ip=None: arch.globaltimer()
    nvvm.read_ptx_sreg_smid = lambda dtype, loc=None, ip=None: arch.smid()
    dialects.nvvm = nvvm
    mlir.dialects = dialects

    cutlass_dsl = types.ModuleType("cutlass.cutlass_dsl")
    cutlass_dsl.T = types.SimpleNamespace(i64=lambda: "i64", i32=lambda: "i32")
    cutlass_dsl.dsl_user_op = lambda fn: fn

    modules = {
        "cutlass": cutlass,
        "cutlass.cute": cute,
        "cutlass.cute.runtime": cute.runtime,
        "cutlass._mlir": mlir,
        "cutlass._mlir.dialects": dialects,
        "cutlass._mlir.dialects.nvvm": nvvm,
        "cutlass.cutlass_dsl": cutlass_dsl,
    }
    sys.modules.update(modules)


ARCH = FakeArch()
_install_fake_cutlass(ARCH)
_spec = importlib.util.spec_from_file_location("fi_cute_under_test", FRONTEND)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {FRONTEND}")
FC = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = FC
_spec.loader.exec_module(FC)


def _runtime(n_slots=1, cap=8):
    records = FakeTensor((n_slots, cap, 2))
    overflow = FakeTensor((1,))
    return records, overflow


def _read_dump(path):
    blob = Path(path).read_bytes()
    if blob[:8] != b"FILITE02":
        raise AssertionError("bad magic")
    n_slots, cap, wpc, ovf, timer, meta_len = struct.unpack_from("<6I", blob, 8)
    offset = 32
    meta = json.loads(blob[offset:offset + meta_len].decode("utf-8"))
    offset += meta_len
    counts = list(struct.unpack_from(f"<{n_slots}I", blob, offset))
    offset += 4 * n_slots
    records = []
    for count in counts:
        slot = []
        for _ in range(count):
            slot.append(struct.unpack_from("<QII", blob, offset))
            offset += 16
        records.append(slot)
    if offset != len(blob):
        raise AssertionError("dump length mismatch")
    return {
        "n_slots": n_slots,
        "cap": cap,
        "wpc": wpc,
        "overflow": ovf,
        "timer": timer,
        "meta": meta,
        "counts": counts,
        "records": records,
    }


def _load_decoder():
    spec = importlib.util.spec_from_file_location(
        "fi_lite_decode_under_test", ROOT / "4_validate" / "tools" / "fi_lite_decode.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load decoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecorderBehaviorTests(unittest.TestCase):
    def setUp(self):
        ARCH.reset()

    def test_open_and_event_pack_aux_in_high_32_bits(self):
        records, overflow = _runtime(n_slots=2, cap=8)
        ARCH._smid = 0x1234
        ctx = FC.fi_open(
            records,
            overflow,
            FC.Int32(7),
            FC.Int32(2),
            fi_level=2,
            fi_gran=0,
        )
        self.assertTrue(ctx.active)
        self.assertEqual(records[0, 0, 1], (0x1234 << 32) | FC.TAG_OPEN | 7)

        ctx = FC.fi_b(ctx, FC.Int32(3), FC.Int32(-2), fi_level=2)
        expected = (((-2) & 0xFFFFFFFF) << 32) | 3
        self.assertEqual(records[0, 1, 1], expected)
        self.assertEqual(ctx.count.value, 2)
        self.assertEqual(overflow[0], 0)

    def test_gran1_selects_linear_thread_zero_only(self):
        records, overflow = _runtime(n_slots=1, cap=8)
        ARCH._block_dim = (4, 2, 1)
        active = []
        for ty in range(2):
            for tx in range(4):
                ARCH._thread_idx = (tx, ty, 0)
                ctx = FC.fi_open(
                    records,
                    overflow,
                    FC.Int32(1),
                    FC.Int32(1),
                    fi_level=2,
                    fi_gran=1,
                )
                active.append(bool(ctx.active))
        self.assertEqual(active, [True] + [False] * 7)
        # OPEN writes timestamp and packed tag exactly once.
        self.assertEqual(records.write_count, 2)
        self.assertEqual(records[0, 0, 1] & FC.TAG_IDMASK, 1)
        self.assertNotEqual(records[0, 0, 1] & FC.TAG_OPEN, 0)

    def test_gran0_selects_one_lane_per_warp(self):
        records, overflow = _runtime(n_slots=2, cap=8)
        for warp in range(2):
            for lane in range(4):
                ARCH._warp_idx = warp
                ARCH._lane_idx = lane
                FC.fi_open(
                    records,
                    overflow,
                    FC.Int32(warp),
                    FC.Int32(2),
                    fi_level=2,
                    fi_gran=0,
                )
        self.assertEqual(records.write_count, 4)
        self.assertNotEqual(records[0, 0, 1], 0)
        self.assertNotEqual(records[1, 0, 1], 0)

    def test_store_and_open_fail_closed_on_capacity_and_slot(self):
        records, overflow = _runtime(n_slots=1, cap=2)
        full = FC.FiCtx(
            records,
            overflow,
            FC.Int32(0),
            FC.Int32(2),
            2,
            FC.Boolean(True),
        )
        result = FC._fi_store(full, FC.Int32(5), FC.Int32(9))
        self.assertEqual(result.count.value, 3)
        self.assertEqual(overflow[0] & 1, 1)
        self.assertEqual(records.write_count, 0)

        overflow.data[0] = 0
        bad_slot = FC.FiCtx(
            records,
            overflow,
            FC.Int32(1),
            FC.Int32(0),
            2,
            FC.Boolean(True),
        )
        FC._fi_store(bad_slot, FC.Int32(5), FC.Int32(9))
        self.assertEqual(overflow[0] & 2, 2)
        self.assertEqual(records.write_count, 0)

        overflow.data[0] = 0
        ARCH._warp_idx = 1
        ARCH._lane_idx = 0
        ctx = FC.fi_open(
            records,
            overflow,
            FC.Int32(1),
            FC.Int32(2),
            fi_level=2,
            fi_gran=0,
        )
        self.assertFalse(ctx.active)
        self.assertEqual(overflow[0] & 2, 2)
        self.assertEqual(records.write_count, 0)

        zero_cap_records, zero_cap_overflow = _runtime(n_slots=1, cap=1)
        zero_cap_records.shape = (1, 0, 2)
        zero_cap_records.data = [[]]
        ARCH._warp_idx = 0
        ctx = FC.fi_open(
            zero_cap_records,
            zero_cap_overflow,
            FC.Int32(1),
            FC.Int32(1),
            fi_level=2,
            fi_gran=0,
        )
        self.assertFalse(ctx.active)
        self.assertEqual(zero_cap_overflow[0] & 2, 2)
        self.assertEqual(zero_cap_records.write_count, 0)


class DumpBehaviorTests(unittest.TestCase):
    def setUp(self):
        ARCH.reset()

    def _buffer(self, **kwargs):
        params = dict(
            n_ctas=1,
            warps_per_cta=1,
            cap=8,
            gran=0,
            device="cpu",
            arch="sm_103a",
        )
        params.update(kwargs)
        return FC.FiBuffer(**params)

    def test_meta_has_arch_backend_numeric_gran_and_decodes_as_nvidia(self):
        buf = self._buffer()
        buf.records_torch[0, 0, 0] = 100
        buf.records_torch[0, 0, 1] = (0x12345678 << 32) | FC.TAG_OPEN | 3
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "cutedsl.dump"
            buf.dump(str(dump))
            parsed = _read_dump(dump)
            meta = parsed["meta"]
            self.assertIs(type(meta["gran"]), int)
            self.assertEqual(meta["gran"], 0)
            self.assertEqual(meta["gran_name"], "warp")
            self.assertEqual(meta["arch"], "sm_103a")
            self.assertEqual(meta["backend"], "nvidia_cutedsl")
            self.assertEqual(meta["platform"], "nv")
            self.assertEqual(parsed["timer"], 0)
            self.assertEqual(parsed["counts"], [1])
            decoded = _load_decoder().read_dump(str(dump))
            self.assertFalse(decoded["amd"])
            self.assertEqual(decoded["platform_source"], "meta.backend")
            self.assertEqual(decoded["gran"], 0)

    def test_dump_counts_by_low_tag_not_timestamp_or_aux(self):
        buf = self._buffer(cap=8)
        buf.records_torch[0, 0, 0] = 111
        buf.records_torch[0, 0, 1] = FC.TAG_OPEN | 1
        # A stale timestamp/aux with a zero tag must not count as a record.
        buf.records_torch[0, 1, 0] = 222
        buf.records_torch[0, 1, 1] = 0x12345678 << 32
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "tag-count.dump"
            buf.dump(str(dump))
            parsed = _read_dump(dump)
            self.assertEqual(parsed["counts"], [1])
            self.assertEqual(len(parsed["records"][0]), 1)

    def test_dump_rejects_noncontiguous_zero_tag_gap(self):
        buf = self._buffer(cap=8)
        buf.records_torch[0, 0, 1] = FC.TAG_OPEN | 1
        buf.records_torch[0, 2, 1] = FC.TAG_CLOSE
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "tag=0 gap"):
                buf.dump(str(Path(tmp) / "broken.dump"))

    def test_gran_strings_conflicts_and_amd_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "gran must be numeric"):
            self._buffer(gran="warp")
        with self.assertRaisesRegex(ValueError, "gran must be numeric"):
            self._buffer(gran="sm")
        with self.assertRaisesRegex(ValueError, "arch must be"):
            self._buffer(arch="gfx942")
        with self.assertRaisesRegex(ValueError, "backend must be NVIDIA"):
            self._buffer(backend="amd_amdgcn")

        buf = self._buffer()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "conflicts with recorder contract"):
                buf.dump(str(Path(tmp) / "bad-meta.dump"), meta={"gran": "warp"})
            with self.assertRaisesRegex(ValueError, "conflicts with recorder contract"):
                buf.dump(str(Path(tmp) / "bad-arch.dump"), meta={"arch": "gfx942"})

    def test_gran1_records_numeric_sm_geometry(self):
        buf = self._buffer(gran=1, warps_per_cta=8)
        self.assertEqual(buf.warps_per_cta, 1)
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "gran1.dump"
            buf.dump(str(dump))
            meta = _read_dump(dump)["meta"]
            self.assertEqual(meta["gran"], 1)
            self.assertEqual(meta["gran_name"], "sm")


class NamesBehaviorTests(unittest.TestCase):
    def setUp(self):
        ARCH.reset()

    def test_names_do_not_invent_tile_aux_and_accept_constexpr_keys(self):
        buf = FC.FiBuffer(
            n_ctas=1,
            warps_per_cta=1,
            cap=8,
            gran=0,
            device="cpu",
            arch="sm_90",
        )
        doc = buf.names(
            roles={0: "main"},
            events={ConstexprKey(3): "mma"},
        )
        self.assertEqual(doc["events"], {"3": "mma"})
        self.assertEqual(doc["roles"], {"0": "main"})
        self.assertEqual(doc["aux"], {})
        self.assertFalse(doc["label_aux"])

        doc = buf.names(
            roles={0: "main"},
            events={ConstexprKey(3): "mma"},
            aux={ConstexprKey(3): "tile_index"},
        )
        self.assertEqual(doc["aux"], {"3": "tile_index"})
        self.assertTrue(doc["label_aux"])

    def test_write_names_normalizes_keys_and_rejects_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "names.json"
            FC.write_names(
                str(path),
                events={ConstexprKey(4): "softmax", "5": "store"},
                roles={0: "main"},
                aux={ConstexprKey(4): "row"},
            )
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(doc["events"], {"4": "softmax", "5": "store"})
            self.assertEqual(doc["aux"], {"4": "row"})
            self.assertTrue(doc["label_aux"])

        with self.assertRaisesRegex(ValueError, "duplicate normalized key"):
            FC.write_names(
                str(Path(tempfile.gettempdir()) / "never-written.json"),
                events={3: "mma", "3": "gemm"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

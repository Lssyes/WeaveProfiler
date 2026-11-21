# -*- coding: utf-8 -*-
"""fast-instrument CuTeDSL frontend.

This module implements the CUDA macro contract (OPEN/B/E/M/CLOSE) in CUTLASS
CuTeDSL.  Records are kept in a torch int64 tensor with shape
``(n_slots, cap, 2)``; field 0 is ``%globaltimer`` and field 1 packs
``tag`` and ``aux`` as ``(aux << 32) | tag``.  ``FiBuffer.dump`` converts that
layout to the unchanged fast-instrument ``FILITE02`` format.

Environment switches:
  FI_LEVEL  0 disables all writes, 1 records only OPEN/CLOSE, 2 records events
  FI_GRAN   0 records one lane per warp, 1 records only the linear thread 0 per CTA

``FiBuffer`` requires an explicit NVIDIA ``arch`` (``sm_90``, ``sm_100a`` or
``sm_103a`` for the supported cards) and writes both ``arch`` and ``backend`` to
the dump metadata. This prevents a CuTeDSL dump from being classified as AMD
merely because it carries the ``timer_hz=1e9`` globaltimer unit.

The device API is value-style because CuTeDSL compiles Python control flow to
SSA.  A recording operation therefore returns an updated context::

    ctx = fi_open(fi.records_cute, fi.overflow_cute, role, warps_per_cta)
    ctx = fi_b(ctx, EV_WAIT, tile)
    ...
    ctx = fi_close(ctx)
"""
from __future__ import annotations

import json
import os
import operator
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    import numpy as np
except ImportError:  # dump() reports the missing dependency; device code is unaffected
    np = None

FI_LEVEL = int(os.environ.get("FI_LEVEL", "2"))
FI_GRAN = int(os.environ.get("FI_GRAN", "0"))

TAG_END = 0x80000000
TAG_OPEN = 0x40000000
TAG_MARK = 0x20000000
TAG_CLOSE = 0x08000000
TAG_IDMASK = 0x0FFF
AUX_SHIFT = 32
UINT32_MASK = 0xFFFFFFFF
DEFAULT_BACKEND = "nvidia_cutedsl"
NVIDIA_ARCH_RE = re.compile(r"^sm_[0-9]+[a-z]?$")


@dataclass(frozen=True)
class FiCtx:
    """Device-side recorder state.

    ``slot`` and ``count`` are SSA values.  Every recording operation returns a
    new immutable context, matching the value semantics used by CuTeDSL.
    """

    records: cute.Tensor
    overflow: cute.Tensor
    slot: Int32
    count: Int32
    cap: Int32
    active: Boolean


@dsl_user_op
def fi_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read PTX ``%globaltimer`` in nanoseconds."""
    return Int64(nvvm.read_ptx_sreg_globaltimer(T.i64(), loc=loc, ip=ip))


@dsl_user_op
def fi_smid(*, loc=None, ip=None) -> Int32:
    """Read PTX ``%smid``."""
    return Int32(nvvm.read_ptx_sreg_smid(T.i32(), loc=loc, ip=ip))


@cute.jit
def _fi_pack(tag: Int32, aux: Int32) -> Int64:
    """Pack ``(aux << 32) | tag`` without sign extension or multiplication."""
    aux_u32 = Int64(aux) & Int64(UINT32_MASK)
    tag_u32 = Int64(tag) & Int64(UINT32_MASK)
    return (aux_u32 << Int64(AUX_SHIFT)) | tag_u32


@cute.jit
def _fi_store(ctx: FiCtx, tag: Int32, aux: Int32) -> FiCtx:
    next_count = ctx.count
    if ctx.active:
        n_slots = Int32(ctx.records.shape[0])
        if ctx.slot < Int32(0):
            cute.arch.atomic_or(ctx.overflow.iterator, Int32(2))
        else:
            if ctx.slot >= n_slots:
                cute.arch.atomic_or(ctx.overflow.iterator, Int32(2))
            else:
                if ctx.count < Int32(0):
                    cute.arch.atomic_or(ctx.overflow.iterator, Int32(2))
                else:
                    if ctx.count < ctx.cap:
                        ctx.records[ctx.slot, ctx.count, 0] = fi_globaltimer()
                        ctx.records[ctx.slot, ctx.count, 1] = _fi_pack(tag, aux)
                        next_count = ctx.count + Int32(1)
                    else:
                        # Capacity exhaustion is distinct from an invalid slot/count.
                        cute.arch.atomic_or(ctx.overflow.iterator, Int32(1))
                        next_count = Int32(ctx.cap + 1)
    return FiCtx(
        ctx.records,
        ctx.overflow,
        ctx.slot,
        next_count,
        ctx.cap,
        ctx.active,
    )


@cute.jit
def fi_open_by(
    records: cute.Tensor,
    overflow: cute.Tensor,
    pred: Boolean,
    role: Int32,
    warps_per_cta: Int32,
    fi_level: cutlass.Constexpr[int] = FI_LEVEL,
    fi_gran: cutlass.Constexpr[int] = FI_GRAN,
) -> FiCtx:
    """Open a recorder selected by ``pred`` and write an OPEN record."""
    if const_expr(fi_level < 1):
        return FiCtx(
            records,
            overflow,
            Int32(0),
            Int32(0),
            records.shape[1],
            Boolean(False),
        )

    bidx, bidy, bidz = cute.arch.block_idx()
    gidx, gidy, gidz = cute.arch.grid_dim()
    cta = bidx + bidy * gidx + bidz * gidx * gidy
    slot = cta
    if const_expr(fi_gran == 0):
        slot = cta * warps_per_cta + cute.arch.warp_idx()
    else:
        # SM granularity deliberately maps one slot to one CTA.
        slot = cta

    n_slots = Int32(records.shape[0])
    out_slot = Int32(0)
    out_active = Boolean(False)
    if pred:
        if slot >= Int32(0):
            if slot < n_slots:
                if records.shape[1] > 0:
                    out_slot = slot
                    out_active = Boolean(True)
                else:
                    cute.arch.atomic_or(overflow.iterator, Int32(2))
            else:
                cute.arch.atomic_or(overflow.iterator, Int32(2))
        else:
            cute.arch.atomic_or(overflow.iterator, Int32(2))
    opened = FiCtx(
        records,
        overflow,
        out_slot,
        Int32(0),
        records.shape[1],
        out_active,
    )
    return _fi_store(
        opened,
        Int32(TAG_OPEN) | (role & Int32(0xFFFF)),
        fi_smid(),
    )


@cute.jit
def fi_open(
    records: cute.Tensor,
    overflow: cute.Tensor,
    role: Int32,
    warps_per_cta: Int32,
    fi_level: cutlass.Constexpr[int] = FI_LEVEL,
    fi_gran: cutlass.Constexpr[int] = FI_GRAN,
) -> FiCtx:
    """Open a recorder using the default lane for FI_GRAN."""
    if const_expr(fi_gran == 0):
        pred = cute.arch.lane_idx() == 0
    else:
        tidx, tidy, tidz = cute.arch.thread_idx()
        block_x, block_y, _ = cute.arch.block_dim()
        linear_tid = tidx + tidy * block_x + tidz * block_x * block_y
        pred = linear_tid == Int32(0)
    return fi_open_by(
        records,
        overflow,
        pred,
        role,
        warps_per_cta,
        fi_level=fi_level,
        fi_gran=fi_gran,
    )


@cute.jit
def _fi_store_fast(ctx: FiCtx, tag: Int32, aux: Int32) -> FiCtx:
    """Hot-loop recorder path: slot bounds already verified in fi_open_by; overflow atomic hoisted to fi_close."""
    next_count = ctx.count
    if ctx.active:
        if ctx.count < ctx.cap:
            ctx.records[ctx.slot, ctx.count, 0] = fi_globaltimer()
            ctx.records[ctx.slot, ctx.count, 1] = _fi_pack(tag, aux)
            next_count = ctx.count + Int32(1)
        else:
            next_count = Int32(ctx.cap + 1)
    return FiCtx(
        ctx.records,
        ctx.overflow,
        ctx.slot,
        next_count,
        ctx.cap,
        ctx.active,
    )


@cute.jit
def fi_b(
    ctx: FiCtx,
    event: Int32,
    aux: Int32,
    fi_level: cutlass.Constexpr[int] = FI_LEVEL,
) -> FiCtx:
    if const_expr(fi_level < 2):
        return ctx
    return _fi_store_fast(ctx, event & TAG_IDMASK, aux)


@cute.jit
def fi_e(
    ctx: FiCtx,
    event: Int32,
    aux: Int32,
    fi_level: cutlass.Constexpr[int] = FI_LEVEL,
) -> FiCtx:
    if const_expr(fi_level < 2):
        return ctx
    return _fi_store_fast(ctx, (event & TAG_IDMASK) | TAG_END, aux)


@cute.jit
def fi_m(
    ctx: FiCtx,
    event: Int32,
    aux: Int32,
    fi_level: cutlass.Constexpr[int] = FI_LEVEL,
) -> FiCtx:
    if const_expr(fi_level < 2):
        return ctx
    return _fi_store_fast(ctx, (event & TAG_IDMASK) | TAG_MARK, aux)


@cute.jit
def fi_close(ctx: FiCtx, fi_level: cutlass.Constexpr[int] = FI_LEVEL) -> FiCtx:
    if const_expr(fi_level < 1):
        return ctx
    return _fi_store(ctx, TAG_CLOSE, Int32(0))


# CUDA-macro-compatible aliases.
FI_OPEN = fi_open
FI_OPEN_BY = fi_open_by
FI_B = fi_b
FI_E = fi_e
FI_M = fi_m
FI_CLOSE = fi_close


class FiBuffer:
    """Host-side recorder allocation and FILITE02 serializer."""

    def __init__(
        self,
        n_ctas: int,
        warps_per_cta: int,
        cap: int,
        gran: int = FI_GRAN,
        device: str = "cuda",
        *,
        arch: str,
        backend: str = DEFAULT_BACKEND,
    ) -> None:
        gran = _coerce_gran(gran)
        n_ctas = _positive_int("n_ctas", n_ctas)
        warps_per_cta = _positive_int("warps_per_cta", warps_per_cta)
        cap = _positive_int("cap", cap)
        arch = str(arch or "").strip().lower()
        if not NVIDIA_ARCH_RE.fullmatch(arch):
            raise ValueError(
                "CuTeDSL arch must be an NVIDIA sm_* target such as sm_90, "
                f"sm_100a or sm_103a; got {arch!r}"
            )
        backend = str(backend or "").strip().lower()
        if not (backend.startswith("nvidia") or backend == "ptx"):
            raise ValueError(
                "CuTeDSL backend must be NVIDIA (for example "
                f"{DEFAULT_BACKEND!r}); got {backend!r}"
            )
        import torch
        from cutlass.cute.runtime import from_dlpack

        self.gran = gran
        self.arch = arch
        self.backend = backend
        self.n_ctas = n_ctas
        self.warps_per_cta = warps_per_cta if gran == 0 else 1
        self.cap = cap
        self.n_slots = self.n_ctas * self.warps_per_cta
        self.records_torch = torch.zeros(
            (self.n_slots, self.cap, 2), dtype=torch.int64, device=device
        )
        self.overflow_torch = torch.zeros((1,), dtype=torch.int32, device=device)
        self.records_cute = from_dlpack(self.records_torch, assumed_align=16).mark_layout_dynamic()
        self.overflow_cute = from_dlpack(self.overflow_torch, assumed_align=4).mark_layout_dynamic()

    @property
    def cute_args(self):
        return self.records_cute, self.overflow_cute

    def reset(self) -> None:
        self.records_torch.zero_()
        self.overflow_torch.zero_()

    def dump(self, path: str, meta: dict[str, Any] | None = None) -> int:
        """Write the compact FILITE02 dump and return the overflow bitfield."""
        import torch

        if np is None:
            raise RuntimeError("numpy is required to serialize FiBuffer")
        if self.records_torch.is_cuda:
            torch.cuda.synchronize()
        rec = self.records_torch.detach().cpu().numpy()
        ovf = int(self.overflow_torch.detach().cpu().item())

        time = rec[:, :, 0].astype(np.uint64)
        packed = rec[:, :, 1].astype(np.uint64)
        tag = (packed & np.uint64(UINT32_MASK)).astype(np.uint32)
        aux = (packed >> np.uint64(AUX_SHIFT)).astype(np.uint32)
        # A record exists iff its low-32-bit tag is nonzero. Timestamp and aux
        # high bits are payload; they must never extend the slot count.
        nonzero = tag != np.uint32(0)
        # Compute explicit `is_zero` instead of `~nonzero` twice: under CPython 3.14
        # (`LOAD_FAST_BORROW`), NumPy 2.x elision mutates `~nonzero` in-place when refcnt==1.
        is_zero = tag == np.uint32(0)
        counts = np.where(
            np.any(is_zero, axis=1),
            np.argmax(is_zero, axis=1).astype(np.int64),
            np.int64(self.cap),
        )
        for slot in range(self.n_slots):
            n = int(counts[slot])
            if n < self.cap and np.any(nonzero[slot, n + 1:]):
                raise ValueError(
                    f"slot {slot}: tag=0 gap at record {n}, but later records "
                    "carry nonzero tags; refusing to truncate a corrupt dump"
                )

        meta = self._dump_meta(meta)
        meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        header = struct.pack(
            "<6I",
            self.n_slots,
            self.cap,
            self.warps_per_cta,
            ovf,
            0,
            len(meta_bytes),
        )
        with open(path, "wb") as f:
            f.write(b"FILITE02")
            f.write(header)
            f.write(meta_bytes)
            f.write(struct.pack(f"<{self.n_slots}I", *counts.tolist()))
            little_endian_fast = (os.sys.byteorder == "little" and rec.dtype == np.int64 and rec.flags["C_CONTIGUOUS"])
            for slot in range(self.n_slots):
                n = int(counts[slot])
                if n:
                    if little_endian_fast:
                        f.write(rec[slot, :n].tobytes())
                    else:
                        for i in range(n):
                            f.write(struct.pack("<QII", int(time[slot, i]), int(tag[slot, i]), int(aux[slot, i])))
        return ovf

    def _dump_meta(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        meta = dict(extra or {})
        canonical = {
            "fi_version": "fast-instrument-cutedsl-v1",
            "language": "cutedsl",
            "gran": self.gran,
            "gran_name": "warp" if self.gran == 0 else "sm",
            "fi_level": int(FI_LEVEL),
            "timer_hz": 1_000_000_000,
            "timer": 0,
            "arch": self.arch,
            "backend": self.backend,
            "platform": "nv",
        }
        for key, value in canonical.items():
            if key in meta and meta[key] != value:
                raise ValueError(
                    f"dump metadata {key}={meta[key]!r} conflicts with recorder "
                    f"contract {value!r}"
                )
            meta[key] = value
        return meta

    def names(
        self,
        roles: Mapping[Any, str],
        events: Mapping[Any, str],
        aux: Mapping[Any, str] | None = None,
    ) -> dict[str, Any]:
        return _names_document(
            roles,
            events,
            aux=aux,
            group="sm" if self.gran == 1 else "role",
        )

    def write_names(
        self,
        path: str,
        roles: Mapping[Any, str],
        events: Mapping[Any, str],
        aux: Mapping[Any, str] | None = None,
        **extra: Any,
    ) -> str:
        group = str(extra.pop("group", "sm" if self.gran == 1 else "role"))
        return write_names(path, events, roles=roles, aux=aux, group=group, **extra)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return int(result)


def _coerce_gran(value: Any) -> int:
    """Return numeric granularity; reject human-readable strings explicitly."""
    if isinstance(value, str):
        raise ValueError(
            "gran must be numeric 0 (warp) or 1 (CTA); string values such as "
            f"'warp'/'sm' are not accepted: got {value!r}"
        )
    if isinstance(value, bool):
        raise ValueError(f"gran must be 0 or 1, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"gran must be 0 or 1, got {value!r}") from exc
    if result not in (0, 1):
        raise ValueError(f"gran must be 0 or 1, got {result}")
    return int(result)


def _name_key(value: Any, field: str) -> str:
    """Convert ints and constexpr-like objects to stable JSON object keys."""
    seen = 0
    while not isinstance(value, (str, int)) and hasattr(value, "value"):
        inner = value.value
        if inner is value:
            break
        value = inner
        seen += 1
        if seen > 8:
            raise TypeError(f"{field} key has a cyclic .value chain")
    if isinstance(value, bool):
        key = str(int(value))
    elif isinstance(value, str):
        key = value.strip()
    else:
        try:
            key = str(operator.index(value))
        except TypeError as exc:
            raise TypeError(
                f"{field} key must be a string/integer or expose .value; got {value!r}"
            ) from exc
    if not key:
        raise ValueError(f"{field} key must not be empty")
    return key


def _name_mapping(values: Mapping[Any, Any] | None, field: str) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{field} must be a mapping, got {type(values).__name__}")
    result: dict[str, str] = {}
    for key, value in values.items():
        text_key = _name_key(key, field)
        if text_key in result:
            raise ValueError(f"{field} contains duplicate normalized key {text_key!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field}[{text_key!r}] must be a non-empty string")
        result[text_key] = value.strip()
    return result


def _names_document(
    roles: Mapping[Any, str] | None,
    events: Mapping[Any, str],
    *,
    aux: Mapping[Any, str] | None = None,
    group: str = "role",
    pair: str = "stack",
    label_aux: bool | None = None,
) -> dict[str, Any]:
    aux_map = _name_mapping(aux, "aux")
    if label_aux is None:
        label_aux = bool(aux_map)
    return {
        "events": _name_mapping(events, "events"),
        "roles": _name_mapping(roles, "roles"),
        "aux": aux_map,
        "label_aux": bool(label_aux),
        "pair": str(pair),
        "group": str(group),
    }


def write_names(
    path: str,
    events: Mapping[Any, str],
    roles: Mapping[Any, str] | None = None,
    aux: Mapping[Any, str] | None = None,
    group: str = "role",
    pair: str = "stack",
    label_aux: bool | None = None,
    **extra: Any,
) -> str:
    """Write names.json while normalizing int/constexpr keys to JSON strings."""
    obj = _names_document(
        roles,
        events,
        aux=aux,
        group=group,
        pair=pair,
        label_aux=label_aux,
    )
    obj.update(extra)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def open_perfetto(dump_path: str, names_path: str, out_path: str) -> dict[str, Any]:
    """Decode a CuTeDSL dump with the package decoder (weaveprof/pipeline/validate/tools/fi_lite_decode.py)."""
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    decoder = os.path.join(here, "..", "..", "weaveprof", "pipeline", "validate", "tools", "fi_lite_decode.py")
    if not os.path.isfile(decoder):
        raise FileNotFoundError(f"decoder not found: {decoder}")
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-I", decoder, dump_path, "--names", names_path, "-o", out_path],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    summary_path = out_path.replace(".gz", "").replace(".json", "") + ".summary.json"
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


_FI_ACTIVE_BUF: FiBuffer | None = None


def fi_attach(buf: FiBuffer | None = None) -> FiBuffer | None:
    """Attach (or detach when ``None``) a module-level ``FiBuffer`` for zero-signature-edit launches."""
    global _FI_ACTIVE_BUF
    _FI_ACTIVE_BUF = buf
    return buf


def fi_get_attached() -> FiBuffer | None:
    return _FI_ACTIVE_BUF


class _FiCuteProxy:
    """Same-name transparent CuTeDSL entry wrapper (§3.2 2a — 0 Call-Chain Signature Edits)."""

    def __init__(self, l0_fn, inst_fn, buf_getter=fi_get_attached):
        self._l0_fn = l0_fn
        self._inst_fn = inst_fn
        self._buf_getter = buf_getter
        self.__name__ = getattr(l0_fn, "__name__", "cute_entry")

    def __call__(self, *args, **kwargs):
        buf = self._buf_getter()
        if buf is None:
            return self._l0_fn(*args, **kwargs)
        return self._inst_fn(*args, _fi_rec=buf.records_cute, _fi_ovf=buf.overflow_cute, **kwargs)

    def __getattr__(self, name: str):
        buf = self._buf_getter()
        target = self._l0_fn if buf is None else self._inst_fn
        return getattr(target, name)


class _FiCuteDescriptor:
    """Same-name transparent CuTeDSL class-method descriptor wrapper (§3.2 2a & MegaMoE OOP support)."""

    def __init__(self, l0_fn, inst_fn, buf_getter=fi_get_attached):
        self._l0_fn = l0_fn
        self._inst_fn = inst_fn
        self._buf_getter = buf_getter
        self.__name__ = getattr(l0_fn, "__name__", "cute_method")

    def __get__(self, instance, owner=None):
        buf = self._buf_getter()
        if instance is None:
            return self._inst_fn if buf is not None else self._l0_fn
        if buf is None:
            return self._l0_fn.__get__(instance, owner)

        bound_target = self._l0_fn.__get__(instance, owner)
        return _FiCuteProxy(bound_target, self._inst_fn.__get__(instance, owner), self._buf_getter)


# Canonical WeaveProfiler (`wp_*` / `Wp*`) aliases (§3.2)
WpCtx = FiCtx
WpBuffer = FiBuffer
wp_open = fi_open
wp_b = fi_b
wp_e = fi_e
wp_m = fi_m
wp_close = fi_close
wp_attach = fi_attach
wp_get_attached = fi_get_attached
_WpCuteProxy = _FiCuteProxy
wp_wrap = _WpCuteProxy
WP_B = wp_b
WP_E = wp_e
_WpCuteDescriptor = _FiCuteDescriptor


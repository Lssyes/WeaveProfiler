# -*- coding: utf-8 -*-
"""WeaveProfiler (`wp`) canonical CuTeDSL recorder frontend.

Self-contained CUTLASS CuTeDSL recorder implementing the OPEN/B/E/M/CLOSE
contract and serializing records into the standard FILITE02/WPLITE02 dump
format decoded by ``weaveprof/pipeline/validate/tools/wp_decode.py``.

Environment switches:
  WP_LEVEL -> FI_LEVEL   0 disables writes, 1 records OPEN/CLOSE only, 2 records events
  WP_GRAN  -> FI_GRAN    0 one lane per warp, 1 one lane per CTA (linear thread 0)

Device API (value-style, CuTeDSL SSA)::

    ctx = wp_open(buf.records_cute, buf.overflow_cute, role, warps_per_cta)
    ctx = wp_b(ctx, EV_LOAD, tile)
    ctx = wp_e(ctx, EV_LOAD, tile)
    ctx = wp_close(ctx)

Pure-host helpers (``write_names``, ``open_perfetto``, ``WP_LEVEL``/``WP_GRAN``)
work without CUTLASS installed; device symbols raise a clear ImportError on first
use when ``cutlass`` is missing.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import operator
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:  # dump() reports the missing dependency; device code is unaffected
    np = None


def _map_env(canonical: str, legacy: str) -> None:
    value = os.environ.get(canonical)
    if value is None or value == "":
        return
    prior = os.environ.get(legacy)
    if prior not in (None, "") and prior != value:
        raise RuntimeError(f"{canonical}={value!r} conflicts with {legacy}={prior!r}; set only one spelling")
    os.environ[legacy] = value


for _canonical, _legacy in (("WP_LEVEL", "FI_LEVEL"), ("WP_GRAN", "FI_GRAN"), ("WP_TIMER", "FI_TIMER")):
    _map_env(_canonical, _legacy)

_TIMER_ENV_EXPLICIT = ("FI_TIMER" in os.environ and os.environ.get("FI_TIMER", "") != "")
WP_LEVEL = int(os.environ.get("FI_LEVEL", "2"))
WP_GRAN = int(os.environ.get("FI_GRAN", "0"))
WP_TIMER = int(os.environ.get("FI_TIMER", "0"))
if WP_LEVEL not in (0, 1, 2):
    raise ValueError(f"WP_LEVEL must be 0, 1 or 2, got {WP_LEVEL}")
if WP_GRAN not in (0, 1):
    raise ValueError(f"WP_GRAN must be 0 or 1, got {WP_GRAN}")
if WP_TIMER not in (0, 1):
    raise ValueError(f"WP_TIMER must be 0 or 1, got {WP_TIMER}")
FI_LEVEL = WP_LEVEL
FI_GRAN = WP_GRAN
FI_TIMER = WP_TIMER

TAG_END = 0x80000000
TAG_OPEN = 0x40000000
TAG_MARK = 0x20000000
TAG_CLK = 0x10000000
TAG_CLOSE = 0x08000000
TAG_IDMASK = 0x0FFF
AUX_SHIFT = 32
UINT32_MASK = 0xFFFFFFFF
DEFAULT_BACKEND = "nvidia_cutedsl"
NVIDIA_ARCH_RE = re.compile(r"^sm_[0-9]+[a-z]?$")
WP_TAG_END, WP_TAG_OPEN, WP_TAG_MARK, WP_TAG_CLK, WP_TAG_CLOSE, WP_TAG_IDMASK = (
    TAG_END, TAG_OPEN, TAG_MARK, TAG_CLK, TAG_CLOSE, TAG_IDMASK
)

_DEVICE_SYMBOLS = {
    "FiCtx", "WpCtx",
    "fi_globaltimer", "fi_clock64", "fi_smid", "_fi_pack", "_fi_store", "_fi_store_fast",
    "fi_open_by", "fi_open", "fi_b", "fi_e", "fi_m", "fi_close",
    "FI_OPEN", "FI_OPEN_BY", "FI_B", "FI_E", "FI_M", "FI_CLOSE",
    "wp_open", "wp_open_by", "wp_b", "wp_e", "wp_m", "wp_close",
    "WP_OPEN", "WP_OPEN_BY", "WP_B", "WP_E", "WP_M", "WP_CLOSE",
}

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Boolean, Int32, Int64, const_expr
    from cutlass._mlir.dialects import nvvm
    from cutlass.cutlass_dsl import T, dsl_user_op
    _CUTLASS_AVAILABLE = True
    _IMPORT_ERROR: BaseException | None = None
except ImportError as _exc:
    _CUTLASS_AVAILABLE = False
    _IMPORT_ERROR = _exc

if _CUTLASS_AVAILABLE:
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
    def fi_clock64(*, loc=None, ip=None) -> Int64:
        """Read PTX ``%clock64`` cycle counter (high resolution on sm_86 where %globaltimer steps at 1024 ns)."""
        fn = getattr(nvvm, "read_ptx_sreg_clock64", None)
        if fn is None:
            return Int64(nvvm.read_ptx_sreg_globaltimer(T.i64(), loc=loc, ip=ip))
        return Int64(fn(T.i64(), loc=loc, ip=ip))


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
    def _fi_store(ctx: FiCtx, tag: Int32, aux: Int32, use_clock64: cutlass.Constexpr[int] = 0) -> FiCtx:
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
                            if const_expr(use_clock64 == 1):
                                ctx.records[ctx.slot, ctx.count, 0] = fi_clock64()
                            else:
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
        fi_timer: cutlass.Constexpr[int] = -1,
    ) -> FiCtx:
        """Open a recorder selected by ``pred`` and write an OPEN record (plus CLK anchor when fi_timer==1)."""
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
        ctx_open = _fi_store(
            opened,
            Int32(TAG_OPEN) | (role & Int32(0xFFFF)),
            fi_smid(),
            use_clock64=0,
        )
        if const_expr((FI_TIMER if fi_timer < 0 else fi_timer) == 1):
            ctx_open = _fi_store(ctx_open, Int32(TAG_CLK), Int32(0), use_clock64=1)
        return ctx_open


    @cute.jit
    def fi_open(
        records: cute.Tensor,
        overflow: cute.Tensor,
        role: Int32,
        warps_per_cta: Int32,
        fi_level: cutlass.Constexpr[int] = FI_LEVEL,
        fi_gran: cutlass.Constexpr[int] = FI_GRAN,
        fi_timer: cutlass.Constexpr[int] = -1,
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
            fi_timer=fi_timer,
        )


    @cute.jit
    def _fi_store_fast(ctx: FiCtx, tag: Int32, aux: Int32, use_clock64: cutlass.Constexpr[int] = 0) -> FiCtx:
        """Hot-loop recorder path: slot bounds already verified in fi_open_by; overflow atomic hoisted to fi_close."""
        next_count = ctx.count
        if ctx.active:
            if ctx.count < ctx.cap:
                if const_expr(use_clock64 == 1):
                    ctx.records[ctx.slot, ctx.count, 0] = fi_clock64()
                else:
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
        fi_timer: cutlass.Constexpr[int] = -1,
    ) -> FiCtx:
        if const_expr(fi_level < 2):
            return ctx
        return _fi_store_fast(
            ctx, event & TAG_IDMASK, aux,
            use_clock64=const_expr(1 if (FI_TIMER if fi_timer < 0 else fi_timer) == 1 else 0),
        )


    @cute.jit
    def fi_e(
        ctx: FiCtx,
        event: Int32,
        aux: Int32,
        fi_level: cutlass.Constexpr[int] = FI_LEVEL,
        fi_timer: cutlass.Constexpr[int] = -1,
    ) -> FiCtx:
        if const_expr(fi_level < 2):
            return ctx
        return _fi_store_fast(
            ctx, (event & TAG_IDMASK) | TAG_END, aux,
            use_clock64=const_expr(1 if (FI_TIMER if fi_timer < 0 else fi_timer) == 1 else 0),
        )


    @cute.jit
    def fi_m(
        ctx: FiCtx,
        event: Int32,
        aux: Int32,
        fi_level: cutlass.Constexpr[int] = FI_LEVEL,
        fi_timer: cutlass.Constexpr[int] = -1,
    ) -> FiCtx:
        if const_expr(fi_level < 2):
            return ctx
        return _fi_store_fast(
            ctx, (event & TAG_IDMASK) | TAG_MARK, aux,
            use_clock64=const_expr(1 if (FI_TIMER if fi_timer < 0 else fi_timer) == 1 else 0),
        )


    @cute.jit
    def fi_close(
        ctx: FiCtx,
        fi_level: cutlass.Constexpr[int] = FI_LEVEL,
        fi_timer: cutlass.Constexpr[int] = -1,
    ) -> FiCtx:
        if const_expr(fi_level < 1):
            return ctx
        cur = _fi_store(ctx, TAG_CLOSE, Int32(0), use_clock64=0)
        if const_expr((FI_TIMER if fi_timer < 0 else fi_timer) == 1):
            cur = _fi_store(cur, Int32(TAG_CLK), Int32(1), use_clock64=1)
        return cur


    # CUDA-macro-compatible aliases.
    FI_OPEN = fi_open
    FI_OPEN_BY = fi_open_by
    FI_B = fi_b
    FI_E = fi_e
    FI_M = fi_m
    FI_CLOSE = fi_close

    WpCtx = FiCtx
    wp_open = fi_open
    wp_open_by = fi_open_by
    wp_b = fi_b
    wp_e = fi_e
    wp_m = fi_m
    wp_close = fi_close
    WP_OPEN = fi_open
    WP_OPEN_BY = fi_open_by
    WP_B = fi_b
    WP_E = fi_e
    WP_M = fi_m
    WP_CLOSE = fi_close
else:
    def __getattr__(name: str) -> Any:  # PEP 562 lazy failure for device symbols
        if name in _DEVICE_SYMBOLS:
            raise ImportError(
                f"wp_cute.{name} needs the CUTLASS CuTeDSL runtime (pip install nvidia-cutlass-dsl); "
                f"import of cutlass failed: {_IMPORT_ERROR}"
            )
        raise AttributeError(f"module 'wp_cute' has no attribute {name!r}")


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
        timer: int | None = None,
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
        if timer is None:
            resolved_timer = 1 if (arch == "sm_86" and not _TIMER_ENV_EXPLICIT) else FI_TIMER
        else:
            if isinstance(timer, bool) or int(timer) not in (0, 1):
                raise ValueError(f"timer must be 0 or 1, got {timer!r}")
            resolved_timer = int(timer)
        backend = str(backend or "").strip().lower()
        if not (backend.startswith("nvidia") or backend == "ptx"):
            raise ValueError(
                "CuTeDSL backend must be NVIDIA (for example "
                f"{DEFAULT_BACKEND!r}); got {backend!r}"
            )
        import torch
        from cutlass.cute.runtime import from_dlpack

        self.gran = gran
        self.timer = resolved_timer
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
            int(getattr(self, "timer", 0)),
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
            "timer": int(getattr(self, "timer", 0)),
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
    """Decode a CuTeDSL dump with the package decoder (weaveprof/pipeline/validate/tools/wp_decode.py)."""
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    decoder = os.path.join(here, "..", "..", "weaveprof", "pipeline", "validate", "tools", "wp_decode.py")
    if not os.path.isfile(decoder):
        raise FileNotFoundError(f"decoder not found: {decoder}")
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-B", decoder, dump_path, "--names", names_path, "-o", out_path],
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
_ORIG_CUTE_COMPILE = None


def _install_cute_compile_hook() -> None:
    """Transparently route cute.compile() through _FiCuteProxy / _FiCuteDescriptor.

    Without this hook, calling ``cute.compile(obj, ...)`` on an instrumented OOP
    CuTeDSL launcher either drops instrumentation (resolving raw class __call__)
    or—if a user manually compiles ``_inst_fn`` with ``_fi_rec/_fi_ovf``—causes
    ``cudaErrorIllegalAddress`` at call time when the compiled C ABI stub is
    invoked without forwarding ``_fi_rec`` and ``_fi_ovf``.
    """
    global _ORIG_CUTE_COMPILE
    if not _CUTLASS_AVAILABLE:
        return
    cur = getattr(cute, "compile", None)
    if cur is None or getattr(cur, "_wp_hooked", False):
        return
    _ORIG_CUTE_COMPILE = cur

    def _wp_cute_compile(func, *args, **kwargs):
        buf = fi_get_attached()
        # Case 1: standalone _FiCuteProxy function
        if isinstance(func, _FiCuteProxy):
            if buf is None:
                return _ORIG_CUTE_COMPILE(func._l0_fn, *args, **kwargs)
            compiled = _ORIG_CUTE_COMPILE(
                func._inst_fn, *args,
                _fi_rec=buf.records_cute, _fi_ovf=buf.overflow_cute,
                **kwargs,
            )
            def _wrapped_proxy(*cargs, **ckwargs):
                active = fi_get_attached() or buf
                return compiled(*cargs, _fi_rec=active.records_cute, _fi_ovf=active.overflow_cute, **ckwargs)
            _wrapped_proxy._wp_compiled_raw = compiled
            return _wrapped_proxy
        # Case 2: class instance whose __call__ is a _FiCuteDescriptor (e.g. FlashAttentionForwardAmpere, PersistentDenseGemmKernel)
        cls_call = type(func).__dict__.get("__call__")
        if isinstance(cls_call, _FiCuteDescriptor):
            if buf is None:
                return _ORIG_CUTE_COMPILE(cls_call._l0_fn, func, *args, **kwargs)
            compiled = _ORIG_CUTE_COMPILE(
                cls_call._inst_fn, func, *args,
                _fi_rec=buf.records_cute, _fi_ovf=buf.overflow_cute,
                **kwargs,
            )
            def _wrapped_obj(*cargs, **ckwargs):
                active = fi_get_attached() or buf
                return compiled(*cargs, _fi_rec=active.records_cute, _fi_ovf=active.overflow_cute, **ckwargs)
            _wrapped_obj._wp_compiled_raw = compiled
            return _wrapped_obj
        return _ORIG_CUTE_COMPILE(func, *args, **kwargs)

    _wp_cute_compile._wp_hooked = True
    cute.compile = _wp_cute_compile


_install_cute_compile_hook()


def fi_attach(buf: FiBuffer | None = None) -> FiBuffer | None:
    """Attach (or detach when ``None``) a module-level ``FiBuffer`` for zero-signature-edit launches."""
    global _FI_ACTIVE_BUF, FI_TIMER, WP_TIMER
    _FI_ACTIVE_BUF = buf
    if buf is not None:
        FI_TIMER = WP_TIMER = int(getattr(buf, "timer", 0))
    _install_cute_compile_hook()
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


# Canonical WeaveProfiler (`wp_*` / `Wp*`) host aliases
WpBuffer = FiBuffer
wp_attach = fi_attach
wp_get_attached = fi_get_attached
_WpCuteProxy = _FiCuteProxy
wp_wrap = _WpCuteProxy
_WpCuteDescriptor = _FiCuteDescriptor
wp_descriptor = _WpCuteDescriptor


def wp_available() -> bool:
    """True when the CuTeDSL device frontend (cutlass) imported successfully."""
    return _CUTLASS_AVAILABLE


__all__ = sorted(
    (_DEVICE_SYMBOLS if _CUTLASS_AVAILABLE else set())
    | {
        "WP_LEVEL", "WP_GRAN", "WP_TIMER", "FI_LEVEL", "FI_GRAN", "FI_TIMER",
        "TAG_END", "TAG_OPEN", "TAG_MARK", "TAG_CLK", "TAG_CLOSE", "TAG_IDMASK",
        "WP_TAG_END", "WP_TAG_OPEN", "WP_TAG_MARK", "WP_TAG_CLK", "WP_TAG_CLOSE", "WP_TAG_IDMASK",
        "FiBuffer", "WpBuffer", "fi_attach", "wp_attach", "fi_get_attached", "wp_get_attached",
        "_FiCuteProxy", "_WpCuteProxy", "wp_wrap", "_FiCuteDescriptor", "_WpCuteDescriptor", "wp_descriptor",
        "write_names", "open_perfetto", "wp_available",
    }
)

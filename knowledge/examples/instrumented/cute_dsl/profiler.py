"""Fixed-slot CuTe DSL device profiler helpers.

The buffer stores int64 pairs ``(timestamp, tag)``.  The caller supplies a
unique offset for each (CTA, track) interval; no atomic allocation is used.
The timestamp is the real 64-bit PTX ``%globaltimer`` value.
"""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T
from cutlass.cute.arch.nvvm_wrappers import nvvm


SUPPORTED_ARCHES = frozenset({"sm_90", "sm_100a", "sm_103a"})


def _require_arch() -> str:
    """Read the explicit host build target; never guess an architecture."""
    raw = os.environ.get("ARCH")
    if raw is None:
        raise ValueError(
            "ARCH must be set explicitly to one of: sm_90, sm_100a, sm_103a")
    arch = raw.strip()
    if arch not in SUPPORTED_ARCHES:
        raise ValueError(
            f"ARCH={arch!r} is unsupported; expected one of: "
            "sm_90, sm_100a, sm_103a")
    return arch


ARCH = _require_arch()


@cute.jit
def record(profile: cute.Tensor, header_elems: Int32, cta: Int32,
           track: Int32, num_tracks: Int32, max_records: Int32,
           offset: Int32, task_id: cutlass.Constexpr,
           is_start: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx - ((tidx >> Int32(5)) << Int32(5))
    if lane == Int32(0):
        if offset < max_records:
            timestamp = Int64(nvvm.read_ptx_sreg_globaltimer(T.i64()))
            slot = header_elems + (
                cta * num_tracks * max_records + track * max_records + offset
            ) * Int32(2)
            # ABI tag fields: block[31:20], track[19:17], task[8:4], is_start[3].
            tag = (cta << Int32(20)) | (track << Int32(17)) \
                | (Int32(task_id) << Int32(4)) \
                | (Int32(is_start) << Int32(3))
            profile[slot] = timestamp
            profile[slot + Int32(1)] = Int64(tag)


@cute.jit
def interval_start(profile: cute.Tensor, header_elems: Int32, cta: Int32,
                   track: Int32, num_tracks: Int32, max_records: Int32,
                   offset: Int32, task_id: cutlass.Constexpr):
    record(profile, header_elems, cta, track, num_tracks, max_records, offset,
           task_id, True)


@cute.jit
def interval_end(profile: cute.Tensor, header_elems: Int32, cta: Int32,
                 track: Int32, num_tracks: Int32, max_records: Int32,
                 offset: Int32, task_id: cutlass.Constexpr):
    record(profile, header_elems, cta, track, num_tracks, max_records, offset,
           task_id, False)

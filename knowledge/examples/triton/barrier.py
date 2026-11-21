"""Minimal Triton reference for a profiled CTA-local debug barrier."""

import triton
import triton.language as tl

from profiler import interval_end, interval_start

# SEMANTIC-ID: barrier
# EXECUTION-SCOPE: cta; tl.debug_barrier is not a grid barrier.
# ABI: ignored synchronization evidence (not a task interval).
# The diagnostic task id 15 remains within 0..31 for the fixed-slot record;
# consumers must ignore this event according to the ABI.


@triton.jit
def barrier_kernel(src, dst, profile, block_size: tl.constexpr,
                   max_records: tl.constexpr, header_elems: tl.constexpr,
                   num_tracks: tl.constexpr, track: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    values = tl.load(src + offsets)
    cta = tl.program_id(0)
    # TRACE-BEGIN
    interval_start(profile, header_elems, cta, track, num_tracks, max_records,
                   0, 15)
    tl.debug_barrier()
    interval_end(profile, header_elems, cta, track, num_tracks, max_records,
                 1, 15)
    # TRACE-END
    tl.store(dst + offsets, values)

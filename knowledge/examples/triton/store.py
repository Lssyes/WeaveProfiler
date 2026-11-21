"""Minimal Triton reference for a profiled masked global-memory store."""

import triton
import triton.language as tl

from profiler import interval_end, interval_start

# SEMANTIC-ID: store
# EXECUTION-SCOPE: cta; physical warp ownership is compiler-selected.
# ABI task id 9 is in the 5-bit task field (0..31).


@triton.jit
def store_kernel(src, dst, profile, n_elements: tl.constexpr,
                 block_size: tl.constexpr, max_records: tl.constexpr,
                 header_elems: tl.constexpr, num_tracks: tl.constexpr,
                 track: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    values = tl.load(src + offsets, mask=mask, other=0.0)
    cta = tl.program_id(0)
    # TRACE-BEGIN
    interval_start(profile, header_elems, cta, track, num_tracks, max_records,
                   0, 9)
    tl.store(dst + offsets, values, mask=mask)
    interval_end(profile, header_elems, cta, track, num_tracks, max_records,
                 1, 9)
    # TRACE-END

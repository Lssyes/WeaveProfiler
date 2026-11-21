"""Minimal Triton reference for a masked global-memory load."""

import triton
import triton.language as tl

from profiler import interval_end, interval_start

# SEMANTIC-ID: load
# EXECUTION-SCOPE: cta; physical warp ownership is compiler-selected.


@triton.jit
def load_kernel(src, dst, profile, n_elements: tl.constexpr,
                block_size: tl.constexpr, max_records: tl.constexpr,
                header_elems: tl.constexpr, num_tracks: tl.constexpr,
                track: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    cta = tl.program_id(0)
    offset = 0
    # TRACE-BEGIN
    interval_start(profile, header_elems, cta, track, num_tracks, max_records,
                   offset, 1)
    values = tl.load(src + offsets, mask=mask, other=0.0)
    interval_end(profile, header_elems, cta, track, num_tracks, max_records,
                 offset + 1, 1)
    # TRACE-END
    tl.store(dst + offsets, values, mask=mask)

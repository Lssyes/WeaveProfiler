"""Minimal Triton reference for a profiled row sum reduction."""

import triton
import triton.language as tl

from profiler import interval_end, interval_start

# SEMANTIC-ID: reduction
# EXECUTION-SCOPE: cta; lowering may use multiple physical warps.
# ABI task id 12 is in the 5-bit task field (0..31).


@triton.jit
def reduction_kernel(src, dst, profile, width: tl.constexpr,
                     max_records: tl.constexpr, header_elems: tl.constexpr,
                     num_tracks: tl.constexpr, track: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, width)
    values = tl.load(src + row * width + columns)
    # TRACE-BEGIN
    interval_start(profile, header_elems, row, track, num_tracks, max_records,
                   0, 12)
    row_sum = tl.sum(values, axis=0)
    interval_end(profile, header_elems, row, track, num_tracks, max_records,
                 1, 12)
    # TRACE-END
    tl.store(dst + row, row_sum)

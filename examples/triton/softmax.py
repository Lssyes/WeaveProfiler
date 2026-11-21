"""Minimal Triton reference for a profiled numerically stable row softmax."""

import triton
import triton.language as tl

from profiler import interval_end, interval_start

# SEMANTIC-ID: softmax
# EXECUTION-SCOPE: cta; physical warp ownership is compiler-selected.
# ABI task id 10 is in the 5-bit task field (0..31).


@triton.jit
def softmax_kernel(src, dst, profile, width: tl.constexpr,
                   max_records: tl.constexpr, header_elems: tl.constexpr,
                   num_tracks: tl.constexpr, track: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, width)
    values = tl.load(src + row * width + columns)
    # TRACE-BEGIN
    interval_start(profile, header_elems, row, track, num_tracks, max_records,
                   0, 10)
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    probabilities = numerator / tl.sum(numerator, axis=0)
    interval_end(profile, header_elems, row, track, num_tracks, max_records,
                 1, 10)
    # TRACE-END
    tl.store(dst + row * width + columns, probabilities)

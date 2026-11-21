"""Minimal Triton reference for a tensor-core matrix multiply."""

import triton
import triton.language as tl

from profiler import interval_end, interval_start

# SEMANTIC-ID: mma
# EXECUTION-SCOPE: cta; do not infer physical warp owners from tl.dot alone.


@triton.jit
def mma_kernel(a, b, c, profile, m: tl.constexpr, n: tl.constexpr,
               k: tl.constexpr, max_records: tl.constexpr,
               header_elems: tl.constexpr, num_tracks: tl.constexpr,
               track: tl.constexpr):
    rows = tl.arange(0, m)
    cols = tl.arange(0, n)
    inner = tl.arange(0, k)
    a_tile = tl.load(a + rows[:, None] * k + inner[None, :])
    b_tile = tl.load(b + inner[:, None] * n + cols[None, :])
    cta = tl.program_id(0)
    offset = 0
    # TRACE-BEGIN
    interval_start(profile, header_elems, cta, track, num_tracks, max_records,
                   offset, 2)
    accumulator = tl.dot(a_tile, b_tile)
    interval_end(profile, header_elems, cta, track, num_tracks, max_records,
                 offset + 1, 2)
    # TRACE-END
    tl.store(c + rows[:, None] * n + cols[None, :], accumulator)


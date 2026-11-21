"""Minimal Triton reference for profiled program-to-work-item assignment."""

import triton
import triton.language as tl

from profiler import interval_end, interval_start

# SEMANTIC-ID: dispatch
# EXECUTION-SCOPE: cta/program; this is a real phase, not a physical warp label.
# ABI task id 0 is in the 5-bit task field (0..31).


@triton.jit
def assigned_work_item(output, profile, max_records: tl.constexpr,
                       header_elems: tl.constexpr, num_tracks: tl.constexpr,
                       track: tl.constexpr):
    cta = tl.program_id(0)
    # TRACE-BEGIN
    interval_start(profile, header_elems, cta, track, num_tracks, max_records,
                   0, 0)
    work_item = tl.program_id(0)
    grid_size = tl.num_programs(0)
    interval_end(profile, header_elems, cta, track, num_tracks, max_records,
                 1, 0)
    # TRACE-END
    tl.store(output + work_item, grid_size)

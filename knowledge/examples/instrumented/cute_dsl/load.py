"""Minimal CuTe DSL reference for a tiled gmem-to-smem load."""

import cutlass
from cutlass import cute

from profiler import interval_end, interval_start

# SEMANTIC-ID: load
# EXECUTION-SCOPE: determined by the supplied tiled-copy atom.


@cute.jit
def load(tiled_copy, gmem_src, smem_dst, barrier_ptr, profile,
         max_records, header_elems, num_tracks, track, task_id: cutlass.Constexpr):
    cta, _, _ = cute.arch.block_idx()
    # TRACE-BEGIN
    interval_start(profile, header_elems, cta, track,
                   num_tracks, max_records, 0, task_id)
    cute.copy(tiled_copy, gmem_src, smem_dst, tma_bar_ptr=barrier_ptr)
    interval_end(profile, header_elems, cta, track,
                 num_tracks, max_records, 1, task_id)
    # TRACE-END

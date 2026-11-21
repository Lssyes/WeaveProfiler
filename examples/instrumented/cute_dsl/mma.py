"""Minimal CuTe DSL reference for an MMA issue operation."""

import cutlass
from cutlass import cute

from profiler import interval_end, interval_start

# SEMANTIC-ID: mma
# EXECUTION-SCOPE: determined by the MMA atom; tcgen05 cute.gemm is asynchronous.


@cute.jit
def mma(mma_atom, accumulator, tensor_a, tensor_b, profile, max_records,
        header_elems, num_tracks, track, task_id: cutlass.Constexpr):
    cta, _, _ = cute.arch.block_idx()
    # TRACE-BEGIN
    interval_start(profile, header_elems, cta, track,
                   num_tracks, max_records, 0, task_id)
    cute.gemm(mma_atom, accumulator, tensor_a, tensor_b, accumulator)
    interval_end(profile, header_elems, cta, track,
                 num_tracks, max_records, 1, task_id)
    # TRACE-END

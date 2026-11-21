"""Minimal CuTe DSL reference for a tiled gmem-to-smem load.

SEMANTIC-ID: load
EXECUTION-SCOPE: determined by the supplied tiled-copy atom.
ABI: layer=phase.  A load is an operator *internal*, never an operator.

A TMA load lives inside an enclosing operator task (``compress_gemm``,
``indexer_gemm``, ``attention``, ...) that ``megakernel.py`` opens around the
whole ``grid_sync`` segment.  Recording loads as tasks and the segment itself as
nothing is the mistake that leaves the timeline empty outside one operator.
"""

from cutlass import Int32, cute

import profiler as prof


@cute.jit
def load(tiled_copy, gmem_src, smem_dst, barrier_ptr, ctx,
         work_id: Int32 = Int32(0)):
    # TRACE-BEGIN
    prof.phase_begin(ctx, prof.EV_LOAD, work_id)
    cute.copy(tiled_copy, gmem_src, smem_dst, tma_bar_ptr=barrier_ptr)
    prof.phase_end(ctx, prof.EV_LOAD, work_id)
    # TRACE-END

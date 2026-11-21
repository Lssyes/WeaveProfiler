"""Minimal CuTe DSL reference for a tiled register-to-global store.

SEMANTIC-ID: store
EXECUTION-SCOPE: determined by the supplied tiled-copy atom.
ABI: layer=phase.  The store is the write-out stage of the operator that produced
     the values -- the epilogue of ``wo_b_gemm``, the inverse-RoPE store of
     ``attention``.  A store becomes layer=task only when it is a whole
     ``grid_sync`` segment on its own, e.g. ``csa_append`` (CSA Cache Append);
     ``scatter.py`` shows both sides of that fork.
"""

from cutlass import Int32, cute

import profiler as prof


@cute.jit
def store(tiled_copy, register_src, gmem_dst, ctx,
          work_id: Int32 = Int32(0)):
    # TRACE-BEGIN
    prof.phase_begin(ctx, prof.EV_STORE, work_id)
    cute.copy(tiled_copy, register_src, gmem_dst)
    prof.phase_end(ctx, prof.EV_STORE, work_id)
    # TRACE-END

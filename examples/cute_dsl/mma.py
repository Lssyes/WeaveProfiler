"""Minimal CuTe DSL reference for an MMA issue operation.

SEMANTIC-ID: mma
EXECUTION-SCOPE: determined by the MMA atom; tcgen05 ``cute.gemm`` is async.
ABI: layer=phase.  The WGMMA issue is an internal of a GEMM operator.

The visible operator is the whole ``grid_sync`` segment -- ``qkv_a_gemm``,
``wq_b_gemm``, ``compress_gemm``, ``indexer_gemm``, ``logits``, ``wo_a_gemm``,
``wo_b_gemm`` -- and ``megakernel.py`` opens one task span around it.  This
record only splits that bar into its internal stages.

``prof.rec`` has no barrier and no warp-wide vote, so calling it inside the
divergent consumer region (``if wg >= 1``) cannot change the original control
flow.
"""

from cutlass import Int32, cute

import profiler as prof


@cute.jit
def mma(mma_atom, accumulator, tensor_a, tensor_b, ctx,
        work_id: Int32 = Int32(0)):
    # TRACE-BEGIN
    prof.phase_begin(ctx, prof.EV_MMA, work_id)
    cute.gemm(mma_atom, accumulator, tensor_a, tensor_b, accumulator)
    prof.phase_end(ctx, prof.EV_MMA, work_id)
    # TRACE-END

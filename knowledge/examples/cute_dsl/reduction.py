"""Minimal CuTe DSL reference for a register-fragment sum reduction.

SEMANTIC-ID: reduction
EXECUTION-SCOPE: the local fragment owner; a cross-warp reduction needs an
                 explicit primitive.
ABI: layer=phase when the reduction is a step of a larger operator (masked-key
     ballot and online-softmax denominator inside ``attention``); layer=task only
     when the reduction is itself a whole ``grid_sync`` segment (``csa_pool``,
     ``topk``).  ``scatter.py`` shows the same fork in full.
"""

from cutlass import Float32, Int32, cute

import profiler as prof


@cute.jit
def reduction(fragment, ctx, work_id: Int32 = Int32(0)):
    values = fragment.load().to(Float32)
    # TRACE-BEGIN
    prof.phase_begin(ctx, prof.EV_EPILOGUE, work_id)
    result = values.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
    prof.phase_end(ctx, prof.EV_EPILOGUE, work_id)
    # TRACE-END
    return result

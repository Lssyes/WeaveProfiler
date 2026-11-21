"""Minimal CuTe DSL reference for stable softmax on one register fragment.

SEMANTIC-ID: softmax
EXECUTION-SCOPE: the fragment owner; add explicit warp reductions when rows span
                 lanes.
ABI: layer=phase.  The online-softmax rescale is an internal of the enclosing
     attention operator (``EV_ATTENTION``), recorded once per KV chunk right
     after the QK^T WGMMA -- see ``attention_task.py``.  It is never its own
     operator bar: the operator is the whole ``grid_sync`` segment.
"""

from cutlass import Float32, Int32, cute

import profiler as prof


@cute.jit
def softmax(fragment, ctx, work_id: Int32 = Int32(0)):
    values = fragment.load().to(Float32)
    # TRACE-BEGIN
    prof.phase_begin(ctx, prof.EV_EPILOGUE, work_id)
    row_max = values.reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
    numerator = cute.math.exp2(values - row_max, fastmath=True)
    denominator = numerator.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
    probabilities = numerator * cute.arch.rcp_approx(denominator)
    prof.phase_end(ctx, prof.EV_EPILOGUE, work_id)
    # TRACE-END
    fragment.store(probabilities)

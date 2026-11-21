"""Minimal CuTe DSL reference for stable softmax on one register fragment."""

from cutlass import Float32, cute

# SEMANTIC-ID: softmax
# EXECUTION-SCOPE: the fragment owner; add explicit warp reductions when rows span lanes.


@cute.jit
def softmax(fragment):
    values = fragment.load().to(Float32)
    # TRACE-BEGIN
    row_max = values.reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
    numerator = cute.math.exp2(values - row_max, fastmath=True)
    denominator = numerator.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
    probabilities = numerator * cute.arch.rcp_approx(denominator)
    # TRACE-END
    fragment.store(probabilities)

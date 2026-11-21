"""Minimal CuTe DSL reference for a register-fragment sum reduction."""

from cutlass import Float32, cute

# SEMANTIC-ID: reduction
# EXECUTION-SCOPE: the local fragment owner; cross-warp reduction needs an explicit primitive.


@cute.jit
def reduction(fragment):
    values = fragment.load().to(Float32)
    # TRACE-BEGIN
    result = values.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
    # TRACE-END
    return result

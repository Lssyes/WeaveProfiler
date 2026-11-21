"""Minimal Triton reference for a row sum reduction."""

import triton
import triton.language as tl

# SEMANTIC-ID: reduction
# EXECUTION-SCOPE: cta; lowering may use multiple physical warps.


@triton.jit
def reduction_kernel(src, dst, width: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, width)
    values = tl.load(src + row * width + columns)
    # TRACE-BEGIN
    row_sum = tl.sum(values, axis=0)
    # TRACE-END
    tl.store(dst + row, row_sum)

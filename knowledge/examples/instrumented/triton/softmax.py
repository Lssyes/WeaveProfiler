"""Minimal Triton reference for a numerically stable row softmax."""

import triton
import triton.language as tl

# SEMANTIC-ID: softmax
# EXECUTION-SCOPE: cta; physical warp ownership is compiler-selected.


@triton.jit
def softmax_kernel(src, dst, width: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, width)
    values = tl.load(src + row * width + columns)
    # TRACE-BEGIN
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    probabilities = numerator / tl.sum(numerator, axis=0)
    # TRACE-END
    tl.store(dst + row * width + columns, probabilities)

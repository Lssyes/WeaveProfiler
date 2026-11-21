"""Minimal Triton reference for a masked global-memory store."""

import triton
import triton.language as tl

# SEMANTIC-ID: store
# EXECUTION-SCOPE: cta; physical warp ownership is compiler-selected.


@triton.jit
def store_kernel(src, dst, n_elements: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    values = tl.load(src + offsets, mask=mask, other=0.0)
    # TRACE-BEGIN
    tl.store(dst + offsets, values, mask=mask)
    # TRACE-END

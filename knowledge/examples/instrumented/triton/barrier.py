"""Minimal Triton reference for a CTA-local thread barrier."""

import triton
import triton.language as tl

# SEMANTIC-ID: barrier
# EXECUTION-SCOPE: cta; tl.debug_barrier is not a grid barrier.


@triton.jit
def barrier_kernel(src, dst, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    values = tl.load(src + offsets)
    # TRACE-BEGIN
    tl.debug_barrier()
    # TRACE-END
    tl.store(dst + offsets, values)

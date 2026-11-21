import torch
import triton
import triton.language as tl


@triton.jit
def recurrent_fma(X, Y, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = x
    for _ in range(STEPS):
        y = tl.fma(y, 0.9995, x)
    tl.store(Y + offsets, y)


def run_recurrence(x):
    out = torch.empty_like(x)
    recurrent_fma[(x.numel() // 128,)](x, out, 8192, 128, num_warps=4)
    return out

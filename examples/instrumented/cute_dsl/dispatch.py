"""Minimal CuTe DSL reference for deterministic work-item assignment only."""

from cutlass import cute

# SEMANTIC-ID: dispatch
# EXECUTION-SCOPE: cta.


@cute.jit
def assigned_work_item():
    block, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    # TRACE-BEGIN
    work_item = block
    # TRACE-END
    return work_item, grid

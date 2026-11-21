"""Minimal Triton reference for program-to-work-item assignment only."""

import triton
import triton.language as tl

# SEMANTIC-ID: dispatch
# EXECUTION-SCOPE: cta/program.


@triton.jit
def assigned_work_item(output):
    # TRACE-BEGIN
    work_item = tl.program_id(0)
    grid_size = tl.num_programs(0)
    # TRACE-END
    tl.store(output + work_item, grid_size)

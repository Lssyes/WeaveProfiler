"""Minimal CuTe DSL reference for a CTA-local thread barrier."""

from cutlass import cute

# SEMANTIC-ID: barrier
# EXECUTION-SCOPE: all threads in one cta.


@cute.jit
def barrier():
    # TRACE-BEGIN
    cute.arch.sync_threads()
    # TRACE-END

"""Minimal CuTe DSL reference for a tiled register-to-global store."""

from cutlass import cute

# SEMANTIC-ID: store
# EXECUTION-SCOPE: determined by the supplied tiled-copy atom.


@cute.jit
def store(tiled_copy, register_src, gmem_dst):
    # TRACE-BEGIN
    cute.copy(tiled_copy, register_src, gmem_dst)
    # TRACE-END

"""Exercise the Listing 1 context manager on an unchanged Triton launch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import weaveprof as wp
from weaveprof.workspace import cache_root

from triton_kernel import run_recurrence


def main():
    source = Path(__file__).with_name("triton_kernel.py").resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    # Pin a reviewed placement by digest so reruns require no model access.
    cache = cache_root(source) / digest
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "probe.json").write_text(json.dumps({
        "roles": {"math": "all"},
        "probes": [{"id": "compute", "begin": {"role": "math", "line": 11},
                    "end": {"role": "math", "line": 12}}]
    }))
    torch.manual_seed(49103)
    x = torch.randn(1024, device="cuda")
    reference = run_recurrence(x)
    with wp.profile(source=str(source), limit=0.10, out="triton.perfetto-trace"):
        observed = run_recurrence(x)
    assert torch.equal(observed.view(torch.int32), reference.view(torch.int32))
    # Reference each FP32 FMA on the host with one rounding after the addition.
    x64 = x.cpu().numpy().astype(np.float64)
    y = x64.astype(np.float32)
    scale = np.float64(np.float32(0.9995))
    for _ in range(8192):
        y = (scale * y.astype(np.float64) + x64).astype(np.float32)
    np.testing.assert_array_equal(observed.cpu().numpy().view(np.uint32), y.view(np.uint32))
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"Verified all {observed.numel()} output elements, source SHA-256 {digest}")


if __name__ == "__main__":
    main()

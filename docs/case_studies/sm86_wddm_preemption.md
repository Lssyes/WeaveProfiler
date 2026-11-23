# Case Study: Catching Windows WDDM Display Preemption on an 8192³ Triton GEMM (RTX 3050 `sm_86`)

> **Takeaway**: When a `92 ms` kernel runs on a Windows desktop GPU (WDDM driver), the OS Desktop Window Manager (DWM) preempts the GPU for `~1 ms` every ~20 ms. Neither `nsys` hardware curves alone nor software timers alone tell the full story—WeaveProfiler's locked microsecond SW+HW alignment exposes the exact microsecond of preemption across both hardware utilization drops and stretched per-SM software spans.

![RTX 3050 sm_86 8192³ Triton GEMM WDDM Preemption SW+HW Forensic Alignment](../images/case_sm86_wddm_preemption.png)

## What the Aligned View Reveals

1. **Panel (A) — 4 Vertical Preemption Cliffs (`Active SMs % → 0%`) & Initial DVFS Ramp**:
   - At `t ≈ 5.5 ms`, `30.2 ms`, `32.5 ms`, `55.0 ms`, and `72.0 ms`, **all four hardware curves (`Active SMs %`, `Tensor Active %`, `HBM BW %`, `SM Issue %`) drop simultaneously to `0%` for `~1 ms`**.
   - During `0–30 ms`, `HBM BW %` ramps from `30%` to `100%` while `Tensor Active %` holds steady near `48%`, reflecting dynamic laptop power/clock (DVFS) stabilization on a `384 MB` working set (`A/B/C` FP16) far exceeding the `2 MB` L2 cache.

2. **Panel (B) — Exact Microsecond Correspondence with Stretched Blue & Red Blocks**:
   - Looking vertically down from Panel (A)'s `0%` cliffs into Panel (B)'s 16-SM waterfall:
     - **Big Blue Blocks (`t ≈ 5.5 ms` on SM0/3/8/11; `t ≈ 30.2 ms` on SM3/8/12–15)**: Those specific SMs happened to be inside `WP_B(K0 MMA) … WP_E(K0 MMA)` at the exact microsecond WDDM suspended the GPU context, stretching a `~1 µs` stage into a `~1,000 µs` (`1 ms`) blue block.
     - **Big Red-Orange Block (`t ≈ 55.0 ms` on SM8)**: SM8 happened to be inside `WP_B(FP16 Store) … WP_E(FP16 Store)` during the 4th preemption, stretching a `~1 µs` store into a `~1,000 µs` red block.
     - The remaining 12–15 SMs were inside the uninstrumented K-loop body at the moment of preemption, so their pause stretched white background rather than colored spans.

3. **Three Practical Lessons Encoded Back into WeaveProfiler**:
   - **Never peel `k=0` outside a Triton K-loop**: Leaving 255/256 K-loop iterations uninstrumented produces sparse CTAs (`2 events/CTA`). On NVIDIA Triton, always enable `triton_ptx_async_hook` for full K-loop `cp.async` + `mma`/`wgmma` visibility.
   - **Panel (D) minimum-event gate**: `weaveprof/viz.py` now requires `>= 6` per-warp events inside a CTA before zooming into a single CTA, falling back to a multi-CTA wave window (`>= 12` events) on outer-only traces.
   - **Multi-millisecond / desktop GPU timing**: Always scope `nsys profile` with `--capture-range=cudaProfilerApi` (avoiding 50M+ metric rows) and prefer `WP_TIMER=0` (`%globaltimer`) on `> 10 ms` runs where DVFS ramps or WDDM preemption pauses cause `%clock64` linear interpolation drift (`1.582 ms` observed here).

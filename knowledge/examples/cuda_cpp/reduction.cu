// Minimal CUDA C++ reference for a warp sum reduction.
// SEMANTIC-ID: reduction
// EXECUTION-SCOPE: one physical warp.

#include "profiler.cuh"

extern "C" __global__ void reduction_kernel(
    const float* src, float* dst, std::int64_t* profile,
    std::uint32_t max_records, std::uint32_t header_elems,
    std::uint32_t num_tracks) {
  if (threadIdx.x >= 32) return;

  const unsigned lane = threadIdx.x & 31u;
  float value = src[blockIdx.x * 32u + lane];
  const std::uint32_t track = threadIdx.x >> 5;
  const std::uint32_t offset = 0;
  // TRACE-BEGIN
  if (lane == 0) {
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset, 4u);
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  if (lane == 0) {
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset + 1, 4u);
  }
  // TRACE-END
  if (lane == 0) dst[blockIdx.x] = value;
}


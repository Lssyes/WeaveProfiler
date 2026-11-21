// Minimal CUDA C++ reference for a unique-index global-memory scatter.
// SEMANTIC-ID: dispatch
// EXECUTION-SCOPE: physical thread; indices supplied by the caller are unique.

#include "profiler.cuh"

extern "C" __global__ void scatter_kernel(
    const float* src, const int* indices, float* dst, int count,
    std::int64_t* profile, std::uint32_t max_records,
    std::uint32_t header_elems, std::uint32_t num_tracks) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    const std::uint32_t track = threadIdx.x >> 5;
    const std::uint32_t offset = 0;
    // TRACE-BEGIN
    if ((threadIdx.x & 31) == 0) {
      fast_instrument_ref::interval_start(
          profile, header_elems, blockIdx.x, track, num_tracks, max_records,
          offset, 7u);
    }
    // Unique indices make this a race-free scatter without a business atomic.
    dst[indices[index]] = src[index];
    if ((threadIdx.x & 31) == 0) {
      fast_instrument_ref::interval_end(
          profile, header_elems, blockIdx.x, track, num_tracks, max_records,
          offset + 1, 7u);
    }
    // TRACE-END
  }
}


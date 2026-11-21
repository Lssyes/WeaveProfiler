// Minimal CUDA C++ reference for a CTA-local barrier.
// SEMANTIC-ID: barrier
// EXECUTION-SCOPE: all threads in one CTA; launch with blockDim.x == 256.

#include "profiler.cuh"

extern "C" __global__ void barrier_kernel(
    const float* src, float* dst, std::int64_t* profile,
    std::uint32_t max_records, std::uint32_t header_elems,
    std::uint32_t num_tracks) {
  __shared__ float tile[256];
  tile[threadIdx.x] = src[blockIdx.x * blockDim.x + threadIdx.x];
  const std::uint32_t track = threadIdx.x >> 5;
  const std::uint32_t offset = 0;
  // TRACE-BEGIN
  // ABI note: this synchronization span must be marked "ignored" in the ABI;
  // task id 6 is only a reference-local tag for the fixed-slot trace.
  if ((threadIdx.x & 31) == 0) {
    fast_instrument_ref::interval_start(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset, 6u);
  }
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    fast_instrument_ref::interval_end(
        profile, header_elems, blockIdx.x, track, num_tracks, max_records,
        offset + 1, 6u);
  }
  // TRACE-END
  dst[blockIdx.x * blockDim.x + threadIdx.x] =
      tile[255u - threadIdx.x];
}


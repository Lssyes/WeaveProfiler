// Minimal CUDA C++ reference for a warp sum reduction.
// SEMANTIC-ID: reduction
// EXECUTION-SCOPE: one physical warp.

extern "C" __global__ void reduction_kernel(const float* src, float* dst) {
  const unsigned lane = threadIdx.x & 31u;
  float value = src[blockIdx.x * 32u + lane];
  // TRACE-BEGIN
  for (int delta = 16; delta > 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  // TRACE-END
  if (lane == 0) dst[blockIdx.x] = value;
}

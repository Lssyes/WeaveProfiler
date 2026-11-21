// Minimal CUDA C++ reference for a global-memory store.
// SEMANTIC-ID: store
// EXECUTION-SCOPE: physical thread; the CTA collectively covers one tile.

extern "C" __global__ void store_kernel(const float* src, float* dst, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    const float value = src[index];
    // TRACE-BEGIN
    dst[index] = value;
    // TRACE-END
  }
}

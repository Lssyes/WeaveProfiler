// Minimal CUDA C++ reference for a CTA-local barrier.
// SEMANTIC-ID: barrier
// EXECUTION-SCOPE: all threads in one cta.

extern "C" __global__ void barrier_kernel(const float* src, float* dst) {
  __shared__ float tile[256];
  tile[threadIdx.x] = src[blockIdx.x * blockDim.x + threadIdx.x];
  // TRACE-BEGIN
  __syncthreads();
  // TRACE-END
  dst[blockIdx.x * blockDim.x + threadIdx.x] = tile[255u - threadIdx.x];
}

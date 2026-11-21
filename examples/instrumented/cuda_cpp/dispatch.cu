// Minimal CUDA C++ reference for work-item acquisition only.
// SEMANTIC-ID: dispatch
// EXECUTION-SCOPE: one source-declared representative lane.

__device__ unsigned int acquire_work_item(unsigned int* next_work_item) {
  // TRACE-BEGIN
  const unsigned int work_item = atomicAdd(next_work_item, 1u);
  // TRACE-END
  return work_item;
}

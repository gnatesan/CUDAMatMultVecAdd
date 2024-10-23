///
/// vecAddKernel01.cu
/// 
///
/// This Kernel adds two Vectors A and B in C on GPU
/// while using coalesced memory access.
/// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int threadStartIndex = blockIdx.x * blockDim.x + threadIdx.x; // thread ID
    int stride = blockDim.x * gridDim.x; // stride, equal to total # of threads, ensures coalesced memory accesses
    int threadEndIndex   = threadStartIndex + (N*stride);
    for (int i = threadStartIndex; i < threadEndIndex; i += stride) {
        C[i] = A[i] + B[i];
    }
}






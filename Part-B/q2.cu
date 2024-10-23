#include <iostream> 
#include <math.h>
#include <time.h>

__global__ void add(int n, float *x, float *y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}
}

int main(int args, char* argv[]) {
	int K = atoi(argv[1]);
	int numElements = K * 1000000;
	size_t size = K * 1000000 * sizeof(float);
	float *x, *y;
	
	float *hx = (float*)malloc(size);
	float *hy = (float*)malloc(size);

	// initialize x and y arrays on the host
	for (int i = 0; i < K; i++) {
		hx[i] = 1.0f;
		hy[i] = 2.0f;
	}	

	//Allocate vectors in device memory
	cudaMalloc(&x, size);
	cudaMalloc(&y, size);


 	// Copy vectors from host memory to device global memory
 	cudaMemcpy(x, hx, size, cudaMemcpyHostToDevice);
 	cudaMemcpy(y, hy, size, cudaMemcpyHostToDevice);


	float elapsed_time = 0.0;
	struct timespec start, end;

	//1 BLOCK 1 THREADS 

	int blockSize = 1;
	int numBlocks = 1;

	// Invoke kernel for warm up
	add<<<numBlocks, blockSize>>>(numElements, x, y);


	// Synchronize to make sure everyone is done in the warmup.
  	cudaThreadSynchronize();

  	// Set up timer
  	clock_gettime(CLOCK_MONOTONIC, &start);

	add<<<numBlocks, blockSize>>> (numElements, x, y);

	// Synchronize to make sure everyone is done.
  	cudaThreadSynchronize();	

	// Compute and report the timing results
  	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed_time = ((float)end.tv_sec - (float)start.tv_sec) + ((float)end.tv_nsec - (float)start.tv_nsec) / 1000000000.0;

	printf("K=%d Number of elements in array=%d Number of blocks=%d\nBlock size=%d  Execution time=%lf seconds\n\n", K, numElements, numBlocks, blockSize, elapsed_time);






	//1 BLOCK 256 THREADS 

	blockSize = 256;
	numBlocks = 1;

	// Synchronize to make sure everyone is done in the warmup.
  	cudaThreadSynchronize();

  	// Set up timer
  	clock_gettime(CLOCK_MONOTONIC, &start);

	add<<<numBlocks, blockSize>>> (numElements, x, y);

	// Synchronize to make sure everyone is done.
  	cudaThreadSynchronize();	

	// Compute and report the timing results
  	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed_time = ((float)end.tv_sec - (float)start.tv_sec) + ((float)end.tv_nsec - (float)start.tv_nsec) / 1000000000.0;

	printf("K=%d Number of elements in array=%d Number of blocks=%d\nBlock size=%d  Execution time=%lf seconds\n\n", K, numElements, numBlocks, blockSize, elapsed_time);






	//256 THREADS NUMBLOCKS  

	blockSize = 256;
	numBlocks = numElements / blockSize + 1; //Since the numElements is not a multiple 256, we add one more block so that there are enough threads for array elements

	// Synchronize to make sure everyone is done in the warmup.
  	cudaThreadSynchronize();

  	// Set up timer
  	clock_gettime(CLOCK_MONOTONIC, &start);

	add<<<numBlocks, blockSize>>> (numElements, x, y);

	// Synchronize to make sure everyone is done.
  	cudaThreadSynchronize();	

	// Compute and report the timing results
  	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed_time = ((float)end.tv_sec - (float)start.tv_sec) + ((float)end.tv_nsec - (float)start.tv_nsec) / 1000000000.0;

	printf("K=%d Number of elements in array=%d Number of blocks=%d\nBlock size=%d  Execution time=%lf seconds\n\n", K, numElements, numBlocks, blockSize, elapsed_time);




	
	
	cudaMemcpy(hy, y, size, cudaMemcpyDeviceToHost);

	cudaFree(x); 
	cudaFree(y); 
	free(hx); 
	free(hy);

	return 0;

} 
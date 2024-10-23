#include <iostream> 
#include <math.h>
#include <time.h>


//#define FOOTPRINT_SIZE BLOCK_SIZE
#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1



__global__ void compute(double *I_0, double *F, double *O, int ans_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

	for (int i = index; i < ans_size; i += stride) {
        int k = i / (W * H); // Calculate k index
        int remaining = i % (W * H);
        int x = remaining / H; // Calculate x index
        int y = remaining % H; // Calculate y index

        double sum = 0.0;
        for (int c = 0; c < C; ++c) {
            for (int j = 0; j < FH; ++j) {
                for (int i = 0; i < FW; ++i) {
                    sum += F[k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FH + (FH - 1 - j)] *
                           I_0[c * (W + 2) * (H + 2) + (x + i) * (H + 2) + (y + j)];
                }
            }
        }

        // Store the result in output tensor O
        O[i] = sum;
    }
}

double calculateChecksum(double *h_O, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += h_O[i];
    }
    return sum;
}

int main() {
	/*
	int H = 1024;
	int W = 1024;
	int C = 3;
	int FW = 3;
	int FK = 3;
	int K = 64;
	int P = 1;
	*/

	int ans_size = K * W * H;

	size_t h_I_0_size = (C * (W+2) * (H+2)) * sizeof(double);
	size_t h_O_size = (K * W * H) * sizeof(double);
	size_t h_F_size = (K * C * FH * FW) * sizeof(double);


	//Allocate host memory
	double *h_I_0 = (double*)malloc(h_I_0_size); 
	double *h_F = (double*)malloc(h_F_size);
	double *h_O = (double*)malloc(h_O_size);


	double *I_0, *F, *O;

	//Initialize array on the host h_I_0[c, x, y] = h_I_0[c * H * W + x * W + y] = c * (x + y) but if we are on the borders set the value to 0.0
	for (int c = 0; c < C; c++) {
        for (int i = 0; i < H+2; i++) {
            for (int j = 0; j < W+2; j++) {
            	if (i < P || i >= H + P || j < P || j >= W + P) {
                    h_I_0[c * (H + 2 * P) * (W + 2 * P) + i * (W + 2 * P) + j] = 0.0;
                }
                else {
                	int x = i - P;
                    int y = j - P;
                	h_I_0[c * (H+2) * (W+2) + i * (W+2) + j] = c * (x + y);
                }
            }
        }
    }

    //Intialize array on the host h_F[k, c, i, j] = h_F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j)
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    h_F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    //Intialize array on the host, h_O[i] = 0.0
    for (int i = 0; i < ans_size; i++) {
        h_O[i] = 0.0;
    }

    //Allocate vectors in device memory
	cudaMalloc(&I_0, h_I_0_size);
	cudaMalloc(&F, h_F_size);
	cudaMalloc(&O, h_O_size);

	//Copy vectors from host memory to device global memory
 	cudaMemcpy(I_0, h_I_0, h_I_0_size, cudaMemcpyHostToDevice);
 	cudaMemcpy(F, h_F, h_F_size, cudaMemcpyHostToDevice);
 	cudaMemcpy(O, h_O, h_O_size, cudaMemcpyHostToDevice);

 	double elapsed_time = 0.0;
	struct timespec start, stop;

 	//1 thread for each element in answer array (67108864 threads)

	int blockSize = 256;
	int numBlocks = 262144;

 	//Set up timer
  	clock_gettime(CLOCK_MONOTONIC, &start);

  	compute<<<numBlocks, blockSize>>> (I_0, F, O, ans_size);

  	cudaDeviceSynchronize();

  	clock_gettime(CLOCK_MONOTONIC, &stop);

  	//Calculate elapsed time in milliseconds
    elapsed_time = (stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_nsec - start.tv_nsec) / 1000000.0;

  	cudaMemcpy(h_O, O, h_O_size, cudaMemcpyDeviceToHost);

  	double checksum = calculateChecksum(h_O, ans_size);
    printf("%f,%.3f\n", checksum, elapsed_time);

    cudaFree(I_0); 
	cudaFree(F);
	cudaFree(O);  
	free(h_I_0); 
	free(h_F);
	free(h_O);

	return 0;
}
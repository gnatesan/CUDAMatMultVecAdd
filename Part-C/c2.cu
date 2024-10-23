#include <iostream>
#include <cmath>
#include <ctime>

#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1
#define TILE_WIDTH 32

__global__ void compute_shared(double *I_0, double *F, double *O) {
    //Shared memory for the tile of I_0,  account for padding
    __shared__ double I_0_tile[TILE_WIDTH + FH - 1][TILE_WIDTH + FW - 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    //x and y coordinates of the thread globally
    int x_o = bx * TILE_WIDTH + tx;
    int y_o = by * TILE_WIDTH + ty;

    int k = blockIdx.z;
    double sum = 0.0;

    for (int c = 0; c < C; c++) {
        //Load tile of I_0 into shared memory
        int x_i = x_o - (FW - 1) / 2;
        int y_i = y_o - (FH - 1) / 2;

        if (x_i >= 0 && x_i < H && y_i >= 0 && y_i < W) {
            I_0_tile[ty][tx] = I_0[c * (H + 2) * (W + 2) + x_i * (W + 2) + y_i];
        } else { // padding
            I_0_tile[ty][tx] = 0.0;
        }

        __syncthreads();

        //Compute convolution
        for (int j = 0; j < FH; ++j) {
            for (int i = 0; i < FW; ++i) {
                sum += F[k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FH + (FH - 1 - j)] *
                       I_0_tile[ty + j][tx + i];
            }
        }

        __syncthreads();
    }

    if (x_o < H && y_o < W) {
        O[k * H * W + x_o * W + y_o] = sum;
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
    int ans_size = K * W * H;

    size_t h_I_0_size = (C * (W + 2 * P) * (H + 2 * P)) * sizeof(double);
    size_t h_O_size = (K * W * H) * sizeof(double);
    size_t h_F_size = (K * C * FH * FW) * sizeof(double);

    //Allocate host memory
    double *h_I_0 = (double*)malloc(h_I_0_size);
    double *h_F = (double*)malloc(h_F_size);
    double *h_O = (double*)malloc(h_O_size);

    double *I_0, *F, *O;

    //Initialize arrays on the host
    //Initialize array h_I_0
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H + 2 * P; i++) {
            for (int j = 0; j < W + 2 * P; j++) {
                if (i < P || i >= H + P || j < P || j >= W + P) {
                    h_I_0[c * (H + 2 * P) * (W + 2 * P) + i * (W + 2 * P) + j] = 0.0;
                } else {
                    int x = i - P;
                    int y = j - P;
                    h_I_0[c * (H + 2 * P) * (W + 2 * P) + i * (W + 2 * P) + j] = c * (x + y);
                }
            }
        }
    }

    //Initialize array h_F
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    h_F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    //Allocate device memory
    cudaMalloc(&I_0, h_I_0_size);
    cudaMalloc(&F, h_F_size);
    cudaMalloc(&O, h_O_size);

    //Copy data from host to device
    cudaMemcpy(I_0, h_I_0, h_I_0_size, cudaMemcpyHostToDevice);
    cudaMemcpy(F, h_F, h_F_size, cudaMemcpyHostToDevice);

    //Set up kernel 
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    int numBlocksX = (W + TILE_WIDTH - 1) / TILE_WIDTH;
    int numBlocksY = (H + TILE_WIDTH - 1) / TILE_WIDTH;
    int numBlocksK = K;
    dim3 blocks(numBlocksX, numBlocksY, numBlocksK);

    //Set up timer
    clock_t start, stop;
    start = clock();

    compute_shared<<<blocks, blockSize>>>(I_0, F, O);

    cudaDeviceSynchronize();

    stop = clock();

    //Calculate elapsed time in milliseconds
    double elapsed_time = ((double)(stop - start)) * 1000.0 / CLOCKS_PER_SEC;

    cudaMemcpy(h_O, O, h_O_size, cudaMemcpyDeviceToHost);

    double checksum = calculateChecksum(h_O, ans_size);
    printf("%f,%.3f\n", checksum, elapsed_time);

    //Free device and host memory
    cudaFree(I_0);
    cudaFree(F);
    cudaFree(O);
    free(h_I_0);
    free(h_F);
    free(h_O);

    return 0;
}

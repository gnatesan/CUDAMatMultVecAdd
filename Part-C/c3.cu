#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <cuda.h>
#include <cudnn.h>

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
    << "Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(x) do { \
	cudnnStatus_t ___s = (x); \
	if (___s != CUDNN_STATUS_SUCCESS) { \
		fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, \
		__LINE__, cudnnGetErrorString(___s)); \
		exit(-1); \
	} \
} while (0)

double calculateChecksum(double *h_O, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += h_O[i];
    }
    return sum;
}

int main() {
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  int N = 1;
  int C = 3;
  int H = 1024;
  int W = 1024;

  cudnnTensorDescriptor_t input_descriptor;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_descriptor));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
        N, C, H, W));

  double *in_data;
  double *in_data_h = (double*)malloc(N * C * H * W * sizeof(double));
  CUDA_CALL(cudaMalloc(
        &in_data, N * C * H * W * sizeof(double)));

  int filt_k = 64;
  int filt_c = 3;
  int filt_h = 3;
  int filt_w = 3;

  cudnnFilterDescriptor_t filter_descriptor;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_descriptor));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, filt_k, filt_c, filt_h, filt_w));

  double *filt_data;
  double *filt_data_h  = (double*)malloc(filt_k * filt_c * filt_h * filt_w * sizeof(double));
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(double)));

  cudnnConvolutionDescriptor_t conv_descriptor;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_descriptor));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, filter_descriptor, &out_n, &out_c, &out_h, &out_w));

  cudnnTensorDescriptor_t output_descriptor;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_descriptor));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, out_n, out_c, out_h, out_w));

  double *out_data;
  double *out_data_h = (double*)malloc(out_n * out_c * out_h * out_w * sizeof(double));;
  CUDA_CALL(cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(double)));

  int count;
  cudnnConvolutionFwdAlgoPerf_t perf;
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor, conv_descriptor, output_descriptor, 1, &count, &perf));

    cudnnConvolutionFwdAlgo_t algo = perf.algo;

  size_t workspace_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, conv_descriptor, output_descriptor, algo, &workspace_size));

  double *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, workspace_size));

  double alpha = 1.0;
  double beta = 0.0;
 
  //Initialize array on the host h_I_0[c, x, y] = h_I_0[c * H * W + x * W + y] = c * (x + y) 
	for (int c = 0; c < C; c++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                	in_data_h[c * (H) * (W) + i * (W) + j] = c * (i + j);
            }
        }
    }


        //Intialize array on the host h_F[k, c, i, j] = h_F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j)
    for (int k = 0; k < filt_k; k++) {
        for (int c = 0; c < filt_c; c++) {
            for (int i = 0; i < filt_h; i++) {
                for (int j = 0; j < filt_w; j++) {
                    filt_data_h[k * filt_c * filt_h * filt_w + c * filt_h * filt_w + i * filt_w + j] = (c + k) * (i + j);
                }
            }
        }
    }

    //Move arrays to device
    size_t in_size = N * C * H * W * sizeof(double);
    cudaMemcpy(in_data, in_data_h, in_size, cudaMemcpyHostToDevice);
    size_t filt_size = filt_k * filt_c * filt_h * filt_w * sizeof(double);
    cudaMemcpy(filt_data, filt_data_h, filt_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

  CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, in_data, filter_descriptor, filt_data, conv_descriptor, algo, ws_data, workspace_size, &beta, output_descriptor, out_data));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);


  size_t out_size = out_n * out_c * out_h * out_w * sizeof(double);
  cudaMemcpy(out_data_h, out_data, out_size, cudaMemcpyDeviceToHost);

  int ans_size = out_n * out_c * out_h * out_w;
  double checksum = calculateChecksum(out_data_h, ans_size);
  printf("%f,%.3f\n", checksum, milliseconds);

  
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_descriptor));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_descriptor));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_descriptor));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_descriptor));
  CUDNN_CALL(cudnnDestroy(cudnn));
  free(in_data_h);
  free(filt_data_h);
  free(out_data_h);
  return 0;
}
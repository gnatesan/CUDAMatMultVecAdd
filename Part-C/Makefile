###
#MAKEFILE for part C of HW3 HPML
###

SDK_INSTALL_PATH := /usr/local/cuda
NVCC := $(SDK_INSTALL_PATH)/bin/nvcc
OPTIONS := -O3
TAR_FILE_NAME := ggn2104.tar
EXECS := c1 c2 c3

all: $(EXECS)

clean:
	rm -f $(EXECS)

tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt

c1: c1.cu
	$(NVCC) -o c1 c1.cu

c2: c2.cu
	$(NVCC) -o c2 c2.cu

c3: c3.cu
	$(NVCC) -o c3 c3.cu -lcudnn


#c1.cu - program that completes a simple convolution algorithm in CUDA without tiling and without shared memory
#c2.cu - program that completes a convolution algorithm in CUDA using tiling and shared memory
#c3.cu - program that completes a convolution algorithm in CUDA using the CuDNN library
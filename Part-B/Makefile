###
#MAKEFILE for part B of HW3 HPML
###

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME := ggn2104.tar
EXECS := q1 q2 q3

all: $(EXECS)

clean:
	rm -f $(EXECS)

tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt

q1: q1.cpp
	g++ -o q1 q1.cpp

q2: q2.cu
	$(NVCC) -o q2 q2.cu

q3: q3.cu
	$(NVCC) -o q3 q3.cu


#q1.cpp - c++ program that adds elements of two arrays with K million elements each. K is command line parameter. 
#q2.cu - CUDA program to execute add operation as a kernel on GPU. Does not use unified memory.
#q3.cu - CUDA program to execute add operation as a kernel on GPU. This program uses unified memory. 
#q4.py - Python program to plot results from Part B




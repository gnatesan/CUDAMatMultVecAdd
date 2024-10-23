#include <cstdlib>
#include <iostream>
#include <math.h>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <time.h>

using namespace std;
using namespace std::chrono;


// function to add the elements of two arrays
void add(int n, float *x, float *y) {
for (int i = 0; i < n; i++)
y[i] = x[i] + y[i];
}


int main(int args, char* argv[]) {

int K = atoi(argv[1]);
int size = K * 1000000;

//int N = 1<<20; // 1M elements
float *x = (float*)malloc(size * sizeof(float));
float *y = (float*)malloc(size * sizeof(float));

// initialize x and y arrays on the host
for (int i = 0; i < K; i++) {
	x[i] = 1.0f;
	y[i] = 2.0f;
}

float elapsed_time = 0.0;
struct timespec start, end;


// Set up timer
clock_gettime(CLOCK_MONOTONIC, &start);

// Run kernel on 1M elements on the CPU
add(size, x, y);

// Compute and report the timing results
  	clock_gettime(CLOCK_MONOTONIC, &end);
	elapsed_time = ((float)end.tv_sec - (float)start.tv_sec) + ((float)end.tv_nsec - (float)start.tv_nsec) / 1000000000.0;

//time_t end = time(nullptr);
//double duration = difftime(end, start);

//clock_t end = clock();
//double duration = double(end - start) / CLOCKS_PER_SEC;

// Free memory
free(x);
free(y);

cout << "K=" << K << " Number of elements in array=" << size << " Execution time=" << elapsed_time << " seconds" << endl;


return 0;
}

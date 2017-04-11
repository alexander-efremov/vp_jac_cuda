#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>
#include <cmath>
#include "timer.h"

// Host version of the Jacobi method
void jacobiOnHost(double *x, double *A, double *x_prev, double *b, int nx, int ny) {
    float sigma;
    for (int i = 0; i < nx; i++) {
        sigma = 0.0;
        for (int j = 0; j < ny; j++) {
            if (i != j)
                sigma += A[i * ny + j] * x_prev[j];
        }
        x[i] = (b[i] - sigma) / A[i * ny + i];
    }
}

__global__ void kernel(double *x_next, double *A, double *x_now, double *b, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nx) {
        float sigma = 0.0;

        int idx_Ai = idx * ny;

        for (int j = 0; j < ny; j++)
            if (idx != j)
                sigma += A[idx_Ai + j] * x_now[j];

        x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
    }
}

void fill_by_null(double *arr, int size) {
    for (int i = 0; i < size; i++)
        arr[i] = 0.;
}

inline void print_matrix(double* a, int n, int m, int precision = 8) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int k = i * n + j;
			switch (precision) {
			case 1:
				printf("%.1f ", a[k]);
				break;
			case 2:
				printf("%.2f ", a[k]);
				break;
			case 3:
				printf("%.3f ", a[k]);
				break;
			case 4:
				printf("%.4f ", a[k]);
				break;
			case 5:
				printf("%.5f ", a[k]);
				break;
			case 6:
				printf("%.6f ", a[k]);
				break;
			case 7:
				printf("%.7f ", a[k]);
				break;
			case 8:
				printf("%.8f ", a[k]);
				break;
			}
		}
		printf("\n");
	}
}

void solve(char *fname, int nx, int ny, int iter_cnt, int block_size, int print_res) {
    double *x_prev, *x, *A, *b, *x_cpu_result, *x_gpu_result;
    double *x_prev_d, *x_d, *A_d, *b_d;
    int NX = nx, NY = ny, iter = iter_cnt, blockSize = block_size;
    int N = NX * NY;

    cudaEvent_t start, stop;
    float gpu_time = 0.;

    printf("Parameters:\nN=%d, NX=%d, NY=%d, iterations=%d\n", N, NX, NY, iter);

    x = (double *) malloc(NX * sizeof(double));
    x_prev = (double *) malloc(NX * sizeof(double));
    x_cpu_result = (double *) malloc(NX * sizeof(double));
    x_gpu_result = (double *) malloc(NX * sizeof(double));    

    A = (double *) malloc(N * sizeof(double));
    b = (double *) malloc(NX * sizeof(double));

    fill_by_null(x_prev, NX);
    fill_by_null(x, NX);
    fill_by_null(b, NX);

    // Read coefficient matrix from file
    FILE *file = fopen(fname, "r");
    if (file == NULL) {	exit(EXIT_FAILURE); }
	
    char *line;
    size_t len = 0;
    int i = 0;
    while ((getline(&line, &len, file)) != -1) {
        if (i < N) 
			A[i] = atof(line); 
        else 
            b[i - N] = atof(line); 
        i++;
    }
	
	StartTimer();

    for (int k = 0; k < iter; k++) {
        if (k % 2)
            jacobiOnHost(x_prev, A, x, b, NX, NY);
        else
            jacobiOnHost(x, A, x_prev, b, NX, NY);
    }
	
	double cpu_time = GetTimer();

    memcpy(x_cpu_result, x, NY * sizeof(double));
	
	if(print_res == 1)
	{
		printf("A:\n");	
		print_matrix(A, NX, NY);
		printf("b:\n");	
		print_matrix(b, 1, NX);		
	}

    fill_by_null(x_prev, NX);
    fill_by_null(x, NX);

    assert(cudaSuccess == cudaMalloc((void **) &x_d, NX * sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &A_d, N * sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &x_prev_d, NX * sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &b_d, NX * sizeof(double)));

    cudaMemcpy(x_d, x, sizeof(double) * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_prev_d, x_prev, sizeof(double) * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * NX, cudaMemcpyHostToDevice);

    int nTiles = NX / blockSize + (NX % blockSize == 0 ? 0 : 1);
    int gridHeight = NY / blockSize + (NY % blockSize == 0 ? 0 : 1);
    int gridWidth = NX / blockSize + (NX % blockSize == 0 ? 0 : 1);
    printf("gridWidth = %d, gridHeight = %d\n", gridWidth, gridHeight);
    dim3 dGrid(gridHeight, gridWidth), dBlock(blockSize, blockSize);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int k = 0; k < iter; k++) {
        if (k % 2)
            kernel << < nTiles, blockSize >> > (x_prev_d, A_d, x_d, b_d, NX, NY);
        else
            kernel << < nTiles, blockSize >> > (x_d, A_d, x_prev_d, b_d, NX, NY);
    }

    // Data <- device
    cudaMemcpy(x_gpu_result, x_d, sizeof(double) * NX, cudaMemcpyDeviceToHost);

	if(print_res == 1)
	{
		printf("CPU Result:\n");	
		print_matrix(x_cpu_result, 1, NX);
		printf("GPU Result:\n");	
		print_matrix(x_gpu_result, 1, NX);
	}
	
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time /=  1000;
    printf("CPU Computation Time %f s.\n", cpu_time);
    printf("CUDA Computation Time %f s.\n", gpu_time);

    printf("\nResult after %d iterations:\n", iter);
    double err = 0.0;
    for (i = 0; i < NX; i++) err += fabs(x_cpu_result[i] - x_gpu_result[i]);
    printf("Relative error: %f\n", err);

    // Free memory
    free(x);
    free(A);
    free(x_prev);
    free(b);
    free(x_cpu_result);
    cudaFree(x_d);
    cudaFree(A_d);
    cudaFree(x_prev_d);
    cudaFree(b_d);
    cudaFree(x_gpu_result);
}

int main() {
    solve("input_small.dat", 3,3, 10000, 4, 1);
    return 1;
}
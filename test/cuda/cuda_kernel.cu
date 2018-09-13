#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>

/* Includes, cuda */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_kernel.h"


#define BLOCKSIZE 128

__global__
void _print_array(float *arr_d, int n) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid


    // grid-striding loop
    for (int i = start_index; i < n; i += stride) {

        printf("Element[%d]: %.2f\n", i, arr_d[i]);
//        arr_d[i] += 1;
    }
}

void print_array(float *arr_d, int n) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);

    printf("Start\n");
    _print_array << < dimGrid, dimBlock >> > (arr_d, n);
    cudaDeviceSynchronize();
    printf("End\n");
}

__global__
void _calc_res0(float *residuals, const float *params, const float *measurements,
                const int nRes, const int nParams) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a2 = params[0]*params[0];
    float b = params[1];

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        float y = (3*a2)*sin(7*b);
        residuals[i] = y - measurements[i];
        printf("Element: %.2f\n", residuals[i]);
    }
}

void calc_res0(float *residuals, const float *params, const float *measurements,
               const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_res0 << < dimGrid, dimBlock >> > (residuals, params, measurements, nRes, nParams);
}

__global__
void _calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a = params[0];
    float b = params[1];
    float da = 6*a*sin(7*b);
    float dy = 21*a*a*cos(7*b);

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        // Subtration becuase residuals are y - f(x), dx = -f'(x)
        jacobians[0*nRes+i] = da;
        jacobians[1*nRes+i] = dy;
        printf("jacobians[%d]: %.2f\n", 0*nRes+i, jacobians[0*nRes+i]);
        printf("jacobians[%d]: %.2f\n", 1*nRes+i, jacobians[1*nRes+i]);
//        //TODO: Fix column orderness, rightnow we are exploiting row-order mat == mat^T for CublasSgemm
//        jacobians[i*nParams+0] = da;
//        jacobians[i*nParams+1] = dy;
//        printf("jacobians[%d]: %.2f\n", i*nParams+0, jacobians[i*nParams+0]);
//        printf("jacobians[%d]: %.2f\n", i*nParams+1, jacobians[i*nParams+1]);

    }
}

void calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_jacobi0 << < dimGrid, dimBlock >> > (jacobians, params, nRes, nParams);
}
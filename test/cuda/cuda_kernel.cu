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

/****************************************************/

__global__
void _calc_resSimple(float *residuals, const float *params, const float *measurements,
                const int nRes, const int nParams) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a2 = params[0]*params[0];

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        float y = a2;
        residuals[i] = y - measurements[i];
        //printf("Element: %.2f\n", residuals[i]);
    }
}

void calc_resSimple(float *residuals, const float *params, const float *measurements,
               const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_resSimple << < dimGrid, dimBlock >> > (residuals, params, measurements, nRes, nParams);
    cudaDeviceSynchronize();
}

__global__
void _calc_jacobiSimple(float *jacobians, const float *params, const int nRes, const int nParams) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a = params[0];
    float da = 2*a;

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        // residuals cumputed are f(x) - y, dx = f'(x), y is measurement
        jacobians[0] = da;
    }
}

void calc_jacobiSimple(float *jacobians, const float *params, const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_jacobiSimple << < dimGrid, dimBlock >> > (jacobians, params, nRes, nParams);
    cudaDeviceSynchronize();
}


/***************************************************/

__global__
void _calc_resSimple2(float *residuals, const float *params, const float *measurements,
                     const int nRes, const int nParams) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a2 = params[0]*params[0];
    float b = params[1];

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        float y = a2 * b;
        residuals[i] = y - measurements[i];
        //printf("Element: %.2f\n", residuals[i]);
    }
}

void calc_resSimple2(float *residuals, const float *params, const float *measurements,
                    const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_resSimple2 << < dimGrid, dimBlock >> > (residuals, params, measurements, nRes, nParams);
    cudaDeviceSynchronize();
}

__global__
void _calc_jacobiSimple2(float *jacobians, const float *params, const int nRes, const int nParams) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a = params[0];
    float b = params[1];
    float da = 2*a*b;
    float db = a*a;

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        // residuals cumputed are f(x) - y, dx = f'(x), y is measurement
        jacobians[0] = da;
        jacobians[1] = db;
    }
}

void calc_jacobiSimple2(float *jacobians, const float *params, const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_jacobiSimple2 << < dimGrid, dimBlock >> > (jacobians, params, nRes, nParams);
    cudaDeviceSynchronize();
}


/***************************************************/

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
        //printf("Element: %.2f\n", residuals[i]);
    }
}

void calc_res0(float *residuals, const float *params, const float *measurements,
               const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_res0 << < dimGrid, dimBlock >> > (residuals, params, measurements, nRes, nParams);
    cudaDeviceSynchronize();
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
        // residuals cumputed are f(x) - y, dx = f'(x), y is measurement
        jacobians[nRes*0+i] = da;
        jacobians[nRes*1+i] = dy;
        //printf("jacobians[%d]: %.2f\n", 0*nRes+i, jacobians[0*nRes+i]);
        //printf("jacobians[%d]: %.2f\n", 1*nRes+i, jacobians[1*nRes+i]);

//        //row-order
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
    cudaDeviceSynchronize();
}

__global__
void _calc_res2Params(float *residuals, const float *params1, const float *params2, const float *measurements,
                const int nRes, const int nParams1, const int nParams2) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a2 = params1[0]*params1[0];
    float b = params1[1];

    float c3 = params2[0]*params2[0]*params2[0];

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        float y = (3*a2)*sin(7*b) + c3;
        residuals[i] = y - measurements[i];
        //printf("Element: %.2f\n", residuals[i]);
    }
}

void calc_res2Params(float *residuals, const float *params1, const float *params2, const float *measurements,
               const int nRes, const int nParams1, const int nParams2) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_res2Params << < dimGrid, dimBlock >> > (residuals, params1, params2, measurements, nRes, nParams1, nParams2);
    cudaDeviceSynchronize();
}


__global__
void _calc_jacobi2Params(float *jacobians, float *jacobians2, const float *params1, const float *params2,
        const int nRes, const int nParams1, const int nParams2) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a = params1[0];
    float a2 = a*a;
    float b = params1[1];

    float c = params2[0];
    float c2 = c*c;
    float c3 = c2*c;

    // Final Cost = 61.0
//    float da = 6*a*sin(7*b) + c3;
//    float db = 21*a2*cos(7*b) + c3;
//    float dc = 3*a2*cos(7*b) + 3*c2;

    // Final Cost = 51.5
    float da = 6*a*sin(7*b);
    float db = 21*a2*cos(7*b);
    float dc = 3*c2;

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        // residuals cumputed are f(x) - y, dx = f'(x), y is measurement
        jacobians[nRes*0+i] = da;
        jacobians[nRes*1+i] = db;

        jacobians2[nRes*0+i] = dc;
    }
}

void calc_jacobi2Params(float *jacobians, float *jacobians2, const float *params1, const float *params2, const int nRes, const int nParams1, const int nParams2) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_jacobi2Params << < dimGrid, dimBlock >> > (jacobians, jacobians2, params1, params2, nRes, nParams1, nParams2);
    cudaDeviceSynchronize();
}

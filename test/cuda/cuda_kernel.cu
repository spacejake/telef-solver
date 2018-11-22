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
#include "solver/util/cudautil.h"


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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
}

/**************BEALE's Function**************************/

__global__
void _beales_res(float *residuals, const float *params,
                const int nRes, const int nParams) {
//    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    const float x1 = params[0];
    const float x2 = params[1];
    residuals[0] = 1.5 - x1 * (1.0 - x2);
    residuals[1] = 2.25 - x1 * (1.0 - x2 * x2);
    residuals[2] = 2.625 - x1 * (1.0 - x2 * x2 * x2);
}

void beales_res(float *residuals, const float *params,
               const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _beales_res << < 1, 1 >> > (residuals, params, nRes, nParams);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    //printf("residuals:\n");
    //print_array(residuals, nRes);
}

__global__
void _beales_jacobi(float *jacobians, const float *params, const int nRes, const int nParams) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
//    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
//    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    const float x1 = params[0];
    const float x2 = params[1];


    jacobians[nRes*0+0] = x2 - 1.0;
    jacobians[nRes*0+1] = x2 * x2 - 1.0;
    jacobians[nRes*0+2] = x2 * x2 * x2 - 1.0;

    jacobians[nRes*1+0] = x1;
    jacobians[nRes*1+1] = 2 * x1 * x2;
    jacobians[nRes*1+2] = 3 * x1 * x2 * x2;
}

void beales_jacobi(float *jacobians, const float *params, const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _beales_jacobi << < 1, 1 >> > (jacobians, params, nRes, nParams);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    //printf("jacobians:\n");
    //print_array(jacobians, nRes*nParams);
}
/************************************************************/

/**************Generalized Schwefel's Function No. 2.26**************************/


__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__
float blockReduceSum(float val) {

    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}


__global__
void _schwefel_sum(float *sum_f, const float *params, const int nParams) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    float sum = 0.f;
    // grid-striding loop
    for (int i = start_index; i < nParams; i += stride) {
        float fi = -1.f * params[i] * sin( sqrt(abs(params[i])) );
        //printf("F(x(%d):%.4f) = %.4f\n", i, params[i], fi);
        sum += fi;
    }

    sum = blockReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(sum_f, sum);
    }
}

__global__
void _schwefel_res(float *residuals, const float *sum, const int nParams) {
    const float y = 418.98288727*nParams;

    residuals[0] = y + sum[0];
}

void schwefel_res(float *residuals, const float *params,
                const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);

    float *sum;
    SOLVER_CUDA_ALLOC_AND_ZERO(&sum, static_cast<size_t>(1));

    _schwefel_sum << < dimGrid, dimBlock >> > (sum, params, nParams);
    cudaDeviceSynchronize();

    //printf("sum:\n");
    //print_array(sum, nRes);
    _schwefel_res << < 1, 1 >> > (residuals, sum, nParams);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    //printf("residuals:\n");
    //print_array(residuals, nRes);

    SOLVER_CUDA_FREE(sum);
}

__global__
void _schwefel_jacobi(float *jacobians, const float *params, const int nParams) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid


    for (int i = start_index; i < nParams; i += stride) {
        float x = abs(params[i]);
        float x_sqrt = sqrt(x);
        jacobians[i] = -sin(x_sqrt) - 0.5f * x_sqrt * cos(x_sqrt);
    }
}

void schwefel_jacobi(float *jacobians, const float *params, const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _schwefel_jacobi << < dimGrid, dimBlock >> > (jacobians, params, nParams);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    //printf("jacobians:\n");
    //print_array(jacobians, nRes*nParams);
}
/************************************************************/

/**************Generalized Schwefel's Function No. 2.26, multi Residuals*************************

__global__
void _schwefel_res(float *residuals, const float *params,
                   const int nRes, const int nParams) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    float x_solution = 420.968746;
    const float y = -1.f * x_solution * sin( sqrt(abs(x_solution)) );

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        float fi = -1.f * params[i] * sin( sqrt(abs(params[i])) );
        residuals[i] = y - fi;
    }
}

void schwefel_res(float *residuals, const float *params,
                  const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _schwefel_res << < dimGrid, dimBlock >> > (residuals, params, nRes, nParams);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    //printf("residuals:\n");
    //print_array(residuals, nRes);
}

__global__
void _schwefel_jacobi(float *jacobians, const float *params, const int nRes, const int nParams) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    for (int i = start_index; i < nRes; i += stride) {
        for (int j = start_index; j < nRes; j += stride) {
            // residuals cumputed are f(x) - y, dx = f'(x), y is measurement
            float x_sqrt = sqrt(abs(params[i]));
            jacobians[nRes * j + i] = -sin(x_sqrt) - 0.5f * x_sqrt * cos(x_sqrt);
        }
    }
}

void schwefel_jacobi(float *jacobians, const float *params, const int nRes, const int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _schwefel_jacobi << < dimGrid, dimBlock>> > (jacobians, params, nRes, nParams);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    //printf("jacobians:\n");
    //print_array(jacobians, nRes*nParams);
}
************************************************************/

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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
//    float c3 = c2*c;

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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
}



__global__
void _calc_res4Params(float *residuals,
                      const float *params1, const float *params2, const float *params3, const float *params4,
                      const float *measurements, const int nRes,
                      const int nParams1, const int nParams2, const int nParams3, const int nParams4) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a2 = params1[0]*params1[0];
    float b = params1[1];

    float c3 = params2[0]*params2[0]*params2[0];

    float d3_1 = params3[0]*params3[0]*params3[0];
    float d_2 = params3[1];

    float e2_1 = params4[0]*params4[0];
    float e2_2 = params4[1]*params4[1];

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        float y = (3*a2)*sin(7*b) + c3 + d3_1*d_2 + e2_1*e2_2;
        residuals[i] = y - measurements[i];
        //printf("Element: %.2f\n", residuals[i]);
    }
}

void calc_res4Params(float *residuals,
        const float *params1, const float *params2, const float *params3, const float *params4,
        const float *measurements, const int nRes,
        const int nParams1, const int nParams2, const int nParams3, const int nParams4) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_res4Params << < dimGrid, dimBlock >> > (residuals,
            params1, params2, params3, params4,
            measurements, nRes,
            nParams1, nParams2, nParams3, nParams4);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
}


__global__
void _calc_jacobi4Params(float *jacobians1, float *jacobians2, float *jacobians3, float *jacobians4,
                         const float *params1, const float *params2, const float *params3, const float *params4,
                         const int nRes, const int nParams1, const int nParams2, const int nParams3, const int nParams4) {
    // TODO: document column order for jacobians, perhaps give option to switch between.
    // Must Compute Jacobians in Column order!!!!, Due to cublas dependancy
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    float a = params1[0];
    float a2 = a*a;
    float b = params1[1];

    float c = params2[0];
    float c2 = c*c;
//    float c3 = c2*c;

    float d1 = params3[0];
    float d2 = params3[1];

    float e1 = params4[0];
    float e2 = params4[1];

    float da = 6*a*sin(7*b);
    float db = 21*a2*cos(7*b);
    float dc = 3*c2;

    float dd_1 = 3*d1*d1 * d2;
    float dd_2 = d1*d1*d1;

    float de_1 = 2*e1 * e2*e2;
    float de_2 = 2*e1*e1 * e2;

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        // residuals cumputed are f(x) - y, dx = f'(x), y is measurement
        jacobians1[nRes*0+i] = da;
        jacobians1[nRes*1+i] = db;

        jacobians2[nRes*0+i] = dc;

        jacobians3[nRes*0+i] = dd_1;
        jacobians3[nRes*1+i] = dd_2;

        jacobians4[nRes*0+i] = de_1;
        jacobians4[nRes*1+i] = de_2;
    }
}

void calc_jacobi4Params(float *jacobians1, float *jacobians2, float *jacobians3, float *jacobians4,
                        const float *params1, const float *params2, const float *params3, const float *params4,
                        const int nRes, const int nParams1, const int nParams2, const int nParams3, const int nParams4) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_jacobi4Params << < dimGrid, dimBlock >> > (jacobians1, jacobians2, jacobians3, jacobians4,
            params1, params2, params3, params4,
            nRes,
            nParams1, nParams2, nParams3, nParams4);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
}
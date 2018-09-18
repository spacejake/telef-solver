
#include <stdio.h>
#include <cuda_runtime.h>
//#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "solver/gpu/cuda/cu_solver.h"

//using namespace cooperative_groups;

#define BLOCKSIZE 128

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
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
void _calc_error(float* error, const float* residuals, const int nRes){
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    float sum = 0;
    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
//        printf("Thread[%d]::add[%d]: %.2f\n",threadIdx.x, i, residuals[i]);
        sum += residuals[i]*residuals[i];
    }

    sum = blockReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(error, sum);
}

void calc_error(float* error, const float* residuals, const int nRes){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_error << < dimGrid, dimBlock >> >(error, residuals, nRes);
}

__global__
void _cuda_step_down(float* step, float* lambda, const float* factor){
//    printf("update lambda: %.3f * %.3f = ", lambda[0], factor[0]);
    lambda[0] *= factor[0];
//    printf("%.3f\n", lambda[0]);

    step[0] = 1 + lambda[0];
//    printf("step down: 1 + %.3f = %.3f\n", lambda[0], step[0]);
}

/* stepdown (lambda*down)
* step = 1 + lambda;
*/
void cuda_step_down(float* step, float* lambda, const float* factor){
    _cuda_step_down << < 1, 1 >> >(step, lambda, factor);
}

/*
 * step = (1 + lambda * up) / (1 + lambda);
 * stepup (lambda*up)
 */
__global__
void _cuda_step_up(float* step, float* lambda, const float* factor){
    float newLambda = lambda[0] * factor[0];
    step[0] = (1 + newLambda) / (1 + lambda[0]);

//    printf("update lambda: %.3f * %.3f = %.3f\n", lambda[0], factor[0], newLambda);
//    printf("step up: (1 + %.3f) / (1 + %.3f) = %.3f\n", newLambda, lambda[0], step[0]);

    lambda[0] = newLambda;
}

void cuda_step_up(float* step, float* lambda, const float* factor){
    _cuda_step_up << < 1, 1 >> >(step, lambda, factor);
}

__global__
void _update_hessians(float* hessians, float* step, int nParams){
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int i = start_index; i < nParams; i += stride) {
        // Apply step down diagonal
        hessians[i+nParams*i] *= step[0];
//        printf("hessians[%d][%d]: %.4f\n",i, i, hessians[i+nParams*i]);
    }
}

void update_hessians(float* hessians, float* step, int nParams){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);

    _update_hessians << < dimGrid, dimBlock >> >(hessians, step, nParams);
}

__global__
void _update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams){
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int i = start_index; i < nParams; i += stride) {
        // Apply step down diagonal
        newParams[i] += params[i] + newDelta[i];
//        printf("params[%d][%d]: %.4f\n",i, i, newParams[i]);
    }
}

void update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);

    _update_parameters << < dimGrid, dimBlock >> >(newParams, params, newDelta, nParams);
}

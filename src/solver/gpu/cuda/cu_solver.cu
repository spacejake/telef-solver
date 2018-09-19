
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
//#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "solver/gpu/cuda/cu_solver.h"

//#include "util/cudautil.h"

//using namespace telef::solver::utils;

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

/**
 * This function takes a symmetric, positive-definite matrix "matA" and overwrites
 * the the lower half of "matA" with the lower-triangular Cholesky factor l for A = L * LH form.
 * Elements above the diagonal of "matA" are neither used nor modified. The decomposition is performed in place.
 *
 * @param solver_handle
 * @param cublas_handle
 * @param matA, matrix of size nxn
 * @param n, size of nxn matrix "matA"
 * @return true if matrix is positive-definite, otherwise false
 */
bool decompose_cholesky(cusolverDnHandle_t solver_handle, cublasHandle_t cublas_handle,
                        float* matA, const int n ){
    bool decomp_status = true;
    int lda = n;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int *info_d = NULL; // info in gpu (device copy)
//    CUDA_MALLOC(&info_d, static_cast<size_t>(1));
    cudaMalloc((void**)&info_d, sizeof(int));

    int buffer_size = 0;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    // Allocate working space for decomposition
    cusolver_status =
            cusolverDnSpotrf_bufferSize(solver_handle, uplo,
                                        n, matA, lda,
                                        &buffer_size);

    // Should not happen, if it does bad stuff man...
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    float *buffer_d;
//    CUDA_MALLOC(&buffer_d, static_cast<size_t>(buffer_size));
    cudaMalloc((void**)&buffer_d, buffer_size * sizeof(float));

    // Compute A = L*LH, result in matA in lower triangular form
    cusolver_status =
            cusolverDnSpotrf(solver_handle, uplo,
                             n, matA, lda,
                             buffer_d, buffer_size,
                             info_d );

    if (CUSOLVER_STATUS_SUCCESS != cusolver_status) {
        printf("cusolverDnSpotrf failed: status %d", cusolver_status);
        decomp_status = false;
    }

    int info_h;
//    CUDA_CHECK(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);

    if ( 0 != info_h ){
        fprintf(stderr, "Error: Cholesky factorization failed\n");
        if ( 0 > info_h ){
            printf("%d-th parameter is wrong \n", -info_h);
        }
        decomp_status = false;
    }

    // free resources
    if (info_d) cudaFree(info_d);
    if (buffer_d ) cudaFree(buffer_d);

    return decomp_status;
}
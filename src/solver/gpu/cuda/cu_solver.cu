
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
//#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "solver/gpu/cuda/cu_solver.h"

//#include "util/cudautil.h"

//using namespace telef::solver::utils;

using Clock=std::chrono::high_resolution_clock;

#define BLOCKSIZE 128


__global__
void _print_arr(const float *arr_d, const int n) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid


    // grid-striding loop
    for (int i = start_index; i < n; i += stride) {

        printf("Element[%d]: %.5f\n", i, arr_d[i]);
//        arr_d[i] += 1;
    }
}

void print_array(const char* msg, const float *arr_d, const int n) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);

    printf("%s:\n", msg);
    _print_arr << < dimGrid, dimBlock >> > (arr_d, n);
    cudaDeviceSynchronize();
    printf("\n");
}

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
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(error, sum);
    }
}

void calc_error(float* error, const float* residuals, const int nRes){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calc_error << < dimGrid, dimBlock >> >(error, residuals, nRes);
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
}

/*
 * step = (1 + lambda * up) / (1 + lambda);
 * stepup (lambda*up)
 */
__global__
void _cuda_step_update(float* lambda, const float* factor){
    lambda[0] *= factor[0];
}

void cuda_step_update(float* lambda, const float* factor){
    _cuda_step_update << < 1, 1 >> >(lambda, factor);
    cudaDeviceSynchronize();
}

__global__
void _update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    printf("_update_hessians:in");

    // grid-striding loop
    for (int i = start_index; i < nParams; i += stride) {
        int diagonal_index = i+nParams*i;
        // Apply step down diagonal
        //hessians[i+nParams*i] += hessians[i+nParams*i] * step[0];
        if (goodStep)
        {
            hessians[diagonal_index] -= dampeningFactors[i] * lambda[0] / 10.;
        }

        // adaptive scaling
        dampeningFactors[i]
                = std::max(dampeningFactors[i], hessians[diagonal_index]);

        // continuous scaling
        //scaling_vector[parameter_index] = hessian[diagonal_index];

        // initial scaling
        //if (scaling_vector[parameter_index] == 0.)
        //    scaling_vector[parameter_index] = hessian[diagonal_index];

        hessians[diagonal_index] += dampeningFactors[i] * lambda[0];

        printf("_update_hessians:hessians[%d][%d]: %.4f\n",i, i, hessians[diagonal_index]);
    }
}

void update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);

    printf("update_hessians:block[%d] grid[%d] n: %.d\n",BLOCKSIZE, ((nParams + BLOCKSIZE - 1) / BLOCKSIZE), nParams);
    _update_hessians << < dimGrid, dimBlock >> >(hessians, dampeningFactors, lambda, nParams, goodStep);
    cudaDeviceSynchronize();
    print_array("New Hessian",hessians, nParams*nParams);

}

__global__
void _update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams){
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int i = start_index; i < nParams; i += stride) {
        // Apply step down diagonal
        newParams[i] = params[i] + newDelta[i];
//        printf("params[%d]: %.4f\n",i, newParams[i]);
    }
}

void update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);

    _update_parameters << < dimGrid, dimBlock >> >(newParams, params, newDelta, nParams);
    cudaDeviceSynchronize();
    //print_array("New Params", newParams, nParams);
}

void initializeSolverBuffer(cusolverDnHandle_t solver_handle,
        float **solverBuffer, int &solverBufferSize, cublasFillMode_t uplo,
        float *matrix, const int &nRows, const int &nCols) {

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    // Allocate working space for decomposition
    // TODO: Try Allocate working space once in parameterBlock, or every-time hessian is computed
    //       as the matrix should not change much
    cusolver_status =
            cusolverDnSpotrf_bufferSize(solver_handle, uplo,
                                        nRows, matrix, nCols,
                                        &solverBufferSize);

    // Should not happen, if it does bad stuff man...
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cudaMalloc((void**)solverBuffer, solverBufferSize * sizeof(float));
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
bool decompose_cholesky(cusolverDnHandle_t solver_handle, float* matA, const int n ){

//    auto allt1 = Clock::now();
    bool decomp_status = true;
    int lda = n;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;


    int *info_d = NULL; // info in gpu (device copy)
//    CUDA_MALLOC(&info_d, static_cast<size_t>(1));
    cudaMalloc((void**)&info_d, sizeof(int));

    // Allocate working space for decomposition
    // TODO: Try Lazily initialize allocate working space once in parameterBlock, or every-time hessian is computed
    //       as the matrix should not change much?

    int buffer_size = 0;
    float *buffer_d;
    initializeSolverBuffer(solver_handle, &buffer_d, buffer_size, uplo,
                           matA, n, lda);

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    print_array("decompose_cholesky:Hessian:before", matA, n);
    // Compute A = L*LH, result in matA in lower triangular form
//    auto t1 = Clock::now();
    cusolver_status =
            cusolverDnSpotrf(solver_handle, uplo,
                             n, matA, lda,
                             buffer_d, buffer_size,
                             info_d );
    print_array("decompose_cholesky:Decomposed Hessian:after", matA, n);
//    print_array("L",matA,n*n);
//    cudaDeviceSynchronize();
//    auto t2 = Clock::now();


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
//    auto allt2 = Clock::now();
//    std::cout << "cusolverDnSpotrf Time: "
//              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
//              << " nanoseconds" << std::endl;
//    std::cout << "Total Time: "
//              << std::chrono::duration_cast<std::chrono::nanoseconds>(allt2 - allt1).count()
//              << " nanoseconds" << std::endl;

    // free resources
    if (info_d) cudaFree(info_d);
    if (buffer_d ) cudaFree(buffer_d);

    return decomp_status;
}


void solve_system_cholesky(cusolverDnHandle_t solver_handle, float* matA, float* matB, int n){
    int lda = n;
    int nCols_B = 1;

    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int *info_d = NULL; // info in gpu (device copy)
//    CUDA_MALLOC(&info_d, static_cast<size_t>(1));
    cudaMalloc((void**)&info_d, sizeof(int));

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    // Compute A = L*LH, result in matA in lower triangular form
    cusolver_status =
            cusolverDnSpotrs(solver_handle, uplo,
                             n, nCols_B, matA, lda, matB, n,
                             info_d );

//    print_array("Deltas",matB,n);

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    int info_h;
//    CUDA_CHECK(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);

    if ( 0 != info_h ){
        fprintf(stderr, "Error: Cholesky Solver failed\n");
        if ( 0 > info_h ){
            printf("%d-th parameter is wrong \n", -info_h);
        }
    }

    // free resources
    if (info_d) cudaFree(info_d);
}

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

#include "solver/util/cudautil.h"

//using namespace telef::solver::utils;


using Clock=std::chrono::high_resolution_clock;

#define BLOCKSIZE 128


__device__
int floatToOrderedInt( float floatVal )
{
    int intVal = __float_as_int( floatVal );

    return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}

__device__ float orderedIntToFloat( int intVal )
{
    return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
}

__global__
void _print_arr(const float *arr_d, const int n) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid


    // grid-striding loop
    for (int i = start_index; i < n; i += stride) {

        printf("Element[%d]: %.11f\n", i, arr_d[i]);
//        arr_d[i] += 1;
    }
}

void print_array(const char* msg, const float *arr_d, const int n) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);

    printf("%s:\n", msg);
    _print_arr << < dimGrid, dimBlock >> > (arr_d, n);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    printf("\n");
}

__inline__ __device__
float warpReduceMax(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = max(__shfl_down_sync(0xffffffff, val, offset), val);
    }
    return val;
}

__inline__ __device__
float blockReduceMax(float val) {

    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceMax(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceMax(val); //Final reduce within first warp

    return val;
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
void _sum_squares(float *sumSquares, const float *vector, const int nRes){
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    float sum = 0;
    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
//        printf("Thread[%d]::add[%d]: %.2f\n",threadIdx.x, i, residuals[i]);
        sum += vector[i]*vector[i];
    }

    sum = blockReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(sumSquares, sum);
    }
}

void calc_error(float* error, const float* residuals, const int nRes){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _sum_squares << < dimGrid, dimBlock >> >(error, residuals, nRes);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
//    print_array("calc_error:residuals",residuals, nRes);
}

/**
 * Nielsen (1999) LM λ update: good step → μ := μ * max(1/3, 1-(2ρ-1)³), ν:=2; bad step → μ := μ*ν, ν:=2ν.
 * Decreases λ when the step is accepted (so next step can be larger), increases when rejected.
 */
__global__
void _cuda_lambda_update(float *lambda, float *failFactor, const float *gainRatio, const bool goodStep){
    if (goodStep){
        float minLambda = 1.f / 3.f;
        float candidateLambda = 1 - pow(2 * gainRatio[0] - 1, 3);

        lambda[0] *= max(minLambda, candidateLambda);
        failFactor[0] = 2.f;
    } else {
        lambda[0] *= failFactor[0];
        failFactor[0] *= 2.f;
    }
}

void cuda_lambda_update(float *lambda, float *failFactor, const float *gainRatio, const bool goodStep) {
    _cuda_lambda_update << < 1, 1 >> >(lambda, failFactor, gainRatio, goodStep);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

/**
 * Damped Hessian for LM: we solve (H + damping)*Δ = -g.
 * damping_type 0: Classic Levenberg-Marquardt: H_ii += λ (i.e. H + λI). Ensures pos. def. and shrinks step.
 * damping_type 1: Marquardt (1963) scaling: H_ii += λ * max(H_ii, ε). Uses diag(H) as scale so that
 *                directions with small curvature get more damping; avoids slow convergence in narrow valleys.
 * Ref: Madsen et al. "Methods for non-linear least squares"; Marquardt (1963).
 */
__global__
void _update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep, int damping_type, float diag_floor_eps) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start_index; i < nParams; i += stride) {
        int diagonal_index = i + nParams * i;
        float h_ii = hessians[diagonal_index];
        if (damping_type == 1) {
            // Marquardt: H + λ·diag(H), with floor so we never divide by zero or negative
            float d = fmaxf(h_ii, diag_floor_eps);
            hessians[diagonal_index] = h_ii + lambda[0] * d;
        } else {
            // Classic LM: H + λI
            hessians[diagonal_index] = h_ii + lambda[0];
        }
    }
}

void update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep, int damping_type, float diag_floor_eps) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);

    _update_hessians << < dimGrid, dimBlock >> >(hessians, dampeningFactors, lambda, nParams, goodStep, damping_type, diag_floor_eps);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
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
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
//    print_array("New Deltas", newDelta, nParams);
//    print_array("New Params", newParams, nParams);
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

    SOLVER_CUDA_CHECK(cudaMalloc((void**)solverBuffer, solverBufferSize * sizeof(float)));
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
__global__ void _extract_diagonal_scale(float* scale, const float* H, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float d = H[i * n + i];
    const float eps = 1e-10f;
    scale[i] = 1.f / sqrtf(fmaxf(d, eps));
}

void extract_diagonal_scale(float* scale, const float* H_lower, int n) {
    dim3 block(BLOCKSIZE);
    dim3 grid((n + BLOCKSIZE - 1) / BLOCKSIZE);
    _extract_diagonal_scale <<< grid, block >>> (scale, H_lower, n);
    SOLVER_CHECK_ERROR_MSG("extract_diagonal_scale");
}

__global__ void _scale_symmetric_lower(float* H, const float* scale, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = n * n;
    if (idx >= total) return;
    int i = idx / n;
    int j = idx % n;
    H[idx] *= scale[i] * scale[j];
}

void scale_symmetric_lower(float* H_lower, const float* scale, int n) {
    dim3 block(BLOCKSIZE);
    dim3 grid((n * n + BLOCKSIZE - 1) / BLOCKSIZE);
    _scale_symmetric_lower <<< grid, block >>> (H_lower, scale, n);
    SOLVER_CHECK_ERROR_MSG("scale_symmetric_lower");
}

__global__ void _scale_vector_inplace(float* v, const float* scale, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) v[i] *= scale[i];
}

void scale_vector_inplace(float* v, const float* scale, int n) {
    dim3 block(BLOCKSIZE);
    dim3 grid((n + BLOCKSIZE - 1) / BLOCKSIZE);
    _scale_vector_inplace <<< grid, block >>> (v, scale, n);
    SOLVER_CHECK_ERROR_MSG("scale_vector_inplace");
}

/** Dogleg step: out = p_sd + τ*(Δ_gn - p_sd). τ ∈ [0,1] chosen so ||out|| = trust radius. */
__global__ void _dogleg_combine(float* out, const float* delta_gn, const float* p_sd, float tau, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = p_sd[i] + tau * (delta_gn[i] - p_sd[i]);
}

void dogleg_combine(float* out, const float* delta_gn, const float* p_sd, float tau, int n) {
    dim3 block(BLOCKSIZE);
    dim3 grid((n + BLOCKSIZE - 1) / BLOCKSIZE);
    _dogleg_combine <<< grid, block >>> (out, delta_gn, p_sd, tau, n);
    SOLVER_CHECK_ERROR_MSG("dogleg_combine");
}

/** For dogleg: ensure H is pos. def. by H_ii = max(H_ii, floor_eps) + μ. Avoids singular/indefinite H. */
__global__ void _apply_diag_floor_and_regularize(float* H, int n, float floor_eps, float mu) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) H[i * n + i] = fmaxf(H[i * n + i], floor_eps) + mu;
}

void apply_diag_floor_and_regularize(float* H, int n, float floor_eps, float mu) {
    dim3 block(BLOCKSIZE);
    dim3 grid((n + BLOCKSIZE - 1) / BLOCKSIZE);
    _apply_diag_floor_and_regularize <<< grid, block >>> (H, n, floor_eps, mu);
    SOLVER_CHECK_ERROR_MSG("apply_diag_floor_and_regularize");
}

/**
 * Cholesky factorization A = L L^T of the damped normal-equations matrix (H + λI or H+λ·diag(H)).
 * Used to solve (H+damping)*Δ = -g in one forward/back solve. Fails if matrix is not positive definite.
 */
bool decompose_cholesky(cusolverDnHandle_t solver_handle, float* matA, const int n ){
    bool decomp_status = true;
    int lda = n;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;


    int *info_d = NULL; // info in gpu (device copy)
//    CUDA_MALLOC(&info_d, static_cast<size_t>(1));
    SOLVER_CUDA_CHECK(cudaMalloc((void**)&info_d, sizeof(int)));

    // Allocate working space for decomposition
    // TODO: Try Lazily initialize allocate working space once in parameterBlock, or every-time hessian is computed
    //       as the matrix should not change much?

    int buffer_size = 0;
    float *buffer_d;
    initializeSolverBuffer(solver_handle, &buffer_d, buffer_size, uplo,
                           matA, n, lda);

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
//    print_array("decompose_cholesky:Hessian:before", matA, n * n);
    // Compute A = L*LH, result in matA in lower triangular form
//    auto t1 = Clock::now();
    cusolver_status =
            cusolverDnSpotrf(solver_handle, uplo,
                             n, matA, lda,
                             buffer_d, buffer_size,
                             info_d );
//    print_array("decompose_cholesky:Decomposed Hessian:after", matA, n * n);
//    print_array("L",matA,n*n);
//    cudaDeviceSynchronize();
//    auto t2 = Clock::now();


    if (CUSOLVER_STATUS_SUCCESS != cusolver_status) {
        printf("cusolverDnSpotrf failed: status %d", cusolver_status);
        decomp_status = false;
    }

    int info_h;
//    CUDA_CHECK(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    SOLVER_CUDA_CHECK(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

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
    if (info_d) SOLVER_CUDA_CHECK(cudaFree(info_d));
    if (buffer_d ) SOLVER_CUDA_CHECK(cudaFree(buffer_d));

    return decomp_status;
}


/**
 * Solve (L L^T) Δ = B after Cholesky: L y = B (forward), L^T Δ = y (back). Here B = -g, result Δ in matB.
 */
void solve_system_cholesky(cusolverDnHandle_t solver_handle, float* matA, float* matB, int n){
    int lda = n;
    int nCols_B = 1;

    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int *info_d = NULL; // info in gpu (device copy)
//    CUDA_MALLOC(&info_d, static_cast<size_t>(1));
    SOLVER_CUDA_CHECK(cudaMalloc((void**)&info_d, sizeof(int)));

    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

//    print_array("TriHessians",matA,n);
//    print_array("-1 * Gradients",matB,n);
    // Compute A = L*LH, result in matA in lower triangular form
    cusolver_status =
            cusolverDnSpotrs(solver_handle, uplo,
                             n, nCols_B, matA, lda, matB, n,
                             info_d );

//    print_array("Deltas",matB,n);

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    int info_h;
//    CUDA_CHECK(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    SOLVER_CUDA_CHECK(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != info_h ){
        fprintf(stderr, "Error: Cholesky Solver failed\n");
        if ( 0 > info_h ){
            printf("%d-th parameter is wrong \n", -info_h);
        }
    }

    // free resources
    if (info_d) SOLVER_CUDA_CHECK(cudaFree(info_d));
}

/**
 * Kernel: λ = τ * max(diag(H)). Used to set initial LM damping so (H+λI) or (H+λ·diag(H)) is pos. def.
 */
__global__
void _initialize_lambda(int *lambda, float tauFactor, float *hessian, int nParams) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int diagonal_start_index = start_index+nParams*start_index;
    float maxH = hessian[diagonal_start_index];
    for (int i = start_index; i < nParams; i += stride) {
        int diagonal_index = i+nParams*i;

        maxH = max(hessian[diagonal_index], maxH);
    }

    maxH = blockReduceMax(maxH);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        // λ = τ * max(diag(H)); applied once per block
        float weightedMaxH = tauFactor * maxH;
        atomicMax(lambda, floatToOrderedInt(weightedMaxH));
    }
}

__global__
void _convertTo(float* out, int* in) {
    out[0] = orderedIntToFloat(in[0]);
}

/**
 * Initialize λ = τ * max(diag(H)). Ensures first damped system (H+λI) or (H+λ·diag(H)) is pos. def.
 * Ref: Madsen et al. "Methods for non-linear least squares", 2nd ed., LM initial damping.
 */
void initialize_lambda(float *lambda, float tauFactor, float *hessian, int nParams) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);
    int *lamda_int;
    cudaMalloc(&lamda_int, sizeof(int));
    cudaMemset(lamda_int, 0, sizeof(int));

    _initialize_lambda << < dimGrid, dimBlock >> >(lamda_int, tauFactor, hessian, nParams);

    // AtomicMax only works for integres, here we covert lambda into a float.
    _convertTo << <1,1>> >(lambda, lamda_int);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
    cudaDeviceSynchronize();
    print_array("initialize_lambda: lambda", lambda, 1);

    SOLVER_CUDA_CHECK(cudaFree(lamda_int));
}


void cuda_norm_inf(cublasHandle_t cublasHandle, float *infNorm, const float *vector, const int nParams){
    int index = 0;
    cublasIsamax_v2(cublasHandle, nParams, vector, 1, &index);
    // CUBLAS uses 1-based index; convert to 0-based for device pointer
    index -= 1;
    cudaMemcpy(infNorm, vector + index, sizeof(float), cudaMemcpyDeviceToDevice);
}


void cuda_sum_squares(float* sumSquares, const float* vector, const int nRes){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _sum_squares << < dimGrid, dimBlock >> >(sumSquares, vector, nRes);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}
__global__
void _cuda_sqrt(float *vector, int n) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int i = start_index; i < n; i += stride) {
        vector[i] = sqrt(vector[i]);
    }
}

void cuda_sqrt(float* vector, int n){
    int blockSize = min(BLOCKSIZE, n);
    dim3 dimBlock(blockSize);
    dim3 dimGrid((n + blockSize - 1) / blockSize);

    _cuda_sqrt << < dimGrid, dimBlock >> >(vector, n);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

/**
 * Predicted gain for LM gain ratio: ½ Δ'(λΔ - g). (Our gradient is stored as -g, so λΔ - g in code.)
 * Used as denominator in gain_ratio = (f(x)-f(x+Δ)) / predicted_gain. Ref: Nielsen (1999), Madsen et al.
 */
__global__
void _compute_predicted_gain(float* predGain, float *lambda, float *daltaParams, float *gradient, int nParams){
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = start_index; i < nParams; i += stride) {
        // 0.5 * Δ'(λΔ - g); gradient here is -g
        float elemGain = lambda[0] * daltaParams[i] - gradient[i];
        elemGain *= 0.5 * daltaParams[i];
        sum += elemGain;
    }

    sum = blockReduceSum(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(predGain, sum);
    }
}

void compute_predicted_gain(float* predGain, float *lambda, float *daltaParams, float *gradient, int nParams){
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nParams + BLOCKSIZE - 1) / BLOCKSIZE);

    _compute_predicted_gain << < dimGrid, dimBlock >> >(predGain, lambda, daltaParams, gradient, nParams);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}
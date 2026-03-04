#include <stdexcept>
#include <cuda_runtime.h>

#include "solver/gpu/cuda/cu_residual.h"
#include "solver/util/cudautil.h"

#define BLOCKSIZE 128

__global__ void _apply_loss_huber_residuals(float* residuals, float* scale, int nRes, float delta) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nRes) return;
    float r = residuals[i];
    float r_sq = r * r;
    float delta_sq = delta * delta;
    float rho_p;
    if (r_sq <= delta_sq)
        rho_p = 1.f;
    else
        rho_p = (r * r > 1e-20f) ? (delta / sqrtf(r_sq)) : 1.f;
    float s = sqrtf(rho_p);
    scale[i] = s;
    residuals[i] = r * s;
}

void apply_loss_huber_residuals(float* residuals, float* scale_buffer, int nRes, float delta) {
    dim3 block(BLOCKSIZE);
    dim3 grid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);
    _apply_loss_huber_residuals <<< grid, block >>> (residuals, scale_buffer, nRes, delta);
    SOLVER_CHECK_ERROR_MSG("apply_loss_huber_residuals");
}

__global__ void _scale_jacobian_rows(float* J, const float* scale, int nRes, int nParams) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = idx / nParams;
    int j = idx % nParams;
    if (i < nRes)
        J[idx] *= scale[i];
}

void scale_jacobian_rows(float* J, const float* scale, int nRes, int nParams) {
    int total = nRes * nParams;
    dim3 block(BLOCKSIZE);
    dim3 grid((total + BLOCKSIZE - 1) / BLOCKSIZE);
    _scale_jacobian_rows <<< grid, block >>> (J, scale, nRes, nParams);
    SOLVER_CHECK_ERROR_MSG("scale_jacobian_rows");
}

void calc_gradients(cublasHandle_t cublasHandle, float *gradients, float *jacobians, float *residuals, int nRes, int nParams) {
    //Compute g(x) + global_G(x)
    cudaMatMul_ATxB(cublasHandle, gradients, jacobians, nRes, nParams, residuals, nRes, 1, 1.0f, 1.0f);
}


void calc_hessians(cublasHandle_t cublasHandle, float *hessians, float *jacobians, int nRes, int nParams){
    cudaMatMul_ATxB(cublasHandle, hessians, jacobians, nRes, nParams, jacobians, nRes, nParams);
}

void cudaMatMul_ATxB(cublasHandle_t cublasHandle, float *matC, const float *matA, const int aRows, const int aCols,
                     const float *matB, const int bRows, const int bCols, const float alpha, const float beta) {

    cudaMatMul_ATxB(cublasHandle, matC, aCols,
            matA, aRows, aCols,
            matB, bRows, bCols,
            alpha, beta);
}

void cudaMatMul_ATxB(cublasHandle_t cublasHandle, float *matC, const int cCols, const float *matA, int aRows, int aCols,
        const float *matB, int bRows, int bCols, const float alpha, const float beta) {

    // Don't know what this is (scalar?) but examples use this
    cublasStatus_t status;

    /* Perform operation using cublas, inputs/outputs are col-major.
     * vector and array were originally Eigen which defaults to Col-major
     * m is rows for A and C
     * n is cols for B and C
     * k is cols for A and rows for B*/
    // Matrix Mult C = α op ( A ) op ( B ) + β C
    status =
            cublasSgemm(cublasHandle,
                        CUBLAS_OP_T, CUBLAS_OP_N, // Matrix op(A) and op(B): No-op, Transpose, Conjugate
                        aCols, bCols, aRows, //(m,n,k)
                        &alpha,
                        matA, aRows/*leading dim*/, //(mxk) or if A^T: (kxm)
                        matB, bRows/*leading dim*/, //(kxn)
                        &beta,
                        matC, cCols/*leading dim*/); //(mxn) unless C is a much larger matrix

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("MatMul Failed\n");
    }
}

void cudaMatMul(cublasHandle_t cublasHandle, float *matC,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols) {

    // Don't know what this is (scalar?) but examples use this
    cublasStatus_t status;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    /* Perform operation using cublas, inputs/outputs are col-major.
     * vector and array were originally Eigen which defaults to Col-major
     * m is rows for A and C
     * n is cols for B and C
     * k is cols for A and rows for B*/
    // Matrix Mult C = α op ( A ) op ( B ) + β C
    status =
            cublasSgemm(cublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N, // Matrix op(A) and op(B): No-op, Transpose, Conjugate
                        aRows, bCols, aCols, //(m,n,k)
                        alpha,
                        matA, aRows/*leading dim*/, //(mxk)
                        matB, bRows/*leading dim*/, //(kxn)
                        beta,
                        matC, aRows/*leading dim*/); //(mxn)

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("MatMul Failed\n");
    }
}
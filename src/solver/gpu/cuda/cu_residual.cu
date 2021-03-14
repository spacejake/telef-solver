#include <stdexcept>

#include "solver/gpu/cuda/cu_residual.h"


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
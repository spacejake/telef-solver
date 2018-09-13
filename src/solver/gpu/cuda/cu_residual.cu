#include <stdexcept>
#include <cublas_v2.h>

#include "solver/gpu/cuda/cu_resudual.h"


void calc_gradients(float *gradients, float *jacobians, float *residuals, int nRes, int nParams) {

    int rowA=nRes;
    int colA=nParams;
    int rowB=nRes;
    int colB=1;
    cudaMatMul(gradients, jacobians, rowA, colA, residuals, rowB, colB);
}

void cudaMatMul_ATxB(float *matC,
                     const float *matA, int aRows, int aCols,
                     const float *matB, int bRows, int bCols) {

    cublasHandle_t cublasHandle;
    if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Cublas could not be initialized");
    }
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
                        CUBLAS_OP_T, CUBLAS_OP_N, // Matrix op(A) and op(B): No-op, Transpose, Conjugate
                        aCols, bCols, aRows, //(m,n,k)
                        alpha,
                        matA, aRows/*leading dim*/, //(mxk) or if A^T: (kxm)
                        matB, bRows/*leading dim*/, //(kxn)
                        beta,
                        matC, aCols/*leading dim*/); //(mxn)

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("MatMul Failed\n");
    }
}

void cudaMatMul(float *matC,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols) {

    cublasHandle_t cublasHandle;
    if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Cublas could not be initialized");
    }
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
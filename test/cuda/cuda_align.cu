
#include <stdexcept>

#include <cuda_runtime.h>
#include "cuda_align.h"
#include "cuda_quaternion.h"
#include "cuda_loss.h"
#include "solver/util/cudautil.h"

#define BLOCKSIZE 128

__global__
void _calculatePointLoss(float *residuals, const float *data, const float *target, const int nRes) {
    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int i = start_index; i < nRes; i += stride) {
        residuals[i] = target - data;
    }
}

void calculatePointLoss(float *residuals, const float *data, const float *target, const int nRes) {
    dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid((nRes + BLOCKSIZE - 1) / BLOCKSIZE);

    _calculatePointLoss << < dimGrid, dimBlock >> > (residuals, data, target, nRes);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _homogeneousPositions(float *h_position_d, const float *position_d, int nPoints) {

    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int index = start_index; index < nPoints; index += stride) {
        // homogeneous coordinates (x,y,z,1);
        float pos[4] = {position_d[3 * index], position_d[3 * index + 1], position_d[3 * index + 2], 1};
        memcpy(&h_position_d[4 * index], &pos[0], 4 * sizeof(float));
    }
}

__global__
void _hnormalizedPositions(float *position_d, const float *h_position_d, int nPoints) {

    int start_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x; // total number of threads in the grid

    // grid-striding loop
    for (int index = start_index; index < nPoints; index += stride) {

        // homogeneous coordinates (x,y,z,1);
        float hnorm = h_position_d[4 * index + 3];
        position_d[3 * index] = h_position_d[4 * index] / hnorm;
        position_d[3 * index + 1] = h_position_d[4 * index + 1] / hnorm;
        position_d[3 * index + 2] = h_position_d[4 * index + 2] / hnorm;
    }
}

void cudaMatMul(float *matC, cublasHandle_t cnpHandle,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols) {

    cublasStatus_t status;

    // Matrix Mult C = α op ( A ) op ( B ) + β C
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
            cublasSgemm(cnpHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N, // Matrix op(A) and op(B): No-op, Transpose, Conjugate
                        aRows, bCols, aCols, //(m,n,k)
                        alpha,
                        matA, aRows/*leading dim, ROWS?*/, //(4x4) or (mxk)
                        matB, bRows/*leading dim*/, //(4xN) or (kxn)
                        beta,
                        matC, aRows/*leading dim*/); //(4xN) or (mxk)

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("MatMul Failed\n");
    }
}

void applyRigidAlignment(cublasHandle_t cnpHandle, float *align_pos_d,
                         const float *position_d, const float *transMat, int N) {
    int size_homo = 4 * N;
    dim3 grid = ((N + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 block = BLOCKSIZE;

    float *matB, *matC;

    cudaMalloc((void **) &matB, size_homo * sizeof(float));
    cudaMalloc((void **) &matC, size_homo * sizeof(float));


    // Create homogenous matrix (x,y,z,1)
    _homogeneousPositions << < grid, block >> > (matB, position_d, N);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");

    /* Perform operation using cublas, inputs/outputs are col-major.
     * vector and array were originally Eigen which defaults to Col-major
     * m is rows for A and C
     * n is cols for B and C
     * k is cols for A and rows for B*/
    // Matrix Mult C = α op ( A ) op ( B ) + β C
    cudaMatMul(matC, cnpHandle, transMat, 4, 4, matB, 4, N);

    // hnormalized point (x,y,z)
    _hnormalizedPositions << < grid, block >> > (align_pos_d, matC, N);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");

    cudaFree(matB);
    cudaFree(matC);
}

__global__
void _convertQtoTrans(float *trans_d, const float* u, const float *t){
    float rotate_d[9];
    calc_r_from_u(rotate_d, u);
    calc_trans_from_tr(trans_d, t, rotate_d);
}

void convertQtoTrans(float *trans_d, const float* u, const float *t){
    _convertQtoTrans<< < 1, 1 >> > (trans_d, u, t);
}

void alignPoints(cublasHandle_t cnpHandle, float* align_d, const float* source_d, const float* t, const float* u, const int pointCount){
    float *trans_d;
    SOLVER_CUDA_CHECK(cudaMalloc((void **) &trans_d, 16*sizeof(float)));

    convertQtoTrans(trans_d, t, u);

    applyRigidAlignment(cnpHandle, align_d, source_d, trans_d, pointCount);
}


void calculateJacobians(float *dres_dt_d, float *dres_du_d, const float *u_d, const float *source_d, const int pointCount){
        calc_de_dt_lmk(dres_dt_d, pointCount, 1.0);
        calc_de_du_lmk(dres_du_d, u_d, source_d, pointCount, 1.0);
}

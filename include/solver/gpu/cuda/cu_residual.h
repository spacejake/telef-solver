#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

void calc_gradients(cublasHandle_t cublasHandle, float *gradients, float *jacobians, float *residuals, int nRes, int nParams);

/** Huber robust loss: scale[i] = sqrt(rho'(r_i^2)), residuals[i] *= scale[i]. Allocates/uses scale_buffer (nRes). */
void apply_loss_huber_residuals(float* residuals, float* scale_buffer, int nRes, float delta);
/** Scale each row i of J by scale[i]: J(i,:) *= scale[i]. */
void scale_jacobian_rows(float* J, const float* scale, int nRes, int nParams);

void calc_hessians(cublasHandle_t cublasHandle, float *hessians, float *jacobians, int nRes, int nParams);

void cudaMatMul(cublasHandle_t cublasHandle, float *matC,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols);

void cudaMatMul_ATxB(cublasHandle_t cublasHandle, float *matC,
        const float *matA, const int aRows, const int aCols,
        const float *matB, const int bRows, const int bCols,
        const float alpha = 1.0f, const float beta = 0.0f);

void cudaMatMul_ATxB(cublasHandle_t cublasHandle, float *matC, const int cCols,
        const float *matA, int aRows, int aCols,
        const float *matB, int bRows, int bCols,
        const float alpha = 1.0f, const float beta = 0.0f);

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

void calc_gradients(cublasHandle_t cublasHandle, float *gradients, float *jacobians, float *residuals, int nRes, int nParams);

void calc_hessians(cublasHandle_t cublasHandle, float *hessians, float *jacobians, int nRes, int nParams);

void cudaMatMul(cublasHandle_t cublasHandle, float *matC,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols);


void cudaMatMul_ATxB(cublasHandle_t cublasHandle, float *matC, const float *matA, int aRows, int aCols, const float *matB,
                     int bRows, int bCols, const float alpha=1.0f, const float beta=0.0f);
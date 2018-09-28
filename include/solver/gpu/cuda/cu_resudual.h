#pragma once

#include <cuda_runtime_api.h>

void calc_gradients(float *gradients, float *jacobians, float *residuals, int nRes, int nParams);

void calc_hessians(float *hessians, float *jacobians, int nRes, int nParams);

void cudaMatMul(float *matC,
                const float *matA, int aRows, int aCols,
                const float *matB, int bRows, int bCols);

void cudaMatMul_ATxB(float *matC, const float *matA, int aRows, int aCols, const float *matB, int bRows, int bCols,
                     const float alpha=1.0f);
#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

__global__
void _calculatePointLoss(float *residuals, const float *data, const float *target, const int nRes);

void calculatePointLoss(float *residuals, const float *data, const float *target, const int nRes);

__global__
void _convertQtoTrans(float *trans_d, const float* u, const float *t);

void convertQtoTrans(float *trans_d, const float* u, const float *t);

void alignPoints(cublasHandle_t cnpHandle, float* align_d, const float* source_d, const float* ft, const float* fu, const int pointCount);

void calculateJacobians(float *dres_dt_d, float *dres_du_d, const float *u_d, const float *source_d, const int pointCount);
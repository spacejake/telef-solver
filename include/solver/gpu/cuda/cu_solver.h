#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


void print_array(const char* msg, const float *arr_d, const int n);

__global__
void _sum_squares(float *sumSquares, const float *vector, const int nRes);

void calc_error(float* error, const float* residuals, const int nRes);

__global__
void _cuda_lambda_update(float *lambda, float *failFactor, const float *predGain, const bool goodStep);

void cuda_lambda_update(float *lambda, float *failFactor, const float *predGain, const bool goodStep);

__global__
void _update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep);

void update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep);

__global__
void _update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams);

void update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams);

bool decompose_cholesky(cusolverDnHandle_t solver_handle,
                        float* matA, const int n);

void solve_system_cholesky(cusolverDnHandle_t solver_handle, float* matA, float* matB, int n);

__global__
void _initialize_lambda(float *lambda, float tauFactor, float *hessian, int nParams);
void initialize_lambda(float *lambda, float tauFactor, float *hessian, int nParams);


void cuda_norm_inf(cublasHandle_t cublasHandle, float *infNorm, const float *vector, const int nParams);

void cuda_sum_squares(float* sumSquares, const float* vector, const int nRes);

__global__
void _cuda_sqrt(float *vector, int n);

void cuda_sqrt(float* vector, int n);

__global__
void _compute_predicted_gain(float* predGain, float *lambda, float *daltaParams, float *gradient, int nParams);

void compute_predicted_gain(float* predGain, float *lambda, float *daltaParams, float *gradient, int nParams);

__global__
void _fill_Jacobian(float* globalJacobian, const float *jacobian,
                    const int nGlobalParams, const int nGlobalRes,
                    const int nParams, const int nRes,
                    const int paramOffset, const int resOffset);

void fill_Jacobian(float* globalJacobian, const float *jacobian,
                   const int nGlobalParams, const int nGlobalRes,
                   const int nParams, const int nRes,
                   const int paramOffset, const int resOffset);
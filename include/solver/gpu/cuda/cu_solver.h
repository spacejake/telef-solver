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

/** damping_type: 0 = H+λI, 1 = H+λ*max(diag(H),diag_floor_eps) */
__global__
void _update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep, int damping_type, float diag_floor_eps);

void update_hessians(float *hessians, float *dampeningFactors, float *lambda, int nParams, bool goodStep, int damping_type = 0, float diag_floor_eps = 1e-6f);


__global__
void _update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams);

void update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams);

/** Jacobi/diag(H) scaling: scale[i] = 1/sqrt(max(H[i,i], eps)). Extracts diagonal from H (lower triangle). */
void extract_diagonal_scale(float* scale, const float* H_lower, int n);
/** H[i,j] *= scale[i]*scale[j] (symmetric). Modifies lower triangle. */
void scale_symmetric_lower(float* H_lower, const float* scale, int n);
/** v[i] *= scale[i] */
void scale_vector_inplace(float* v, const float* scale, int n);

/** Dogleg: out = p_sd + tau*(delta_gn - p_sd). */
void dogleg_combine(float* out, const float* delta_gn, const float* p_sd, float tau, int n);

/** H[i,i] = max(H[i,i], floor_eps) + mu (for pos-def in dogleg). */
void apply_diag_floor_and_regularize(float* H, int n, float floor_eps, float mu);

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
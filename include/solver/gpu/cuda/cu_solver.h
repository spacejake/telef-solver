#pragma once

#include <cuda_runtime_api.h>
#include <cusolverDn.h>


void print_array(const char* msg, float *arr_d, int n);

__global__
void _calc_error(float* error, const float* residuals, const int nRes);

void calc_error(float* error, const float* residuals, const int nRes);

__global__
void _cuda_step_down(float* step, float* lambda, const float* factor);

void cuda_step_down(float* step, float* lambda, const float* factor);

__global__
void _cuda_step_up(float* step, float* lambda, const float* factor);

void cuda_step_up(float* step, float* lambda, const float* factor);

__global__
void _update_hessians(float* hessians, float* step, int nParams);

void update_hessians(float* hessians, float* step, int nParams);

__global__
void _update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams);

void update_parameters(float* newParams, const float* params, const float* newDelta, const int nParams);

bool decompose_cholesky(cusolverDnHandle_t solver_handle,
                        float* matA, const int n);


void solve_system_cholesky(cusolverDnHandle_t solver_handle, float* matA, float* matB, int n);
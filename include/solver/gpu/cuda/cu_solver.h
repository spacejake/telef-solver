#pragma once

#include <cuda_runtime_api.h>

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
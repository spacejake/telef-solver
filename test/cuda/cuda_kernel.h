#pragma once

#include <cuda_runtime_api.h>


__global__
void _print_array(float *arr_d, int n);

void print_array(float *arr_d, int n);


__global__
void _calc_res0(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);

void calc_res0(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);



__global__
void _calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams);

void calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams);
#pragma once

#include <cuda_runtime_api.h>


__global__
void _print_array(float *arr_d, int n);

void print_array(float *arr_d, int n);



__global__
void _calc_resSimple(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);

void calc_resSimple(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);



__global__
void _calc_jacobiSimple(float *jacobians, const float *params, const int nRes, const int nParams);

void calc_jacobiSimple(float *jacobians, const float *params, const int nRes, const int nParams);

__global__
void _calc_res0(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);

void calc_res0(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);



__global__
void _calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams);

void calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams);


__global__
void _calc_res2Params(float *residuals, const float *params1, const float *params2, const float *measurements, const int nRes, const int nParams);

void calc_res2Params(float *residuals, const float *params1, const float *params2, const float *measurements, const int nRes, const int nParams);



__global__
void _calc_jacobi2Params(float *jacobians, const float *params1, const float *params2, const int nRes, const int nParams);

void calc_jacobi2Params(float *jacobians, const float *params1, const float *params2, const int nRes, const int nParams);
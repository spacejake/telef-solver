#pragma once

#include <cuda_runtime_api.h>

__global__
void _powell_res1(float *residuals, const float *x1, const float *x2);

void powell_res1(float *residuals, const float *x1, const float *x2);


__global__
void _powell_jacobi1(float *jacobians1, float *jacobians2, const float *x1, const float *x2);

void powell_jacobi1(float *jacobians1, float *jacobians2, const float *x1, const float *x2);


__global__
void _powell_res2(float *residuals, const float *x3, const float *x4);
void powell_res2(float *residuals, const float *x3, const float *x4);

__global__
void _powell_jacobi2(float *jacobians3, float *jacobians4, const float *x3, const float *x4);
void powell_jacobi2(float *jacobians3, float *jacobians4, const float *x3, const float *x4);

__global__
void _powell_res3(float *residuals, const float *x2, const float *x3);
void powell_res3(float *residuals, const float *x2, const float *x3);

__global__
void _powell_jacobi3(float *jacobians2, float *jacobians3, const float *x2, const float *x3);
void powell_jacobi3(float *jacobians2, float *jacobians3, const float *x2, const float *x3);


__global__
void _powell_res4(float *residuals, const float *x1, const float *x4);
void powell_res4(float *residuals, const float *x1, const float *x4);

__global__
void _powell_jacobi4(float *jacobians1, float *jacobians4, const float *x1, const float *x4);
void powell_jacobi4(float *jacobians1, float *jacobians4, const float *x1, const float *x4);
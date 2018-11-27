
#include <cuda_runtime.h>

#include "cuda_powell.h"
#include "solver/util/cudautil.h"

#define BLOCKSIZE 128

__global__
void _powell_res1(float *residuals, const float *x1, const float *x2) {
    residuals[0] = x1[0] + 10.0f * x2[0];
}

void powell_res1(float *residuals, const float *x1, const float *x2) {
    _powell_res1 << < 1, 1 >> > (residuals, x1, x2);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _powell_jacobi1(float *jacobians1, float *jacobians2, const float *x1, const float *x2) {
    jacobians1[0] = 1.f;
    jacobians2[0] = 10.0f;
}

void powell_jacobi1(float *jacobians1, float *jacobians2, const float *x1, const float *x2) {
    _powell_jacobi1 << < 1, 1 >> > (jacobians1, jacobians2, x1, x2);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _powell_res2(float *residuals, const float *x3, const float *x4) {
    residuals[0] = sqrt(5.0) * (x3[0] - x4[0]);
}

void powell_res2(float *residuals, const float *x3, const float *x4) {
    _powell_res2 << < 1, 1 >> > (residuals, x3, x4);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _powell_jacobi2(float *jacobians3, float *jacobians4, const float *x3, const float *x4) {
    jacobians3[0] = sqrt(5.0f);
    jacobians4[0] = -sqrt(5.0f);
}

void powell_jacobi2(float *jacobians3, float *jacobians4, const float *x3, const float *x4) {
    _powell_jacobi2 << < 1, 1 >> > (jacobians3, jacobians4, x3, x4);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _powell_res3(float *residuals, const float *x2, const float *x3) {
    residuals[0] = (x2[0] - 2.0f * x3[0]) * (x2[0] - 2.0f * x3[0]);
}

void powell_res3(float *residuals, const float *x2, const float *x3) {
    _powell_res3 << < 1, 1 >> > (residuals, x2, x3);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _powell_jacobi3(float *jacobians2, float *jacobians3, const float *x2, const float *x3) {
    jacobians2[0] = 2.f * (x2[0] - 2.0f * x3[0]);
    jacobians3[0] = 8.0f * x3[0] - 4.f * x2[0];
}

void powell_jacobi3(float *jacobians2, float *jacobians3, const float *x2, const float *x3) {
    _powell_jacobi3 << < 1, 1 >> > (jacobians2, jacobians3, x2, x3);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}


__global__
void _powell_res4(float *residuals, const float *x1, const float *x4) {
    residuals[0] = sqrt(10.0f) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
}

void powell_res4(float *residuals, const float *x1, const float *x4) {
    _powell_res3 << < 1, 1 >> > (residuals, x1, x4);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _powell_jacobi4(float *jacobians1, float *jacobians4, const float *x1, const float *x4) {
    float dx = 2 * sqrt(10.0f) * (x1[0] - x4[0]);
    jacobians1[0] = dx;
    jacobians4[0] = -1.f * dx;
}

void powell_jacobi4(float *jacobians1, float *jacobians4, const float *x1, const float *x4) {
    _powell_jacobi4 << < 1, 1 >> > (jacobians1, jacobians4, x1, x4);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}
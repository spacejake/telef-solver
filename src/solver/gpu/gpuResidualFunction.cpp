#include <iostream>

#include "solver/gpu/gpuResidualFunction.h"
#include "solver/gpu/cuda/cu_resudual.h"

using namespace telef::solver;

void GPUResidualFunction::calcGradients(float *gradients, float *jacobians, float *residuals, int nRes, int nParams) {
    calc_gradients(cublasHandle, gradients, jacobians, residuals, nRes, nParams);
}


void GPUResidualFunction::calcHessians(float *hessians, float *jacobians, int nRes, int nParams) {
    calc_hessians(cublasHandle, hessians, jacobians, nRes, nParams);
}
#include <iostream>

#include "solver/gpu/gpuResidualFunction.h"


using namespace telef::solver;

void GPUResidualFunction::calcGradients(float *gradients, float *residuals, float *jacobians, int nRes, int nParams) {
    std::cout << "Hello calcGradients" << std::endl;

}


void GPUResidualFunction::calcHessians(float *hessians, float *jacobians, int nRes, int nParams) {
    std::cout << "Hello calcHessians" << std::endl;
}
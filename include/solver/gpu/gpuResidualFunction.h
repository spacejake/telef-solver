#pragma once

#include "solver/residualFunction.h"
#include "solver/gpu/gpuResidualBlock.h"

namespace telef::solver {
    class GPUResidualFunction : public ResidualFunction {

        GPUResidualFunction(CostFunction::Ptr costFunc_,
                            GPUResidualBlock::Ptr resBlock_,
                            const int weight_)
                : ResidualFunction(costFunc_, resBlock_, weight_) {}

        virtual ~GPUResidualFunction(){}

        virtual void calcGradients(float* gradients, float* residuals, float* jacobians, int nRes, int nParams){};

        virtual void calcHessians(float* hessians, float* jacobians, int nRes, int nParams){};
    };
}
#pragma once

#include "solver/residualFunction.h"
#include "solver/gpu/gpuResidualBlock.h"

namespace telef::solver {

    class GPUResidualFunction : public ResidualFunction {

    public:
        using Ptr = std::shared_ptr<GPUResidualFunction>;
        using ConstPtr = std::shared_ptr<const GPUResidualFunction>;

        GPUResidualFunction(CostFunction::Ptr costFunc_,
                            GPUResidualBlock::Ptr resBlock_,
                            const float weight_=1.0)
                : ResidualFunction(costFunc_, resBlock_, weight_) {}

        virtual ~GPUResidualFunction(){}

        virtual void calcGradients(float* gradients, float* jacobians, float* residuals, int nRes, int nParams);
        virtual void calcHessians(float* hessians, float* jacobians, int nRes, int nParams);
    };
}
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "solver/residualFunction.h"
#include "solver/gpu/gpuResidualBlock.h"

namespace telef::solver {

    class GPUResidualFunction : public ResidualFunction {

    public:
        using Ptr = std::shared_ptr<GPUResidualFunction>;
        using ConstPtr = std::shared_ptr<const GPUResidualFunction>;

        GPUResidualFunction(CostFunction::Ptr costFunc_,
                            GPUResidualBlock::Ptr resBlock_,
                            const float weight_= 1.0)
                : ResidualFunction(costFunc_, resBlock_, weight_), lossScale_d(nullptr) {}

        virtual ~GPUResidualFunction();

        void setCublasHandle(cublasHandle_t cublasHandle_){
            cublasHandle = cublasHandle_;
        }

        void setLossFunction(LossFunction::Ptr loss) override;
        void applyLoss() override;

    protected:
        cublasHandle_t cublasHandle;
        float* lossScale_d;
    };
}
#pragma once

#include <cublas_v2.h>
#include "solver/residualFunction.h"
#include "solver/gpu/gpuResidualBlock.h"

namespace telef::solver {

    // TODO: Remove GPUResidualFunction class??? Seems we don't need this currently
    class GPUResidualFunction : public ResidualFunction {

    public:
        using Ptr = std::shared_ptr<GPUResidualFunction>;
        using ConstPtr = std::shared_ptr<const GPUResidualFunction>;

        GPUResidualFunction(CostFunction::Ptr costFunc_,
                            GPUResidualBlock::Ptr resBlock_,
                            const float weight_= 1.0)
                : ResidualFunction(costFunc_, resBlock_, weight_) {}

        virtual ~GPUResidualFunction(){
        }

        void setCublasHandle(cublasHandle_t cublasHandle_){
            cublasHandle = cublasHandle_;
        }

    protected:
        cublasHandle_t cublasHandle;
    };
}
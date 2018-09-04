#pragma once

#include "solver/residualBlock.h"
#include "solver/gpu/gpuParameterBlock.h"
#include "util/cudautil.h"

namespace telef::solver {

    class GPUResidualBlock : public ResidualBlock {
    public:
        using Ptr = std::shared_ptr<GPUResidualBlock>;
        using ConstPtr = std::shared_ptr<const GPUResidualBlock>;

        GPUResidualBlock(const int nRes): ResidualBlock(nRes) {
            CUDA_ALLOC_AND_ZERO(residuals, static_cast<size_t>(nRes));
        }

        /**
         * Allocates Residual Block and all required Parameter Blocks in one call
         *
         * @param nRes
         * @param nParamsList
         */
        GPUResidualBlock(const int nRes, const std::vector<int> nParamsList): ResidualBlock(nRes) {
            CUDA_ALLOC_AND_ZERO(residuals, static_cast<size_t>(nRes));

            for(int nParams : nParamsList){
                GPUParameterBlock::Ptr paramObj = std::make_shared(nRes, nParams);
            }
        }

        virtual ~GPUResidualBlock(){
            CUDA_FREE(residuals);
        }

        virtual float* getResiduals(){
            return residuals;
        };

        void addParameterBlock(ParameterBlock::Ptr param) {
            params.push_back(param);
        }

        const std::vector<ParameterBlock::Ptr>& getParameterBlocks() const {
            return parameterBlocks;
        }

    private:
        float* residuals;
    };

}

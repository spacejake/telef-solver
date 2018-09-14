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
            utils::CUDA_ALLOC_AND_ZERO(&residuals, static_cast<size_t>(nRes));
        }

        /**
         * Allocates Residual Block and all required Parameter Blocks in one call
         *
         * @param nRes
         * @param nParamsList
         */
        GPUResidualBlock(const int nRes, const std::vector<int> nParamsList): ResidualBlock(nRes) {
            utils::CUDA_ALLOC_AND_ZERO(&residuals, static_cast<size_t>(nRes));

//            std::cout << "Num ParamBlocks: " << nParamsList.size() << std::endl;
            for(int nParams : nParamsList){
//                std::cout << "Num Params: " << nParams << std::endl;
                GPUParameterBlock::Ptr paramObj = std::make_shared<GPUParameterBlock>(nRes, nParams);
                parameterBlocks.push_back(paramObj);
            }
        }

        virtual ~GPUResidualBlock(){
            utils::CUDA_FREE(residuals);
        }

        virtual float* getResiduals(){
            return residuals;
        };

    private:
        float* residuals;
    };

}

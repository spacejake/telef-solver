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
            initialize(nRes);
        }

        /**
         * Allocates Residual Block and all required Parameter Blocks in one call
         *
         * @param nRes
         * @param nParamsList
         */
        GPUResidualBlock(const int nRes, const std::vector<int>& nParamsList): ResidualBlock(nRes) {
            initialize(nRes);
            initializeParams(nRes, nParamsList);
        }

        virtual ~GPUResidualBlock(){
            utils::CUDA_FREE(residuals);
            utils::CUDA_FREE(step);
            utils::CUDA_FREE(lambda);
        }

        virtual float* getResiduals(){
            return residuals;
        };

        virtual float* getStep(){
            return step;
        };

        virtual float* getLambda(){
            return lambda;
        };

    private:
        float* residuals;
        float* step;
        float* lambda;

        void initialize(const int& nRes){
            utils::CUDA_ALLOC_AND_ZERO(&residuals, static_cast<size_t>(nRes));
            utils::CUDA_ALLOC_AND_ZERO(&step, static_cast<size_t>(1));
            utils::CUDA_ALLOC_AND_ZERO(&lambda, static_cast<size_t>(1));
        }

        void initializeParams(const int& nRes, const std::vector<int>& nParamsList){
            for(int nParams : nParamsList){
                GPUParameterBlock::Ptr paramObj = std::make_shared<GPUParameterBlock>(nRes, nParams);
                parameterBlocks.push_back(paramObj);
            }
        }
    };

}

#pragma once

#include "solver/residualBlock.h"
#include "solver/gpu/gpuParameterBlock.h"
#include "solver/util/cudautil.h"

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
            SOLVER_CUDA_FREE(residuals);
            SOLVER_CUDA_FREE(workingError);
        }

        // TODO: Change to use override keyword instead of virtual
        virtual float* getResiduals(){
            return residuals;
        };

        virtual float* getWorkingError(){
            return workingError;
        }

    private:
        float* residuals;

        float* workingError;
        void initialize(const int& nRes){
            SOLVER_CUDA_ALLOC_AND_ZERO(&residuals, static_cast<size_t>(nRes));
            SOLVER_CUDA_ALLOC_AND_ZERO(&workingError, static_cast<size_t>(1));
        }

        void initializeParams(const int& nRes, const std::vector<int>& nParamsList){
            for(int nParams : nParamsList){
                GPUParameterBlock::Ptr paramObj = std::make_shared<GPUParameterBlock>(nRes, nParams);
                parameterBlocks.push_back(paramObj);
            }
        }
    };

}

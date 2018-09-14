#pragma once

#include "solver/parameterBlock.h"

#include "util/cudautil.h"

namespace telef::solver {
    class GPUParameterBlock : public ParameterBlock {
    public:
        using Ptr = std::shared_ptr<GPUParameterBlock>;
        using ConstPtr = std::shared_ptr<const GPUParameterBlock>;

        GPUParameterBlock(const int nRes, const int nParams) : ParameterBlock(nRes, nParams){
            printf("ParamBlock with Res:%d and Params:%d\n", nRes, nParams);
            utils::CUDA_ALLOC_AND_ZERO(&parameters, static_cast<size_t>(nParams));
            utils::CUDA_ALLOC_AND_ZERO(&resultParameters, static_cast<size_t>(nParams));
            utils::CUDA_ALLOC_AND_ZERO(&deltaParams, static_cast<size_t>(nParams));

            utils::CUDA_ALLOC_AND_ZERO(&jacobians, static_cast<size_t>(nParams * nRes));
            utils::CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(nParams));
            utils::CUDA_ALLOC_AND_ZERO(&hessians, static_cast<size_t>(nParams * nParams));
        }

        virtual ~GPUParameterBlock(){
            utils::CUDA_FREE(parameters);
            utils::CUDA_FREE(resultParameters);
            utils::CUDA_FREE(deltaParams);

            utils::CUDA_FREE(jacobians);
            utils::CUDA_FREE(gradients);
            utils::CUDA_FREE(hessians);
        }

        virtual float* getParameters(){
            return parameters;
        }

        virtual float* getResultParameters(){
            return resultParameters;
        }

        virtual float* getDeltaParameters(){
            return deltaParams;
        }

        virtual float* getJacobians(){
            return jacobians;
        }

        virtual float* getGradients(){
            return gradients;
        }

        virtual float* getHessians(){
            return hessians;
        }

    private:
        float* parameters;
        float* resultParameters;
        float* deltaParams;

        float* jacobians;
        float* gradients;
        float* hessians;
    };
}
#pragma once

#include "solver/parameterBlock.h"

#include "util/cudautil.h"

namespace telef::solver {
    class GPUParameterBlock : public ParameterBlock {
    public:
        using Ptr = std::shared_ptr<GPUParameterBlock>;
        using ConstPtr = std::shared_ptr<const GPUParameterBlock>;

        GPUParameterBlock(const int nRes, const int nParams) : ParameterBlock(nRes, nParams){
            // TODO: Allocate cuda space for params and partial derivitives
            CUDA_ALLOC_AND_ZERO(&parameters, static_cast<size_t>(nParams));
            CUDA_ALLOC_AND_ZERO(&resultParameters, static_cast<size_t>(nParams));
            CUDA_ALLOC_AND_ZERO(&deltaParams, static_cast<size_t>(nParams));

            CUDA_ALLOC_AND_ZERO(&jacobians, static_cast<size_t>(nParams * nRes));
            CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(nParams));
            CUDA_ALLOC_AND_ZERO(&hessians, static_cast<size_t>(nParams * nParams));
        }

        virtual ~GPUParameterBlock(){
            //TODO: Free Cuda memory
            CUDA_FREE(parameters);
            CUDA_FREE(resultParameters);
            CUDA_FREE(deltaParams);

            CUDA_FREE(jacobians);
            CUDA_FREE(gradients);
            CUDA_FREE(hessians);
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
#pragma once

#include "solver/parameterBlock.h"

#include "util/cudautil.h"

namespace telef::solver {
    class GPUParameterBlock : public ParameterBlock {
    public:
        using Ptr = std::shared_ptr<GPUParameterBlock>;
        using ConstPtr = std::shared_ptr<const GPUParameterBlock>;

        GPUParameterBlock(const int nRes, const int nParams)
                : ParameterBlock(nRes, nParams) {
            initDeviceMemory();
        }

        virtual ~GPUParameterBlock(){
            if (workingParameters) utils::CUDA_FREE(workingParameters);
            if (parameters) utils::CUDA_FREE(parameters);


            utils::CUDA_FREE(jacobians);
            utils::CUDA_FREE(gradients);
        }


        virtual void setInitialParams(float* initialParams_) {
            if (!isShared()) {
                resultParameters = initialParams_;
                initializeParameters();
            }
        }

        virtual void initializeParameters(){
            if (!isShared()) {
                cudaMemcpy(workingParameters, resultParameters, nParameters * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(parameters, workingParameters, nParameters * sizeof(float), cudaMemcpyDeviceToDevice);
            }
        }

        virtual float* getResultParameters() {
            cudaMemcpy(resultParameters, getParameters(), nParameters*sizeof(float), cudaMemcpyDeviceToHost);
            return resultParameters;
        }

        virtual float* getWorkingParameters(){
            if (isShared()) {
                return shared_parameter->getWorkingParameters();
            } else {
                return workingParameters;
            }
        }

        virtual float* getParameters(){
            if (isShared()) {
                return shared_parameter->getParameters();
            } else {
                return parameters;
            }
        }

        virtual float* getJacobians(){
            return jacobians;
        }

        virtual float* getGradients(){
            return gradients;
        }

        virtual void onShare(){
            // We will be using the shared parameters now
            utils::CUDA_FREE(workingParameters);
            utils::CUDA_FREE(parameters);
        }

    private:
        float* workingParameters;
        float* parameters;

        float* jacobians;
        float* gradients;

    private:
        void initDeviceMemory() {
//            printf("ParamBlock with Res:%d and Params:%d\n", nRes, nParams);
            utils::CUDA_ALLOC_AND_ZERO(&workingParameters, static_cast<size_t>(numParameters()));
            utils::CUDA_ALLOC_AND_ZERO(&parameters, static_cast<size_t>(numParameters()));

            utils::CUDA_ALLOC_AND_ZERO(&jacobians, static_cast<size_t>(numParameters() * numResiduals()));
            utils::CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(numParameters()));
        }
    };
}
#pragma once

#include "solver/parameterBlock.h"

#include "solver/util/cudautil.h"

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
            if (workingParameters) SOLVER_CUDA_FREE(workingParameters);
            if (parameters) SOLVER_CUDA_FREE(parameters);


            SOLVER_CUDA_FREE(jacobians);
            SOLVER_CUDA_FREE(gradients);
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
            // FIXME: What if user uses same pointer or doesn't but still considard shared?? Just overwrite it?
            if (!isShared()) {
                cudaMemcpy(resultParameters, getParameters(), nParameters * sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpy(resultParameters, shared_parameter->getParameters(), nParameters * sizeof(float), cudaMemcpyDeviceToHost);
            }

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
            SOLVER_CUDA_FREE(workingParameters);
            SOLVER_CUDA_FREE(parameters);
        }

    private:
        float* workingParameters;
        float* parameters;

        float* jacobians;
        float* gradients;

    private:
        void initDeviceMemory() {
//            printf("ParamBlock with Res:%d and Params:%d\n", nRes, nParams);
            SOLVER_CUDA_ALLOC_AND_ZERO(&workingParameters, static_cast<size_t>(numParameters()));
            SOLVER_CUDA_ALLOC_AND_ZERO(&parameters, static_cast<size_t>(numParameters()));

            SOLVER_CUDA_ALLOC_AND_ZERO(&jacobians, static_cast<size_t>(numParameters() * numResiduals()));
            SOLVER_CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(numParameters()));
        }
    };
}
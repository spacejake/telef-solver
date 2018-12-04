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
            if (parameters) SOLVER_CUDA_FREE(parameters);
            if (bestParameters) SOLVER_CUDA_FREE(bestParameters);
            if (jacobians) SOLVER_CUDA_FREE(jacobians);
            if (gradients) SOLVER_CUDA_FREE(gradients);
        }


        virtual void setInitialParams(float* initialParams_) {
            if (!isShared()) {
                resultParameters = initialParams_;
                initializeParameters();
            } else {
                resultParameters = initialParams_;
            }
        }

        virtual void initializeParameters(){
            if (!isShared()) {
                SOLVER_CUDA_CHECK(cudaMemcpy(parameters, resultParameters, nParameters * sizeof(float), cudaMemcpyHostToDevice));
                SOLVER_CUDA_CHECK(cudaMemcpy(bestParameters, parameters, nParameters * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }

        virtual float* getResultParameters() {
            // FIXME: What if user uses same pointer or doesn't but still considard shared?? Just overwrite it?
            if (!isShared()) {
                SOLVER_CUDA_CHECK(cudaMemcpy(resultParameters, getBestParameters(), nParameters * sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                SOLVER_CUDA_CHECK(cudaMemcpy(resultParameters, shared_parameter->getBestParameters(), nParameters * sizeof(float), cudaMemcpyDeviceToHost));
            }

            return resultParameters;
        }

        virtual float* getParameters(){
            if (isShared()) {
                return shared_parameter->getParameters();
            } else {
                return parameters;
            }
        }

        virtual float* getBestParameters(){
            if (isShared()) {
                return shared_parameter->getBestParameters();
            } else {
                return bestParameters;
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
            //SOLVER_CUDA_FREE(parameters);
            //SOLVER_CUDA_FREE(bestParameters);
        }

    private:
        float* parameters;
        float* bestParameters;

        float* jacobians;
        float* gradients;

    private:
        void initDeviceMemory() {
//            printf("ParamBlock with Res:%d and Params:%d\n", nRes, nParams);
            SOLVER_CUDA_ALLOC_AND_ZERO(&parameters, static_cast<size_t>(numParameters()));
            SOLVER_CUDA_ALLOC_AND_ZERO(&bestParameters, static_cast<size_t>(numParameters()));

            SOLVER_CUDA_ALLOC_AND_ZERO(&jacobians, static_cast<size_t>(numParameters() * numResiduals()));
            SOLVER_CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(numParameters()));
        }
    };
}
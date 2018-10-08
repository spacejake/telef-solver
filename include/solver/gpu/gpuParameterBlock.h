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
            /*, solverBuffer(NULL), solverBufferSize(0)*/
            initDeviceMemory();
        }

        virtual ~GPUParameterBlock(){
            utils::CUDA_FREE(workingParameters);
            utils::CUDA_FREE(parameters);
            utils::CUDA_FREE(deltaParams);

            utils::CUDA_FREE(dampeningFactors);

            utils::CUDA_FREE(jacobians);
            utils::CUDA_FREE(gradients);
            utils::CUDA_FREE(hessians);
            utils::CUDA_FREE(hessianLowTri);
//            utils::CUDA_FREE(solverBuffer);
        }


        virtual void setInitialParams(float* initialParams_) {
            resultParameters = initialParams_;
            initializeParameters();
        }

        virtual void initializeParameters(){
            cudaMemcpy(workingParameters, resultParameters, nParameters*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(parameters, workingParameters,nParameters*sizeof(float), cudaMemcpyDeviceToDevice);
        }

        virtual float* getResultParameters() {
            cudaMemcpy(resultParameters, parameters, nParameters*sizeof(float), cudaMemcpyDeviceToHost);
            return resultParameters;
        };

        virtual float* getWorkingParameters(){
            return workingParameters;
        }

        virtual float* getParameters(){
            return parameters;
        }

        virtual float* getDeltaParameters(){
            return deltaParams;
        }


        virtual float* getDampeningFactors(){
            return dampeningFactors;
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

        virtual float* getHessianLowTri(){
            return hessianLowTri;
        }

//        virtual float* getSolverBuffer(){
//            return solverBuffer;
//        }
//
//        virtual int& getSolverBufferSize(){
//            return solverBufferSize;
//        }

    private:
        float* workingParameters;
        float* parameters;
        float* deltaParams;

        float* dampeningFactors;

        float* jacobians;
        float* gradients;
        float* hessians;
        float* hessianLowTri;

//        float* solverBuffer;
//        int solverBufferSize;

    private:
        void initDeviceMemory(){
//            printf("ParamBlock with Res:%d and Params:%d\n", nRes, nParams);
            utils::CUDA_ALLOC_AND_ZERO(&workingParameters, static_cast<size_t>(numParameters()));
            utils::CUDA_ALLOC_AND_ZERO(&parameters, static_cast<size_t>(numParameters()));
            utils::CUDA_ALLOC_AND_ZERO(&deltaParams, static_cast<size_t>(numParameters()));

            utils::CUDA_ALLOC_AND_ZERO(&dampeningFactors, static_cast<size_t>(numParameters()));

            utils::CUDA_ALLOC_AND_ZERO(&jacobians, static_cast<size_t>(numParameters() * numResiduals()));
            utils::CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(numParameters()));
            utils::CUDA_ALLOC_AND_ZERO(&hessians, static_cast<size_t>(numParameters() * numParameters()));
            utils::CUDA_ALLOC_AND_ZERO(&hessianLowTri, static_cast<size_t>(numParameters() * numParameters()));
        }
    };
}
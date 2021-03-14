#pragma once

#include <memory>
#include <vector>

#include "solver/util/cudautil.h"

#include "solver/problem.h"
#include "solver/gpu/gpuResidualFunction.h"
#include "solver/gpu/cuda/cu_residual.h"
#include "solver/gpu/cuda/cu_solver.h"

namespace telef::solver {
    class GPUProblem : public Problem {
    public:
        using Ptr = std::shared_ptr<GPUProblem>;
        using ConstPtr = std::shared_ptr<const GPUProblem>;

        GPUProblem() : Problem() {
        }

        virtual ~GPUProblem() {
            SOLVER_CUDA_FREE(workingError);
            SOLVER_CUDA_FREE(lambda);
            SOLVER_CUDA_FREE(failFactor);
            SOLVER_CUDA_FREE(predictedGain);
            SOLVER_CUDA_FREE(parameters2norm);
            SOLVER_CUDA_FREE(deltaParams);
            SOLVER_CUDA_FREE(dampeningFactors);
            SOLVER_CUDA_FREE(gradients);
            SOLVER_CUDA_FREE(hessian);
            SOLVER_CUDA_FREE(hessianLowTri);
        }

        void setCublasHandle(cublasHandle_t cublasHandle_){
            cublasHandle = cublasHandle_;
        }

        virtual float* getWorkingError() {
            return workingError;
        }

        virtual float* getLambda() {
            return lambda;
        }

        virtual float* getFailFactor(){
            return failFactor;
        }

        virtual float* getPredictedGain() {
            return predictedGain;
        }

        virtual float* getParams2Norm() {
            return parameters2norm;
        }

        // Global combined Matricies
        virtual float* getDeltaParameters() {
            return deltaParams;
        }

        virtual float* getDampeningFactors() {
            return dampeningFactors;
        }

        virtual float* getGradient() {
            return gradients;
        }

        virtual float* getHessian() {
            return hessian;
        }

        virtual float* getHessianLowTri() {
            return hessianLowTri;
        }

        /**
         * compute and allocate size for global matrices
         * Call befor running solver or when the Problem space has been modified, i.e. add more ResidualBlocks
         */
        virtual void onInitialize() {
            // Allocate cuda space
            SOLVER_CUDA_ALLOC_AND_ZERO(&workingError, static_cast<size_t>(1));
            SOLVER_CUDA_ALLOC_AND_ZERO(&lambda, static_cast<size_t>(1));
            SOLVER_CUDA_ALLOC_AND_ZERO(&failFactor, static_cast<size_t>(1));
            SOLVER_CUDA_ALLOC_AND_ZERO(&predictedGain, static_cast<size_t>(1));
            SOLVER_CUDA_ALLOC_AND_ZERO(&parameters2norm, static_cast<size_t>(1));
            SOLVER_CUDA_ALLOC_AND_ZERO(&deltaParams, static_cast<size_t>(nEffectiveParams));
            SOLVER_CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(nEffectiveParams));

            SOLVER_CUDA_ALLOC_AND_ZERO(&dampeningFactors, static_cast<size_t>(nEffectiveParams));

            SOLVER_CUDA_ALLOC_AND_ZERO(&hessian, static_cast<size_t>(nEffectiveParams * nEffectiveParams));
            SOLVER_CUDA_ALLOC_AND_ZERO(&hessianLowTri, static_cast<size_t>(nEffectiveParams * nEffectiveParams));
        }


        void calcGradients(float *gradients, float *jacobians, float *residuals, int nRes, int nParams) {
            calc_gradients(cublasHandle, gradients, jacobians, residuals, nRes, nParams);
        }

        virtual void
        calcHessianBlock(float *hessianBlock, const int nEffectiveParams,
                         const float *jacobianA, const int nParamsA,
                         const float *jacobianB, const int nParamsB,
                         const int nResiduals) {
            cudaMatMul_ATxB(cublasHandle, hessianBlock, nEffectiveParams,
                    jacobianA, nResiduals, nParamsA,
                    jacobianB, nResiduals, nParamsB,
                    1.0f, 1.0f);
        }

        virtual ResidualFunction::Ptr createResidualFunction(CostFunction::Ptr costFunc_) {
            auto resBlock = std::make_shared<GPUResidualBlock>(costFunc_->numResiduals(), costFunc_->getParameterSizes());
            return std::make_shared<GPUResidualFunction>(costFunc_, resBlock);
        }

    protected:
        cublasHandle_t cublasHandle;

    private:
        float* workingError;

        float* lambda;
        float* failFactor;
        float* predictedGain;
        float* parameters2norm;

        float* deltaParams;

        float* dampeningFactors;

        float* gradients;
        float* hessian;
        float* hessianLowTri;

    };
}
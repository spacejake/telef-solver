#pragma once

#include <memory>
#include <vector>

#include "util/cudautil.h"

#include "solver/problem.h"
#include "solver/gpu/cuda/cu_resudual.h"
#include "solver/gpu/cuda/cu_solver.h"

namespace telef::solver {
    class GPUProblem : public Problem {
    public:
        using Ptr = std::shared_ptr<GPUProblem>;
        using ConstPtr = std::shared_ptr<const GPUProblem>;

        GPUProblem() : Problem() {}
        virtual ~GPUProblem() {
            cudaFree(workingError);
            cudaFree(lambda);
            cudaFree(deltaParams);
            cudaFree(dampeningFactors);
            cudaFree(gradients);
            cudaFree(hessian);
            cudaFree(hessianLowTri);
        }

        virtual float* getWorkingError() {
            return workingError;
        }

        virtual float* getLambda() {
            return lambda;
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
            utils::CUDA_ALLOC_AND_ZERO(&workingError, static_cast<size_t>(1));
            utils::CUDA_ALLOC_AND_ZERO(&lambda, static_cast<size_t>(1));
            utils::CUDA_ALLOC_AND_ZERO(&deltaParams, static_cast<size_t>(nEffectiveParams));
            utils::CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(nEffectiveParams));

            utils::CUDA_ALLOC_AND_ZERO(&dampeningFactors, static_cast<size_t>(nEffectiveParams));

            utils::CUDA_ALLOC_AND_ZERO(&hessian, static_cast<size_t>(nEffectiveParams * nEffectiveParams));
            utils::CUDA_ALLOC_AND_ZERO(&hessianLowTri, static_cast<size_t>(nEffectiveParams * nEffectiveParams));
        }

        virtual void
        calculateHessianBlock(float *hessianBlock, const int nEffectiveParams,
                const float *jacobianA, const int nParamsA,
                const float *jacobianB, const int nParamsB,
                const int nResiduals) {
            cudaMatMul_ATxB(cublasHandle, hessianBlock, nEffectiveParams,
                    jacobianA, nResiduals, nParamsA,
                    jacobianB, nResiduals, nParamsB,
                    1.0f, 1.0f);
        }

        void setCublasHandle(cublasHandle_t cublasHandle_){
            cublasHandle = cublasHandle_;
        }

    protected:
        cublasHandle_t cublasHandle;

    private:
        float* workingError;

        float* lambda;
        float* deltaParams;

        float* dampeningFactors;

        float* gradients;
        float* hessian;
        float* hessianLowTri;

    };
}
#pragma once

#include <memory>
#include <vector>

#include "util/cudautil.h"

#include "solver/problem.h"

namespace telef::solver {
    class GPUProblem : public Problem {
    public:
        using Ptr = std::shared_ptr<GPUProblem>;
        using ConstPtr = std::shared_ptr<const GPUProblem>;

        GPUProblem() : Problem() {}
        virtual ~GPUProblem() {
            cudaFree(lambda);
            cudaFree(deltaParams);
            cudaFree(dampeningFactors);
            cudaFree(gradients);
            cudaFree(hessian);
            cudaFree(hessianLowTri);
        }

        virtual float* getLambda() {
            return lambda;
        };

        // Global combined Matricies
        virtual float* getDeltaParameters() {
            return deltaParams;
        }

        virtual float* getDampeningFactors() {
            return dampeningFactors;
        }

        virtual float* getGradients() {
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
            utils::CUDA_ALLOC_AND_ZERO(&deltaParams, static_cast<size_t>(nEffectiveParams));
            utils::CUDA_ALLOC_AND_ZERO(&gradients, static_cast<size_t>(nEffectiveParams));

            utils::CUDA_ALLOC_AND_ZERO(&dampeningFactors, static_cast<size_t>(nEffectiveParams));

            utils::CUDA_ALLOC_AND_ZERO(&hessian, static_cast<size_t>(nEffectiveParams * nEffectiveParams));
            utils::CUDA_ALLOC_AND_ZERO(&hessianLowTri, static_cast<size_t>(nEffectiveParams * nEffectiveParams));
        }

    private:
        float* lambda;
        float* deltaParams;

        float* dampeningFactors;

        float* gradients;
        float* hessian;
        float* hessianLowTri;
    };
}
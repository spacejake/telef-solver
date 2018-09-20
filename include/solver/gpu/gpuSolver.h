#pragma once

#include <cusolver_common.h>
#include <cusolverDn.h>
#include "util/cudautil.h"

#include "solver/solver.h"

namespace telef::solver {
    class GPUSolver : public Solver {
    public:
        using Ptr = std::shared_ptr<GPUSolver>;
        using ConstPtr = std::shared_ptr<const GPUSolver>;
        //TODO: Add params for Cublas handler. What does Cusolver need?
        GPUSolver():Solver(){
            initHandlers();
            initDeviceMemory();
        }

        GPUSolver(Options opts) : Solver(opts) {
            initHandlers();
            initDeviceMemory();
        }

        virtual ~GPUSolver(){
            cudaFree(error_d);
            cudaFree(down_factor_d);
            cudaFree(up_factor_d);

            cusolverDnDestroy(solver_handle);
        }

    protected:
        float *error_d;
        float *down_factor_d;
        float *up_factor_d;
        cusolverDnHandle_t solver_handle;

        virtual void initialize_solver();

        virtual void finalize_result();

        virtual float calcError(const float* residuals, const int nRes);

        virtual bool solveSystem(float *deltaParams, float* hessianLowTri,
                                 const float* hessians, const float* gradients,
                                 const int nParams);

        virtual void updateParams(float* newParams, const float* params, const float* newDelta, const int nParams);

        virtual void copyParams(float *destParams, const float *srcParams, const int nParams);

        // Step Functions
        virtual void updateHessians(float* hessians, float* step, const int nParams);

        virtual void stepUp(float* step, float* lambda);
        virtual void stepDown(float* step, float* lambda);

    private:
        void initHandlers() {
            cusolverDnCreate(&solver_handle);
        }

        void initDeviceMemory(){
            utils::CUDA_ALLOC_AND_ZERO(&error_d, static_cast<size_t>(1));
            utils::CUDA_ALLOC_AND_ZERO(&down_factor_d, static_cast<size_t>(1));
            utils::CUDA_ALLOC_AND_ZERO(&up_factor_d, static_cast<size_t>(1));
        }
    };
}
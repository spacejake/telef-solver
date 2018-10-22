#pragma once

#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include "solver/util/cudautil.h"

#include "solver/solver.h"

namespace telef::solver {
    class GPUSolver : public Solver {
    public:
        using Ptr = std::shared_ptr<GPUSolver>;
        using ConstPtr = std::shared_ptr<const GPUSolver>;
        //TODO: Add params for Cublas handler. What does Cusolver need?
        GPUSolver() : Solver(){
            initHandlers();
            initDeviceMemory();
        }

        GPUSolver(Options opts) : Solver(opts) {
            initHandlers();
            initDeviceMemory();
        }

        virtual ~GPUSolver(){
            cudaFree(down_factor_d);
            cudaFree(up_factor_d);

            cusolverDnDestroy(solver_handle);
            cublasDestroy_v2(cublasHandle);
        }

    protected:
        float *down_factor_d;
        float *up_factor_d;
        cusolverDnHandle_t solver_handle;
        cublasHandle_t cublasHandle;

        virtual void initialize_run(Problem::Ptr problem);

        virtual void finalize_result(Problem::Ptr problem);

        virtual float calcError(float *error, const float *residuals, const int nRes);

        virtual bool solveSystem(float *deltaParams, float* hessianLowTri,
                                 const float* hessians, const float* gradients,
                                 const int nParams);

        virtual void updateParams(float* newParams, const float* params, const float* newDelta, const int nParams);

        virtual void copyParams(float *destParams, const float *srcParams, const int nParams);

        // Step Functions
        virtual void
        updateHessians(float *hessians, float *dampeningFactors, float *lambda, const int nParams, bool goodStep);

        virtual void updateStep(float* lambda, bool goodStep);

    private:
        void initHandlers() {
            cusolverDnCreate(&solver_handle);
            if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Cublas could not be initialized");
            }
        }

        void initDeviceMemory(){
            SOLVER_CUDA_ALLOC_AND_ZERO(&down_factor_d, static_cast<size_t>(1));
            SOLVER_CUDA_ALLOC_AND_ZERO(&up_factor_d, static_cast<size_t>(1));
        }
    };
}
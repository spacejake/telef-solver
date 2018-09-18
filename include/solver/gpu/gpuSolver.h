#pragma once

#include "util/cudautil.h"

#include "solver/solver.h"

namespace telef::solver {
    class GPUSolver : public Solver {
    public:
        //TODO: Add params for Cublas handler. What does Cusolver need?
        GPUSolver():Solver(){
            initDeviceMemory();
        }

        GPUSolver(Options opts) : Solver(opts) {
            initDeviceMemory();
        }

        virtual ~GPUSolver(){
            cudaFree(error_d);
            cudaFree(down_factor_d);
            cudaFree(up_factor_d);
        }

    protected:
        float *error_d;
        float *down_factor_d;
        float *up_factor_d;

        virtual void initialize_solver();

        virtual float calcError(const float* residuals, const int nRes);

        virtual bool solveSystem(float *deltaParams,
                                 const float* hessians, const float* gradients,
                                 const int nRes, const int nParams){
            //TODO: implement using cuslover for Dense matrices cusolverDnCgesvd and cusolverDnCsytrf or cusolverDnCgeqrf
        }

        virtual void updateParams(float* newParams, const float* params, const float* newDelta, const int nParams);

        virtual void copyParams(float *destParams, const float *srcParams, const int nParams);

        // Step Functions
        virtual void updateHessians(float* hessians, float* step, const int nParams);

        virtual void stepUp(float* step, float* lambda);
        virtual void stepDown(float* step, float* lambda);

    private:
        void initDeviceMemory(){
            utils::CUDA_ALLOC_AND_ZERO(&error_d, static_cast<size_t>(1));
            utils::CUDA_ALLOC_AND_ZERO(&down_factor_d, static_cast<size_t>(1));
            utils::CUDA_ALLOC_AND_ZERO(&up_factor_d, static_cast<size_t>(1));
        }
    };
}
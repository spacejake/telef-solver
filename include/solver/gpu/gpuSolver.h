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
            cusolverDnDestroy(solver_handle);
            cublasDestroy_v2(cublasHandle);
        }

    protected:
        cusolverDnHandle_t solver_handle;
        cublasHandle_t cublasHandle;

        void computePredictedGain(float *predGain, float *lambda, float *daltaParams, float *gradient, int nParams);

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


        virtual bool evaluateGradient(float *gradient, int nParams, float tolerance);

        /**
         * convergence reached if
         *
         * ||x_new − x|| ≤ ε_2 (||x|| + ε_2) .
         *
         * or
         *
         * ||h_lm|| ≤ ε_2 (||x|| + ε_2)
         *
         * h_lm == deltas
         *
         * @param problem
         * @param tolerance
         * @return
         */
        virtual bool evaluateStep(Problem::Ptr problem, float tolerance);
        virtual void calcParams2Norm(float* params2Norm, Problem::Ptr problem);

        // TODO: assert lambda is not 0
        /**
         * Compute Gain Ratio
         * gainRatio = (error - newError)/(0.5*Delta^T (lambda * delta + -g))
         *
         * Gradient is computed as -g
         * hlm garuntieed not be 0 because we check before, lambda cannot be 0
         *
         * @param gainRatio
         * @param error
         * @param newError
         * @param lambda
         * @param deltaParams
         * @param gradient
         * @param nParams
         * @return
         */
        virtual float computeGainRatio(float *predGain,
                                       float error, float newError, float *lambda,
                                       float *deltaParams, float *gradient, int nParams);

        virtual void initializeLambda(float *lambda, float tauFactor, float *hessian, int nParams);

        virtual void updateLambda(float *lambda, float *failFactor, float *predGain, bool goodStep);

    private:
        void initHandlers() {
            cusolverDnCreate(&solver_handle);
            if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Cublas could not be initialized");
            }
        }

        void initDeviceMemory(){
        }

    };
}
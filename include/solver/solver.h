#pragma once

#include "solver/costFunction.h"
#include "solver/problem.h"

namespace telef::solver {


    enum class Status {
        UNKNOWN,
        RUNNING,
        CONVERGENCE,
        CONVERGENCE_FAILED,
        MAX_ITERATIONS,
    };

    using Options = struct Options {
        // lambda = tau * max(Diag(Initial_Hessian)) as initial Dampening factor,
        // initial_dampening_factor == tau
        // tau = 1e-6 is considard good if initial parameters are good approximations
        // use 1e-3 or 1 (Default) otherwise
        // See "Methods For Non-linear Least Square Problems", 2nd edition 2004, Madsen, Nielsen, and Tingleff
        // This implementation is based on the Lavenberg-Marquart method in the paper above.
        float initial_dampening_factor;

        float gain_ratio_threashold;

        // Termination targets
        int max_iterations;
        int max_num_consecutive_invalid_steps;
        float step_tolerance;
        float gradient_tolerance;

        bool verbose;
    };


    class Solver {
    public:
        using Ptr = std::shared_ptr<Solver>;
        using ConstPtr = std::shared_ptr<const Solver>;
        Options options;

        Solver(){
            // Decrease for good starting parameter guesses, for really good guesses use 1e-6
            options.initial_dampening_factor = 1;

            options.max_iterations = 500;
            options.max_num_consecutive_invalid_steps = 5;
            options.step_tolerance = 1e-8;
            options.gradient_tolerance = 1e-8;
            options.gain_ratio_threashold = 0; // Nielsen (1999)

            options.verbose = false;
        }

        Solver(Options opts) : options(opts) {}

        virtual ~Solver(){};

        // Result can be obtained via the given ParameterBlocks.getParameters(),
        // the user gives the solver the working memory space
        Status solve(Problem::Ptr problem, bool initProblem = true);
    protected:

        /**
         * Must be called each run to initialize the solver
         */
        virtual void initialize_run(Problem::Ptr problem) = 0;
        virtual void finalize_result(Problem::Ptr problem) = 0;

        /****Interface to be implemented for CPU and GPU implementations****/
        /**
         * Chi-squares calculation, sum(res^2)
         * We return to allow memory management to be inherited to allow user to decide
         * This is more overhead on the user, but provides a flexability that may be desired
         *
         * @param residuals
         * @return error, a single float value on host (float) representing the sum of squared residuals
         *
         */
        virtual float calcError(float *error, const float *residuals, const int nRes) = 0;


        virtual bool solveSystem(float *deltaParams, float* hessianLowTri,
                                 const float* hessians, const float* gradients,
                                 const int nParams) = 0;

        virtual void updateParams(float* newParams, const float* params, const float* newDelta, const int nParams) = 0;
        virtual void copyParams(float *destParams, const float *srcParams, const int nParams) = 0;

        // Step Functions
        virtual void
        updateHessians(float *hessians, float *dampeningFactors, float *lambda, const int nParams, bool goodStep) = 0;

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
        virtual bool evaluateStep(Problem::Ptr problem, float tolerance) = 0;

//        //TODO: functions to help evaluate convergence??
//        virtual bool evaluateStep() = 0;
//        virtual bool evaluateConvergence() = 0;

        /**
         * sum(gradient) < tolerance
         * @param gradient
         * @param nParams
         * @param tolerance, must be grater than 0
         * @return True if sum(Gradient) is below tolerance
         */
        virtual bool evaluateGradient(float &inf_norm_grad, float *gradient, int nParams, float tolerance) = 0;



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
         * @return CPU copied version of Gainratio
         */
        virtual float computeGainRatio(float *predGain,
                                       float error, float newError, float *lambda,
                                       float *deltaParams, float *gradient, int nParams) = 0;

        /**
         * lambda = tau * max(Diag(Initial_Hessian))
         *
         * @param lambda
         * @param tauFactor
         * @param hessian
         * @param nParams
         */
        virtual void initializeLambda(float *lambda, float tauFactor, float *hessian, int nParams) = 0;

        /**
         *  if (good_iteration) {
         *      μ := μ ∗ max{ 1/3, 1 − (2*gainRatio − 1)^3 }; ν := 2
         *  } else {
         *      μ := μ ∗ ν; ν := 2 ∗ ν
         *  }
         *
         *  ν = Consecutive Failure Factor (failFactor)
         * @param lambda
         * @param failFactor
         * @param predGain
         * @param goodStep
         */
        virtual void updateLambda(float *lambda, float *failFactor, float *predGain, bool goodStep) = 0;

        virtual void calcParams2Norm(float *params2Norm, Problem::Ptr problem) = 0;
    };
}
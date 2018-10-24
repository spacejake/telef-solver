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
        // use 1e-3 or 1 otherwise
        // See "Methods For Non-linear Least Square Problems", 2nd edition 2004, Madsen, Nielsen, and Tingleff
        // This implementation is based on the Lavenberg-Marquart method in the paper above.
        float initial_dampening_factor;

        // Termination targets
        int max_iterations;
        int max_num_consecutive_invalid_steps;
        float error_change_tolerance;
        float gradient_tolerance;

        bool verbose;
    };


    class Solver {
    public:
        using Ptr = std::shared_ptr<Solver>;
        using ConstPtr = std::shared_ptr<const Solver>;
        Options options;

        Solver(){
            options.initial_dampening_factor = 1e-3; // for good starting parameter guesses use 1e-6.

            options.max_iterations = 100;
            options.max_num_consecutive_invalid_steps = 5;
            options.error_change_tolerance = 1e-8;
            options.gradient_tolerance = 1e-8;

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

        virtual void updateStep(float* lambda, bool goodStep) = 0;

//        //TODO: functions to help evaluate convergence??
//        virtual bool evaluateStep() = 0;
//        virtual bool evaluateConvergence() = 0;
    };
}
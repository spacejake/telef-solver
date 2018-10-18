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
        float lambda_initial;
        float step_up;
        float step_down;
        float target_error_change;
        int max_iterations;
        bool verbose;
    };


    class Solver {
    public:
        using Ptr = std::shared_ptr<Solver>;
        using ConstPtr = std::shared_ptr<const Solver>;
        Options options;

        Solver(){
            options.lambda_initial = 1e-2;
            options.max_iterations = 1000;
            options.step_up = 10;
            options.step_down = 10;
            options.target_error_change = 1e-4;
            options.verbose = false;
        }

        Solver(Options opts) : options(opts) {}

        virtual ~Solver(){};

        // Result can be obtained via the given ParameterBlocks.getParameters(),
        // the user gives the solver the working memory space
        Status solve(Problem::Ptr problem);
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
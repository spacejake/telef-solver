#pragma once

#include "solver/costFunction.h"
#include "solver/residualFunction.h"

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
        Options options;

        Solver(){
            options.lambda_initial = 0.1;
            options.max_iterations = 10000;
            options.step_up = 10;
            options.step_down = 10;
            options.target_error_change = 0.01;
            options.verbose = false;
        }

        Solver(Options opts) : options(opts) {}

        virtual ~Solver(){};

        void addResidualFunction(ResidualFunction::Ptr resFunc_,
                                 std::vector<float*> initialParams_){
            residualFuncs.push_back(resFunc_);

        }

        // Result can be obtained via the given ParameterBlocks.getResultParameters(),
        // the user gives the solver the working memory space
        Status solve();
    protected:

        virtual void initialize_solver() = 0;

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
        virtual float calcError(const float* residuals, const int nRes) = 0;


        virtual bool solveSystem(float *deltaParams,
                                 const float* hessians, const float* gradients,
                                 const int nRes, const int nParams) = 0;

        virtual void updateParams(float* params, const float* newDelta, const int nParams) = 0;
        virtual void copyParams(float *dest_Params, const float *src_params, const int nParams) = 0;

        // Step Functions
        virtual void updateHessians(float* hessians, float* step, const int nParams) = 0;
        virtual void stepUp(float* step, float* lambda) = 0;
        virtual void stepDown(float* step, float* lambda) = 0;

//        //TODO: functions to help evaluate convergence??
//        virtual bool evaluateStep() = 0;
//        virtual bool evaluateConvergence() = 0;

    protected:
        std::vector<ResidualFunction::Ptr> residualFuncs;
        Status status;
    };
}
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#include "solver/solver.h"

using namespace std;
using namespace telef::solver;

Status Solver::solve(Problem::Ptr problem, bool initProblem) {
    auto residualFuncs = problem->getResidualFunctions();

    if (residualFuncs.size() == 0) {
        throw std::invalid_argument("Problem must have a ResidualFunction");
    }

    Status status = Status::RUNNING;

    // Initialize
    if (initProblem) {
        problem->initialize();
    }
    initialize_run(problem);

    //loop through each cost function, initialize all memory with results from given starting params
    problem->evaluate(true);

    float init_error = 0;
    for (auto resFunc : residualFuncs) {
        auto resBlock = resFunc->getResidualBlock();
        float block_error = calcError(resBlock->getWorkingError(), resBlock->getResiduals(), resBlock->numResiduals());
        resBlock->setError(block_error);
        init_error += block_error;
    }

    if (evaluateGradient(problem->getGradient(), options.gradient_tolerance)){
        status = Status::CONVERGENCE;
        if (options.verbose) {
            std::stringstream logmsg;
            logmsg << "Convergence occured in initial step.";
            std::cout << logmsg.str() << std::endl;
        }
    } else {
        // lambda = tau * max(Diag(Initial_Hessian))
        initializeLambda(problem->getLambda(), options.initial_dampening_factor, problem->getHessian());
    }

    // outerIter and innerIter is for reporting how many iterations until converging params found
    int outerIter = 1;
    int innerIter = 0;

    float error = init_error;
    // Delta error, between current params and new params we step too
    float iterDerr = 0;
    bool prev_good_iteration = false;

    int consecutive_invalid_steps = 0;
    int iter;
    for (iter = 0; iter < options.max_iterations && status == Status::RUNNING; ++iter) {

        float newError = 0;
        bool good_step = true; // Determine if all steps are good across parameter and residual blocks
        bool good_iteration = false; // Is this iteration good (better fit than best fit)

        // TODO: Parallelize?
        //if good_step, update 1+lambda
        //else, update with (1 + lambda * up) / (1 + lambda)
        updateHessians(problem->getHessian(), problem->getDampeningFactors(), problem->getLambda(),
                       problem->numEffectiveParams(), prev_good_iteration);

        // Check if decompsition results in a symmetric positive-definite matrix
        /* TODO: Use steepest descent for non positive-definite matrixies (where solution is not guaranteed to be downhill )
         * if F00(x) is positive definite
         *      h := h_n (Cholesky decomposition)
         * else
         *      h := h_sd (Steepest descent Direction, h_sd = −F'(x) or -g(x).)
         * x := x + αh
         */
        // Solves delta = -(H(x) + lambda * I)^-1 * g(x), x+1 = x + delta
        bool solveSystemSuccess = solveSystem(problem->getDeltaParameters(), problem->getHessianLowTri(),
                problem->getHessian(), problem->getGradient(),
                problem->numEffectiveParams());

        // If the system of equations failed to be evaluated with current step, make another step and try again.
        if (solveSystemSuccess) {
            // Update Params
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                auto paramBlocks = resBlock->getParameterBlocks();
                // Compute next step for each parameter
                for (ParameterBlock::Ptr paramBlock : paramBlocks) {
                    if (!paramBlock->isShared()) {
                        updateParams(paramBlock->getParameters(),
                                     paramBlock->getBestParameters(),
                                     problem->getDeltaParameters() + paramBlock->getOffset(),
                                     paramBlock->numParameters());
                    }
                }
            }

            // Evaluate step and new params
            // if prev error was derr <= 0, don't do jacobian recalc
            problem->evaluate(prev_good_iteration);
            float problemError = 0;
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                float blockError = calcError(resBlock->getWorkingError(), resBlock->getResiduals(), resBlock->numResiduals());
                resBlock->setError(blockError);
                problemError += blockError;
            }

            //TODO: Add Loss Funciton for each Resdidual Function, 0.5 * Loss(chi-squared-error or bloack error)
            newError += 0.5 * problemError; //Compute Cost: 0.5 * chi-squared-error, as ceres does

            //printf("BlockError:%.4f\n", blockError);

            // accumulated good steps, 1 bad step means retry with new step on all?
            if( evaluateStep(problem, options.error_change_tolerance) ) {
                status = Status::CONVERGENCE;
            }

            good_step = true;
            consecutive_invalid_steps = 0;
        } else {
            good_step = false;
            if (consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps) {
                status = Status::CONVERGENCE_FAILED;
            }

            consecutive_invalid_steps++;
        }

        if(good_step && status == Status::RUNNING) {
            //TODO: Convert to use Gain Ratio
//            iterDerr = newError - error;
            gainRatio = computeGainRatio(problem->getGainRatio(),
                    error, newError,
                    problem->getDeltaParameters(), problem->getGradient());

            good_iteration = gainRatio > 0;
        } else {
            lambda *= step_up_factor;

        }


        if (good_iteration) {
            error = newError;

            //Copy Params that result in improved error
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                auto paramBlocks = resBlock->getParameterBlocks();
                for (ParameterBlock::Ptr paramBlock : paramBlocks) {
                    if (!paramBlock->isShared()) {
                        copyParams(paramBlock->getBestParameters(),
                                   paramBlock->getParameters(), paramBlock->numParameters());
                    }
                }
            }

            //TODO: Check Sum of gradients, gradients near zero means minimum likly found. As Ceres Does
            // Convergence achieved
            if (evaluateGradient()) {
                status = Status::CONVERGENCE;
            }
        }

        if (options.verbose) {
            std::stringstream logmsg;

            if (prev_good_iteration || iter == 0) {
                logmsg << "OuterIter: " << std::to_string(outerIter);
            }
            else {
                logmsg << "\tInnerIter: " << std::to_string(innerIter);
            }
            logmsg << std::scientific;

            logmsg << "\t\tError: " << error;
            logmsg << "\t\tChange: " << iterDerr;

            std::cout << logmsg.str() << std::endl;

            // Counters are for logging
            if (good_iteration) {
                outerIter++;
                innerIter = 0;
            } else {
                innerIter++;
            }
        }

        if (status == Status::RUNNING){
            // Setup next iteration
            updateStep(problem->getLambda(), good_iteration);
            prev_good_iteration = good_iteration;
        }
    }

    finalize_result(problem);

    if (options.verbose) {
        std::stringstream logmsg;
        string header = "Final Result:";
        logmsg << header << std::endl;
        logmsg << "\tTotal Iterations: " << iter << std::endl;

        logmsg << std::scientific;
        logmsg << "\tInitial Error:\t"  << init_error << std::endl;
        logmsg << "\tFinal Error:\t"  << error << std::endl;
        logmsg << "\tTotal Change:\t" << error-init_error << std::endl;

        std::cout << std::endl << logmsg.str() << std::endl;
    }

    if (iter == options.max_iterations) {
        status = Status::MAX_ITERATIONS;
    } else if (Status::CONVERGENCE !=status) {
        // TODO: when to consider convergence failed, when inner loop iteration is too large?
        status = Status::CONVERGENCE_FAILED;
    }

    return status;
}
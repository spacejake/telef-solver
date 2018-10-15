#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#include "solver/solver.h"

using namespace std;
using namespace telef::solver;

Status Solver::solve() {
    if (residualFuncs.size() == 0) {
        throw std::invalid_argument("Solver must have a ResidualFunction");
    }

    Status status = Status::RUNNING;

    initialize_run();

    //loop through each cost function, initialize all memory with results from given starting params
    float init_error = 0;
    for (auto resFunc : residualFuncs) {
        ResidualBlock::Ptr resBlock = resFunc->evaluate(true);
        float block_error = calcError(resBlock->getWorkingError(), resBlock->getResiduals(), resBlock->numResiduals());
        resBlock->setError(block_error);
        init_error += block_error;
    }


    // outerIter and innerIter is for reporting how many iterations until converging params found
    int outerIter = 1;
    int innerIter = 0;

    float error = init_error;
    // Delta error, between current params and new params we step too
    float iterDerr = 0;
    bool prev_good_iteration = false;

    //TODO: Use just one
    float* lambda = residualFuncs[0]->getResidualBlock()->getLambda();

    int iter;
    for (iter = 0; iter < options.max_iterations; ++iter) {

        float newError = 0;
        bool good_step = true; // Determine if all steps are good across parameter and residual blocks
        bool good_iteration = false; // Is this iteration good (better fit than best fit)

        for (auto resFunc : residualFuncs) {
            auto resBlock = resFunc->getResidualBlock();
            auto paramBlocks = resBlock->getParameterBlocks();

            bool solveSystemSuccess = false;

            // TODO: Parallelize?
            // Compute next step for each parameter
            for (ParameterBlock::Ptr paramBlock : paramBlocks) {
                //if good_step, update 1+lambda
                //else, update with (1 + lambda * up) / (1 + lambda)
                updateHessians(paramBlock->getHessians(), paramBlock->getDampeningFactors(), lambda,
                               paramBlock->numParameters(), prev_good_iteration);

                // Check if decompsition results in a symmetric positive-definite matrix
                /* TODO: Use steepest descent for non positive-definite matrixies (where solution is not guaranteed to be downhill )
                 * if F00(x) is positive definite
                 *      h := h_n (Cholesky decomposition)
                 * else
                 *      h := h_sd (Steepest descent Direction, h_sd = −F'(x) or -g(x).)
                 * x := x + αh
                 */
                // Solves delta = -(H(x) + lambda * I)^-1 * g(x), x+1 = x + delta
                solveSystemSuccess = solveSystem(paramBlock->getDeltaParameters(), paramBlock->getHessianLowTri(),
                                                 paramBlock->getHessians(), paramBlock->getGradients(),
                                                 paramBlock->numParameters());

                // Check if decomoposition worked, if not, then iterate again with new step (good_step == false, break;)
                if (!solveSystemSuccess) {
                    break;
                }

                updateParams(paramBlock->getWorkingParameters(), paramBlock->getParameters(),
                             paramBlock->getDeltaParameters(), paramBlock->numParameters());
            }


            // If the system of equations failed to be evaluated with current step, make another step and try again.
            if (solveSystemSuccess) {
                // if derr <= 0, don't do jacobian recalc
                ResidualBlock::Ptr resBlock = resFunc->evaluate(prev_good_iteration);
                float blockError = calcError(resBlock->getWorkingError(), resBlock->getResiduals(), resBlock->numResiduals());

                //TODO: Add Loss Funciton for each Resdidual Function, 0.5 * Loss(chi-squared-error or bloack error)
                newError += 0.5 * blockError; //Compute Cost: 0.5 * chi-squared-error, as ceres does

                //printf("BlockError:%.4f\n", blockError);

                // accumulated good steps, 1 bad step means retry with new step on all?
                good_step &= true;
            } else {
                good_step = false;
            }

        } //END: Residual Blocks

        if(good_step) {
            //TODO: Convert to use Dog leg method or classical LM method
            iterDerr = newError - error;
            good_iteration = iterDerr <= 0;
        }


        if (good_iteration) {
            error = newError;

            //Copy Params that result in improved error
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                auto paramBlocks = resBlock->getParameterBlocks();
                for (ParameterBlock::Ptr paramBlock : paramBlocks) {
                    copyParams(paramBlock->getParameters(),
                               paramBlock->getWorkingParameters(), paramBlock->numParameters());
                }
            }

            //TODO: Check Sum of gradients, gradients near zero means minimum likly found. As Ceres Does
            // Convergence achieved
            if (-iterDerr < options.target_error_change) {
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

        if (status != Status::RUNNING){
            // Stop we have converged or failed
            break;
        }

        // Setup next iteration
        updateStep(lambda, good_iteration);
        prev_good_iteration = good_iteration;
    }

    finalize_result();

    if (options.verbose) {
        std::stringstream logmsg;
        string header = "Final Result:";
        logmsg << header << std::endl;
        logmsg << "\tTotal Iterations: " << iter << std::endl;

        logmsg << std::scientific;
        logmsg << "\tError: " << error << std::endl;

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
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>
using Clock=std::chrono::high_resolution_clock;

#include "solver/util/profile.h"
#include "solver/solver.h"

using namespace std;
using namespace telef::solver;



Status Solver::solve(Problem::Ptr problem, bool initProblem) {
    long residual_Ttime = 0;
    long derivative_Ttime = 0;
    long linSolver_Ttime = 0;
    long solver_Ttime = 0;
    PROFILE_DECLARE(init);
    PROFILE_DECLARE(solver);
    PROFILE_DECLARE(linear_solver);

    auto residualFuncs = problem->getResidualFunctions();

    if (residualFuncs.size() == 0) {
        throw std::invalid_argument("Problem must have a ResidualFunction");
    }

    Status status = Status::RUNNING;
    PROFILE_START(solver);


    // Initialize
    PROFILE_START(init);
    if (initProblem) {
        problem->initialize();
    }
    initialize_run(problem);
    PROFILE_END(init);
    auto init_time = PROFILE_GET(init);

    //loop through each cost function, initialize all memory with results from given starting params
    residual_Ttime += PROFILE(
            problem->evaluate(););

    derivative_Ttime += PROFILE(
            problem->computeDerivatives(););

    float init_error = 0;
    for (auto resFunc : residualFuncs) {
        auto resBlock = resFunc->getResidualBlock();
        float block_error = calcError(resBlock->getWorkingError(), resBlock->getResiduals(), resBlock->numResiduals());
        resBlock->setError(block_error);
        init_error += block_error;
    }

    if (evaluateGradient(problem->getGradient(), problem->numEffectiveParams(), options.gradient_tolerance)){
        status = Status::CONVERGENCE;
        if (options.verbose) {
            std::stringstream logmsg;
            logmsg << "CONVERGENCE: occured in initial step. Gradient less than "
                      "tolerace:" << options.gradient_tolerance;
            std::cout << logmsg.str() << std::endl;
        }
    } else {
        // lambda = tau * max(Diag(Initial_Hessian))
        initializeLambda(problem->getLambda(), options.initial_dampening_factor,
                problem->getHessian(), problem->numEffectiveParams());
    }

    // outerIter and innerIter is for reporting how many iterations until converging params found
    int outerIter = 1;
    int innerIter = 1;

    float error = init_error;
    // Delta error, between current params and new params we step too
    float iterDerr = 0;
    bool prev_good_iteration = false;

    int consecutive_invalid_steps = 0;
    int iter = 0;
    while (status == Status::RUNNING && iter++ < options.max_iterations) {

        float newError = 0;
        bool good_step = true; // Determine if all steps are good across parameter and residual blocks
        bool good_iteration = false; // Is this iteration good (better fit than best fit)

        // TODO: Parallelize?
        //if good_step, update 1+lambda
        //else, update with (1 + lambda * up) / (1 + lambda)
        updateHessians(problem->getHessian(), problem->getDampeningFactors(), problem->getLambda(),
                       problem->numEffectiveParams(), prev_good_iteration);

        // Solves delta = -(H(x) + lambda * I)^-1 * g(x), x+1 = x + delta

        PROFILE_START(linear_solver);
        bool solveSystemSuccess = solveSystem(problem->getDeltaParameters(), problem->getHessianLowTri(),
                problem->getHessian(), problem->getGradient(),
                problem->numEffectiveParams());
        PROFILE_END(linear_solver);
        linSolver_Ttime += PROFILE_GET(linear_solver);

        // Check if decompsition results in a symmetric positive-definite matrix
        // If the system of equations failed to be evaluated with current step, make another step and try again.

        // convergence reached if ||h_lm|| ≤ ε_2 (||x|| + ε_2)
        if( solveSystemSuccess && evaluateStep(problem, options.step_tolerance) ) {
            status = Status::CONVERGENCE;

            if (options.verbose) {
                std::stringstream logmsg;
                logmsg << "CONVERGENCE: Cannot calculate step with magnitude greater than "
                          "tolerace:" << options.step_tolerance;
                std::cout << logmsg.str() << std::endl;
            }
            // Save parameters?
            good_iteration = false;
        } else if (solveSystemSuccess) {
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
            residual_Ttime += PROFILE(
                    problem->evaluate(););

            float problemError = 0;
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                float blockError = calcError(resBlock->getWorkingError(), resBlock->getResiduals(), resBlock->numResiduals());
                resBlock->setError(blockError);
                problemError += blockError;
            }

            //TODO: Add Loss Funciton for each Resdidual Function, 0.5 * Loss(chi-squared-error or bloack error)
            newError += 0.5 * problemError; //Compute Cost: 0.5 * chi-squared-error, as ceres does
            iterDerr = newError - error;
            //printf("BlockError:%.4f\n", blockError);

            /* Compute Gain Ratio
             * gainRatio = (error - newError)/(0.5*Delta^T (lambda * delta + -g))
             *
             * Gradient is computed as -g
             * hlm garuntieed not be 0 because we check above, lambda cannot be 0
             *
             */
            float gainRatio = computeGainRatio(problem->getPredictedGain(),
                                               error, newError, problem->getLambda(),
                                               problem->getDeltaParameters(), problem->getGradient(),
                                               problem->numEffectiveParams());

            //printf("GainRatio:%.4f\n", gainRatio);

            good_iteration = gainRatio > options.gain_ratio_threashold;

            consecutive_invalid_steps = 0;
        } else {
            good_iteration = false;
            if (consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps) {
                status = Status::CONVERGENCE_FAILED;
//                printf("CONVERGENCE_FAILED: Max consecutive bad steps reached");
            }

            iterDerr = 0.0f;
            consecutive_invalid_steps++;
        }

        if (good_iteration) {
            error = newError;

            //save Params that result in improved error
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

            // Recompute derivatives (jacobian, gradients, Hessian) for best found parameters
            derivative_Ttime += PROFILE(
                    problem->computeDerivatives(););

            //TODO: Check Sum of gradients, gradients near zero means minimum likly found. As Ceres Does
            // Convergence achieved?
            if (evaluateGradient(problem->getGradient(), problem->numEffectiveParams(), options.gradient_tolerance)) {
                status = Status::CONVERGENCE;
                if (options.verbose) {
                    std::stringstream logmsg;
                    logmsg << "CONVERGENCE: Gradient less than "
                              "tolerace:" << options.gradient_tolerance;
                    std::cout << logmsg.str() << std::endl;
                }
            } else {
                // for next iteration, we should recalculate the 2-norm of our best fitted parameters
                calcParams2Norm(problem->getParams2Norm(), problem);
            }
        }

        if (status == Status::RUNNING){
            // Setup next iteration
            /**
             *  if (good_iteration) {
             *      μ := μ ∗ max{ 1/3, 1 − (2*gainRatio − 1)^3 }; ν := 2
             *  } else {
             *      μ := μ ∗ ν; ν := 2 ∗ ν
             *  }
             *
             *  ν = Consecutive Failure Factor (failFactor)
             */
            updateLambda(problem->getLambda(), problem->getFailFactor(), problem->getPredictedGain(), good_iteration);

            prev_good_iteration = good_iteration;
        }

        // Verbose logging
        if (options.verbose) {
            std::stringstream logmsg;

            if (prev_good_iteration || iter == 1) {
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
                innerIter = 1;
            } else {
                innerIter++;
            }
        }
    }

    PROFILE_END(solver);
    solver_Ttime = PROFILE_GET(solver);

    auto post_time = PROFILE(
            finalize_result(problem););

    if (options.verbose) {
        std::stringstream logmsg;
        string header = "Final Result:";
        logmsg << header << std::endl;
        logmsg << "\tTotal Iterations: " << iter << std::endl;

        logmsg << std::scientific;
        logmsg << "\tInitial Error:\t"  << init_error << std::endl;
        logmsg << "\tFinal Error:\t"  << error << std::endl;
        logmsg << "\tTotal Change:\t" << error-init_error << std::endl;

        logmsg << std::defaultfloat << std::setprecision(4);
        logmsg << "\nTime: (in Seconds)" << std::endl;
        // Convert ns to seconds
        logmsg << "\tPreprocess:\t"
               << init_time * 1e-9
               << std::endl;
        logmsg << "\n\tResiduals:\t"
               << residual_Ttime * 1e-9
               << std::endl;
        logmsg << "\tDerivatives:\t"
               << derivative_Ttime * 1e-9
               << std::endl;
        logmsg << "\tLinear Solver:\t"
               << linSolver_Ttime * 1e-9
               << std::endl;
        logmsg << "\n\tPostprocess:\t"
               << post_time * 1e-9
               << std::endl;
        logmsg << "\tTotal:\t\t"
                << solver_Ttime * 1e-9
                << std::endl;
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



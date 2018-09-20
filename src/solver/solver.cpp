#include "solver/solver.h"

using namespace std;
using namespace telef::solver;

Status Solver::solve() {
    if (residualFuncs.size() == 0) {
        throw std::invalid_argument("Solver must have a ResidualFunction");
    }

    Status status = Status::RUNNING;

    initialize_solver();

    //TODO: loop through each cost function, for now we assume one
    ResidualFunction::Ptr resFunc = residualFuncs.at(0);

    // Initial evaluation with given params
    ResidualBlock::Ptr resBlock = resFunc->evaluate(true);
    float error = calcError(resBlock->getResiduals(), resBlock->numResiduals());

    const std::vector<ParameterBlock::Ptr> paramBlocks = resBlock->getParameterBlocks();

    // Good Step is used to indicate converging params found, and when to recompute jacobians
    bool good_step = false;

    // Delta error, between current params and new params we step too
    float derr = 0;

    // innerIter is for reporting how many iterations until converging params found
    int innerIter = 0;

    int iter;
    for (iter = 0; iter < options.max_iterations; ++iter) {
        bool solveSystemSuccess = false;
        // TODO: Parallelize?
        // Compute next step for each parameter
        for (ParameterBlock::Ptr paramBlock : paramBlocks) {
            //if good_step, update 1+lambda
            //else, update with (1 + lambda * up) / (1 + lambda)
            updateHessians(paramBlock->getHessians(), resBlock->getStep(), paramBlock->numParameters());

            // Check if decompsition results in a symmetric positive-definite matrix
            solveSystemSuccess = solveSystem(paramBlock->getDeltaParameters(), paramBlock->getHessianLowTri(),
                                             paramBlock->getHessians(), paramBlock->getGradients(),
                                             paramBlock->numParameters());

            // Check if decomoposition worked, if not, then iterate again with new step (good_step == false, break;)
            if (!solveSystemSuccess){
                break;
            }

            updateParams(paramBlock->getWorkingParameters(), paramBlock->getParameters(),
                         paramBlock->getDeltaParameters(), paramBlock->numParameters());
        }

        float newError = 0;
        // If the system of equations failed to be evaluated with current step, make another step and try again.
        if (solveSystemSuccess) {
            // if derr < 0, don't do jacobian recalc
            ResidualBlock::Ptr resBlock = resFunc->evaluate(good_step);

            newError = calcError(resBlock->getResiduals(), resBlock->numResiduals());

            derr = newError - error;
            good_step = derr <= 0;
        } else {
            good_step = false;
        }

        if (good_step) {
            //Evaluate Jacobians during next iteration
            for (ParameterBlock::Ptr paramBlock : paramBlocks) {
                copyParams(paramBlock->getParameters(),
                           paramBlock->getWorkingParameters(), paramBlock->numParameters());
            }

            error = newError;

            if(-derr < options.target_error_change){
                status = Status::CONVERGENCE;
                break;
            }

            /* stepdown (lambda*down)
             * mult = 1 + lambda;
             */
            stepDown(resBlock->getStep(), resBlock->getLambda());
            innerIter = 0;
        } else {
            /*
             * mult = (1 + lambda * up) / (1 + lambda);
             * stepup (lambda*up)
             */
            stepUp(resBlock->getStep(), resBlock->getLambda());
            innerIter++;
        }
    }

    finalize_result();

    if (iter == options.max_iterations) {
        status = Status::MAX_ITERATIONS;
    } else if (Status::CONVERGENCE !=status) {
        // TODO: when to consider convergence failed, when inner loop iteration is too large?
        status = Status::CONVERGENCE_FAILED;
    }

    return status;
}
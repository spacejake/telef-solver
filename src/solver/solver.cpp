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

    // loop through each cost function, initialize all memory with results from given starting params
    // Evaluate residuals r(x) and derivatives (J, H=J'J, g=J'r) at initial x. Cost f(x) = ½||r||².
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

    // First-order optimality: ||∇f||_∞ = ||J'r||_∞ small ⇒ near stationary point (necessary for min).
    float init_norm_inf_grad = 0;
    if (evaluateGradient(init_norm_inf_grad, problem->getGradient(), problem->numEffectiveParams(), options.gradient_tolerance)){
        status = Status::CONVERGENCE;
        if (options.verbose) {
            std::stringstream logmsg;
            logmsg << "CONVERGENCE: occured in initial step. Gradient less than "
                      "tolerace:" << options.gradient_tolerance;
            std::cout << logmsg.str() << std::endl;
        }
    } else {
        // Initialize LM damping: λ = τ * max(diag(H)). Ref: Madsen et al. "Methods for non-linear least squares".
        initializeLambda(problem->getLambda(), options.initial_dampening_factor,
                problem->getHessian(), problem->numEffectiveParams());

        if (options.verbose) {
            std::stringstream logmsg;
            logmsg << "Initial Gradient:" << init_norm_inf_grad;
            std::cout << logmsg.str() << std::endl;
        }
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
    float norm_inf_grad = init_norm_inf_grad;

    while (status == Status::RUNNING && iter++ < options.max_iterations) {

        float new_norm_inf_grad = norm_inf_grad;
        float change_norm_inf_grad = 0;
        float newError = 0;
        bool good_step = true; // Determine if all steps are good across parameter and residual blocks
        bool good_iteration = false; // Is this iteration good (better fit than best fit)

        // --- LM: Form damped normal equations (H + damping)*Δ = -g ---
        // Only for DAMPED_LM: we add λI or λ·diag(H) to H so the step is (H+damping)^{-1}(-g).
        // For DOGLEG we use raw H (with small diag floor) and a trust-region step instead.
        if (options.step_type == StepType::DAMPED_LM) {
            updateHessians(problem->getHessian(), problem->getDampeningFactors(), problem->getLambda(),
                          problem->numEffectiveParams(), prev_good_iteration,
                          options.damping_type, options.diag_floor_epsilon);
        }

        // --- Solve for the step Δ: (H + damping)*Δ = -g (LM) or dogleg step inside trust region ---
        // Theory: LM minimizes the local quadratic model m(Δ)=f(x)+g'Δ + ½Δ'HΔ; damping ensures
        // H+damping is pos. def. and controls step size. Ref: Madsen et al. "Methods for non-linear least squares".
        PROFILE_START(linear_solver);
        bool solveSystemSuccess = solveSystem(problem->getDeltaParameters(), problem->getHessianLowTri(),
                problem->getHessian(), problem->getGradient(),
                problem->numEffectiveParams(), problem->getScaleBuffer(),
                options.step_type, problem->getTrustRadius(), problem->getAuxBuffer());
        PROFILE_END(linear_solver);
        linSolver_Ttime += PROFILE_GET(linear_solver);

        // Cholesky can fail if matrix is not positive definite; then we retry with different λ next iteration.

        // --- Step-size convergence: ||Δ|| ≤ ε₂(||x||+ε₂) means we are not moving much (local minimum) ---
        // Ref: Madsen et al. termination criteria.
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
            // --- Tentative update: x_new = x + Δ (or manifold Plus for rotation, etc.) ---
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                auto paramBlocks = resBlock->getParameterBlocks();
                // Compute next step for each parameter
                for (ParameterBlock::Ptr paramBlock : paramBlocks) {
                    if (!paramBlock->isShared()) {
                        updateParams(paramBlock->getParameters(),
                                     paramBlock->getBestParameters(),
                                     problem->getDeltaParameters() + paramBlock->getOffset(),
                                     paramBlock->numParameters(),
                                     paramBlock.get());
                    }
                }
            }

            // --- Evaluate cost at x_new: f(x_new) = ½ Σ r_i^2 ---
            residual_Ttime += PROFILE(
                    problem->evaluate(););

            float problemError = 0;
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                float blockError = calcError(resBlock->getWorkingError(), resBlock->getResiduals(), resBlock->numResiduals());
                resBlock->setError(blockError);
                problemError += blockError;
            }

            // Cost = ½ * chi-squared (sum of squared residuals), as in Ceres and standard LM.
            newError += 0.5 * problemError;
            iterDerr = newError - error;

            // --- Step acceptance: trust-region ratio ρ = (actual reduction) / (predicted reduction) ---
            // Theory: m(Δ) = f(x) + g'Δ + ½Δ'HΔ. Predicted reduction = m(0) - m(Δ) = -g'Δ - ½Δ'HΔ.
            // ρ = (f(x)-f(x+Δ)) / (m(0)-m(Δ)). If ρ > threshold (e.g. 0.25) the quadratic model is
            // trustworthy and we accept; else we reject and increase damping / shrink trust region.
            if (options.use_rho_accept) {
                float pred_red = computeModelReduction(problem->getDeltaParameters(), problem->getGradient(),
                        problem->getHessian(), problem->numEffectiveParams(), problem->getAuxBuffer());
                float actual_red = error - newError;
                float rho = (pred_red > 1e-14f) ? (actual_red / pred_red) : 0.f;
                good_iteration = rho > options.rho_accept_threshold;
            } else {
                // Legacy: gain ratio from LM (Nielsen 1999 style), denominator = 0.5*Δ'(λΔ - g).
                float gainRatio = computeGainRatio(problem->getPredictedGain(),
                        error, newError, problem->getLambda(),
                        problem->getDeltaParameters(), problem->getGradient(),
                        problem->numEffectiveParams());
                good_iteration = gainRatio > options.gain_ratio_threashold;
            }

            consecutive_invalid_steps = 0;
        } else {
            good_iteration = false;
            if (consecutive_invalid_steps >= options.max_num_consecutive_invalid_steps) {
                status = Status::CONVERGENCE_FAILED;

                if (options.verbose) {
                    std::stringstream logmsg;
                    logmsg << "CONVERGENCE_FAILED: Max consecutive bad steps "
                              "reached " << options.max_num_consecutive_invalid_steps;
                    std::cout << logmsg.str() << std::endl;
                }
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

            //TODO: Check Sum of gradients, gradients near zero means minimum likely found. As Ceres Does
            // Convergence achieved?
            if (evaluateGradient(new_norm_inf_grad, problem->getGradient(), problem->numEffectiveParams(), options.gradient_tolerance)) {
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
            change_norm_inf_grad = new_norm_inf_grad - norm_inf_grad;
            norm_inf_grad = new_norm_inf_grad;
        }

        if (status == Status::RUNNING){
            // --- Update damping (LM) or trust radius (dogleg) for next iteration ---
            // LM: if good step, decrease λ (Nielsen: μ := μ * max{1/3, 1-(2ρ-1)³}, ν:=2);
            //     if bad step, increase λ (μ := μ*ν, ν := 2ν). So next solve uses (H + new_λ)*Δ = -g.
            updateLambda(problem->getLambda(), problem->getFailFactor(), problem->getPredictedGain(), good_iteration);
            // Dogleg: increase trust radius on accept, decrease on reject (standard trust-region update).
            if (options.step_type == StepType::DOGLEG)
                updateTrustRadius(problem, good_iteration);

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
            logmsg << "\t\tGradient: " << new_norm_inf_grad;
            logmsg << "\t\tGradient Change: " << change_norm_inf_grad;

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

        logmsg << std::endl << "\tInitial Gradient:\t"  << init_norm_inf_grad << std::endl;
        logmsg << "\tFinal Gradient:\t"  << norm_inf_grad << std::endl;
        logmsg << "\tTotal Change:\t" << norm_inf_grad-init_norm_inf_grad << std::endl;

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



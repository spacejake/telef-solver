#pragma once

#include "solver/costFunction.h"
#include "solver/parameterBlock.h"
#include "solver/problem.h"

namespace telef::solver {


    enum class Status {
        UNKNOWN,
        RUNNING,
        CONVERGENCE,
        CONVERGENCE_FAILED,
        MAX_ITERATIONS,
    };

    /** Damping: add λI (classic LM) or λ·diag(H) (Marquardt-style). */
    enum class DampingType {
        LAMBDA_I,       ///< H + λI
        LAMBDA_DIAG_H,  ///< H + λ·diag(H), diag(H) ← max(diag(H), diag_floor_epsilon)
    };

    /** Step computation: damped LM solve or dogleg trust-region. */
    enum class StepType {
        DAMPED_LM,  ///< Solve (H + damping)*Δ = -g
        DOGLEG,     ///< Trust-region dogleg: combine Gauss-Newton and steepest descent
    };

    using Options = struct Options {
        // lambda = tau * max(Diag(Initial_Hessian)) as initial damping
        float initial_dampening_factor;

        float gain_ratio_threashold;  ///< legacy: accept if gain_ratio > this when !use_rho_accept

        /// Use ρ = (f(x)-f(x+Δ)) / (m(0)-m(Δ)) for accept/reject; if true, accept when ρ > rho_accept_threshold
        bool use_rho_accept;
        float rho_accept_threshold;   ///< e.g. 0.25; accept step if ρ > this

        DampingType damping_type;
        float diag_floor_epsilon;     ///< For LAMBDA_DIAG_H: diag(H) ← max(diag(H), ε)

        StepType step_type;
        float initial_trust_radius;   ///< For DOGLEG: initial trust region radius (e.g. 1.0 or from ||Δ_gn||)

        // Termination
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
            options.initial_dampening_factor = 1.f;
            options.gain_ratio_threashold = 0.f;
            options.use_rho_accept = true;
            options.rho_accept_threshold = 0.25f;
            options.damping_type = DampingType::LAMBDA_DIAG_H;
            options.diag_floor_epsilon = 1e-6f;
            options.step_type = StepType::DAMPED_LM;
            options.initial_trust_radius = 1.f;

            options.max_iterations = 500;
            options.max_num_consecutive_invalid_steps = 5;
            options.step_tolerance = 1e-8f;
            options.gradient_tolerance = 1e-8f;
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


        virtual bool solveSystem(float *deltaParams, float* hessianLowTri, const float* hessians, const float* gradients,
                                 const int nParams, float* scaleBuffer = nullptr,
                                 StepType stepType = StepType::DAMPED_LM, float* trustRadius = nullptr, float* auxBuffer = nullptr) = 0;

        /** If paramBlock has a LocalParameterization, uses it (host callback); else Euclidean update on GPU. */
        virtual void updateParams(float* newParams, const float* params, const float* newDelta, const int nParams, ParameterBlock* paramBlock = nullptr) = 0;
        virtual void copyParams(float *destParams, const float *srcParams, const int nParams) = 0;

        // Step Functions
        virtual void updateHessians(float *hessians, float *dampeningFactors, float *lambda, const int nParams, bool goodStep,
                                    DampingType dampingType = DampingType::LAMBDA_I, float diagFloorEpsilon = 1e-6f) = 0;

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

        /** Predicted reduction m(0)-m(Δ) = gradient''*Δ - 0.5*Δ''*H*Δ for ρ = actual_red / pred_red. auxBuffer (nParams) used for H*delta when non-null. */
        virtual float computeModelReduction(float *deltaParams, float *gradient, const float *hessianDamped, int nParams, float* auxBuffer = nullptr) = 0;

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

        /** For DOGLEG: update trust radius (e.g. increase on good step, decrease on bad). Default no-op. */
        virtual void updateTrustRadius(Problem::Ptr problem, bool goodStep) { (void)problem; (void)goodStep; }

        virtual void calcParams2Norm(float *params2Norm, Problem::Ptr problem) = 0;
    };
}
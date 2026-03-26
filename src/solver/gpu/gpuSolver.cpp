#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "solver/gpu/cuda/cu_solver.h"

#include "solver/util/cudautil.h"
#include "solver/gpu/gpuSolver.h"
#include "solver/gpu/gpuProblem.h"
#include "solver/gpu/gpuParameterBlock.h"

using namespace telef::solver;

float GPUSolver::calcError(float *error, const float *residuals, const int nRes) {
    // TODO: Use Residual based error, use error_d as total error for all residuals
    float error_h = 0;
    //Reset to 0
    SOLVER_CUDA_CHECK(cudaMemset(error,0, sizeof(float)));
    calc_error(error, residuals, nRes);

    SOLVER_CUDA_CHECK(cudaMemcpy(&error_h, error, sizeof(float), cudaMemcpyDeviceToHost));
    return error_h;
}


void GPUSolver::initialize_run(Problem::Ptr problem) {
    // Initialize step factors
    auto residualFuncs = problem->getResidualFunctions();
    std::shared_ptr<GPUProblem> gpuProblem = std::dynamic_pointer_cast<GPUProblem>(problem);
    if (gpuProblem != nullptr) {
        gpuProblem->setCublasHandle(cublasHandle);
    }
    // Trust-region radius for dogleg: initial value from options; updated each iteration on accept/reject.
    float* tr = problem->getTrustRadius();
    if (tr) {
        float r = options.initial_trust_radius;
        SOLVER_CUDA_CHECK(cudaMemcpy(tr, &r, sizeof(float), cudaMemcpyHostToDevice));
    }

    //Initialize Total Error
    SOLVER_CUDA_CHECK(cudaMemset(problem->getWorkingError(), 0, sizeof(float)));
    SOLVER_CUDA_CHECK(cudaMemset(problem->getPredictedGain(), 0, sizeof(float)));
    SOLVER_CUDA_CHECK(cudaMemset(problem->getParams2Norm(), 0, sizeof(float)));

    float initLambda = 1e-1;
    float initFailFactor = 2;
    SOLVER_CUDA_CHECK(cudaMemcpy(problem->getLambda(), &initLambda, sizeof(float), cudaMemcpyHostToDevice));
    SOLVER_CUDA_CHECK(cudaMemcpy(problem->getFailFactor(), &initFailFactor, sizeof(float), cudaMemcpyHostToDevice));

    // Initialize Dampening Factors
    float* dampeningFactors = problem->getDampeningFactors();
    SOLVER_CUDA_CHECK(cudaMemset(dampeningFactors, 0, problem->numEffectiveParams()*sizeof(float)));

    for(ResidualFunction::Ptr resFunc : residualFuncs) {
        std::shared_ptr<GPUResidualFunction> gpuResFunc = std::dynamic_pointer_cast<GPUResidualFunction>(resFunc);
        if (gpuResFunc != nullptr) {
            gpuResFunc->setCublasHandle(cublasHandle);
        }

        // Iitialize step values
        auto resBlock = resFunc->getResidualBlock();
        SOLVER_CUDA_CHECK(cudaMemset(resBlock->getWorkingError(), 0, sizeof(float)));

        //TODO: Initialize Parameters in init step, currently in setInitialParams
        //      This is so CPU? and GPU implementations can copy parameters to working params or on to GPU
//        for(auto paramBlock : resBlock->getParameterBlocks()) {
//            paramBlock->initializeParameters();
//        }
    }

}

void GPUSolver::finalize_result(Problem::Ptr problem) {
    auto residualFuncs = problem->getResidualFunctions();
    for(auto resFunc : residualFuncs) {
        auto resBlock = resFunc->getResidualBlock();
        for(auto paramBlock : resBlock->getParameterBlocks()){
            // TODO: What if user uses same pointer to parameter or doesn't but still considered shared?? Just overwrite it?
//            if (!paramBlock->isShared()) {
                // Copys Results from GPU onto CPU into user maintained parameter array.
                paramBlock->getResultParameters();
//            }
        }
    }
}


/**
 * Form damped Hessian for LM step: we solve (H + damping)*Δ = -g.
 * LAMBDA_I:      classic Levenberg: H + λI  (Marquardt 1963, earlier Levenberg).
 * LAMBDA_DIAG_H: Marquardt-style:    H + λ·diag(H), with diag(H) ← max(diag(H), ε) for stability.
 * The latter scales damping by curvature so ill-conditioned directions are damped more.
 */
void GPUSolver::updateHessians(float *hessians, float *dampeningFactors, float *lambda, const int nParams, bool goodStep,
                               DampingType dampingType, float diagFloorEpsilon) {
    int dtype = (dampingType == DampingType::LAMBDA_DIAG_H) ? 1 : 0;
    update_hessians(hessians, dampeningFactors, lambda, nParams, goodStep, dtype, diagFloorEpsilon);
}

/**
 * Updates params: if block has a LocalParameterization, copy to host and call Plus(); else Euclidean kernel.
 */
void GPUSolver::updateParams(float* newParams, const float* params, const float* newDelta, const int nParams, ParameterBlock* paramBlock) {
    auto* gpuBlock = paramBlock ? dynamic_cast<GPUParameterBlock*>(paramBlock) : nullptr;
    if (gpuBlock) {
        auto localParam = gpuBlock->getLocalParameterization();
        float* hostBuf = localParam ? gpuBlock->getHostBufferForPlus() : nullptr;
        if (localParam && hostBuf) {
            int n = nParams;
            SOLVER_CUDA_CHECK(cudaMemcpy(hostBuf, params, n * sizeof(float), cudaMemcpyDeviceToHost));
            SOLVER_CUDA_CHECK(cudaMemcpy(hostBuf + n, newDelta, n * sizeof(float), cudaMemcpyDeviceToHost));
            localParam->Plus(hostBuf, hostBuf + n, hostBuf + 2 * n, n);
            SOLVER_CUDA_CHECK(cudaMemcpy(newParams, hostBuf + 2 * n, n * sizeof(float), cudaMemcpyHostToDevice));
            return;
        }
    }
    update_parameters(newParams, params, newDelta, nParams);
}

void GPUSolver::copyParams(float *destParams, const float *srcParams, const int nParams) {
    // TODO: verify copy-kernel vs cudaMemcpyDeviceToDevice performance (Time)
    // According to documentation, cudaMemcpyDeviceToDevice is generally preferable over a copu kernel
    /**
     * Tested on GeForce GTX960,
     * see https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
     *
     * N           cudaMemcpyDeviceToDevice           copy kernel
     * 1000        0.0075                             0.029
     * 10000       0.0078                             0.072
     * 100000      0.019                              0.068
     * 1000000     0.20                               0.22
     */
    SOLVER_CUDA_CHECK(cudaMemcpy(destParams, srcParams, nParams*sizeof(float), cudaMemcpyDeviceToDevice));
}

/**
 * Solve for the step Δ. Two modes:
 * (1) DAMPED_LM: (H + damping)*Δ = -g, with H already damped by updateHessians. Cholesky on H_lowTri.
 * (2) DOGLEG:    Trust-region step: combine Gauss-Newton step Δ_gn = -H^{-1}g with steepest-descent
 *                p_sd = -α·g (α = (g'g)/(g'Hg)); pick step on the "dogleg" curve so ||Δ|| ≤ radius.
 * Ref: Nocedal & Wright, Trust-region methods; Madsen et al. for LM.
 */
bool GPUSolver::solveSystem(float *deltaParams, float *hessianLowTri, const float *hessians, const float *gradients, const int nParams, float* scaleBuffer,
                             StepType stepType, float* trustRadius, float* auxBuffer) {

    // Copy hessians(A) and hessianLowTri(will be L), since it is inplace decomposition, for A=LL*
    SOLVER_CUDA_CHECK(cudaMemcpy(hessianLowTri, hessians, nParams*nParams*sizeof(float), cudaMemcpyDeviceToDevice));
    SOLVER_CUDA_CHECK(cudaMemcpy(deltaParams, gradients, nParams*sizeof(float), cudaMemcpyDeviceToDevice));

    if (stepType == StepType::DOGLEG && trustRadius && auxBuffer && scaleBuffer) {
        // Dogleg path: H is undamped; ensure pos. def. with diag floor + small μ.
        apply_diag_floor_and_regularize(hessianLowTri, nParams, 1e-6f, 1e-8f);
        bool ok = decompose_cholesky(solver_handle, hessianLowTri, nParams);
        if (!ok) return false;
        // Δ_gn = -H^{-1}g (solve H*Δ_gn = -g; deltaParams currently holds -g).
        float alpha = -1.f;
        cublasSscal(cublasHandle, nParams, &alpha, deltaParams, 1);
        solve_system_cholesky(solver_handle, hessianLowTri, deltaParams, nParams);
        SOLVER_CUDA_CHECK(cudaMemcpy(auxBuffer, deltaParams, nParams*sizeof(float), cudaMemcpyDeviceToDevice));
        float norm_gn = 0.f;
        cublasSnrm2_v2(cublasHandle, nParams, auxBuffer, 1, &norm_gn);
        // Steepest-descent: α = (g'g)/(g'Hg), p_sd = -α·g. (gradient here is stored as -g in some conventions; we use g=gradients.)
        float g_dot_g = 0.f, g_dot_Hg = 0.f;
        cublasSdot(cublasHandle, nParams, gradients, 1, gradients, 1, &g_dot_g);
        float one = 1.f, zero = 0.f;
        cublasSsymv(cublasHandle, CUBLAS_FILL_MODE_LOWER, nParams, &one, hessians, nParams, gradients, 1, &zero, scaleBuffer, 1);
        cublasSdot(cublasHandle, nParams, gradients, 1, scaleBuffer, 1, &g_dot_Hg);
        float alpha_sd = (g_dot_Hg > 1e-20f) ? (g_dot_g / g_dot_Hg) : 0.f;
        float neg_alpha = -alpha_sd;
        cublasScopy(cublasHandle, nParams, gradients, 1, scaleBuffer, 1);
        cublasSscal(cublasHandle, nParams, &neg_alpha, scaleBuffer, 1);
        float norm_sd = 0.f;
        cublasSnrm2_v2(cublasHandle, nParams, scaleBuffer, 1, &norm_sd);
        float radius_h = 1.f;
        SOLVER_CUDA_CHECK(cudaMemcpy(&radius_h, trustRadius, sizeof(float), cudaMemcpyDeviceToHost));
        // Dogleg: if ||Δ_gn|| ≤ radius use full GN; else if ||p_sd|| ≥ radius use scaled p_sd; else point on dogleg with ||Δ||=radius.
        if (norm_gn <= radius_h) {
            SOLVER_CUDA_CHECK(cudaMemcpy(deltaParams, auxBuffer, nParams*sizeof(float), cudaMemcpyDeviceToDevice));
        } else if (norm_sd >= radius_h && norm_sd > 1e-14f) {
            float scale = radius_h / norm_sd;
            cublasScopy(cublasHandle, nParams, scaleBuffer, 1, deltaParams, 1);
            cublasSscal(cublasHandle, nParams, &scale, deltaParams, 1);
        } else {
            // Δ = p_sd + τ*(Δ_gn - p_sd), choose τ so ||Δ|| = radius (solve quadratic in τ).
            float minus_one = -1.f;
            cublasScopy(cublasHandle, nParams, auxBuffer, 1, deltaParams, 1);
            cublasSaxpy(cublasHandle, nParams, &minus_one, scaleBuffer, 1, deltaParams, 1);
            float p_dot_d = 0.f, d_dot_d = 0.f;
            cublasSdot(cublasHandle, nParams, scaleBuffer, 1, deltaParams, 1, &p_dot_d);
            p_dot_d *= 2.f;
            cublasSdot(cublasHandle, nParams, deltaParams, 1, deltaParams, 1, &d_dot_d);
            float c = norm_sd * norm_sd - radius_h * radius_h;
            float disc = p_dot_d * p_dot_d - 4.f * d_dot_d * c;
            float tau = (disc >= 0.f && d_dot_d > 1e-20f) ? ((-p_dot_d + std::sqrt(disc)) / (2.f * d_dot_d)) : 1.f;
            if (tau > 1.f) tau = 1.f;
            if (tau < 0.f) tau = 0.f;
            dogleg_combine(deltaParams, auxBuffer, scaleBuffer, tau, nParams);
        }
        return true;
    }

    // DAMPED_LM path: optional Jacobi scaling then solve (H already damped in place).
    if (scaleBuffer) {
        extract_diagonal_scale(scaleBuffer, hessianLowTri, nParams);
        scale_symmetric_lower(hessianLowTri, scaleBuffer, nParams);
        scale_vector_inplace(deltaParams, scaleBuffer, nParams);
    }

    bool isPosDefMat = decompose_cholesky(solver_handle, hessianLowTri, nParams);

    if (isPosDefMat) {
        //Multipy gradients by -1, we must solve got H * x = -g
        // Solve (H+damping)*Δ = -g: we have -g in deltaParams, overwrite with solution Δ.
        float alpha = -1.f;
        cublasSscal(cublasHandle, nParams, &alpha, deltaParams, 1);
        solve_system_cholesky(solver_handle, hessianLowTri, deltaParams, nParams);
        if (scaleBuffer)
            scale_vector_inplace(deltaParams, scaleBuffer, nParams);
    }

    return isPosDefMat;
}

bool GPUSolver::evaluateGradient(float &norm_inf_grad, float *gradient, int nParams, float tolerance) {
    // Return math::norm_inf(g) <= e_1
    int index = 0;
    float absMaxGrad_h = 0;

    cublasIsamax_v2(cublasHandle, nParams, gradient, 1, &index);

    // Fortran 1-based indexing, covert to 0-based index
    index -= 1;

    SOLVER_CUDA_CHECK(cudaMemcpy(&absMaxGrad_h, gradient + index, sizeof(float), cudaMemcpyDeviceToHost));
    norm_inf_grad = abs(absMaxGrad_h);
//    printf("norm-inf(gradient[%d]): %.4f \n", index, iNorm_h);

    return norm_inf_grad <= tolerance;
}

/**
 * Step-length convergence: ||Δ|| ≤ ε₂(||x|| + ε₂). If the step is tiny relative to current params,
 * we are effectively at a minimum (no point taking more steps). Ref: Madsen et al. termination criteria.
 */
bool GPUSolver::evaluateStep(Problem::Ptr problem, float tolerance) {
    // 2-norm(deltas) ||h_lm||
    float delta_2norm = 0.0f;
    cublasSnrm2_v2(cublasHandle, problem->numEffectiveParams(), problem->getDeltaParameters(), 1, &delta_2norm);

    float param_2norm = 0.0f;
    SOLVER_CUDA_CHECK(cudaMemcpy(&param_2norm, problem->getParams2Norm(), sizeof(float), cudaMemcpyDeviceToHost));

    // return ||h_lm|| ≤ ε_2 (||x|| + ε_2)
    return delta_2norm <= tolerance * (param_2norm + tolerance);
}


void GPUSolver::calcParams2Norm(float* params2Norm, Problem::Ptr problem) {
    SOLVER_CUDA_CHECK(cudaMemset(params2Norm, 0, sizeof(float)));

    auto residualFuncs = problem->getResidualFunctions();
    for(auto resFunc : residualFuncs) {
        auto resBlock = resFunc->getResidualBlock();
        for(auto paramBlock : resBlock->getParameterBlocks()){
            if (!paramBlock->isShared()) {
                // Sum Squared values
                cuda_sum_squares(params2Norm, paramBlock->getBestParameters(), paramBlock->numParameters());
            }
        }
    }

    cuda_sqrt(params2Norm, 1);
}

/**
 * Predicted reduction of the quadratic model: m(0) - m(Δ) = -g'Δ - ½Δ'HΔ.
 * Here gradient is stored as -g, so g'Δ = -gradient'Δ; we compute gradient'Δ - ½Δ'HΔ
 * which equals -g'Δ - ½Δ'HΔ = m(0)-m(Δ). Used for trust-region ratio ρ = (actual_red)/(pred_red).
 * Ref: Nocedal & Wright, eq. for model reduction in trust-region methods.
 */
float GPUSolver::computeModelReduction(float *deltaParams, float *gradient, const float *hessianDamped, int nParams, float* auxBuffer) {
    if (!auxBuffer) return 0.f;
    float g_dot_d = 0.f, d_dot_Hd = 0.f;
    cublasSdot(cublasHandle, nParams, gradient, 1, deltaParams, 1, &g_dot_d);
    float one = 1.f, zero = 0.f;
    cublasSsymv(cublasHandle, CUBLAS_FILL_MODE_LOWER, nParams, &one, hessianDamped, nParams, deltaParams, 1, &zero, auxBuffer, 1);
    cublasSdot(cublasHandle, nParams, deltaParams, 1, auxBuffer, 1, &d_dot_Hd);
    return g_dot_d - 0.5f * d_dot_Hd;
}

/**
 * Gain ratio (LM / Nielsen 1999): (f(x)-f(x+Δ)) / predicted_gain.
 * Predicted gain = ½ Δ'(λΔ - g), the denominator in the LM step quality. Good step if gainRatio > threshold.
 */
float GPUSolver::computeGainRatio(float *predGain, float error, float newError, float *lambda, float *deltaParams,
                                  float *gradient, int nParams) {
    //TODO: Compute Gain ratio
    /*
     * double l = (F_x - F_xnew) / predictedGain;
     */
     // Gain ratio = (f(x)-f(x+Δ)) / predicted_gain, with predicted_gain = ½Δ'(λΔ - g). Nielsen (1999).
    float actualGain = error - newError;
    float predictGain = 0;
    computePredictedGain(predGain, lambda, deltaParams, gradient, nParams);

    SOLVER_CUDA_CHECK(cudaMemcpy(&predictGain, predGain, sizeof(float), cudaMemcpyDeviceToHost));

    float gainRatio = actualGain / predictGain;

    return gainRatio;
}

/** Predicted gain for LM: ½ Δ'(λΔ - g). Used in gain ratio. (Gradient stored as -g.) */
void GPUSolver::computePredictedGain(float *predGain, float *lambda, float *daltaParams, float *gradient, int nParams) {
    // predicted_gain = 0.5 * delta^T (lambda * delta + -g)
    compute_predicted_gain(predGain, lambda, daltaParams, gradient, nParams);
}

/**
 * Initialize λ = τ * max(diag(H)). Ensures initial (H+λI) or (H+λ·diag(H)) is sufficiently pos. def.
 * Ref: Madsen et al. "Methods for non-linear least squares", initial damping.
 */
void GPUSolver::initializeLambda(float *lambda, float tauFactor, float *hessian, int nParams) {
    // lambda = tau * max(Diag(Initial_Hessian))
    assert(tauFactor > 0 && "Tau Factor must be greater than 0");
    initialize_lambda(lambda, tauFactor, hessian, nParams);
}

/**
 * LM λ update (Nielsen 1999): if good step, μ := μ * max{1/3, 1-(2ρ-1)³}, ν:=2;
 * if bad step, μ := μ*ν, ν := 2ν. So we decrease λ when the step is accepted, increase when rejected.
 */
void GPUSolver::updateLambda(float *lambda, float *failFactor, float *predGain, bool goodStep){
    /*
     * if (good_iteration) {
     *    μ := μ ∗ max{ 1/3, 1 − (2*gainRatio − 1)^3 }; ν := 2
     * } else {
     *    μ := μ ∗ ν; ν := 2 ∗ ν
     * }
     *
     * ν = Consecutive Failure Factor (failFactor)
     */
    cuda_lambda_update(lambda, failFactor, predGain, goodStep);
}

/** Trust-region radius for dogleg: double on accept, halve on reject (standard heuristic). */
void GPUSolver::updateTrustRadius(Problem::Ptr problem, bool goodStep) {
    float* tr = problem->getTrustRadius();
    if (!tr) return;
    float r = 1.f;
    SOLVER_CUDA_CHECK(cudaMemcpy(&r, tr, sizeof(float), cudaMemcpyDeviceToHost));
    if (goodStep) r *= 2.f;
    else r *= 0.5f;
    if (r < 1e-10f) r = 1e-10f;
    if (r > 1e10f) r = 1e10f;
    SOLVER_CUDA_CHECK(cudaMemcpy(tr, &r, sizeof(float), cudaMemcpyHostToDevice));
}
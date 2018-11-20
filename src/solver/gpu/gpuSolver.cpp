
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "solver/gpu/cuda/cu_solver.h"

#include "solver/util/cudautil.h"
#include "solver/gpu/gpuSolver.h"
#include "solver/gpu/gpuProblem.h"

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
    //cudaDeviceReset();
    // Initialize step factors
    auto residualFuncs = problem->getResidualFunctions();

    std::shared_ptr<GPUProblem> gpuProblem = std::dynamic_pointer_cast<GPUProblem>(problem);
    if (gpuProblem != nullptr) {
        gpuProblem->setCublasHandle(cublasHandle);
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


void
GPUSolver::updateHessians(float *hessians, float *dampeningFactors, float *lambda, const int nParams, bool goodStep) {
//    print_array("updateHessians::hessians::before", hessians, nParams*nParams);
//    print_array("updateHessians::dampeningFactors::before", dampeningFactors, nParams);
//    print_array("updateHessians::lambda", lambda, 1);
    update_hessians(hessians, dampeningFactors, lambda, nParams, goodStep);
//    print_array("updateHessians::hessians::after", hessians, nParams*nParams);
//    print_array("updateHessians::dampeningFactors::after", dampeningFactors, nParams);
}

/**
 * Updates params, p_1 = p_0 + delta
 * @param params
 * @param newDelta
 * @param nParams
 */
void GPUSolver::updateParams(float* newParams, const float* params, const float* newDelta, const int nParams) {
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

bool GPUSolver::solveSystem(float *deltaParams, float *hessianLowTri, const float *hessians, const float *gradients, const int nParams) {


//    print_array("solveSystem::hessians", hessians, nParams*nParams);
    // Copy hessians(A) to hessianLowTri(will be L), since it is inplace decomposition, for A=LL*
    SOLVER_CUDA_CHECK(cudaMemcpy(hessianLowTri, hessians, nParams*nParams*sizeof(float), cudaMemcpyDeviceToDevice));
//    print_array("InitL:", hessianLowTri, nParams*nParams);
//    print_array("solveSystem::hessianLowTri", hessians, nParams*nParams);

    // Copy gradients(x) to deltaParams(will be b), since it shares the same in/out param, for Ax=b
    SOLVER_CUDA_CHECK(cudaMemcpy(deltaParams, gradients, nParams*sizeof(float), cudaMemcpyDeviceToDevice));


    bool isPosDefMat = decompose_cholesky(solver_handle, hessianLowTri, nParams);

    if (isPosDefMat) {
        //Multipy gradients by -1, we must solve got H * x = -g
        float alpha = -1.f;
        cublasSscal(cublasHandle, nParams, &alpha, deltaParams, 1);
        solve_system_cholesky(solver_handle, hessianLowTri, deltaParams, nParams);
    }

    return isPosDefMat;
}

bool GPUSolver::evaluateGradient(float *gradient, int nParams, float tolerance) {
    // Return math::norm_inf(g) <= e_1
    int index = 0;
    float iNorm_h = 0;

    cublasIsamax_v2(cublasHandle, nParams, gradient, 1, &index);

    // Fortran 1-based indexing, covert to 0-based index
    index -= 1;

    SOLVER_CUDA_CHECK(cudaMemcpy(&iNorm_h, gradient + index, sizeof(float), cudaMemcpyDeviceToHost));
    iNorm_h = abs(iNorm_h);
//    printf("norm-inf(gradient[%d]): %.4f \n", index, iNorm_h);

    return iNorm_h <= tolerance;
}

bool GPUSolver::evaluateStep(Problem::Ptr problem, float tolerance) {
    //2-norm(deltas) ||h_lm||

    float delta_2norm=0.0f;
    cublasSnrm2_v2(cublasHandle, problem->numEffectiveParams(), problem->getDeltaParameters(), 1, &delta_2norm);

    //2-norm(x_params) ||x||
    float param_2norm = 0.0f;
    SOLVER_CUDA_CHECK(cudaMemcpy(&param_2norm, problem->getParams2Norm(), sizeof(float), cudaMemcpyDeviceToHost));

    //printf("delta-2Norm: %.4f param-2Norm: %.4f changeXTol: %.4f\n", delta_2norm, param_2norm, tolerance * (param_2norm + tolerance));

    //return ||h_lm|| ≤ ε_2 (||x|| + ε_2)
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

float GPUSolver::computeGainRatio(float *predGain, float error, float newError, float *lambda, float *deltaParams,
                                  float *gradient, int nParams) {
    //TODO: Compute Gain ratio
    /*
     * double l = (F_x - F_xnew) / predictedGain;
     */

    float actualGain = error - newError;
    float predictGain = 0;
    computePredictedGain(predGain, lambda, deltaParams, gradient, nParams);

    SOLVER_CUDA_CHECK(cudaMemcpy(&predictGain, predGain, sizeof(float), cudaMemcpyDeviceToHost));

    float gainRatio = actualGain / predictGain;

    return gainRatio;
}

void GPUSolver::computePredictedGain(float *predGain, float *lambda, float *daltaParams, float *gradient, int nParams) {
    //predGain = 0.5*delta^T (lambda * delta + -g)
    compute_predicted_gain(predGain, lambda, daltaParams, gradient, nParams);
}

void GPUSolver::initializeLambda(float *lambda, float tauFactor, float *hessian, int nParams) {
    // lambda = tau * max(Diag(Initial_Hessian))
    assert(tauFactor > 0 && "Tau Factor must be greater than 0");
    initialize_lambda(lambda, tauFactor, hessian, nParams);
}


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
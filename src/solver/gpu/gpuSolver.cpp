
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <solver/gpu/gpuResidualFunction.h>

#include "solver/gpu/cuda/cu_solver.h"

#include "util/cudautil.h"
#include "solver/gpu/gpuSolver.h"
#include "solver/gpu/gpuProblem.h"
#include "solver/gpu/gpuResidualFunction.h"

using namespace telef::solver;

float GPUSolver::calcError(float *error, const float *residuals, const int nRes) {
    // TODO: Use Residual based error, use error_d as total error for all residuals
    float error_h = 0;
    //Reset to 0
    cudaMemset(error,0, sizeof(float));
    calc_error(error, residuals, nRes);

    cudaMemcpy(&error_h, error, sizeof(float), cudaMemcpyDeviceToHost);
    return error_h;
}


void GPUSolver::initialize_run(Problem::Ptr problem) {
    //cudaDeviceReset();
    // Initialize step factors
    float downFactor = 1 / options.step_down;
    cudaMemcpy(down_factor_d, &downFactor, sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(up_factor_d, &options.step_up, sizeof(float), cudaMemcpyHostToDevice );

    float inital_step = options.lambda_initial;

    auto residualFuncs = problem->getResidualFunctions();

    std::shared_ptr<GPUProblem> gpuProblem = std::dynamic_pointer_cast<GPUProblem>(problem);
    if (gpuProblem != nullptr) {
        gpuProblem->setCublasHandle(cublasHandle);
    }

    // Iitialize step values
    float* lambda = problem->getLambda();

    //Initialize Total Error
    float* problemWorkError = problem->getWorkingError();

    CUDA_CHECK(cudaMemcpy(lambda, &options.lambda_initial, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(problemWorkError, 0, sizeof(float)));

    // Initialize Dampening Factors
    float* dampeningFactors = problem->getDampeningFactors();
    CUDA_CHECK(cudaMemset(dampeningFactors, 0, problem->numEffectiveParams()*sizeof(float)));

    for(ResidualFunction::Ptr resFunc : residualFuncs) {
        std::shared_ptr<GPUResidualFunction> gpuResFunc = std::dynamic_pointer_cast<GPUResidualFunction>(resFunc);
        if (gpuResFunc != nullptr) {
            gpuResFunc->setCublasHandle(cublasHandle);
        }

        // Iitialize step values
        auto resBlock = resFunc->getResidualBlock();
        float* workError = resBlock->getWorkingError();

        CUDA_CHECK(cudaMemset(workError, 0, sizeof(float)));

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
            if (!paramBlock->isShared()) {
                // Copys Results from GPU onto CPU into user maintained parameter array.
                paramBlock->getResultParameters();
            }
        }
    }
}

void GPUSolver::updateStep(float* lambda, bool goodStep) {
    if (goodStep) {
        cuda_step_update(lambda, down_factor_d);
    } else {
        cuda_step_update(lambda, up_factor_d);
    }
}

void
GPUSolver::updateHessians(float *hessians, float *dampeningFactors, float *lambda, const int nParams, bool goodStep) {
    print_array("updateHessians::hessians::before", hessians, nParams*nParams);
    print_array("updateHessians::dampeningFactors::before", dampeningFactors, nParams);
    print_array("updateHessians::lambda", lambda, 1);
    update_hessians(hessians, dampeningFactors, lambda, nParams, goodStep);
    print_array("updateHessians::hessians::after", hessians, nParams*nParams);
    print_array("updateHessians::dampeningFactors::after", dampeningFactors, nParams);
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
    CUDA_CHECK(cudaMemcpy(destParams, srcParams, nParams*sizeof(float), cudaMemcpyDeviceToDevice));
}

bool GPUSolver::solveSystem(float *deltaParams, float *hessianLowTri, const float *hessians, const float *gradients, const int nParams) {


    print_array("solveSystem::hessians", hessians, nParams*nParams);
    // Copy hessians(A) to hessianLowTri(will be L), since it is inplace decomposition, for A=LL*
    CUDA_CHECK(cudaMemcpy(hessianLowTri, hessians, nParams*nParams*sizeof(float), cudaMemcpyDeviceToDevice));
//    print_array("InitL:", hessianLowTri, nParams*nParams);
    print_array("solveSystem::hessianLowTri", hessians, nParams*nParams);

    // Copy gradients(x) to deltaParams(will be b), since it shares the same in/out param, for Ax=b
    CUDA_CHECK(cudaMemcpy(deltaParams, gradients, nParams*sizeof(float), cudaMemcpyDeviceToDevice));


    bool isPosDefMat = decompose_cholesky(solver_handle, hessianLowTri, nParams);

    if (isPosDefMat) {
        solve_system_cholesky(solver_handle, hessianLowTri, deltaParams, nParams);
    }

    return isPosDefMat;
}
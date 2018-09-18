#include "solver/gpu/cuda/cu_solver.h"

#include "util/cudautil.h"
#include "solver/gpu/gpuSolver.h"

using namespace telef::solver;

float GPUSolver::calcError(const float* residuals, const int nRes) {
    // TODO: Use Residual based error, use error_d as total error for all residuals
    float error = 0;
    calc_error(error_d, residuals, nRes);

    cudaMemcpy(&error, error_d, sizeof(float), cudaMemcpyDeviceToHost);
    return error;
}


void GPUSolver::initialize_solver(){

    // Initialize step factors
    float downFactor = 1 / options.step_down;
    cudaMemcpy(down_factor_d, &downFactor, sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(up_factor_d, &options.step_up, sizeof(float), cudaMemcpyHostToDevice );

    float inital_step = 1 + options.lambda_initial;
    for(ResidualFunction::Ptr resFunc : residualFuncs) {
        // Iitialize step values
        float* lambda = resFunc->getResidualBlock()->getLambda();
        float* step = resFunc->getResidualBlock()->getStep();

        CUDA_CHECK(cudaMemcpy(lambda, &options.lambda_initial, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(step, &inital_step, sizeof(float), cudaMemcpyHostToDevice));

        //TODO: Initialize Parameters
        // Initilaize Parameters
        // Copy Parameters to Result Parameters

        //TODO: Initialize errors, one per residual block
    }

    //Initialize Total Error
    CUDA_CHECK(cudaMemset(error_d, 0, sizeof(float)));
}

void GPUSolver::stepUp(float* step, float* lambda){
    cuda_step_up(step, lambda, up_factor_d);
}


void GPUSolver::stepDown(float* step, float* lambda) {
    cuda_step_down(step, lambda, down_factor_d);
}

void GPUSolver::updateHessians(float* hessians, float* step, const int nParams){
    update_hessians(hessians, step, nParams);
}

void copyParams(float *destParams, const float *srcParams, const int nParams){
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
    CUDA_CHECK(cudaMemcpy(&destParams, srcParams, nParams*sizeof(float), cudaMemcpyDeviceToDevice));
}

/**
 * Updates params, p_1 = p_0 + delta
 * @param params
 * @param newDelta
 * @param nParams
 */
void updateParams(float* newParams, const float* params, const float* newDelta, const int nParams){
    update_parameters(newParams, params, newDelta, nParams);
}

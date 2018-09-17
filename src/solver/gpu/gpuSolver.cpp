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


#include <gtest/gtest.h> // googletest header file

#include <iostream>
#include <string>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "solver/gpu/gpuResidualFunction.h"

#include "util/cudautil.h"
#include "cuda/cuda_kernel.h"
#include "solver/gpu/cuda/cu_solver.h"

using namespace std;
using namespace telef::solver;
using namespace testing;

TEST(GPUSolverTest_cuda, calcError) {

    float residuals[] = {-10.2631, -3.2631, -4.2631, -1.2631};
    float *residuals_d;
    float *error_d;

    utils::CUDA_ALLOC_AND_ZERO(&error_d, static_cast<size_t >(1));
    utils::CUDA_ALLOC_AND_COPY(&residuals_d, residuals, static_cast<size_t >(4));

    calc_error(error_d, residuals_d, 4);

    float error;
    cudaMemcpy(&error, error_d, sizeof(float), cudaMemcpyDeviceToHost);

    float real_err = 135.748;

    float ferr = 1e-3;
    EXPECT_THAT(error, FloatNear(real_err, ferr));

    cudaFree(residuals_d);
    cudaFree(error_d);
}

TEST(GPUSolverTest_cuda, stepDown) {
    float ferr = 1e-3;

    float *step;
    float *lambda;
    float *factor;

    float step_h;
    float lambda_h = 0.1;
    float factor_h = 1/10.0;

    utils::CUDA_ALLOC_AND_ZERO(&step, static_cast<size_t >(1));
    utils::CUDA_ALLOC_AND_COPY(&lambda, &lambda_h, static_cast<size_t >(1));
    utils::CUDA_ALLOC_AND_COPY(&factor, &factor_h, static_cast<size_t >(1));

    cuda_step_down(step, lambda, factor);
    cudaDeviceSynchronize();

    cudaMemcpy(&step_h, step, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);

    float real_step = 1.010;
    float real_lambda = 0.010;

    EXPECT_THAT(step_h, FloatNear(real_step, ferr));
    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));

    cuda_step_down(step, lambda, factor);
    cudaDeviceSynchronize();

    cudaMemcpy(&step_h, step, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);

    real_step = 1.001;
    real_lambda = 0.001;

    EXPECT_THAT(step_h, FloatNear(real_step, ferr));
    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));

    cudaFree(step);
    cudaFree(lambda);
    cudaFree(factor);
}

TEST(GPUSolverTest_cuda, stepUp) {
    float ferr = 1e-3;

    float *step;
    float *lambda;
    float *factor;

    float step_h;
    float lambda_h = 0.1;
    float factor_h = 10.0;

    utils::CUDA_ALLOC_AND_ZERO(&step, static_cast<size_t >(1));
    utils::CUDA_ALLOC_AND_COPY(&lambda, &lambda_h, static_cast<size_t >(1));
    utils::CUDA_ALLOC_AND_COPY(&factor, &factor_h, static_cast<size_t >(1));

    cuda_step_up(step, lambda, factor);
    cudaDeviceSynchronize();

    cudaMemcpy(&step_h, step, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);

    float real_step = 1.818;
    float real_lambda = 1.0;

    EXPECT_THAT(step_h, FloatNear(real_step, ferr));
    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));

    cuda_step_up(step, lambda, factor);
    cudaDeviceSynchronize();

    cudaMemcpy(&step_h, step, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);

    real_step = 5.500;
    real_lambda = 10.000;

    EXPECT_THAT(step_h, FloatNear(real_step, ferr));
    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));

//    float down_factor_h = 1/10.0;
//    float *down_factor;
//    utils::CUDA_ALLOC_AND_COPY(&down_factor, &down_factor_h, static_cast<size_t >(1));
//    cuda_step_down(step, lambda, down_factor);
//
//    cuda_step_up(step, lambda, factor);
//    cudaFree(down_factor);
//    cudaDeviceSynchronize();

    cudaFree(step);
    cudaFree(lambda);
    cudaFree(factor);
}


TEST(GPUSolverTest_cuda, updateHessian) {
    float hessian[] = {4.42934,  20.6941,
                        20.6941, 96.684};
    float step = 1.81818;
    float *hessian_d;
    float *step_d;
    int nParams = 2;

    utils::CUDA_ALLOC_AND_COPY(&hessian_d, hessian, static_cast<size_t >(nParams*nParams));
    utils::CUDA_ALLOC_AND_COPY(&step_d, &step, static_cast<size_t >(1));

    update_hessians(hessian_d, step_d, nParams);

    cudaMemcpy(hessian, hessian_d, nParams*nParams*sizeof(float), cudaMemcpyDeviceToHost);

    float real_update_hessian[] = {8.05333,  20.6941,
                                    20.6941, 175.7889};

    float ferr = 1e-3;
    EXPECT_THAT(hessian,
                Pointwise(FloatNear(ferr), real_update_hessian));
}
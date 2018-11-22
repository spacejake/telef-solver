

#include <gtest/gtest.h> // googletest header file

#include <iostream>
#include <string>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "solver/gpu/gpuResidualFunction.h"

#include "solver/util/cudautil.h"
#include "cuda/cuda_kernel.h"
#include "solver/gpu/cuda/cu_solver.h"
#include "mock_gpu.h"

using namespace std;
using namespace telef::solver;
using namespace testing;


TEST_F(GPUSolverTestSimple, solve1) {
//    solver->options.max_iterations = 500;
    solver->options.verbose = true;

    Status status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    vector<float> real_fit_params = {3.162278};
    float ferr = 1e-4;
    EXPECT_THAT(params,
                Pointwise(FloatNear(ferr), real_fit_params));
}

TEST_F(GPUSolverTest, solve2) {
    solver->options.initial_dampening_factor = 1e-3;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    //FIXME: Debug mode results in different params {-3.89191, -0.46297}, why?
    vector<float> real_fit_params = {-2.01896, 0.0538367};

    // Actual Ceres minimizad params, but this is a sinosoidal and can have multiple minimums
    // the above is equivilat in error (22.5000 = .5*lse) and the result our minimizer results to.
//    vector<float> real_fit_params = {-2.60216, 0.0318891};

    float ferr = 1e-3;
    EXPECT_THAT(params,
                Pointwise(FloatNear(ferr), real_fit_params));

}

TEST_F(BealesTest, solve2) {
    solver->options.initial_dampening_factor = 1e-3;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    vector<float> real_fit_params = {3, 0.5};

    // Actual Ceres minimizad params, but this is a sinosoidal and can have multiple minimums
    // the above is equivilat in error (22.5000 = .5*lse) and the result our minimizer results to.
//    vector<float> real_fit_params = {-2.60216, 0.0318891};

    float ferr = 1e-3;
    EXPECT_THAT(params,
                Pointwise(FloatNear(ferr), real_fit_params));

}

TEST_F(SchwefelTest, solve) {
    solver->options.initial_dampening_factor = 1;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    vector<float> real_fit_params(n, 420.968746f);

    float ferr = 1e-3;
    EXPECT_THAT(params,
                Pointwise(FloatNear(ferr), real_fit_params));

}

TEST_F(GPUSolverMultiParam, MultiParams) {
    solver->options.max_iterations = 20;
    solver->options.initial_dampening_factor = 1e-3;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    vector<float> real_fit_params1 = { -0.804031, -1.22526 };
    vector<float> real_fit_params2 = {1.81251};

    // Actual Ceres minimizad params, but this is a sinosoidal and can have multiple minimums
    // the above is equivilat in error (22.5000 = .5*lse) and the result our minimizer results to.
//    vector<float> real_fit_params = {-2.60216, 0.0318891};

    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));
    EXPECT_THAT(params2,
                Pointwise(FloatNear(ferr), real_fit_params2));

}

TEST_F(GPUSolver4Param, MultiParams) {
    // TODO: This is a bad example, fix!!!!
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    vector<float> real_fit_params1 = {-1.00897, 0.044994};
    vector<float> real_fit_params2 = {1.52608};
    vector<float> real_fit_params3 = {-1, 1};
    vector<float> real_fit_params4 = {1, 0.5};


    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));
    EXPECT_THAT(params2,
                Pointwise(FloatNear(ferr), real_fit_params2));
    EXPECT_THAT(params3,
                Pointwise(FloatNear(ferr), real_fit_params3));
    EXPECT_THAT(params4,
                Pointwise(FloatNear(ferr), real_fit_params4));

}

TEST_F(GPUSolverMultiResidual, MultiObjective) {
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    // Independant: Our current best LS Error is 22.5, ceres is 22.5
    vector<float> real_fit_params1 = {1.57844, 4.01355};
    vector<float> real_fit_params2 = {1.62091, 0.361981};

    // Actual Ceres minimizad params, but this is a sinosoidal and can have multiple minimums
    // the above is equivilat in error (22.5000 = .5*lse) and the result our minimizer results to.
//    vector<float> real_fit_params = {-2.60216, 0.0318891};

    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));
    EXPECT_THAT(params2,
                Pointwise(FloatNear(ferr), real_fit_params2));

}

TEST_F(GPUSolverMultiResidual, MultiObjectiveShared) {
    resFunc1->getResidualBlock()->getParameterBlocks()[0]->share(
            resFunc2->getResidualBlock()->getParameterBlocks()[0]);

    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    // Shared: Our current best LS Error is 22.5, ceres is 22.5
    vector<float> real_fit_params1 = {4.77069, 0.439377};


    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));
    EXPECT_THAT(params2,
                Pointwise(FloatNear(ferr), real_fit_params1));

}

//
//TEST(GPUSolverTest_cuda, calcError) {
//
//    float residuals[] = {-10.2631, -3.2631, -4.2631, -1.2631};
//    float *residuals_d;
//    float *error_d;
//
//    utils::CUDA_ALLOC_AND_ZERO(&error_d, static_cast<size_t >(1));
//    utils::CUDA_ALLOC_AND_COPY(&residuals_d, residuals, static_cast<size_t >(4));
//
//    calc_error(error_d, residuals_d, 4);
//
//    float error;
//    cudaMemcpy(&error, error_d, sizeof(float), cudaMemcpyDeviceToHost);
//
//    float real_err = 135.748;
//
//    float ferr = 1e-3;
//    EXPECT_THAT(error, FloatNear(real_err, ferr));
//
//    cudaFree(residuals_d);
//    cudaFree(error_d);
//}
//
//TEST(GPUSolverTest_cuda, stepDown) {
//    float ferr = 1e-3;
//
//    float *lambda;
//    float *factor;
//
//    float lambda_h = 0.1;
//    float factor_h = 1/10.0;
//
//    utils::CUDA_ALLOC_AND_COPY(&lambda, &lambda_h, static_cast<size_t >(1));
//    utils::CUDA_ALLOC_AND_COPY(&factor, &factor_h, static_cast<size_t >(1));
//
//    cuda_lambda_update(lambda, factor);
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);
//
//    float real_lambda = 0.010;
//
//    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));
//
//    cuda_lambda_update(lambda, factor);
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);
//
//    real_lambda = 0.001;
//
//    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));
//
//    cudaFree(lambda);
//    cudaFree(factor);
//}
//
//TEST(GPUSolverTest_cuda, stepUp) {
//    float ferr = 1e-3;
//
//    float *lambda;
//    float *factor;
//
//    float lambda_h = 0.1;
//    float factor_h = 10.0;
//
//    utils::CUDA_ALLOC_AND_COPY(&lambda, &lambda_h, static_cast<size_t >(1));
//    utils::CUDA_ALLOC_AND_COPY(&factor, &factor_h, static_cast<size_t >(1));
//
//    cuda_lambda_update(lambda, factor);
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);
//
//    float real_lambda = 1.0;
//
//    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));
//
//    cuda_lambda_update(lambda, factor);
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(&lambda_h, lambda, sizeof(float), cudaMemcpyDeviceToHost);
//
//    real_lambda = 10.000;
//
//    EXPECT_THAT(lambda_h, FloatNear(real_lambda, ferr));
//
////    float down_factor_h = 1/10.0;
////    float *down_factor;
////    utils::CUDA_ALLOC_AND_COPY(&down_factor, &down_factor_h, static_cast<size_t >(1));
////    cuda_step_down(step, lambda, down_factor);
////
////    cuda_step_up(step, lambda, factor);
////    cudaFree(down_factor);
////    cudaDeviceSynchronize();
//
//    cudaFree(lambda);
//    cudaFree(factor);
//}
//
//
//TEST(GPUSolverTest_cuda, updateHessian) {
//    float hessian[] = {4.42934,  20.6941,
//                        20.6941, 96.684};
//    float dampeningFactors[] = {0.0f, 0.0f};
//
//    float step = 0.1f;
//    float *hessian_d;
//    float *dampeningFactors_d;
//    float *step_d;
//    int nParams = 2;
//
//    utils::CUDA_ALLOC_AND_COPY(&hessian_d, hessian, static_cast<size_t >(nParams*nParams));
//    utils::CUDA_ALLOC_AND_COPY(&dampeningFactors_d, dampeningFactors, static_cast<size_t >(nParams));
//    utils::CUDA_ALLOC_AND_COPY(&step_d, &step, static_cast<size_t >(1));
//
//    update_hessians(hessian_d, dampeningFactors_d, step_d, nParams, false);
//
//    cudaMemcpy(hessian, hessian_d, nParams*nParams*sizeof(float), cudaMemcpyDeviceToHost);
//    cudaMemcpy(dampeningFactors, dampeningFactors_d, nParams*sizeof(float), cudaMemcpyDeviceToHost);
//
//    float real_update_hessian[] = {4.872274,  20.6941,
//                                    20.6941, 106.3524};
//    float real_dampeningFactors[] = {4.42934, 96.68400};
//
//    float ferr = 1e-3;
//    EXPECT_THAT(hessian,
//                Pointwise(FloatNear(ferr), real_update_hessian));
//    EXPECT_THAT(dampeningFactors,
//                Pointwise(FloatNear(ferr), real_dampeningFactors));
//
//    cudaFree(hessian_d);
//    cudaFree(step_d);
//}
//
//
//TEST(GPUSolverTest_cuda, updateParams) {
//    float params[] = {0.5, 0.5, 0.5, 0.5};
//    float paramsDelta[] = {0.42934,  0.6941, 0.6941, 0.684};
//    float newParams[4];
//    int nParams = 4;
//
//    float *params_d;
//    float *deltaParams_d;
//    float *newParams_d;
//
//    utils::CUDA_ALLOC_AND_COPY(&params_d, params, static_cast<size_t>(nParams));
//    utils::CUDA_ALLOC_AND_COPY(&deltaParams_d, paramsDelta, static_cast<size_t>(nParams));
//    utils::CUDA_MALLOC(&newParams_d, static_cast<size_t>(nParams));
//
//    update_parameters(newParams_d, params_d, deltaParams_d, nParams);
//
//    cudaMemcpy(newParams, newParams_d, nParams*sizeof(float), cudaMemcpyDeviceToHost);
//
//    float real_new_params[] = {0.92934, 1.1941, 1.1941, 1.184};
//
//    float ferr = 1e-3;
//    EXPECT_THAT(newParams,
//                Pointwise(FloatNear(ferr), real_new_params));
//
//
//    cudaFree(params_d);
//    cudaFree(newParams_d);
//    cudaFree(deltaParams_d);
//}
//
//TEST(GPUSolverTest_cuda, CholeskyDecompseHessian) {
//    // --- CUDA solver initialization
//    cusolverDnHandle_t solver_handle;
//    cusolverDnCreate(&solver_handle);
//
//    float hessian[] = {4.872274,  20.6941,
//                       20.6941, 106.3524};
//    float *hessian_d;
//    int nParams = 2;
//    utils::CUDA_ALLOC_AND_COPY(&hessian_d, hessian, static_cast<size_t >(nParams*nParams));
//    decompose_cholesky(solver_handle, hessian_d, nParams);
//
//    cudaMemcpy(hessian, hessian_d, nParams*nParams*sizeof(float), cudaMemcpyDeviceToHost);
//
//    // Column-order lower triangular matric, upper left unchanged.
//    float real_decomposed[] = {2.20732, 9.37520,
//                               20.6941, 4.29627};
//
//    float ferr = 1e-4;
//    EXPECT_THAT(hessian,
//                Pointwise(FloatNear(ferr), real_decomposed));
//
//    cudaFree(hessian_d);
//
//    if (solver_handle) cusolverDnDestroy(solver_handle);
//}
//
//TEST(GPUSolverTest_cuda, CholeskySolve) {
//    // --- CUDA solver initialization
//    cusolverDnHandle_t solver_handle;
//    cusolverDnCreate(&solver_handle);
//
////    // --- CUBLAS initialization
////    cublasHandle_t cublas_handle;
////    cublasCreate(&cublas_handle);
//
//    float gradiants[] = {20.0488, 93.6692};
//    float decomposed_hessian[] = {2.20732, 9.37520,
//                                  20.6941, 4.29627};
//    float *paramsDelta_d;
//    float *decomposed_hessian_d;
//    int nParams = 2;
//
//    utils::CUDA_ALLOC_AND_COPY(&paramsDelta_d, gradiants, static_cast<size_t >(nParams));
//    utils::CUDA_ALLOC_AND_COPY(&decomposed_hessian_d, decomposed_hessian, static_cast<size_t >(nParams*nParams));
//    solve_system_cholesky(solver_handle, decomposed_hessian_d, paramsDelta_d, nParams);
//
//    float paramsDelta[2];
//    cudaMemcpy(paramsDelta, paramsDelta_d, nParams*sizeof(float), cudaMemcpyDeviceToHost);
//
//
//    // Column-order lower triangular matric, upper left unchanged.
//    float real_deltas[] = {2.1554, 0.461345};
//
//    float ferr = 1e-4;
//    EXPECT_THAT(paramsDelta,
//                Pointwise(FloatNear(ferr), real_deltas));
//
//    cudaFree(paramsDelta_d);
//    cudaFree(decomposed_hessian_d);
//
//    if (solver_handle) cusolverDnDestroy(solver_handle);
//}
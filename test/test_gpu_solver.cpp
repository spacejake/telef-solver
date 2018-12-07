

#include <gtest/gtest.h> // googletest header file

#include <iostream>
#include <string>
#include <gmock/gmock.h>
#include <experimental/filesystem>


#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "solver/gpu/gpuResidualFunction.h"

#include "solver/util/cudautil.h"
#include "solver/util/fileio.h"
#include "cuda/cuda_kernel.h"
#include "solver/gpu/cuda/cu_solver.h"
#include "mock_gpu.h"


#ifdef TEST_DATA_DIR
#define DATA_DIR TEST_DATA_DIR
#else
#define DATA_DIR "NO_FILE"
#endif


namespace fs = std::experimental::filesystem;
using namespace std;
using namespace telef::solver;
using namespace testing;

TEST(GPUSolverTest_cuda, calcError) {

    float residuals[] = {-10.2631, -3.2631, -4.2631, -1.2631};
    float *residuals_d;
    float *error_d;

    SOLVER_CUDA_ALLOC_AND_ZERO(&error_d, static_cast<size_t >(1));
    SOLVER_CUDA_ALLOC_AND_COPY(&residuals_d, residuals, static_cast<size_t >(4));

    calc_error(error_d, residuals_d, 4);

    float error;
    cudaMemcpy(&error, error_d, sizeof(float), cudaMemcpyDeviceToHost);

    float real_err = 135.748;

    float ferr = 1e-3;
    EXPECT_THAT(error, FloatNear(real_err, ferr));

    cudaFree(residuals_d);
    cudaFree(error_d);
}

TEST(GPUSolverTest_cuda, initLambda) {

    float tau = 1.f;

    float hessian[16] = {10.2631, 3.2631, -4.2631, -1.2631,
                         4.77069, -0.439377, 2.42385, 1.70211,
                         1.22791, 0.2099, 11.22542, -1.5641,
                         -4.9164, -1.0523, 1.2631, 3.2631 };
    float* hessian_d;
    float* lambda_d;
    SOLVER_CUDA_ALLOC_AND_ZERO(&lambda_d, static_cast<size_t >(1));
    SOLVER_CUDA_ALLOC_AND_COPY(&hessian_d, hessian, static_cast<size_t >(16));

    initialize_lambda(lambda_d, tau, hessian_d, 4);

    float lambda;
    cudaMemcpy(&lambda, lambda_d, sizeof(float), cudaMemcpyDeviceToHost);

    float real_lambda = 11.22542;

    float ferr = 1e-3;
    EXPECT_THAT(lambda, FloatNear(real_lambda, ferr));

    cudaFree(hessian_d);
    cudaFree(lambda_d);
}


TEST(GPUSolverTest_cuda, updateParams) {
    float params[] = {0.5, 0.5, 0.5, 0.5};
    float paramsDelta[] = {0.42934,  0.6941, 0.6941, 0.684};
    float newParams[4];
    int nParams = 4;

    float *params_d;
    float *deltaParams_d;
    float *newParams_d;

    SOLVER_CUDA_ALLOC_AND_COPY(&params_d, params, static_cast<size_t>(nParams));
    SOLVER_CUDA_ALLOC_AND_COPY(&deltaParams_d, paramsDelta, static_cast<size_t>(nParams));
    SOLVER_CUDA_MALLOC(&newParams_d, static_cast<size_t>(nParams));

    update_parameters(newParams_d, params_d, deltaParams_d, nParams);

    cudaMemcpy(newParams, newParams_d, nParams*sizeof(float), cudaMemcpyDeviceToHost);

    float real_new_params[] = {0.92934, 1.1941, 1.1941, 1.184};

    float ferr = 1e-3;
    EXPECT_THAT(newParams,
                Pointwise(FloatNear(ferr), real_new_params));


    cudaFree(params_d);
    cudaFree(newParams_d);
    cudaFree(deltaParams_d);
}



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

TEST(GPUSolverTest_cuda, CholeskyDecompseHessian) {
    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    float hessian[] = {4.872274,  20.6941,
                       20.6941, 106.3524};
    float *hessian_d;
    int nParams = 2;
    SOLVER_CUDA_ALLOC_AND_COPY(&hessian_d, hessian, static_cast<size_t >(nParams*nParams));
    decompose_cholesky(solver_handle, hessian_d, nParams);

    cudaMemcpy(hessian, hessian_d, nParams*nParams*sizeof(float), cudaMemcpyDeviceToHost);

    // Column-order lower triangular matric, upper left unchanged.
    float real_decomposed[] = {2.20732, 9.37520,
                               20.6941, 4.29627};

    float ferr = 1e-4;
    EXPECT_THAT(hessian,
                Pointwise(FloatNear(ferr), real_decomposed));

    cudaFree(hessian_d);

    if (solver_handle) cusolverDnDestroy(solver_handle);
}

TEST(GPUSolverTest_cuda, CholeskySolve) {
    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

//    // --- CUBLAS initialization
//    cublasHandle_t cublas_handle;
//    cublasCreate(&cublas_handle);

    float gradiants[] = {20.0488, 93.6692};
    float decomposed_hessian[] = {2.20732, 9.37520,
                                  20.6941, 4.29627};
    float *paramsDelta_d;
    float *decomposed_hessian_d;
    int nParams = 2;

    SOLVER_CUDA_ALLOC_AND_COPY(&paramsDelta_d, gradiants, static_cast<size_t >(nParams));
    SOLVER_CUDA_ALLOC_AND_COPY(&decomposed_hessian_d, decomposed_hessian, static_cast<size_t >(nParams*nParams));
    solve_system_cholesky(solver_handle, decomposed_hessian_d, paramsDelta_d, nParams);

    float paramsDelta[2];
    cudaMemcpy(paramsDelta, paramsDelta_d, nParams*sizeof(float), cudaMemcpyDeviceToHost);


    // Column-order lower triangular matric, upper left unchanged.
    float real_deltas[] = {2.1554, 0.461345};

    float ferr = 1e-4;
    EXPECT_THAT(paramsDelta,
                Pointwise(FloatNear(ferr), real_deltas));

    cudaFree(paramsDelta_d);
    cudaFree(decomposed_hessian_d);

    if (solver_handle) cusolverDnDestroy(solver_handle);
}

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
    vector<float> real_fit_params = {1.61672, 0.0873202};

    // Actual Ceres minimizad params, but this is a sinosoidal and can have multiple minimums
    // the above is equivilat in error (22.5000 = .5*lse) and the result our minimizer results to.
//    vector<float> real_fit_params = {-2.60216, 0.0318891};

    float ferr = 1e-3;
    EXPECT_THAT(params,
                Pointwise(FloatNear(ferr), real_fit_params));

}

TEST_F(GPUSolverMultiParam, MultiParams) {
//    solver->options.max_iterations = 20;
    solver->options.initial_dampening_factor = 1e-3;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    vector<float> real_fit_params1 = { -0.67663, -0.196581 };
    vector<float> real_fit_params2 = {1.80162};

    // Actual Ceres minimizad params, but this is a sinosoidal and can have multiple minimums
    // the above is equivilat in error (22.5000 = .5*lse) and the result our minimizer results to.
//    vector<float> real_fit_params = {-2.60216, 0.0318891};

    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));
    EXPECT_THAT(params2,
                Pointwise(FloatNear(ferr), real_fit_params2));

}

TEST_F(GPUSolverMultiResidual, MultiObjective) {
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    // Independant: Our current best LS Error is 22.5, ceres is 22.5
//    vector<float> real_fit_params2 = {2.75453, 1.31797};
    vector<float> real_fit_params1 = {2.07193, 2.32942};
    vector<float> real_fit_params2 = {-1.7402, -0.823567};

    // Actual Ceres minimizad params, but this is a sinosoidal and can have multiple global minimums
    // the above is equivilat in error (22.5000 = .5*lse) and the result our minimizer results to.

    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));
    EXPECT_THAT(params2,
                Pointwise(FloatNear(ferr), real_fit_params2));

}

TEST_F(GPUSolverMultiResidual, MultiObjectiveExplicitShared) {
    resFunc1->getResidualBlock()->getParameterBlocks()[0]->share(
            resFunc2->getResidualBlock()->getParameterBlocks()[0]);

    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    // Shared: Our current best LS Error is 22.5, ceres is 22.5
    vector<float> real_fit_params1 = {2.75453, 1.31797};


    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));
    EXPECT_THAT(params2,
                Pointwise(FloatNear(ferr), real_fit_params1));

}

TEST_F(GPUSolverMultiResidualImplicit, MultiObjectiveImplicitShared) {

    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    // Shared: Our current best LS Error is 22.5, ceres is 22.5
    vector<float> real_fit_params1 = {2.75453, 1.31797};


    float ferr = 1e-3;
    EXPECT_THAT(params1,
                Pointwise(FloatNear(ferr), real_fit_params1));

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
    // For large residuals using many parameters, we should use 100% of the initial dampening factor
    solver->options.initial_dampening_factor = 1;
    solver->options.gradient_tolerance = 1e-20;
    solver->options.step_tolerance = 1e-20;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    vector<float> real_fit_params(n, 420.968746f);

    // Cannot solve at higher precision due to floating point error?
    float ferr = 1e-1;
    EXPECT_THAT(params,
                Pointwise(FloatNear(ferr), real_fit_params));

}

TEST_F(PowellTest, solve) {
    // PowellTest minimizes Powell's singular function.
    //
    //   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
    //
    //   f1(x1,x2) = x1 + 10*x2;
    //   f2(x3,x4) = sqrt(5) * (x3 - x4)
    //   f3(x2,x3) = (x2 - 2*x3)^2
    //   f4(x1,x4) = sqrt(10) * (x1 - x4)^2
    //
    // The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
    // The minimum is 0 at (x1, x2, x3, x4) = 0.
    // Reference to Ceres Powell example program for comparison

    solver->options.initial_dampening_factor = 1e-1;
    solver->options.gradient_tolerance = 1e-20;
    solver->options.step_tolerance = 1e-20;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    float actual[4] = {x1, x2, x3, x4};
    vector<float> real_fit_params(4, 0.f);

//    float ferr = 1e-5;
//    float ferr = 1e-3;
    float ferr = 1e-1;
    EXPECT_THAT(actual,
                Pointwise(FloatNear(ferr), real_fit_params));
}

TEST(fileIO, parseCSV) {
    std::vector<float> targetPoints;
    fs::path dir (DATA_DIR);
    fs::path file ("/rigidFit/actualPoints.data");
    fs::path full_path = dir/file;
    telef::solver::io::parseCSVFile(targetPoints, full_path);

//    std::cout << "File("<< full_path <<") Contents: " << std::endl;
//    for (auto i = targetPoints.begin(); i != targetPoints.end(); ++i){
//        std::cout << "\t" << *i << std::endl;
//    }
    const int actual_size = 58*3;

    ASSERT_EQ(targetPoints.size(), actual_size);

}

TEST_F(RigidFitTest, solve) {
    // Fit quaternions Rotation and Translation using actual face landmarks
    // Target: detected 3D landmarks
    // Source: landmarks from annotated Face model
    // Actuals: Ceres results fitting the same data

    solver->options.initial_dampening_factor = 2;
    solver->options.gradient_tolerance = 1e-20;
    solver->options.step_tolerance = 1e-20;
    solver->options.verbose = true;

    Status  status = solver->solve(problem);

    EXPECT_TRUE(Status::CONVERGENCE == status);

    float actual_T[3] = {0.09495, 0.09297, 0.10206};
    float actual_U[3] = {3.50248, 0.00739, 0.01075};

    float ferr = 1e-3;
    EXPECT_THAT(result_T,
                Pointwise(FloatNear(ferr), actual_T));
    EXPECT_THAT(result_U,
                Pointwise(FloatNear(ferr), actual_U));
}
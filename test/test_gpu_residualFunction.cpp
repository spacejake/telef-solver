

#include <gtest/gtest.h> // googletest header file

#include <iostream>
#include <string>
#include "gmock/gmock.h"

#include "solver/gpu/gpuResidualFunction.h"

#include "util/cudautil.h"
#include "cuda/cuda_kernel.h"

using namespace telef::solver;
using namespace testing;

class TestCostFunction : public CostFunction {
public:
    TestCostFunction(){
        float measurements[] = {10,3,4,1};

        utils::CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
    }

    virtual ~TestCostFunction(){
        utils::CUDA_FREE(measurements_d);
    }

    virtual void evaluate(ResidualBlock::Ptr residualBlock, const bool computeJacobians) const {
        ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
        calc_res0(residualBlock->getResiduals(), params->getParameters(), measurements_d,
                  residualBlock->numResiduals(), params->numParameters());

        if (computeJacobians) {
            calc_jacobi0(params->getJacobians(), params->getParameters(),
                         residualBlock->numResiduals(), params->numParameters());
        }
    }
private:
    float* measurements_d;
};


//TEST(GPU_Residual_Function, interfaceTest) {
//
//    std::vector<int> params = {1, 2};
//    int nRes = 4;
//    GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, params);
//    CostFunction::Ptr cost = std::make_shared<TestCostFunction>();
//
//    GPUResidualFunction::Ptr resFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);
//    resFunc->evaluate(true);
//
//    EXPECT_TRUE(true);
//}

TEST(GPU_Residual_Function, evaluate) {
    /**
     * Measurement[0]:10.00
     * residuals[0]:-10.2631
     * Measurement[1]:3.00
     * residuals[1]:-3.2631
     * Measurement[2]:4.00
     * residuals[2]:-4.2631
     * Measurement[3]:1.00
     * residuals[3]:-1.2631
     * jacobians[0][0]:-1.0523
     * jacobians[0][1]:-4.9164
     * jacobians[1][0]:-1.0523
     * jacobians[1][1]:-4.9164
     * jacobians[2][0]:-1.0523
     * jacobians[2][1]:-4.9164
     * jacobians[3][0]:-1.0523
     * jacobians[3][1]:-4.9164
     */
    std::vector<int> params = {2};
    int nRes = 4;

    float params_init[] = {0.5,0.5};
    GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, params);
    ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks().at(0);
    cudaMemcpy(paramBlock->getParameters(), params_init, 2* sizeof(float), cudaMemcpyHostToDevice);

    CostFunction::Ptr cost = std::make_shared<TestCostFunction>();

    GPUResidualFunction::Ptr resFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);

    resFunc->evaluate(true);
    cudaDeviceSynchronize();

    float residuals[4];
    float jacobians[8];

    cudaMemcpy(residuals, resBlock->getResiduals(), nRes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy( jacobians,resBlock->getParameterBlocks()[0]->getJacobians(), nRes*2*sizeof(float), cudaMemcpyDeviceToHost);

    float real_res[] = {-10.2631, -3.2631, -4.2631, -1.2631};
    float real_jacobi[] = {-1.0523, -4.9164,
                           -1.0523, -4.9164,
                           -1.0523, -4.9164,
                           -1.0523, -4.9164};

    float ferr = 1e-4;
    EXPECT_THAT(residuals,
                Pointwise(FloatNear(ferr), real_res));
    EXPECT_THAT(jacobians,
                Pointwise(FloatNear(ferr), real_jacobi));
}

//TEST(GPU_Residual_Function, gradientTest) {
//
//    std::vector<int> params = {2};
//    int nRes = 4;
//    GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, params);
//    GPUParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks()[0];
//    CostFunction::Ptr cost = std::make_shared<TestCostFunction>();
//
//    GPUResidualFunction::Ptr resFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);
//
//
//    resFunc->calcGradients(paramBlock->getGradients(), paramBlock->getJacobians(), paramBlock->getParameters(),
//                           resBlock->numResiduals(), paramBlock->numParameters());
//
//
//    EXPECT_TRUE(true);
//}
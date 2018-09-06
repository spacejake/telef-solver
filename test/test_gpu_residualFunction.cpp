

#include <gtest/gtest.h> // googletest header file

#include <iostream>
#include <string>

#include "solver/gpu/gpuResidualFunction.h"

using namespace telef::solver;

class TestCostFunction : public CostFunction {
public:
    TestCostFunction(){}
    virtual ~TestCostFunction(){}

    virtual void evaluate(ResidualBlock::Ptr residualBlock) const {
        std::cout << "TestCostFunction Evaluate residuals and Jacobians!!" << std::endl;
    }
};


TEST(GPU_Residual_Function, interfaceTest) {

    std::vector<int> params = {1, 2};
    int nRes = 4;
    GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, params);
    CostFunction::Ptr cost = std::make_shared<TestCostFunction>();

    GPUResidualFunction::Ptr resFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);
    resFunc->evaluate(true);

    EXPECT_TRUE(true);
}
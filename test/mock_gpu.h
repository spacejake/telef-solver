#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>


#include "solver/gpu/gpuResidualFunction.h"

class TestCostFunction : public telef::solver::CostFunction {
public:
    TestCostFunction();
    virtual ~TestCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock, const bool computeJacobians) const;
private:
    float* measurements_d;
};

// A new one of these is created for each test
class GPUResidualFunctionTest : public testing::Test
{
public:

    telef::solver::GPUResidualFunction::Ptr residualFunc;

    virtual void SetUp();
    virtual void TearDown(){}

    void initResiduals();
    void initJacobians();
};
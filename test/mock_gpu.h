#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <solver/gpu/gpuSolver.h>
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

// A new one of these is created for each test
class GPUSolverTest : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    std::vector<float> params;

    virtual void SetUp();
    virtual void TearDown(){}
};
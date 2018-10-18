#pragma once

#include <cublas_v2.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <solver/gpu/gpuSolver.h>
#include <solver/gpu/gpuProblem.h>
#include "solver/gpu/gpuResidualFunction.h"

class TestCostFunctionSimple : public telef::solver::CostFunction {
public:
    TestCostFunctionSimple();
    virtual ~TestCostFunctionSimple();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock, const bool computeJacobians) const;
private:
    float* measurements_d;
};


class TestCostFunctionSimple2 : public telef::solver::CostFunction {
public:
    TestCostFunctionSimple2();
    virtual ~TestCostFunctionSimple2();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock, const bool computeJacobians) const;
private:
    float* measurements_d;
};


class TestCostFunction : public telef::solver::CostFunction {
public:
    TestCostFunction();
    virtual ~TestCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock, const bool computeJacobians) const;
private:
    float* measurements_d;
};

class TestMultiParamCostFunction : public telef::solver::CostFunction {
public:
    TestMultiParamCostFunction();
    virtual ~TestMultiParamCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock, const bool computeJacobians) const;
private:
    float* measurements_d;
};

// A new one of these is created for each test
class GPUResidualFunctionTest : public testing::Test
{
public:

    telef::solver::GPUResidualFunction::Ptr residualFunc;
    cublasHandle_t cublasHandle;

    virtual void SetUp();
    virtual void TearDown(){
        cublasDestroy_v2(cublasHandle);
    }

    void initResiduals();
    void initJacobians();
};

// A new one of these is created for each test
class GPUSolverTestSimple : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params;

    virtual void SetUp();
    virtual void TearDown();
};

// A new one of these is created for each test
class GPUSolverTest : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params;

    virtual void SetUp();
    virtual void TearDown();
};

// A new one of these is created for each test
class GPUSolverMultiParam : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params1;
    std::vector<float> params2;

    virtual void SetUp();
    virtual void TearDown();
};

// A new one of these is created for each test
class GPUSolverMultiResidual : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params1;
    std::vector<float> params2;

    virtual void SetUp();
    virtual void TearDown();
};
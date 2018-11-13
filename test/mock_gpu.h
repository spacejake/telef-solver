#pragma once

#include <cublas_v2.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <solver/gpu/gpuSolver.h>
#include <solver/gpu/gpuProblem.h>
#include "solver/gpu/gpuResidualFunction.h"

class TestCostFunctionSimple : public telef::solver::CostFunction {
public:
    TestCostFunctionSimple(int nRes, const std::vector<int>& paramSizes_);
    virtual ~TestCostFunctionSimple();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
private:
    float* measurements_d;
};


class TestCostFunctionSimple2 : public telef::solver::CostFunction {
public:
    TestCostFunctionSimple2(int nRes, const std::vector<int>& paramSizes_);
    virtual ~TestCostFunctionSimple2();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
private:
    float* measurements_d;
};


class TestCostFunction : public telef::solver::CostFunction {
public:
    TestCostFunction(int nRes, const std::vector<int>& paramSizes_);
    virtual ~TestCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
private:
    float* measurements_d;
};

class TestMultiParamCostFunction : public telef::solver::CostFunction {
public:
    TestMultiParamCostFunction(int nRes, const std::vector<int>& paramSizes_);
    virtual ~TestMultiParamCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
private:
    float* measurements_d;
};

class Test4ParamCostFunction : public telef::solver::CostFunction {
public:
    Test4ParamCostFunction(int nRes, const std::vector<int>& paramSizes_);
    virtual ~Test4ParamCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
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
class GPUSolver4Param : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params1;
    std::vector<float> params2;
    std::vector<float> params3;
    std::vector<float> params4;

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
    telef::solver::ResidualFunction::Ptr resFunc1;
    telef::solver::ResidualFunction::Ptr resFunc2;

    virtual void SetUp();
    virtual void TearDown();
};

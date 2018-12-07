#pragma once

#include <cublas_v2.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <solver/gpu/gpuSolver.h>
#include <solver/gpu/gpuProblem.h>
#include "solver/gpu/gpuResidualFunction.h"



/*************************Benchmarks*************************************/

//BealesTest Start
// A new one of these is created for each test
class BealesTest : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params;

    virtual void SetUp();
    virtual void TearDown();
};

class BealesCostFunction : public telef::solver::CostFunction {
public:
    BealesCostFunction(int nRes, const std::vector<int>& paramSizes_);
    virtual ~BealesCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
private:
    //float* measurements_d;
};
//BealesTest End

//SchwefelTest Start
class SchwefelTest : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params;
    int n;

    virtual void SetUp();
    virtual void TearDown();
};

class SchwefelCostFunction : public telef::solver::CostFunction {
public:
    SchwefelCostFunction(int nRes, const std::vector<int>& paramSizes_);
    virtual ~SchwefelCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
private:
    //float* measurements_d;
};
//SchwefelTest End

//PowellTest Start
class PowellTest : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;
    std::vector<float> params;

    float x1;
    float x2;
    float x3;
    float x4;

    virtual void SetUp();
    virtual void TearDown();
};

class PowellCostFunction1 : public telef::solver::CostFunction {
public:
    PowellCostFunction1(int nRes, const std::vector<int>& paramSizes_)
            : telef::solver::CostFunction(nRes, paramSizes_){}
    virtual ~PowellCostFunction1(){}
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
};

class PowellCostFunction2 : public telef::solver::CostFunction {
public:
    PowellCostFunction2(int nRes, const std::vector<int>& paramSizes_)
            : telef::solver::CostFunction(nRes, paramSizes_){}
    virtual ~PowellCostFunction2(){}
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
};

class PowellCostFunction3 : public telef::solver::CostFunction {
public:
    PowellCostFunction3(int nRes, const std::vector<int>& paramSizes_)
            : telef::solver::CostFunction(nRes, paramSizes_){}
    virtual ~PowellCostFunction3(){}
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
};

class PowellCostFunction4 : public telef::solver::CostFunction {
public:
    PowellCostFunction4(int nRes, const std::vector<int>& paramSizes_)
        : telef::solver::CostFunction(nRes, paramSizes_){}
    virtual ~PowellCostFunction4(){}
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
};
//PowellTest End


/*************************Benchmarks End*************************************/
/*************************Rigid Fitting Start*************************************/
class RigidFitTest : public testing::Test
{
public:

    telef::solver::GPUSolver::Ptr solver;
    telef::solver::GPUProblem::Ptr problem;

    std::vector<float> result_T;
    std::vector<float> result_U;

    int n;

    void loadData(std::vector<float> &data, std::string file);

    virtual void SetUp();
    virtual void TearDown();
};

class RigidFitCostFunction : public telef::solver::CostFunction {
public:
    RigidFitCostFunction(std::vector<float> source, std::vector<float> target);
    virtual ~RigidFitCostFunction();
    virtual void evaluate(telef::solver::ResidualBlock::Ptr residualBlock);
    virtual void computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock);
private:
    cublasHandle_t cublasHandle;

    float* source_d;
    float* target_d;
    float* fitted_d;
};
/*************************Rigid Fitting End*************************************/


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
private:
};

// A new one of these is created for each test
class GPUSolverMultiResidualImplicit : public GPUSolverMultiResidual
{
public:
    virtual void SetUp();
private:
};
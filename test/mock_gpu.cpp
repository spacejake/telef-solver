#include "mock_gpu.h"

#include "util/cudautil.h"
#include "cuda/cuda_kernel.h"

using namespace telef::solver::utils;
using namespace telef::solver;

//TestCostFunctionSimple START
TestCostFunctionSimple::TestCostFunctionSimple(){
    float measurements[] = {10};
    CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestCostFunctionSimple::~TestCostFunctionSimple(){
    CUDA_FREE(measurements_d);
}

void TestCostFunctionSimple::evaluate(ResidualBlock::Ptr residualBlock, const bool computeJacobians) const {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_resSimple(residualBlock->getResiduals(), params->getWorkingParameters(), measurements_d,
            residualBlock->numResiduals(), params->numParameters());

    if (computeJacobians) {
        calc_jacobiSimple(params->getJacobians(), params->getWorkingParameters(),
                     residualBlock->numResiduals(), params->numParameters());
    }
}
//TestCostFunctionSimple END

//TestCostFunctionSimple START
TestCostFunctionSimple2::TestCostFunctionSimple2(){
    float measurements[] = {10};
    CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestCostFunctionSimple2::~TestCostFunctionSimple2(){
    CUDA_FREE(measurements_d);
}

void TestCostFunctionSimple2::evaluate(ResidualBlock::Ptr residualBlock, const bool computeJacobians) const {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_resSimple2(residualBlock->getResiduals(), params->getWorkingParameters(), measurements_d,
                   residualBlock->numResiduals(), params->numParameters());

    if (computeJacobians) {
        calc_jacobiSimple2(params->getJacobians(), params->getWorkingParameters(),
                          residualBlock->numResiduals(), params->numParameters());
    }
}
//TestCostFunctionSimple END

//TestCostFunction START
TestCostFunction::TestCostFunction(){
    float measurements[] = {10,3,4,1};
    CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestCostFunction::~TestCostFunction(){
    CUDA_FREE(measurements_d);
}

void TestCostFunction::evaluate(ResidualBlock::Ptr residualBlock, const bool computeJacobians) const {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_res0(residualBlock->getResiduals(), params->getWorkingParameters(), measurements_d,
              residualBlock->numResiduals(), params->numParameters());

    if (computeJacobians) {
        calc_jacobi0(params->getJacobians(), params->getWorkingParameters(),
                     residualBlock->numResiduals(), params->numParameters());
    }
}
//TestCostFunction END

//TestMultiParamCostFunction START
TestMultiParamCostFunction::TestMultiParamCostFunction(){
    float measurements[] = {10,3,4,1};
    CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestMultiParamCostFunction::~TestMultiParamCostFunction(){
    CUDA_FREE(measurements_d);
}

void TestMultiParamCostFunction::evaluate(ResidualBlock::Ptr residualBlock, const bool computeJacobians) const {
    ParameterBlock::Ptr params1 = residualBlock->getParameterBlocks()[0];
    ParameterBlock::Ptr params2 = residualBlock->getParameterBlocks()[1];

    calc_res2Params(residualBlock->getResiduals(), params1->getWorkingParameters(), params2->getWorkingParameters(),
            measurements_d, residualBlock->numResiduals(), params1->numParameters(), params2->numParameters());

    if (computeJacobians) {
        calc_jacobi2Params(params1->getJacobians(), params2->getJacobians(),
                params1->getWorkingParameters(), params2->getWorkingParameters(),
                residualBlock->numResiduals(), params1->numParameters(), params2->numParameters());
    }
}
//TestMultiParamCostFunction END

//GPUResidualFunctionTest START
void GPUResidualFunctionTest::SetUp()
{
    if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Cublas could not be initialized");
    }

    std::vector<int> nParams = {2};
    int nRes = 4;
    std::vector<float> params = {0.5,0.5};
    GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, nParams);
    ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks().at(0);
    paramBlock->setInitialParams(params.data());
    paramBlock->initializeParameters();

    CostFunction::Ptr cost = std::make_shared<TestCostFunction>();

    residualFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);
    residualFunc->setCublasHandle(cublasHandle);


//        float residuals[] = {10, 3, 4, 1};
//        float jacobians[] = {1, 1, 1, 1,
//                             4, 4, 4, 4};
    float residuals[] = {-10.2631, -3.2631, -4.2631, -1.2631};
    float jacobians[] = {-1.0523, -1.0523, -1.0523, -1.0523,
                         -4.9164, -4.9164, -4.9164, -4.9164};
}

void GPUResidualFunctionTest::initResiduals(){
    auto resBlock = residualFunc->getResidualBlock();
    int nRes = resBlock->numResiduals();
    float residuals[] = {-10.2631, -3.2631, -4.2631, -1.2631};
    cudaMemcpy(resBlock->getResiduals(), residuals, nRes*sizeof(float), cudaMemcpyHostToDevice);
}

void GPUResidualFunctionTest::initJacobians(){
    auto resBlock = residualFunc->getResidualBlock();
    auto paramBlock = resBlock->getParameterBlocks()[0];
    int nRes = resBlock->numResiduals();
    int nParams = paramBlock->numParameters();
    float jacobians[] = {-1.0523, -1.0523, -1.0523, -1.0523,
                         -4.9164, -4.9164, -4.9164, -4.9164};
    // row-order
//        float jacobians[] = {-1.0523, -4.9164,
//                             -1.0523, -4.9164,
//                             -1.0523, -4.9164,
//                             -1.0523, -4.9164};
    cudaMemcpy(paramBlock->getJacobians(), jacobians, nRes*nParams*sizeof(float), cudaMemcpyHostToDevice);
}
//GPUResidualFunctionTest END

//GPUSolverTestSimple START
void GPUSolverTestSimple::SetUp()
{

    std::vector<int> nParams = {1};
    int nRes = 1;
    auto resBlock = std::make_shared<GPUResidualBlock>(nRes, nParams);
    auto cost = std::make_shared<TestCostFunctionSimple>();
    auto residualFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);

    solver = std::make_shared<GPUSolver>();
    params = {0.5f};
    std::vector<float*> initParams = {params.data()};

    problem = std::make_shared<GPUProblem>();
    problem->addResidualFunction(residualFunc, initParams);
    //problem->initialize();
}

void GPUSolverTestSimple::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverTestSimple END

//GPUSolverTest START
void GPUSolverTest::SetUp()
{
    std::vector<int> nParams = {2};
    int nRes = 4;
    auto resBlock = std::make_shared<GPUResidualBlock>(nRes, nParams);
    auto cost = std::make_shared<TestCostFunction>();
    auto residualFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);

    solver = std::make_shared<GPUSolver>();
    params = {0.5,0.5};
//    params = {-2.60216,0.0318891};
    std::vector<float*> initParams = {params.data()};

    problem = std::make_shared<GPUProblem>();
    problem->addResidualFunction(residualFunc, initParams);
    problem->initialize();
}

void GPUSolverTest::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverTest END

//GPUSolverMultiParam START
void GPUSolverMultiParam::SetUp()
{
    std::vector<int> nParams = {2,1};
    int nRes = 4;
    auto resBlock = std::make_shared<GPUResidualBlock>(nRes, nParams);
    auto cost = std::make_shared<TestMultiParamCostFunction>();
    auto residualFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);

    solver = std::make_shared<GPUSolver>();
    params1 = {0.5,0.5};
    params2 = {0.5};

    std::vector<float*> initParams = {params1.data(), params2.data()};

    problem = std::make_shared<GPUProblem>();
    problem->addResidualFunction(residualFunc, initParams);
    problem->initialize();
}

void GPUSolverMultiParam::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverMultiParam END

//GPUSolverMultiResidual START
void GPUSolverMultiResidual::SetUp()
{
    solver = std::make_shared<GPUSolver>();

    std::vector<int> nParams2 = {2};
    int nRes2 = 4;
    auto resBlock2 = std::make_shared<GPUResidualBlock>(nRes2, nParams2);
    auto cost2 = std::make_shared<TestCostFunction>();
    auto residualFunc2 = std::make_shared<GPUResidualFunction>(cost2, resBlock2, 1.0);
    params2 = {0.5,0.5};
    std::vector<float*> initParams2 = {params2.data()};

    problem = std::make_shared<GPUProblem>();
    problem->addResidualFunction(residualFunc2, initParams2);

    std::vector<int> nParams1 = {2};
    int nRes1 = 1;
    auto resBlock1 = std::make_shared<GPUResidualBlock>(nRes1, nParams1);
    auto cost1 = std::make_shared<TestCostFunctionSimple2>();
    auto residualFunc1 = std::make_shared<GPUResidualFunction>(cost1, resBlock1, 1.0);
    resBlock1->getParameterBlocks()[0]->share(resBlock2->getParameterBlocks()[0]);
    params1 = {0.5f,0.5};
    std::vector<float*> initParams1 = {params2.data()};
    problem->addResidualFunction(residualFunc1, initParams1);
    problem->initialize();
}

void GPUSolverMultiResidual::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverMultiResidual END
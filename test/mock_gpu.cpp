#include "mock_gpu.h"

#include "util/cudautil.h"
#include "cuda/cuda_kernel.h"

using namespace telef::solver::utils;

TestCostFunction::TestCostFunction(){
    float measurements[] = {10,3,4,1};
    CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestCostFunction::~TestCostFunction(){
    CUDA_FREE(measurements_d);
}

void TestCostFunction::evaluate(telef::solver::ResidualBlock::Ptr residualBlock, const bool computeJacobians) const {
    telef::solver::ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_res0(residualBlock->getResiduals(), params->getParameters(), measurements_d,
              residualBlock->numResiduals(), params->numParameters());

    if (computeJacobians) {
        calc_jacobi0(params->getJacobians(), params->getParameters(),
                     residualBlock->numResiduals(), params->numParameters());
    }
}

void GPUResidualFunctionTest::SetUp()
{
    std::vector<int> params = {2};
    int nRes = 4;
    float params_init[] = {0.5,0.5};
    telef::solver::GPUResidualBlock::Ptr resBlock = std::make_shared<telef::solver::GPUResidualBlock>(nRes, params);
    telef::solver::ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks().at(0);
    cudaMemcpy(paramBlock->getParameters(), params_init, params[0]* sizeof(float), cudaMemcpyHostToDevice);

    telef::solver::CostFunction::Ptr cost = std::make_shared<TestCostFunction>();

    residualFunc = std::make_shared<telef::solver::GPUResidualFunction>(cost, resBlock, 1.0);



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
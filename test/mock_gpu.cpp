#include "mock_gpu.h"

#include "util/cudautil.h"
#include "cuda/cuda_kernel.h"

using namespace telef::solver::utils;
using namespace telef::solver;


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

//GPUResidualFunctionTest START
void GPUResidualFunctionTest::SetUp()
{
    std::vector<int> nParams = {2};
    int nRes = 4;
    std::vector<float> params = {0.5,0.5};
    GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, nParams);
    ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks().at(0);
    paramBlock->setInitialParams(params.data());
    paramBlock->initializeParameters();

    CostFunction::Ptr cost = std::make_shared<TestCostFunction>();

    residualFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);



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
    std::vector<float*> initParams = {params.data()};
    solver->addResidualFunction(residualFunc, initParams);
}
//GPUSolverTest END

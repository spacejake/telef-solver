#include "mock_gpu.h"

#include "solver/util/cudautil.h"
#include "cuda/cuda_kernel.h"
#include "cuda/cuda_powell.h"

using namespace telef::solver;

//PowellTest START
void PowellTest::SetUp()
{
    // Initial Params
    x1 =  3.0;
    x2 = -1.0;
    x3 =  0.0;
    x4 =  1.0;

    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams = {1,1};
    int nRes = 1;

    // Add residual terms to the problem using the using the autodiff
    // wrapper to get the derivatives automatically. The parameters, x1 through
    // x4, are modified in place.
    auto resFunc1 = problem->addResidualFunction(
            std::make_shared<PowellCostFunction1>(nRes, nParams), {&x1, &x2});
    auto resFunc2 = problem->addResidualFunction(
            std::make_shared<PowellCostFunction2>(nRes, nParams), {&x3, &x4});
    auto resFunc3 = problem->addResidualFunction(
            std::make_shared<PowellCostFunction3>(nRes, nParams), {&x2, &x3});
    auto resFunc4 = problem->addResidualFunction(
            std::make_shared<PowellCostFunction4>(nRes, nParams), {&x1, &x4});


    // Explicitly share parameters, TODO: add implicit base on pointer -> parameterBlock, share in addResidualFunction.
//    resFunc3->getResidualBlock()->getParameterBlocks()[0]->share(
//            resFunc1->getResidualBlock()->getParameterBlocks()[1]);
//    resFunc3->getResidualBlock()->getParameterBlocks()[1]->share(
//            resFunc2->getResidualBlock()->getParameterBlocks()[0]);
//
//    resFunc4->getResidualBlock()->getParameterBlocks()[0]->share(
//            resFunc1->getResidualBlock()->getParameterBlocks()[0]);
//    resFunc4->getResidualBlock()->getParameterBlocks()[1]->share(
//            resFunc2->getResidualBlock()->getParameterBlocks()[1]);
}

void PowellTest::TearDown() {
}
//PowellTest END


//PowellCostFunction 1 START
void PowellCostFunction1::evaluate(ResidualBlock::Ptr residualBlock) {
    float* x1 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x2 = residualBlock->getParameterBlocks()[1]->getParameters();
    powell_res1(residualBlock->getResiduals(), x1, x2);
}

void PowellCostFunction1::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    float* x1 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x2 = residualBlock->getParameterBlocks()[1]->getParameters();

    float* j1 = residualBlock->getParameterBlocks()[0]->getJacobians();
    float* j2 = residualBlock->getParameterBlocks()[1]->getJacobians();
    powell_jacobi1(j1, j2, x1, x2);
}
//PowellCostFunction 1 END

//PowellCostFunction 2 START
void PowellCostFunction2::evaluate(ResidualBlock::Ptr residualBlock) {
    float* x3 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x4 = residualBlock->getParameterBlocks()[1]->getParameters();
    powell_res2(residualBlock->getResiduals(), x3, x4);
}

void PowellCostFunction2::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    float* x3 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x4 = residualBlock->getParameterBlocks()[1]->getParameters();

    float* j3 = residualBlock->getParameterBlocks()[0]->getJacobians();
    float* j4 = residualBlock->getParameterBlocks()[1]->getJacobians();
    powell_jacobi2(j3, j4, x3, x4);

}
//PowellCostFunction2 END

//PowellCostFunction3 START
void PowellCostFunction3::evaluate(ResidualBlock::Ptr residualBlock) {
    float* x2 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x3 = residualBlock->getParameterBlocks()[1]->getParameters();
    powell_res3(residualBlock->getResiduals(), x2, x3);

}

void PowellCostFunction3::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    float* x2 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x3 = residualBlock->getParameterBlocks()[1]->getParameters();

    float* j2 = residualBlock->getParameterBlocks()[0]->getJacobians();
    float* j3 = residualBlock->getParameterBlocks()[1]->getJacobians();
    powell_jacobi3(j2, j3, x2, x3);

}
//PowellCostFunction3 END

//PowellCostFunction4 START
void PowellCostFunction4::evaluate(ResidualBlock::Ptr residualBlock) {
    float* x1 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x4 = residualBlock->getParameterBlocks()[1]->getParameters();
    powell_res4(residualBlock->getResiduals(), x1, x4);
}

void PowellCostFunction4::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    float* x1 = residualBlock->getParameterBlocks()[0]->getParameters();
    float* x4 = residualBlock->getParameterBlocks()[1]->getParameters();

    float* j1 = residualBlock->getParameterBlocks()[0]->getJacobians();
    float* j4 = residualBlock->getParameterBlocks()[1]->getJacobians();
    powell_jacobi4(j1, j4, x1, x4);

}
//PowellCostFunction4 END

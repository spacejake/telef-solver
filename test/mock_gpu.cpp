#include "mock_gpu.h"

#include "solver/util/cudautil.h"
#include "cuda/cuda_kernel.h"

using namespace telef::solver;

//TestCostFunctionSimple START
TestCostFunctionSimple::TestCostFunctionSimple(int nRes, const std::vector<int>& paramSizes_) : telef::solver::CostFunction(nRes, paramSizes_){
    float measurements[] = {10};
    SOLVER_CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestCostFunctionSimple::~TestCostFunctionSimple(){
    SOLVER_CUDA_FREE(measurements_d);
}

void TestCostFunctionSimple::evaluate(ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_resSimple(residualBlock->getResiduals(), params->getParameters(), measurements_d,
            residualBlock->numResiduals(), params->numParameters());
}

void TestCostFunctionSimple::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_jacobiSimple(params->getJacobians(), params->getParameters(),
                      residualBlock->numResiduals(), params->numParameters());

}
//TestCostFunctionSimple END

//TestCostFunctionSimple START
TestCostFunctionSimple2::TestCostFunctionSimple2(int nRes, const std::vector<int>& paramSizes_) : telef::solver::CostFunction(nRes, paramSizes_){
    float measurements[] = {10};
    SOLVER_CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestCostFunctionSimple2::~TestCostFunctionSimple2(){
    SOLVER_CUDA_FREE(measurements_d);
}

void TestCostFunctionSimple2::evaluate(ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_resSimple2(residualBlock->getResiduals(), params->getParameters(), measurements_d,
                   residualBlock->numResiduals(), params->numParameters());
}

void TestCostFunctionSimple2::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_jacobiSimple2(params->getJacobians(), params->getParameters(),
                       residualBlock->numResiduals(), params->numParameters());

}
//TestCostFunctionSimple END

//TestCostFunction START
TestCostFunction::TestCostFunction(int nRes, const std::vector<int>& paramSizes_) : telef::solver::CostFunction(nRes, paramSizes_){
    float measurements[] = {10,3,4,1};
    SOLVER_CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestCostFunction::~TestCostFunction(){
    SOLVER_CUDA_FREE(measurements_d);
}

void TestCostFunction::evaluate(ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_res0(residualBlock->getResiduals(), params->getParameters(), measurements_d,
              residualBlock->numResiduals(), params->numParameters());
}

void TestCostFunction::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    calc_jacobi0(params->getJacobians(), params->getParameters(),
                 residualBlock->numResiduals(), params->numParameters());
}
//TestCostFunction END

//BealesCostFunction START
BealesCostFunction::BealesCostFunction(int nRes, const std::vector<int>& paramSizes_) : telef::solver::CostFunction(nRes, paramSizes_){
    //float measurements[] = {1.5,2.25,2.625};
    //SOLVER_CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

BealesCostFunction::~BealesCostFunction(){
    //SOLVER_CUDA_FREE(measurements_d);
}

void BealesCostFunction::evaluate(ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    beales_res(residualBlock->getResiduals(), params->getParameters(), residualBlock->numResiduals(), params->numParameters());
}

void BealesCostFunction::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    beales_jacobi(params->getJacobians(), params->getParameters(),
                 residualBlock->numResiduals(), params->numParameters());
}
//BealesCostFunction END

//SchwefelCostFunction START
SchwefelCostFunction::SchwefelCostFunction(int nRes, const std::vector<int>& paramSizes_) : telef::solver::CostFunction(nRes, paramSizes_){
    //float measurements[] = {1.5,2.25,2.625};
    //SOLVER_CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

SchwefelCostFunction::~SchwefelCostFunction(){
    //SOLVER_CUDA_FREE(measurements_d);
}

void SchwefelCostFunction::evaluate(ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    schwefel_res(residualBlock->getResiduals(), params->getParameters(), residualBlock->numResiduals(), params->numParameters());
}

void SchwefelCostFunction::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
    schwefel_jacobi(params->getJacobians(), params->getParameters(),
                  residualBlock->numResiduals(), params->numParameters());
}
//SchwefelCostFunction END

//TestMultiParamCostFunction START
TestMultiParamCostFunction::TestMultiParamCostFunction(int nRes, const std::vector<int>& paramSizes_) : telef::solver::CostFunction(nRes, paramSizes_){
    float measurements[] = {10,3,4,1};
    SOLVER_CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
}

TestMultiParamCostFunction::~TestMultiParamCostFunction(){
    SOLVER_CUDA_FREE(measurements_d);
}

void TestMultiParamCostFunction::evaluate(ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params1 = residualBlock->getParameterBlocks()[0];
    ParameterBlock::Ptr params2 = residualBlock->getParameterBlocks()[1];

    calc_res2Params(residualBlock->getResiduals(), params1->getParameters(), params2->getParameters(),
            measurements_d, residualBlock->numResiduals(), params1->numParameters(), params2->numParameters());
}

void TestMultiParamCostFunction::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params1 = residualBlock->getParameterBlocks()[0];
    ParameterBlock::Ptr params2 = residualBlock->getParameterBlocks()[1];

    calc_jacobi2Params(params1->getJacobians(), params2->getJacobians(),
                       params1->getParameters(), params2->getParameters(),
                       residualBlock->numResiduals(), params1->numParameters(), params2->numParameters());
}
//TestMultiParamCostFunction END

//Test4ParamCostFunction START
Test4ParamCostFunction::Test4ParamCostFunction(int nRes, const std::vector<int>& paramSizes_) : telef::solver::CostFunction(nRes, paramSizes_){
    float measurements[] = {10,3,4,1,76,43,23,89,23,44};
    SOLVER_CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(10));
}

Test4ParamCostFunction::~Test4ParamCostFunction(){
    SOLVER_CUDA_FREE(measurements_d);
}

void Test4ParamCostFunction::evaluate(ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params1 = residualBlock->getParameterBlocks()[0];
    ParameterBlock::Ptr params2 = residualBlock->getParameterBlocks()[1];
    ParameterBlock::Ptr params3 = residualBlock->getParameterBlocks()[2];
    ParameterBlock::Ptr params4 = residualBlock->getParameterBlocks()[3];

    calc_res4Params(residualBlock->getResiduals(),
                    params1->getParameters(), params2->getParameters(), params3->getParameters(),
                    params4->getParameters(),
                    measurements_d, residualBlock->numResiduals(),
                    params1->numParameters(), params2->numParameters(), params3->numParameters(), params4->numParameters());
}

void Test4ParamCostFunction::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    ParameterBlock::Ptr params1 = residualBlock->getParameterBlocks()[0];
    ParameterBlock::Ptr params2 = residualBlock->getParameterBlocks()[1];
    ParameterBlock::Ptr params3 = residualBlock->getParameterBlocks()[2];
    ParameterBlock::Ptr params4 = residualBlock->getParameterBlocks()[3];

    calc_jacobi4Params(params1->getJacobians(), params2->getJacobians(), params3->getJacobians(), params4->getJacobians(),
                       params1->getParameters(), params2->getParameters(), params3->getParameters(),
                       params4->getParameters(),
                       residualBlock->numResiduals(),
                       params1->numParameters(), params2->numParameters(), params3->numParameters(), params4->numParameters());
}
//Test4ParamCostFunction END

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

    CostFunction::Ptr cost = std::make_shared<TestCostFunction>(nRes, nParams);

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
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams = {1};
    int nRes = 1;
    auto cost = std::make_shared<TestCostFunctionSimple>(nRes, nParams);

    params = {0.5f};
    std::vector<float*> initParams = {params.data()};

    auto resFunc = problem->addResidualFunction(cost, initParams);
    //problem->initialize();
}

void GPUSolverTestSimple::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverTestSimple END

//GPUSolverTest START
void GPUSolverTest::SetUp()
{
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams = {2};
    int nRes = 4;
    auto cost = std::make_shared<TestCostFunction>(nRes, nParams);

    params = {0.5,0.5};
//    params = {-2.60216,0.0318891};
    std::vector<float*> initParams = {params.data()};

    auto resFunc = problem->addResidualFunction(cost, initParams);
//    problem->initialize();
}

void GPUSolverTest::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverTest END

//BaelesTest START
void BealesTest::SetUp()
{
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams = {2};
    int nRes = 3;
    auto cost = std::make_shared<BealesCostFunction>(nRes, nParams);

    params = {4, 1};
    std::vector<float*> initParams = {params.data()};

    auto resFunc = problem->addResidualFunction(cost, initParams);
}

void BealesTest::TearDown() {
}

//BaelesTest END

//SchwefelTest START
void SchwefelTest::SetUp()
{
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();
    n = 1024;

    std::vector<int> nParams = {n};
    int nRes = 1;
    auto cost = std::make_shared<SchwefelCostFunction>(nRes, nParams);

    // due to d/dx abs(x) = abs(x)/x, non-differentiable at 0. Don't start at 0,
    // otherwise solver will considards converged (gradient will be 0).
    // This is a flaw of the LM algorithm, when x0 is a local minimizer and not x* (global minimizer),
    // LM quits if initial gradient is 0.
    params = std::vector<float>(n, 1.f);
    std::vector<float*> initParams = {params.data()};

    auto resFunc = problem->addResidualFunction(cost, initParams);
}

void SchwefelTest::TearDown() {
}
//SchwefelTest END



//GPUSolverMultiParam START
void GPUSolverMultiParam::SetUp()
{
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams = {2,1};
    int nRes = 4;
    auto cost = std::make_shared<TestMultiParamCostFunction>(nRes, nParams);

    params1 = {0.5,0.5};
    params2 = {0.5};

    std::vector<float*> initParams = {params1.data(), params2.data()};
    auto resFunc = problem->addResidualFunction(cost, initParams);
//    problem->initialize();
}

void GPUSolverMultiParam::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverMultiParam END

//GPUSolverMultiParam START
void GPUSolver4Param::SetUp()
{
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams = {2,1,2,2};
    int nRes = 10;
    auto cost = std::make_shared<Test4ParamCostFunction>(nRes, nParams);

    params1 = {0.5,0.5};
    params2 = {0.5};
    params3 = {0.5,0.5};
    params4 = {0.5,0.5};

    std::vector<float*> initParams = {params1.data(), params2.data(), params3.data(), params4.data()};
    auto resFunc = problem->addResidualFunction(cost, initParams);
//    problem->initialize();
}

void GPUSolver4Param::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverMultiParam END

//GPUSolverMultiResidual START
void GPUSolverMultiResidual::SetUp()
{
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams1 = {2};
    int nRes1 = 1;
    auto cost1 = std::make_shared<TestCostFunctionSimple2>(nRes1, nParams1);
    params1 = {0.5f,0.5};
    std::vector<float*> initParams1 = {params1.data()};

    std::vector<int> nParams2 = {2};
    int nRes2 = 4;
    auto cost2 = std::make_shared<TestCostFunction>(nRes2, nParams2);
    params2 = {0.5,0.5};
    std::vector<float *> initParams2 = {params2.data()};

    resFunc1 = problem->addResidualFunction(cost1, initParams1);
    resFunc2 = problem->addResidualFunction(cost2, initParams2);

    //resFunc1->getResidualBlock()->getParameterBlocks()[0]->share(
    //        resFunc2->getResidualBlock()->getParameterBlocks()[0]);

//    problem->initialize();
}

void GPUSolverMultiResidual::TearDown() {
//    cudaDeviceReset();
}

//GPUSolverMultiResidual END

//GPUSolverMultiResidualImplicit START
void GPUSolverMultiResidualImplicit::SetUp()
{
    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<int> nParams1 = {2};
    int nRes1 = 1;
    auto cost1 = std::make_shared<TestCostFunctionSimple2>(nRes1, nParams1);
    params1 = {0.5f,0.5};
    std::vector<float*> initParams1 = {params1.data()};

    std::vector<int> nParams2 = {2};
    int nRes2 = 4;
    auto cost2 = std::make_shared<TestCostFunction>(nRes2, nParams2);

    resFunc1 = problem->addResidualFunction(cost1, initParams1);
    resFunc2 = problem->addResidualFunction(cost2, initParams1);

    //resFunc1->getResidualBlock()->getParameterBlocks()[0]->share(
    //        resFunc2->getResidualBlock()->getParameterBlocks()[0]);

//    problem->initialize();
}

//GPUSolverMultiResidualImplicit END

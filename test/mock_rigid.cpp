#include "mock_gpu.h"

#include <experimental/filesystem>
#include <iostream>
#include <string>

#include "solver/util/cudautil.h"
#include "solver/util/fileio.h"
#include "cuda/cuda_align.h"

#ifdef TEST_DATA_DIR
#define DATA_DIR TEST_DATA_DIR
#else
#define DATA_DIR "NO_FILE"
#endif

#define ROTATE_COEFF 3
#define TRANSLATE_COEFF 3

namespace fs = std::experimental::filesystem;
using namespace std;
using namespace telef::solver;
using namespace telef::solver::io;


//RigidFitTest START

void RigidFitTest::loadData(std::vector<float> &data, std::string file){
    fs::path dir (DATA_DIR);
    fs::path full_path = dir / fs::path(file);
    parseCSVFile(data, full_path);
}

void RigidFitTest::SetUp()
{
    // Initial Params
    result_T = {0.f, 0.f, 0.f};
    result_U = {3.14f, 0.f, 0.f};


    solver = std::make_shared<GPUSolver>();
    problem = std::make_shared<GPUProblem>();

    std::vector<float> targetPoints;
    loadData(targetPoints, "/rigidFit/targetPoints.data");

    std::vector<float> sourcePoints;
    loadData(sourcePoints, "/rigidFit/sourcePoints.data");

    // Actually fitted points using ceres, for comparison
//    std::vector<float> actualPoints;
//    loadData(targetPoints, "/rigidFit/actualPoints.data");

    // Add residual terms to the problem using the using the autodiff
    // wrapper to get the derivatives automatically. The parameters, x1 through
    // x4, are modified in place.
    problem->addResidualFunction(
            std::make_shared<RigidFitCostFunction>(sourcePoints, targetPoints),
            {
                result_T.data(),
                result_U.data()
            });
}

void RigidFitTest::TearDown() {
}
//RigidFitTest END

//RigidFitCostFunction START
RigidFitCostFunction::RigidFitCostFunction(std::vector<float> source, std::vector<float> target)
    : CostFunction(source.size(), {TRANSLATE_COEFF, ROTATE_COEFF})
{
    assert(source.size() == target.size() && "Source and Target are different sizes, cannot establish correspondance");

    if(cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Cublas could not be initialized");
    }

    SOLVER_CUDA_ALLOC_AND_COPY(&source_d, source.data(), static_cast<size_t>(source.size()));
    SOLVER_CUDA_ALLOC_AND_COPY(&target_d, target.data(), static_cast<size_t>(target.size()));
    SOLVER_CUDA_MALLOC(&fitted_d, static_cast<size_t>(numResiduals()));
}

RigidFitCostFunction::~RigidFitCostFunction() {
    cublasDestroy(cublasHandle);
    SOLVER_CUDA_FREE(source_d);
    SOLVER_CUDA_FREE(target_d);
    SOLVER_CUDA_FREE(fitted_d);
}

void RigidFitCostFunction::evaluate(ResidualBlock::Ptr residualBlock) {
    auto ftParams = residualBlock->getParameterBlocks()[0];
    auto fuParams = residualBlock->getParameterBlocks()[1];

    float* residuals_d = residualBlock->getResiduals();
    int pointCount = residualBlock->numResiduals()/3;

    auto ft = ftParams->getParameters();
    auto fu = fuParams->getParameters();
//    print_array("source", source_d, residualBlock->numResiduals());
    alignPoints(cublasHandle, fitted_d, source_d, ft, fu, pointCount);
//    print_array("fitted", fitted_d, residualBlock->numResiduals());
//    print_array("target", target_d, residualBlock->numResiduals());

    SOLVER_CUDA_ZERO(&residuals_d, static_cast<size_t>(pointCount));
    calculatePointLoss(residuals_d, fitted_d, target_d, residualBlock->numResiduals());
//    print_array("residuals", residuals_d, residualBlock->numResiduals());
}

void RigidFitCostFunction::computeJacobians(telef::solver::ResidualBlock::Ptr residualBlock) {
    auto ftParams = residualBlock->getParameterBlocks()[0];
    auto fuParams = residualBlock->getParameterBlocks()[1];
    int pointCount = residualBlock->numResiduals();

    auto fu = fuParams->getParameters();

    auto ftJacobian_d = ftParams->getJacobians();
    int numtj = residualBlock->numResiduals() * ftParams->numParameters();

    auto fuJacobian_d = fuParams->getJacobians();
    int numuj = residualBlock->numResiduals() * fuParams->numParameters();

    SOLVER_CUDA_CHECK(cudaMemset(ftJacobian_d, 0, numtj*sizeof(float)));
    SOLVER_CUDA_CHECK(cudaMemset(fuJacobian_d, 0, numuj*sizeof(float)));
    calculateJacobians(ftJacobian_d, fuJacobian_d, fu, source_d, pointCount/3.f);

//    print_array("Jacobians_T", ftJacobian_d, numtj);
//    print_array("Jacobians_U", fuJacobian_d, numuj);
}
//RigidFitCostFunction END
#include <iostream>

#include "solver/gpu/gpuResidualFunction.h"
#include "solver/gpu/cuda/cu_residual.h"
#include "solver/lossFunction.h"
#include "solver/util/cudautil.h"

using namespace telef::solver;

GPUResidualFunction::~GPUResidualFunction() {
    if (lossScale_d) {
        SOLVER_CUDA_FREE(lossScale_d);
        lossScale_d = nullptr;
    }
}

void GPUResidualFunction::setLossFunction(LossFunction::Ptr loss) {
    ResidualFunction::setLossFunction(std::move(loss));
    if (lossScale_d) {
        SOLVER_CUDA_FREE(lossScale_d);
        lossScale_d = nullptr;
    }
    if (getLossFunction() && getResidualBlock()) {
        int nRes = getResidualBlock()->numResiduals();
        SOLVER_CUDA_MALLOC(&lossScale_d, static_cast<size_t>(nRes));
    }
}

void GPUResidualFunction::applyLoss() {
    if (!getLossFunction() || !getResidualBlock()) return;
    HuberLoss* huber = dynamic_cast<HuberLoss*>(getLossFunction().get());
    if (!huber || !lossScale_d) return;
    float* residuals = getResidualBlock()->getResiduals();
    int nRes = getResidualBlock()->numResiduals();
    apply_loss_huber_residuals(residuals, lossScale_d, nRes, huber->getDelta());
    for (auto& paramBlock : getResidualBlock()->getParameterBlocks()) {
        scale_jacobian_rows(paramBlock->getJacobians(), lossScale_d, nRes, paramBlock->numParameters());
    }
}

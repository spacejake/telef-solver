#include <iostream>

#include "solver/residualFunction.h"
#include "solver/gpu/cuda/cu_solver.h"

using namespace telef::solver;

void ResidualFunction::evaluate() {
    costFunction->evaluate(residualBlock);
}

void ResidualFunction::computeJacobians(/*float *gradient_*/) {
    costFunction->computeJacobinas(residualBlock);

//    auto ParamBlocks  = residualBlock->getParameterBlocks();
//    for (ParameterBlock::Ptr paramBlock : ParamBlocks) {
//        calcGradients(gradient_+paramBlock->getOffset(),
//                      paramBlock->getJacobians(), residualBlock->getResiduals(),
//                      residualBlock->numResiduals(), paramBlock->numParameters());
//    }
}
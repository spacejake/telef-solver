#include <iostream>

#include "solver/residualFunction.h"
#include "solver/gpu/cuda/cu_solver.h"

using namespace telef::solver;

void ResidualFunction::evaluate() {
//    int idx = 0;
//    for (auto param : residualBlock->getParameterBlocks()) {
//        printf("B4:param[%d]\n", idx++);
//        print_array("\tB4", param->getParameters(), param->numParameters());
//    }
    costFunction->evaluate(residualBlock);
}

void ResidualFunction::computeJacobians(/*float *gradient_*/) {
    costFunction->computeJacobians(residualBlock);

//    auto ParamBlocks  = residualBlock->getParameterBlocks();
//    for (ParameterBlock::Ptr paramBlock : ParamBlocks) {
//        calcGradients(gradient_+paramBlock->getOffset(),
//                      paramBlock->getJacobians(), residualBlock->getResiduals(),
//                      residualBlock->numResiduals(), paramBlock->numParameters());
//    }
}
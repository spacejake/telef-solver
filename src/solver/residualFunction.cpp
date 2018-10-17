#include <iostream>

#include "solver/residualFunction.h"

using namespace telef::solver;

void ResidualFunction::evaluate(float *gradient, bool evalJacobians_) {
    costFunction->evaluate(residualBlock, evalJacobians_);

    if (evalJacobians_) {
        auto ParamBlocks  = residualBlock->getParameterBlocks();
//        std::cout << "Num ParamBlocks: " << ParamBlocks.size() << std::endl;
        for (ParameterBlock::Ptr paramBlock : ParamBlocks) {

//            std::cout << "Num Params: " << paramBlock->numParameters() << std::endl;
            calcGradients(gradient+paramBlock->getOffset(),
                          paramBlock->getJacobians(), residualBlock->getResiduals(),
                          residualBlock->numResiduals(), paramBlock->numParameters());

//
//            calcHessians(paramBlock->getHessians(),
//                         paramBlock->getJacobians(),
//                         residualBlock->numResiduals(), paramBlock->numParameters());
        }
    }

    return residualBlock;
}
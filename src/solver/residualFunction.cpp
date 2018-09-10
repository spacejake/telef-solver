#include <iostream>

#include "solver/residualFunction.h"

using namespace telef::solver;

ResidualBlock::Ptr ResidualFunction::evaluate(bool evalJacobians_) {
    costFunction->evaluate(residualBlock, evalJacobians_);

    if (evalJacobians_) {
        auto ParamBlocks  = residualBlock->getParameterBlocks();
        std::cout << "Num ParamBlocks: " << ParamBlocks.size() << std::endl;
        for (ParameterBlock::Ptr paramBlock : ParamBlocks) {

            std::cout << "Num Params: " << paramBlock->numParameters() << std::endl;
            calcGradients(paramBlock->getGradients(),
                          residualBlock->getResiduals(), paramBlock->getJacobians(),
                          residualBlock->numResiduals(), paramBlock->numResiduals());


            calcHessians(paramBlock->getHessians(),
                         paramBlock->getJacobians(),
                         residualBlock->numResiduals(), paramBlock->numResiduals());
        }
    }

    return residualBlock;
}
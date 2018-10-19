#include <iostream>

#include "solver/residualFunction.h"
#include "solver/gpu/cuda/cu_solver.h"

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

            print_array("evaluate::Gradient", gradient+paramBlock->getOffset(), paramBlock->numParameters());
//
//            calcHessians(paramBlock->getHessians(),
//                         paramBlock->getJacobians(),
//                         residualBlock->numResiduals(), paramBlock->numParameters());
        }
    }
}
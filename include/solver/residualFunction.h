#pragma once

#include "solver/costFunction.h"


namespace telef::solver {
    /**
     * Residual Funciton Pairs Cost function and ResidualBlocks together, so they can be maintained in same data structure
     */
    class ResidualFunction {
    public:
        using Ptr = std::shared_ptr<ResidualFunction>;
        using ConstPtr = std::shared_ptr<const ResidualFunction>;

        ResidualFunction(CostFunction::Ptr costFunc_,
                         ResidualBlock::Ptr resBlock_,
                         const float weight_=1.0)
                : costFunction(costFunc_),
                  residualBlock(resBlock_),
                  weight(weight_)
        {}

        virtual ~ResidualFunction(){}

        void evaluate();
        void computeJacobians();

//        void initParams(std::vector<float*> initParams) {
//            residualBlock->initParams(initParams);
//        }

        void setWeight(float weight_) {
            weight = weight_;
        }

        ResidualBlock::Ptr getResidualBlock() {
            return residualBlock;
        }

        void setInitialParams(const std::vector<float*> & initialParams_){
            residualBlock->setInitialParams(initialParams_);
        }

    protected:
        CostFunction::Ptr costFunction;
        //TODO:: LossFunction::Ptr lossFunction;
        ResidualBlock::Ptr residualBlock;
        float weight;
    };
}
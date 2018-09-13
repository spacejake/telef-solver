#pragma once

#include "solver/costFunction.h"


namespace telef::solver {
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

        ResidualBlock::Ptr evaluate(bool evalJacobians_);

        virtual void calcGradients(float* gradients, float* residuals, float* jacobians, int nRes, int nParams) = 0;
        virtual void calcHessians(float* hessians, float* jacobians, int nRes, int nParams) = 0;

//        void initParams(std::vector<float*> initParams) {
//            residualBlock->initParams(initParams);
//        }

        void setWeight(float weight_) {
            weight = weight_;
        }

        ResidualBlock::Ptr getResidualBlock() {
            return residualBlock;
        }


    protected:
        CostFunction::Ptr costFunction;
        //TODO:: LossFunction::Ptr lossFunction;
        ResidualBlock::Ptr residualBlock;
        float weight;
    };
}
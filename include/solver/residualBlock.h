#pragma once

#include <memory>
#include <vector>

#include "solver/parameterBlock.h"

namespace telef::solver {
    class ResidualBlock {
    public:
        using Ptr = std::shared_ptr<ResidualBlock>;
        using ConstPtr = std::shared_ptr<const ResidualBlock>;

        ResidualBlock(const int nRes): error(0.0f), nResiduals(nRes) {}
        virtual ~ResidualBlock(){}

        virtual float* getResiduals() = 0;
        virtual float* getStep() = 0;
        virtual float* getLambda() = 0;

        virtual float* getWorkingError() = 0;

        float setError(float error_) {
            error = error_;
        };

        float getError() {
            return error;
        };

        int numResiduals() {
            return nResiduals;
        }

        void addParameterBlock(ParameterBlock::Ptr param) {
            parameterBlocks.push_back(param);
        }

//        ParameterBlock::Ptr getParameterBlock(int idx) {
//            return params[idx];
//        }
//
//        int numParameters() {
//            return params.size();
//        };

        std::vector<ParameterBlock::Ptr>& getParameterBlocks() {
            return parameterBlocks;
        }

        void setInitialParams(const std::vector<float*> &initialParams_){
            assert(initialParams_.size() == parameterBlocks.size());

            for (int idx = 0; idx < parameterBlocks.size(); idx++) {
                parameterBlocks[idx]->setInitialParams(initialParams_[idx]);
            }
        }

    protected:
        float error;
        int nResiduals;
        std::vector<ParameterBlock::Ptr> parameterBlocks;
    };
}
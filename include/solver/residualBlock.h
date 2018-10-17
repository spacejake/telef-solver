#pragma once

#include <memory>
#include <vector>

#include "solver/parameterBlock.h"

namespace telef::solver {
    class ResidualBlock {
    public:
        using Ptr = std::shared_ptr<ResidualBlock>;
        using ConstPtr = std::shared_ptr<const ResidualBlock>;

        ResidualBlock(const int nRes): offset(0), error(0.0f), nResiduals(nRes) {}
        virtual ~ResidualBlock(){}

        virtual float* getResiduals() = 0;
        virtual float* getStep() = 0;
        virtual float* getLambda() = 0;

        virtual float* getWorkingError() = 0;

        virtual float* getGradient() = 0;
        virtual float* getHessian() = 0;

        virtual void initialize() = 0;

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

        std::vector<ParameterBlock::Ptr>& getParameterBlocks() {
            return parameterBlocks;
        }

        void setInitialParams(const std::vector<float*> &initialParams_){
            assert(initialParams_.size() == parameterBlocks.size());

            for (int idx = 0; idx < parameterBlocks.size(); idx++) {
                parameterBlocks[idx]->setInitialParams(initialParams_[idx]);
            }
        }

        int getOffset(){
            return offset;
        }

        void setOffset(int offset_){
            offset = offset_;
        }

    protected:
        int nEffectiveParams;
        int offset; // Memory offest in Global space

        float error;
        int nResiduals;
        std::vector<ParameterBlock::Ptr> parameterBlocks;
    };
}
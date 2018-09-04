#pragma once

#include <memory>
#include <vector>

#include "solver/parameterBlock.h"

namespace telef::solver {
    class ResidualBlock {
    public:
        using Ptr = std::shared_ptr<ResidualBlock>;
        using ConstPtr = std::shared_ptr<const ResidualBlock>;

        ResidualBlock(const int nRes): nResiduals(nRes) {}
        virtual ~ResidualBlock(){}

        virtual float* getResiduals() = 0;

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

        const std::vector<ParameterBlock::Ptr>& getParameterBlocks() const {
            return parameterBlocks;
        }

    protected:
        int nResiduals;
        std::vector<ParameterBlock::Ptr> parameterBlocks;
    };
}
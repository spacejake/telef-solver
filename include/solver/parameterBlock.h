#pragma once

#include <assert.h>
#include "solver/residualBlock.h"

namespace telef::solver {
    class ParameterBlock {
    public:
        using Ptr = std::shared_ptr<ParameterBlock>;
        using ConstPtr = std::shared_ptr<const ParameterBlock>;

        ParameterBlock(const int nRes, const int nParams)
                : nResiduals(nRes), nParameters(nParams){}

        virtual ~ParameterBlock(){}

        virtual void setInitialParams(float* initialParams_) {
            resultParameters = initialParams_;
        }

        virtual float* getResultParameters() {
            return resultParameters;
        };

        virtual void initializeParameters() = 0;
        virtual float* getWorkingParameters() = 0;
        virtual float* getParameters() = 0;
        virtual float* getDeltaParameters() = 0;

        virtual float* getDampeningFactors() = 0;

        virtual float* getJacobians() = 0;
        virtual float* getGradients() = 0;
        virtual float* getHessians() = 0;
        virtual float* getHessianLowTri() = 0;

        int numParameters() {
            return nParameters;
        }

        int numResiduals() {
            return nResiduals;
        }

        bool isShared(){
            return shared;
        }

    protected:
        int nResiduals;
        float* resultParameters;
        int nParameters;

        // Shared Parameters will have same Pointer to parameter and global gradient,
        // but different offsets in global Jacobian
        bool shared;
        std::shared_ptr<ParameterBlock::Ptr> shared_parameter;
//        int shared_owner_index;
    };
}
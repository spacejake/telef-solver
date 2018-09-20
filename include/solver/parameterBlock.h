#pragma once

#include <assert.h>

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
    protected:
        int nResiduals;
        float* resultParameters;
        int nParameters;
    };
}
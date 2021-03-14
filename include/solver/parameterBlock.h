#pragma once

#include <assert.h>
#include "solver/residualBlock.h"

namespace telef::solver {
//    using Parameters = struct Parameters {
//        float* workingParameters;
//        float* parameters;
//        int nParameters;
//        int offset;
//    };

    class ParameterBlock {
    public:
        using Ptr = std::shared_ptr<ParameterBlock>;
        using ConstPtr = std::shared_ptr<const ParameterBlock>;

        ParameterBlock(const int nRes, const int nParams)
                : nResiduals(nRes), nParameters(nParams),
                resultParameters(nullptr), shared_parameter(),
                offset(0), paramBlockIndex(0) {}

        virtual ~ParameterBlock(){
            //Subclasses must Destroy their pointers
        }

        virtual void setInitialParams(float* initialParams_) {
            resultParameters = initialParams_;
        }

        virtual float* getResultParameters() {
            return resultParameters;
        };

        virtual void initializeParameters() = 0;
        virtual float* getParameters() = 0;
        virtual float* getBestParameters() = 0;

        virtual float* getJacobians() = 0;
        virtual float* getGradients() = 0;

        const int numParameters() const {
            return nParameters;
        }

        const int numResiduals() const {
            return nResiduals;
        }

        int getOffset() const {
            if (isShared()){
                return shared_parameter->getOffset();
            }
            else {
                return offset;
            }
        }

        void setOffset(int offset_){
            if (isShared()){
                shared_parameter->setOffset(offset_);
            }
            else {
                offset = offset_;
            }
        }

        void share(ParameterBlock::Ptr original_parameter) {
            assert(!original_parameter->isShared() && "original parameter must not be shared");
            assert(original_parameter.get() != this && "Parameter Block Can't be shared with itself");

            shared_parameter = std::move(original_parameter);
            onShare();
        }

        virtual void onShare() = 0;

        bool isShared() const {
            return shared_parameter != nullptr;
        }

        ParameterBlock::Ptr getSharedParameter() const {
            return shared_parameter;
        }

        const int getParamBlockIndex() const {
            if (isShared()){
                return shared_parameter->getParamBlockIndex();
            }
            else {
                return paramBlockIndex;
            }
        }

        void setParamBlockIndex(int idx){
            if (isShared()){
                shared_parameter->setParamBlockIndex(idx);
            }
            else {
                paramBlockIndex = idx;
            }
        }

    protected:
        int nResiduals;
        int nParameters;
        float* resultParameters;

        // TODO: Transition to use a shared parameter block
//        std::shared_ptr<Parameters> parameters;

        // Shared Parameters will have same Pointer to parameter and global gradient,
        // but different offsets in global Jacobian
        ParameterBlock::Ptr shared_parameter;
//        int shared_owner_index;

    private:
        // Offset in global parameter space for deltas, Gradiant, and Hessian
        int offset;

        // Index used in the global Jacobian Block structure
        int paramBlockIndex;
    };
}
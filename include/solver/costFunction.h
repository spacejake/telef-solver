#pragma once

#include <vector>
#include <memory>

#include "solver/residualBlock.h"

/**
 * Nearly identical to Ceres::CostFunction, for easy migration of telef implementation and others that use Ceres
 */
namespace telef::solver {
    class ResidualBlock;

    class CostFunction {
    public:
        using Ptr = std::shared_ptr<CostFunction>;
        using ConstPtr = std::shared_ptr<const CostFunction>;

        //TODO: Should we give number of residuals/params for validation(assert) ResidualBlock and cost match up???
        //      I see this as benificial incase there is a mistake since both are independant of one another.
        //      Alothough it could be done in the evaluate function too?
        CostFunction(int nRes, const std::vector<int>& paramSizes_) : nResiduals(nRes), parameterSizes(paramSizes_){}

        virtual ~CostFunction() {}


        const int numResiduals() const {
            return nResiduals;
        }

        const std::vector<int>& getParameterSizes() const {
            return parameterSizes;
        }

        /**
         * Evaluate residuals (y - f(x)) and the jacobians of f(x) given the parameters x.
         * Subclasses should provide the interface for the measurements y.
         *
         * Jacobians are computeted if computeJacobians == True
         *
         * @param parameters
         * @param computeJacobians
         * @return Cost, container of the produced Residuals and Jacobians
         */
        virtual void evaluate(ResidualBlock::Ptr residualBlock, const bool computeJacobians) const = 0;

    protected:
        // TODO: Changing after adding to a "Problem" will likely cause CUDA allocated space to become invalid, fix
        std::vector<int> parameterSizes;

    private:
        int nResiduals;
    };
}
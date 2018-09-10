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

        CostFunction(){}

        virtual ~CostFunction() {}


//        int getNumResiduals() const {
//            return numResiduals;
//        }

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


//    private:
//        int numResiduals;
    };
}
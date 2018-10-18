#pragma once

#include <memory>
#include <vector>
#include <cuda_runtime.h>

#include "solver/residualFunction.h"

namespace telef::solver {
    class Problem {
    public:
        using Ptr = std::shared_ptr<Problem>;
        using ConstPtr = std::shared_ptr<const Problem>;

        Problem(): nEffectiveParams(0), nEffectiveResiduals(0), error(0.0f),
                nJacobianBlockRows(0), nJacobianBlockCols(0), jacobianBlocks(){}
        virtual ~Problem(){}

        float evaluate(bool evalJacobians_){
            if (evalJacobians_) {
                // Set Global Gradients and Hessian to zero
                cudaMemset(getGradient(), 0, nEffectiveParams*sizeof(float));
                cudaMemset(getHessian(), 0, nEffectiveParams*nEffectiveParams*sizeof(float));

            }

            for (auto resFunc : residualFuncs) {
                resFunc->evaluate(getGradient(), evalJacobians_);
            }

            if (evalJacobians_) {
                //TODO: use cublas<t>geam() to transpose the upper half into the lower half instead, modify loop accordingly
                int jBlkRow = 0;
                for (auto resFunc : residualFuncs) {
                    for (int jBlkCol = 0; jBlkCol < nJacobianBlockCols; jBlkCol++) {
                        for (int i = 0; i < nJacobianBlockRows; i++) {
                            auto paramT = getFromJBlock(i, jBlkRow);
                            auto param = getFromJBlock(i, jBlkCol);
                            if (paramT == nullptr || param == nullptr) {
                                // Skip block, will result in 0s since we add the result to the Global hessian
                                continue;
                            }

                            int colOffset = paramT->getOffset();
                            int rowOffset = param->getOffset();

                            int hessianBlocklOffset = colOffset * nJacobianBlockRows + rowOffset;

                            // Compute upper triagle J()
                            // J(i,row)' * J(i,col) will share same number of residuals
                            // Because the hessian matrix can be much larger and we want to insert/add the computed results
                            // into the hessian, we send the method the total size of the hessian so proper offsets can be
                            // computed. We will only be computing a square Hessian.
                            calculateHessianBlock(getHessian() + hessianBlocklOffset, nEffectiveParams,
                                                  paramT->getJacobians(), paramT->numParameters(),
                                                  param->getJacobians(), param->numParameters(),
                                                  paramT->numResiduals());

                            // Compute lower triagle H(row,col) += H(col,row)'
//                        int hessianBlockTOffset = rowOffset * nJacobianBlockRows + colOffset;
                        }
                    }
                    jBlkRow++;
                }
            }
        }

        float setError(float error_) {
            error = error_;
        }

        const float getError() const {
            return error;
        }

        void addResidualFunction(ResidualFunction::Ptr resFunc_,
                                 const std::vector<float*> &params_){
            resFunc_->setInitialParams(params_);
            residualFuncs.push_back(resFunc_);
        }


        const std::vector<ResidualFunction::Ptr>& getResidualFunctions() {
            return residualFuncs;
        }

        const int numEffectiveParams() const {
            return nEffectiveParams;
        }

        const int numEffectiveResiduals() const {
            return nEffectiveResiduals;
        }

        /**
         * When computing the global Hessian, we will need to compute J'J (' is Transpose)
         * to compute the J'J(x,y) Block in latex, J'J(x,y) = Sum_{i=0}^{nJRows}(J(i,x)' * J(i,y))
         * For the upper triangle we will compute J'J(x,y) + H[param->offest]
         * For the lower triangle we will compute J'J(x,y) = J'J(y,x)' + H[param->offest]
         *
         * @param row
         * @param col
         * @return
         */
        ParameterBlock::Ptr getFromJBlock(int row, int col){
            int index = row * nJacobianBlockCols + col;
            assert( index  < jacobianBlocks.size() && "Requesting invalid index from JBlock");
            return jacobianBlocks[row * nJacobianBlockCols + col];
        }

        void initialize() {
            nEffectiveResiduals = 0;
            nEffectiveParams = 0;

            /*
             * The global Jacobian consists of an structure of parameter block's computed Jacobian
             * grouped in the row of their residual block
             *      P0   P1   P2   P3
             *    ---------------------
             * R0 | J0 | J1 | 0  | 0  |
             *    ---------------------
             * R1 | J2 | 0  | J3 | 0  |
             *    ---------------------
             * R2 | 0  | 0  | 0  | J4 |
             *    ---------------------
             *
             * Parameters for J(0,0) are shared with J(1,0)
             *
             */
            nJacobianBlockRows = residualFuncs.size();
            nJacobianBlockCols = 0;
            jacobianBlocks.clear();

            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                auto paramBlocks = resBlock->getParameterBlocks();

                for (auto paramBlock : paramBlocks) {
                    if (!paramBlock->isShared()){
                        // The original parameter cannot be shared, this is forbidden in the interface
                        paramBlock->setOffset(nEffectiveParams);
                        nEffectiveParams += paramBlock->numParameters();

                        // Do not count shared parameters
                        paramBlock->setParamBlockIndex(nJacobianBlockCols);
                        nJacobianBlockCols++;
                    }
                }

                resBlock->setOffset(nEffectiveResiduals);
                nEffectiveResiduals += resBlock->numResiduals();
            }

            // Initialize Jacobian Block to NULL
            for (int jBlkIdx = 0; jBlkIdx < nJacobianBlockRows*nJacobianBlockCols; jBlkIdx++) {
                jacobianBlocks.emplace_back();
            }

            int jBlkRow = 0;
            for (auto resFunc : residualFuncs) {
                auto resBlock = resFunc->getResidualBlock();
                auto paramBlocks = resBlock->getParameterBlocks();

                for (auto paramBlock : paramBlocks) {
                    int jBlkCol = paramBlock->getParamBlockIndex();
                    int jBlkIdx = jBlkRow * nJacobianBlockCols + jBlkCol;
                    jacobianBlocks.at(jBlkIdx) = paramBlock;
                }
                jBlkRow++;
            }

            // Subclass initialization
            onInitialize();
        }

        virtual void
        calculateHessianBlock(float *hessianBlock, const int nEffectiveParams, const float *jacobianA, const int nParamsA,
                                      const float *jacobianB, const int nParamsB, const int nResiduals) = 0;

        virtual float* getLambda() = 0;

        virtual float* getWorkingError() = 0;

        // Global combined Matricies
        virtual float* getDeltaParameters() = 0;
        virtual float* getDampeningFactors() = 0;
        virtual float* getGradient() = 0;
        virtual float* getHessian() = 0;

        /**
         * Used for Hessian decomposition
         * @return
         */
        virtual float* getHessianLowTri() = 0;

    protected:
        //Computed during initilization, after all functions and parameters defined
        int nEffectiveParams;
        int nEffectiveResiduals;

        float error;

        std::vector<ResidualFunction::Ptr> residualFuncs;

        /**
         * compute and allocate size for global matrices
         * Call befor running solver or when the Problem space has been modified, i.e. add more ResidualBlocks
         */
        virtual void onInitialize() = 0;

    private:
        int nJacobianBlockRows;
        int nJacobianBlockCols;
        std::vector<ParameterBlock::Ptr> jacobianBlocks;
    };
}
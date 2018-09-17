

#include <gtest/gtest.h> // googletest header file

#include <iostream>
#include <string>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace std;

#include "solver/gpu/gpuResidualFunction.h"

#include "util/cudautil.h"
#include "cuda/cuda_kernel.h"
#include "solver/gpu/cuda/cu_resudual.h"

using namespace telef::solver;
using namespace testing;

class TestCostFunction : public CostFunction {
public:
    TestCostFunction(){
        float measurements[] = {10,3,4,1};

        utils::CUDA_ALLOC_AND_COPY(&measurements_d, measurements, static_cast<size_t >(4));
    }

    virtual ~TestCostFunction(){
        utils::CUDA_FREE(measurements_d);
    }

    virtual void evaluate(ResidualBlock::Ptr residualBlock, const bool computeJacobians) const {
        ParameterBlock::Ptr params = residualBlock->getParameterBlocks()[0];
        calc_res0(residualBlock->getResiduals(), params->getParameters(), measurements_d,
                  residualBlock->numResiduals(), params->numParameters());

        if (computeJacobians) {
            calc_jacobi0(params->getJacobians(), params->getParameters(),
                         residualBlock->numResiduals(), params->numParameters());
        }
    }
private:
    float* measurements_d;
};

// A new one of these is created for each test
class GPUResidualFunctionTest : public testing::Test
{
public:
    GPUResidualFunction::Ptr residualFunc;

    virtual void SetUp()
    {
        std::vector<int> params = {2};
        int nRes = 4;
        float params_init[] = {0.5,0.5};
        GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, params);
        ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks().at(0);
        cudaMemcpy(paramBlock->getParameters(), params_init, params[0]* sizeof(float), cudaMemcpyHostToDevice);

        CostFunction::Ptr cost = std::make_shared<TestCostFunction>();

        residualFunc = std::make_shared<GPUResidualFunction>(cost, resBlock, 1.0);



//        float residuals[] = {10, 3, 4, 1};
//        float jacobians[] = {1, 1, 1, 1,
//                             4, 4, 4, 4};
        float residuals[] = {-10.2631, -3.2631, -4.2631, -1.2631};
        float jacobians[] = {-1.0523, -1.0523, -1.0523, -1.0523,
                             -4.9164, -4.9164, -4.9164, -4.9164};
    }

    virtual void TearDown()
    {
    }

    void initResiduals(){
        auto resBlock = residualFunc->getResidualBlock();
        int nRes = resBlock->numResiduals();
        float residuals[] = {-10.2631, -3.2631, -4.2631, -1.2631};
        cudaMemcpy(resBlock->getResiduals(), residuals, nRes*sizeof(float), cudaMemcpyHostToDevice);
    }

    void initJacobians(){
        auto resBlock = residualFunc->getResidualBlock();
        auto paramBlock = resBlock->getParameterBlocks()[0];
        int nRes = resBlock->numResiduals();
        int nParams = paramBlock->numParameters();
        float jacobians[] = {-1.0523, -1.0523, -1.0523, -1.0523,
                             -4.9164, -4.9164, -4.9164, -4.9164};
        // row-order
//        float jacobians[] = {-1.0523, -4.9164,
//                             -1.0523, -4.9164,
//                             -1.0523, -4.9164,
//                             -1.0523, -4.9164};
        cudaMemcpy(paramBlock->getJacobians(), jacobians, nRes*nParams*sizeof(float), cudaMemcpyHostToDevice);
    }
};

TEST(Matrix, gpu_multiply) {


    float A[] = {1, 2,
                 1, 2,
                 1, 2};

    float B[] = {1,2,3};

    float *A_d;
    float *B_d;

    int rowA=2;
    int colA=3;
    int rowB=3;
    int colB=1;

    //Rows C=2
    //Cols C=1
    float C[2];
    float *C_d;
    utils::CUDA_MALLOC(&A_d, static_cast<size_t>(rowA*colA));
    utils::CUDA_MALLOC(&B_d, static_cast<size_t>(rowB*colB));
    utils::CUDA_ALLOC_AND_ZERO(&C_d, static_cast<size_t>(2));

    cudaMemcpy(A_d, A, rowA*colA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, rowB*colB*sizeof(float), cudaMemcpyHostToDevice);

//    print_array(A_d, rowA*colA);
//    print_array(B_d, rowB*colB);
    cudaMatMul(C_d, A_d, rowA, colA, B_d, rowB, colB);

    cudaMemcpy(C, C_d, 2*sizeof(float), cudaMemcpyDeviceToHost);
    float real_C[] = {6, 12};

//    print_array(C_d, 2);
    float ferr = 1e-4;
    EXPECT_THAT(C,
                Pointwise(FloatNear(ferr), real_C));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


TEST(Matrix, gpu_multiply_ATxB) {


//    float A[] = {1, 2,
//                 1, 2,
//                 1, 2};
    float A[] = {1, 1, 1,
                 2, 2, 2};

    float B[] = {1,2,3};

    float *A_d;
    float *B_d;

//    int rowA=3;
//    int colA=2;
//    int rowB=3;
//    int colB=1;
    int rowA=3;
    int colA=2;
    int rowB=3;
    int colB=1;

    //Rows C=2
    //Cols C=1
    float C[2];
    float *C_d;
    utils::CUDA_MALLOC(&A_d, static_cast<size_t>(rowA*colA));
    utils::CUDA_MALLOC(&B_d, static_cast<size_t>(rowB*colB));
    utils::CUDA_ALLOC_AND_ZERO(&C_d, static_cast<size_t>(2));

    cudaMemcpy(A_d, A, rowA*colA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, rowB*colB*sizeof(float), cudaMemcpyHostToDevice);

//    print_array(A_d, rowA*colA);
//    print_array(B_d, rowB*colB);
    cudaMatMul_ATxB(C_d, A_d, rowA, colA, B_d, rowB, colB);

    cudaMemcpy(C, C_d, 2*sizeof(float), cudaMemcpyDeviceToHost);
    float real_C[] = {6, 12};

//    print_array(C_d, 2);
    float ferr = 1e-4;
    EXPECT_THAT(C,
                Pointwise(FloatNear(ferr), real_C));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

TEST(CostFunctionTest, costEvaluate) {
    /**
     * Measurement[0]:10.00
     * residuals[0]:-10.2631
     * Measurement[1]:3.00
     * residuals[1]:-3.2631
     * Measurement[2]:4.00
     * residuals[2]:-4.2631
     * Measurement[3]:1.00
     * residuals[3]:-1.2631
     * jacobians[0][0]:-1.0523
     * jacobians[0][1]:-4.9164
     * jacobians[1][0]:-1.0523
     * jacobians[1][1]:-4.9164
     * jacobians[2][0]:-1.0523
     * jacobians[2][1]:-4.9164
     * jacobians[3][0]:-1.0523
     * jacobians[3][1]:-4.9164
     */
    std::vector<int> params = {2};
    int nRes = 4;

    float params_init[] = {0.5,0.5};
    GPUResidualBlock::Ptr resBlock = std::make_shared<GPUResidualBlock>(nRes, params);
    ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks().at(0);
    cudaMemcpy(paramBlock->getParameters(), params_init, 2* sizeof(float), cudaMemcpyHostToDevice);

    CostFunction::Ptr cost = std::make_shared<TestCostFunction>();

    cost->evaluate(resBlock, true);

    float residuals[4];
    float jacobians[8];

    cudaMemcpy(residuals, resBlock->getResiduals(), nRes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(jacobians, resBlock->getParameterBlocks()[0]->getJacobians(), nRes*2*sizeof(float), cudaMemcpyDeviceToHost);

    float real_res[] = {-10.2631, -3.2631, -4.2631, -1.2631};
    float real_jacobi[] = {-1.0523, -1.0523, -1.0523, -1.0523,
                           -4.9164, -4.9164, -4.9164, -4.9164};


    float ferr = 1e-4;
    EXPECT_THAT(residuals,
                Pointwise(FloatNear(ferr), real_res));
    EXPECT_THAT(jacobians,
                Pointwise(FloatNear(ferr), real_jacobi));
}

TEST_F(GPUResidualFunctionTest, gradientTest) {
    initResiduals();
    initJacobians();

    ResidualBlock::Ptr resBlock = residualFunc->getResidualBlock();
    ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks()[0];

    int nRes = resBlock->numResiduals();
    int nParams = paramBlock->numParameters();

//    print_array(paramBlock->getJacobians(), nParams*nRes);
//    print_array(resBlock->getResiduals(), nRes*1);

    residualFunc->calcGradients(paramBlock->getGradients(), paramBlock->getJacobians(), resBlock->getResiduals(),
                                resBlock->numResiduals(), paramBlock->numParameters());
//    print_array(paramBlock->getGradients(), nParams);

    // size of nParams
    float gradients[2];

    cudaMemcpy(gradients, paramBlock->getGradients(), nParams*sizeof(float), cudaMemcpyDeviceToHost);
    float real_gradients[] = {20.0488, 93.6692};


    float ferr = 1e-4;
    EXPECT_THAT(gradients,
                Pointwise(FloatNear(ferr), real_gradients));
}

TEST_F(GPUResidualFunctionTest, hessiansTest) {
    initResiduals();
    initJacobians();

    ResidualBlock::Ptr resBlock = residualFunc->getResidualBlock();
    ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks()[0];

    int nRes = resBlock->numResiduals();
    int nParams = paramBlock->numParameters();

//    print_array(paramBlock->getJacobians(), nParams*nRes);

    residualFunc->calcHessians(paramBlock->getHessians(), paramBlock->getJacobians(),
                                resBlock->numResiduals(), paramBlock->numParameters());
//    print_array(paramBlock->getHessians(), nParams*nParams);

    // size of nParams*nParams
    float hessians[4];

    cudaMemcpy(hessians, paramBlock->getHessians(), nParams*nParams*sizeof(float), cudaMemcpyDeviceToHost);
    // column-Order, psst. this mat is same regardless, lol
    float real_hessians[] = {4.42934,  20.6941,
                             20.6941, 96.684};

    float ferr = 1e-4;
    EXPECT_THAT(hessians,
                Pointwise(FloatNear(ferr), real_hessians));
}



TEST_F(GPUResidualFunctionTest, evaluate) {

    ResidualBlock::Ptr resBlock = residualFunc->getResidualBlock();
    ParameterBlock::Ptr paramBlock = resBlock->getParameterBlocks()[0];

    int nRes = resBlock->numResiduals();
    int nParams = paramBlock->numParameters();

    residualFunc->evaluate(true);

    float residuals[4];
    float jacobians[8];
    float gradients[2];
    float hessiens[4];

    cudaMemcpy(residuals, resBlock->getResiduals(), nRes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(jacobians, paramBlock->getJacobians(), nRes*nParams*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gradients, paramBlock->getGradients(), nParams*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hessiens, paramBlock->getHessians(), nParams*nParams*sizeof(float), cudaMemcpyDeviceToHost);

    float real_res[] = {-10.2631, -3.2631, -4.2631, -1.2631};
    float real_jacobi[] = {-1.0523, -1.0523, -1.0523, -1.0523,
                           -4.9164, -4.9164, -4.9164, -4.9164};
    float real_gradients[] = {20.0488, 93.6692};
    float real_hessiens[] = {4.42934,  20.6941,
                             20.6941, 96.684};


    float ferr = 1e-3;
    EXPECT_THAT(residuals,
                Pointwise(FloatNear(ferr), real_res));
    EXPECT_THAT(jacobians,
                Pointwise(FloatNear(ferr), real_jacobi));
    EXPECT_THAT(gradients,
                Pointwise(FloatNear(ferr), real_gradients));
    EXPECT_THAT(hessiens,
                Pointwise(FloatNear(ferr), real_hessiens));
}
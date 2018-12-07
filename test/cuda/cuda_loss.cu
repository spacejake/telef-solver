#include <device_launch_parameters.h>
#include <math.h>

#include "cuda_loss.h"
#include "cuda_quaternion.h"
#include "util/cudautil.h"

namespace {
    const int NUM_THREAD = 512;
    const int DIM_X_THREAD = 16;
    const int DIM_Y_THREAD = 16;
}

__global__
void _calc_dx_m_dt_lmk(float *dx_m_dt, const int num_points) {
    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = num_points * 3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = 3;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j = y_start; j < y_size; j += y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            if(i==j) {
                dx_m_dt[3*3*k + 3*i + j] = -1.0f;
            }
            else {
                dx_m_dt[3*3*k + 3*i + j] = 0.0;
            }
        }
    }
}

void calc_de_dt_lmk(float *de_dt_d, const int num_points, const float weight) {
    const int xRequired = 3*num_points;
    const int yRequired = 3;
    dim3 dimGrid(SOLVER_GET_DIM_GRID(xRequired, DIM_X_THREAD), SOLVER_GET_DIM_GRID(yRequired, DIM_Y_THREAD));
    dim3 dimBlock(DIM_X_THREAD, DIM_Y_THREAD);
    _calc_dx_m_dt_lmk<<<dimGrid, dimBlock>>>(de_dt_d, num_points);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
//    scale_array<<<SOLVER_GET_DIM_GRID(num_points*3*3, NUM_THREAD), NUM_THREAD>>>
//                    (de_dt_d, num_points*3*3, weight*1.0f/sqrtf(num_points));
//    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

__global__
void _calc_dx_m_du_lmk(float *dx_m_du, const float *u_d, const float *source_d, const int pointCount) {
    float dr_du[27];
    calc_dr_du(dr_du, u_d);

    const int x_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int x_size = pointCount * 3;
    const int x_step = blockDim.x * gridDim.x;
    const int y_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int y_size = 3;
    const int y_step = blockDim.y * gridDim.y;
    for(int ind=x_start; ind<x_size; ind+=x_step) {
        for (int j=y_start; j<y_size; j+=y_step) {
            const int i = ind % 3;
            const int k = ind / 3;
            float sum = 0.0f;

            sum += dr_du[9*j + 3*0 + i] * source_d[3*k + 0];
            sum += dr_du[9*j + 3*1 + i] * source_d[3*k + 1];
            sum += dr_du[9*j + 3*2 + i] * source_d[3*k + 2];

            dx_m_du[3*3*k + 3*i + j] = -sum;
        }
    }
}

void calc_de_du_lmk(float *de_du_d, const float *u_d, const float *source_d, const int pointCount, const float weight) {
    const int xRequired = 3*pointCount;
    const int yRequired = 3;
    dim3 dimGrid(SOLVER_GET_DIM_GRID(xRequired, DIM_X_THREAD), SOLVER_GET_DIM_GRID(yRequired, DIM_Y_THREAD));
    dim3 dimBlock(DIM_X_THREAD, DIM_Y_THREAD);
    _calc_dx_m_du_lmk<<<dimGrid, dimBlock>>>(de_du_d, u_d, source_d, pointCount);
    SOLVER_CHECK_ERROR_MSG("Kernel Error");
//    scale_array<<<SOLVER_GET_DIM_GRID(3*3*point_pair.point_count, NUM_THREAD),NUM_THREAD>>>
//                   (de_du_d, point_pair.point_count*3*3, weight*1.0f/sqrtf(point_pair.point_count));
//    SOLVER_CHECK_ERROR_MSG("Kernel Error");
}

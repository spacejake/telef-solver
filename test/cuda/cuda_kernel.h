#pragma once

#include <cuda_runtime_api.h>


__global__
void _print_array(float *arr_d, int n);

void print_array(float *arr_d, int n);



__global__
void _calc_resSimple(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);

void calc_resSimple(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);



__global__
void _calc_jacobiSimple(float *jacobians, const float *params, const int nRes, const int nParams);

void calc_jacobiSimple(float *jacobians, const float *params, const int nRes, const int nParams);



__global__
void _calc_resSimple2(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);

void calc_resSimple2(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);



__global__
void _calc_jacobiSimple2(float *jacobians, const float *params, const int nRes, const int nParams);

void calc_jacobiSimple2(float *jacobians, const float *params, const int nRes, const int nParams);



__global__
void _calc_res0(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);

void calc_res0(float *residuals, const float *params, const float *measurements, const int nRes, const int nParams);



__global__
void _calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams);

void calc_jacobi0(float *jacobians, const float *params, const int nRes, const int nParams);


__global__
void _calc_res2Params(float *residuals, const float *params1, const float *params2, const float *measurements,
                      const int nRes, const int nParams1, const int nParams2);

void calc_res2Params(float *residuals, const float *params1, const float *params2, const float *measurements,
                     const int nRes, const int nParams1, const int nParams2);



__global__
void _calc_jacobi2Params(float *jacobians, float *jacobians2, const float *params1, const float *params2,
        const int nRes, const int nParams1, const int nParams2);

void calc_jacobi2Params(float *jacobians, float *jacobians2, const float *params1, const float *params2,
        const int nRes, const int nParams1, const int nParams2);



/******************************************************************/
__global__
void _calc_res4Params(float *residuals,
                      const float *params1, const float *params2, const float *params3, const float *params4,
                      const float *measurements, const int nRes,
                      const int nParams1, const int nParams2, const int nParams3, const int nParams4);

void calc_res4Params(float *residuals,
                     const float *params1, const float *params2, const float *params3, const float *params4,
                     const float *measurements, const int nRes,
                     const int nParams1, const int nParams2, const int nParams3, const int nParams4);


__global__
void _calc_jacobi4Params(float *jacobians1, float *jacobians2, float *jacobians3, float *jacobians4,
                         const float *params1, const float *params2, const float *params3, const float *params4,
                         const int nRes, const int nParams1, const int nParams2, const int nParams3, const int nParams4);

void calc_jacobi4Params(float *jacobians1, float *jacobians2, float *jacobians3, float *jacobians4,
                        const float *params1, const float *params2, const float *params3, const float *params4,
                        const int nRes, const int nParams1, const int nParams2, const int nParams3, const int nParams4);
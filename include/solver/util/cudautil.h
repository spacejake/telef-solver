#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>

#ifdef DEBUG
#define SOLVER_CUDA_CHECK(errarg)   __solver_checkErrorFunc(errarg, __FILE__, __LINE__)
#define SOLVER_CHECK_ERROR_MSG(errstr) __solver_checkErrMsgFunc(errstr, __FILE__, __LINE__)
#else
#define SOLVER_CUDA_CHECK(arg)   arg
#define SOLVER_CHECK_ERROR_MSG(str) do {} while (0)
#endif

inline void __solver_checkErrorFunc(cudaError_t errarg, const char *file,
                                    const int line) {
    if (errarg) {
        fprintf(stderr, "Error \"%s\" at %s(%i)\n", cudaGetErrorName(errarg), file, line);
        exit(EXIT_FAILURE);
    }
}


inline void __solver_checkErrMsgFunc(const char *errstr, const char *file,
                                     const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s at %s(%i): %s\n",
                errstr, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template<typename T>
inline void SOLVER_CUDA_MALLOC(T **dst_d, size_t count) {
    SOLVER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(dst_d), count * sizeof(T)));
}

template<typename T>
inline void SOLVER_CUDA_ALLOC_AND_COPY(T **dst_d, const T *src_h, size_t count) {
    SOLVER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(dst_d), count * sizeof(T)));
    SOLVER_CUDA_CHECK(cudaMemcpy(*dst_d, src_h, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
inline void SOLVER_CUDA_ALLOC_AND_ZERO(T **dst_d, size_t count) {
    SOLVER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(dst_d), count * sizeof(T)));
    SOLVER_CUDA_CHECK(cudaMemset(*dst_d, 0, count * sizeof(T)));
}

template<typename T>
inline void SOLVER_CUDA_ZERO(T **dst_d, size_t count) {
    SOLVER_CUDA_CHECK(cudaMemset(*dst_d, 0, count * sizeof(T)));
}

template<typename T>
inline void SOLVER_CUDA_RETRIEVE(T *dst_h, const T *src_d, size_t count) {
    SOLVER_CUDA_CHECK(cudaMemcpy(*dst_h, src_d, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
inline void SOLVER_CUDA_FREE(T *p_d) {
    SOLVER_CUDA_CHECK(cudaFree(p_d));
}

inline constexpr unsigned int SOLVER_GET_DIM_GRID(int required_thread, int num_thread) {
    return static_cast<unsigned int>((required_thread + num_thread - 1) / num_thread);
}

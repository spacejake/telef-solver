cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

include_directories("${PROJECT_SOURCE_DIR}")

set( cudaTestHdr
        cuda/cuda_kernel.h
        )

set( cudaTestSrc
        cuda/cuda_kernel.cu
        )

#source_group("Interface Header Files" FILES ${interfaceHdr})
#source_group("Interface Source Files" FILES ${interfaceSrc})
#source_group("GPU Interface Header Files" FILES ${gpuHdr})
#source_group("Cuda Utils Header Files" FILES ${cudaUtil})

######## linking and properties ########
#add_library(${PROJECT_NAME}
#        ##### HDR #####
#        ${cudaTestHdr}
#
#        ##### SRC #####
#        ${cudaTestSrc}
#        )

target_sources(${PROJECT_NAME}
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/mock_gpu.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mock_gpu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_kernel.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_kernel.cu"
)

#set(SOURCE
#    ${SOURCE}
#    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_kernel.cu
#    PARENT_SCOPE
#)
#set(HEADERS
#    ${HEADERS}
#    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_kernel.h
#    PARENT_SCOPE
#)

# Add Google Tests
myaddtest(test_gpu_residualFunction test_gpu_residualFunction.cpp)
myaddtest(test_gpu_solver test_gpu_solver.cpp)
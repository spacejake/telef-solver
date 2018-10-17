cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)

option (BUILD_TESTS "Build All Tests" ON)

project(telef-solver VERSION 1.0.0 LANGUAGES CXX CUDA DESCRIPTION "telef-solver: A general perpouse GPU Least Squares solver")

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Required because of relocated static libray. We can delete it after we get rid of it.
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -DUSE_SSE4_INSTRUCTIONS=ON")

set(CUDA_GENCODES
        "-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 "
        )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -rdc=true ${CUDA_GENCODES}")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_HOST_COMPILER gcc-7)

find_package(CUDA REQUIRED)
find_package(CBLAS REQUIRED)

############## Define headers and sourcse ################
set( interfaceHdr
        include/solver/costFunction.h
        include/solver/solver.h
        include/solver/residualBlock.h
        include/solver/residualFunction.h
        include/solver/parameterBlock.h
        include/solver/problem.h
)

set( interfaceSrc
        src/solver/solver.cpp
        src/solver/residualFunction.cpp
)

set( gpuHdr
        include/solver/gpu/gpuParameterBlock.h
        include/solver/gpu/gpuResidualBlock.h
        include/solver/gpu/gpuResidualFunction.h
        include/solver/gpu/gpuSolver.h
        include/solver/gpu/gpuProblem.h
)


set( gpuSrc
        src/solver/gpu/gpuResidualFunction.cpp
        src/solver/gpu/gpuSolver.cpp
)

set( cudaHdr
        include/util/cudautil.h
        include/solver/gpu/cuda/cu_resudual.h
        include/solver/gpu/cuda/cu_solver.h
)

set( cudaSrc
        src/solver/gpu/cuda/cu_residual.cu
        src/solver/gpu/cuda/cu_solver.cu
)

#source_group("Interface Header Files" FILES ${interfaceHdr})
#source_group("Interface Source Files" FILES ${interfaceSrc})
#source_group("GPU Interface Header Files" FILES ${gpuHdr})
#source_group("Cuda Utils Header Files" FILES ${cudaUtil})

######## linking and properties ########
add_library(${PROJECT_NAME}
        ##### HDR #####
        ${interfaceHdr}

        ${cudaHdr}

        # GPU Solver
        ${gpuHdr}

        ##### SRC #####
        ${interfaceSrc}

        # GPU Solver
        ${gpuSrc}

        ${cudaSrc}
        )

set_target_properties(${PROJECT_NAME}
        PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION 1
            CUDA_SEPARABLE_COMPILATION ON
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
            #CXX_VISIBILITY_PRESET hidden
            BUILD_SHARED_LIBS ON
        #PUBLIC_HEADER
            #include/solver/test.h
        )

target_include_directories(${PROJECT_NAME}
        PUBLIC ${CUDA_INCLUDE_DIRS}
        PUBLIC ${CMAKE_SOURCE_DIR}/include/
        PUBLIC /usr/local/include )

target_link_libraries(${PROJECT_NAME}
            ${CUDA_LIBRARIES}
            -lcublas
            -lcusolver
        )

###### INSTALL ######
#configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)

#install(TARGETS ${PROJECT_NAME}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
#        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

###### EXAMPLES ######
add_executable(TestApp examples/test_app.cpp)
target_link_libraries(TestApp ${PROJECT_NAME}
        )

###### Tests ######
if (BUILD_TESTS)
    function(myaddtest testname testpath)
        add_executable(${testname} ${testpath})
        target_link_libraries(${testname} gtest gtest_main gmock ${CBLAS_LIBRARIES})
        target_link_libraries(${testname} ${PROJECT_NAME} -lm)

        target_include_directories(${testname}
                PRIVATE ${CMAKE_SOURCE_DIR}/include
                PRIVATE ${CMAKE_BINARY_DIR}/lib/googletest/googletest/include
                PRIVATE ${CMAKE_BINARY_DIR}/lib/googletest/googlemock/include
                PRIVATE ${CBLAS_INCLUDE_DIRS}
                )
        add_test(NAME ${testname} COMMAND ${testname})
    endfunction(myaddtest)

    include(CTest)
    find_package(CBLAS REQUIRED)

    # Download and unpack googletest at configure time
    configure_file(CMakeLists.txt.in
            lib/googletest/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib/googletest )
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib/googletest )

    # Prevent GoogleTest from overriding our compiler/linker options
    # when building with Visual Studio
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This adds
    # the following targets: gtest, gtest_main, gmock
    # and gmock_main
    add_subdirectory(${CMAKE_BINARY_DIR}/lib/googletest/src
            ${CMAKE_BINARY_DIR}/lib/googletest/build)

    add_subdirectory(${CMAKE_BINARY_DIR}/lib/googletest)
    add_subdirectory(test)
    enable_testing()

endif()

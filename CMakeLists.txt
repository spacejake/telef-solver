cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
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
        #"-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 "
        "-gencode arch=compute_86,code=sm_86" # RTX 30x
        )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -rdc=true ${CUDA_GENCODES}")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_HOST_COMPILER gcc-7)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)

# sudo apt-get install libatlas-base-dev
#find_package(CBLAS REQUIRED)
find_package(BLAS REQUIRED)

############## Define Preprocessor Vars ##################
add_definitions ( -DTEST_DATA_DIR=\"${CMAKE_SOURCE_DIR}/test/data\")

############## Define headers and sourcse ################
set( interfaceHdr
        include/solver/costFunction.h
        include/solver/solver.h
        include/solver/residualBlock.h
        include/solver/residualFunction.h
        include/solver/parameterBlock.h
        include/solver/problem.h
        include/solver/util/profile.h
        include/solver/util/fileio.h
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
        include/solver/util/cudautil.h
        include/solver/gpu/cuda/cu_residual.h
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

        #$#{cudaHdr}

        # GPU Solver
        ${gpuHdr}

        ##### SRC #####
        ${interfaceSrc}

        # GPU Solver
        ${gpuSrc}

        #$#{cudaSrc}
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
            -lcudart
            -lcublas
            -lcusolver
            cutelef-solver
        )

add_library(cutelef-solver STATIC
            ${cudaHdr}
            ${cudaSrc}
        )

target_include_directories(cutelef-solver
            PUBLIC ${CUDA_INCLUDE_DIRS}
            PUBLIC ${CMAKE_SOURCE_DIR}/include/
        )

target_link_libraries(cutelef-solver
            stdc++fs
            -lcudart
            -lcublas
            -lcusolver
        )

set_target_properties(cutelef-solver PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)


###### INSTALL ######
configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(TARGETS ${PROJECT_NAME} cutelef-solver
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

###### EXAMPLES ######
add_executable(TestApp examples/test_app.cpp)
target_link_libraries(TestApp ${PROJECT_NAME}
        )

###### Tests ######
if (BUILD_TESTS)
    set(INSTALL_GTEST 0)
    set(INSTALL_GMOCK 0)
    function(myaddtest testname testpath)
        add_executable(${testname} ${testpath})

        # git clone https://github.com/google/googletest.git, cmake, make, then install
        target_link_libraries(${testname} gtest gtest_main gmock ${BLAS_LIBRARIES})
        target_link_libraries(${testname} ${PROJECT_NAME} -lm)

        target_include_directories(${testname}
                PRIVATE ${CMAKE_SOURCE_DIR}/include
                PRIVATE ${CMAKE_BINARY_DIR}/lib/googletest/googletest/include
                PRIVATE ${CMAKE_BINARY_DIR}/lib/googletest/googlemock/include
                PRIVATE ${BLAS_INCLUDE_DIRS}
                )
        add_test(NAME ${testname} COMMAND ${testname})
    endfunction(myaddtest)

    include(CTest)
    find_package(BLAS REQUIRED)

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

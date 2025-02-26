#=============================================================================
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=============================================================================
enable_testing()

include(rapids-test)
#rapids_test_init()

###################################################################################################
# - set rapids dataset path ----------------------------------------------------------------------
if(RAPIDS_DATASET_ROOT_DIR)
    message(STATUS "setting default RAPIDS_DATASET_ROOT_DIR to: ${RAPIDS_DATASET_ROOT_DIR}")
    string(CONCAT CMAKE_C_FLAGS ${CMAKE_C_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
endif(RAPIDS_DATASET_ROOT_DIR)

###################################################################################################
# - common C++ test utils -----------------------------------------------------------------------------

add_library(cugraphtestutil STATIC
            utilities/thrust_wrapper.cu
            utilities/misc_utilities.cpp)

target_compile_options(cugraphtestutil
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
           "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

set_property(TARGET cugraphtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cugraphtestutil
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio"
    PRIVATE
        "${CUGRAPH_SOURCE_DIR}/src"
)

target_link_libraries(cugraphtestutil
    PUBLIC
        cugraph::cugraph
    PRIVATE
        GTest::gtest
)


if(BUILD_CUGRAPH_MG_TESTS)
    include(../cmake/thirdparty/get_nccl.cmake)

    ###############################################################################################
    # - find MPI - only enabled if MG tests are to be built
    set(MPI_ROOT "/home/seunghwak/Programs/openmpi-aarch64")
    set(MPI_CXX_COMPILER /home/seunghwak/Programs/openmpi-aarch64/bin/mpic++)
    find_package(MPI REQUIRED COMPONENTS CXX)


    ###################################################################################################
    # - common C++ mg test utils -----------------------------------------------------------------------
    add_library(cugraphmgtestutil STATIC
                utilities/device_comm_wrapper.cu
                utilities/mg_utilities.cpp
                utilities/conversion_utilities_mg.cu
                )

    set_property(TARGET cugraphmgtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

    target_compile_options(cugraphmgtestutil
                PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                       "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
    )

    target_include_directories(cugraphmgtestutil
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CUGRAPH_SOURCE_DIR}/src"
    )

    target_link_libraries(cugraphmgtestutil
        PRIVATE
            cugraph::cugraph
            NCCL::NCCL
            MPI::MPI_CXX
            GTest::gtest
    )
endif()

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME)
    set(options)
    set(one_value GPUS PERCENT)
    set(multi_value)
    cmake_parse_arguments(_CUGRAPH_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
    if(NOT DEFINED _CUGRAPH_TEST_GPUS AND NOT DEFINED _CUGRAPH_TEST_PERCENT)
        set(_CUGRAPH_TEST_GPUS 1)
        set(_CUGRAPH_TEST_PERCENT 25)
    endif()
    if(NOT DEFINED _CUGRAPH_TEST_GPUS)
        set(_CUGRAPH_TEST_GPUS 1)
    endif()
    if(NOT DEFINED _CUGRAPH_TEST_PERCENT)
        set(_CUGRAPH_TEST_PERCENT 100)
    endif()

    add_executable(${CMAKE_TEST_NAME} ${_CUGRAPH_TEST_UNPARSED_ARGUMENTS})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraphtestutil
            GTest::gtest
            GTest::gtest_main
    )
    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
                       CXX_STANDARD                        17
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       17
                       CUDA_STANDARD_REQUIRED              ON)

    #rapids_test_add(
    #    NAME ${CMAKE_TEST_NAME}
    #    COMMAND ${CMAKE_TEST_NAME}
    #    GPUS ${_CUGRAPH_TEST_GPUS}
    #    PERCENT ${_CUGRAPH_TEST_PERCENT}
    #    INSTALL_COMPONENT_SET testing
    #)
    set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "CUGRAPH")
endfunction()

function(ConfigureTestMG CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    target_include_directories(${CMAKE_TEST_NAME} PRIVATE "${CUGRAPH_SOURCE_DIR}/src")
    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        cugraphmgtestutil
        cugraphtestutil
        GTest::gtest
        GTest::gtest_main
        NCCL::NCCL
        MPI::MPI_CXX
    )
    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib"
                       CXX_STANDARD                        17
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       17
                       CUDA_STANDARD_REQUIRED              ON)

    rapids_test_add(
        NAME ${CMAKE_TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE}
             "--noprefix"
             ${MPIEXEC_NUMPROC_FLAG}
             ${GPU_COUNT}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_TEST_NAME}
             ${MPIEXEC_POSTFLAGS}
        GPUS ${GPU_COUNT}
        PERCENT 100
        INSTALL_COMPONENT_SET testing_mg
        INSTALL_TARGET ${CMAKE_TEST_NAME}
    )
    set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "CUGRAPH_MG")

endfunction()

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - MG tests --------------------------------------------------------------------------------------

if(BUILD_CUGRAPH_MG_TESTS)
    # Set the GPU count to 1.  If the caller wants to execute MG tests using
    # more than 1, override from the command line using -DGPU_COUNT=<gpucount>
    if (NOT DEFINED GPU_COUNT)
      set(GPU_COUNT "1")
    endif()

    ###############################################################################################
    # - MG BFS tests ------------------------------------------------------------------------------
    ConfigureTestMG(MG_BFS_TEST traversal/mg_bfs_test.cpp)

    #rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing_mg DESTINATION bin/gtests/libcugraph_mg)

endif()

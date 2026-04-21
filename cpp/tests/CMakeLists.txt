#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#
#=============================================================================
enable_testing()

include(rapids-test)
rapids_test_init()

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

add_library(cugraphtestutil SHARED
            utilities/thrust_wrapper.cu
)

target_compile_options(cugraphtestutil
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
           "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

set_property(TARGET cugraphtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET cugraphtestutil PROPERTY NO_CUDART_DEP ON)

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
        CUDA::toolkit
)

###################################################################################################
# - common C API test utils -----------------------------------------------------------------------

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
    set_property(TARGET ${CMAKE_TEST_NAME} PROPERTY NO_CUDART_DEP ON)

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraphtestutil
            CUDA::cudart
    )
    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUGRAPH_BINARY_DIR}/gtests>"
                       INSTALL_RPATH "\$ORIGIN/../../../lib"
                       CXX_STANDARD                        20
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       20
                       CUDA_STANDARD_REQUIRED              ON
                       LINK_FLAGS "-Wl,--allow-shlib-undefined")

    rapids_test_add(
        NAME ${CMAKE_TEST_NAME}
        COMMAND ${CMAKE_TEST_NAME}
        GPUS ${_CUGRAPH_TEST_GPUS}
        PERCENT ${_CUGRAPH_TEST_PERCENT}
        INSTALL_COMPONENT_SET testing
    )
    set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "CUGRAPH")
endfunction()

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - TEMPORAL NBR SAMPLING tests -------------------------------------------------------------------
ConfigureTest(
    TEMPORAL_NEIGHBOR_SAMPLING_TEST sampling/temporal_neighbor_sampling.cpp
        GPUS 1 PERCENT 75)

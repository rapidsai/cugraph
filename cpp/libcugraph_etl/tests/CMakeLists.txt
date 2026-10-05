#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================
enable_testing()

include(rapids-test)
rapids_test_init()

add_executable(RENUMBER_CUDF_TABLES_TEST renumber_cudf_tables_test.cpp)

target_link_libraries(RENUMBER_CUDF_TABLES_TEST
    PRIVATE
        cugraph::cugraph_etl
        cudf::cudf
        GTest::gtest
        GTest::gtest_main
)

set_target_properties(RENUMBER_CUDF_TABLES_TEST
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gtests>"
        INSTALL_RPATH "\$ORIGIN/../../../lib"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
)

rapids_test_add(
    NAME RENUMBER_CUDF_TABLES_TEST
    COMMAND RENUMBER_CUDF_TABLES_TEST
    GPUS 1
    PERCENT 100
    INSTALL_COMPONENT_SET testing
)

rapids_test_install_relocatable(
    INSTALL_COMPONENT_SET testing
    DESTINATION bin/gtests/libcugraph_etl
)

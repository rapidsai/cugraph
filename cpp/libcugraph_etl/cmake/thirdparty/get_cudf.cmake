#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_and_configure_cudf)

    set(oneValueArgs VERSION FORK PINNED_TAG)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    rapids_cpm_find(cudf ${PKG_VERSION}
      GLOBAL_TARGETS      cudf::cudf
      BUILD_EXPORT_SET    cugraph_etl-exports
      INSTALL_EXPORT_SET  cugraph_etl-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/cudf.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
              "BUILD_BENCHMARKS OFF"
              "BUILD_TESTS OFF"
    )

    message(VERBOSE "CUGRAPH_ETL: Using CUDF located in ${cudf_SOURCE_DIR}")

endfunction()

set(CUGRAPH_ETL_MIN_VERSION_cudf "${CUGRAPH_ETL_VERSION_MAJOR}.${CUGRAPH_ETL_VERSION_MINOR}.00")

# Change pinned tag and fork here to test a commit in CI
# To use a different cuDF locally, set the CMake variable
# CPM_cudf_SOURCE=/path/to/local/cudf
find_and_configure_cudf(VERSION    ${CUGRAPH_ETL_MIN_VERSION_cudf}
                        FORK       rapidsai
                        PINNED_TAG ${rapids-cmake-checkout-tag}
                        )

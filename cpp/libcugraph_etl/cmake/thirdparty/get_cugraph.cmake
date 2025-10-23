#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_and_configure_cugraph)

    set(oneValueArgs VERSION FORK PINNED_TAG)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    rapids_cpm_find(cugraph ${PKG_VERSION}
      GLOBAL_TARGETS      cugraph::cugraph
      BUILD_EXPORT_SET    cugraph_etl-exports
      INSTALL_EXPORT_SET  cugraph_etl-exports
        CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/cugraph.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS "BUILD_TESTS OFF"
    )

    message(VERBOSE "CUGRAPH_ETL: Using CUGRAPH located in ${cugraph_SOURCE_DIR}")

endfunction()

set(CUGRAPH_ETL_MIN_VERSION_cugraph "${CUGRAPH_ETL_VERSION_MAJOR}.${CUGRAPH_ETL_VERSION_MINOR}.00")


# Change pinned tag and fork here to test a commit in CI
# To use a different cuGraph locally, set the CMake variable
# CPM_cugraph_SOURCE=/path/to/local/cugraph
find_and_configure_cugraph(VERSION    ${CUGRAPH_ETL_MIN_VERSION_cugraph}
                           FORK       rapidsai
                           PINNED_TAG ${rapids-cmake-checkout-tag}
                           )

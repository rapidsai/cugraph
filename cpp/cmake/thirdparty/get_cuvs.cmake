#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

set(CUGRAPH_MIN_VERSION_cuvs "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL USE_CUVS_STATIC CLONE_ON_PIN)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message(STATUS "CUGRAPH: Pinned tag found: ${PKG_PINNED_TAG}. Cloning cuVS locally.")
        set(CPM_DOWNLOAD_cuvs ON)
    elseif(PKG_USE_CUVS_STATIC AND (NOT CPM_cuvs_SOURCE))
      message(STATUS "CUGRAPH: Cloning cuVS locally to build static libraries.")
      set(CPM_DOWNLOAD_cuvs ON)
    endif()

    # Disable multi-GPU algorithms to avoid NCCL dependency
    set(CUVS_BUILD_MG_ALGOS OFF)

    rapids_cpm_find(cuvs ${PKG_VERSION}
      GLOBAL_TARGETS      cuvs::cuvs
      CPM_ARGS
        GIT_REPOSITORY         https://github.com/${PKG_FORK}/cuvs.git
        GIT_TAG                ${PKG_PINNED_TAG}
        SOURCE_SUBDIR          cpp
        EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_CAGRA_HNSWLIB OFF"
          "BUILD_CUVS_BENCH OFF"
          "BUILD_MG_ALGOS ${CUVS_BUILD_MG_ALGOS}"
    )

    if(cuvs_ADDED)
        message(VERBOSE "CUGRAPH: Using cuVS located in ${cuvs_SOURCE_DIR}")
    else()
        message(VERBOSE "CUGRAPH: Using cuVS located in ${cuvs_DIR}")
    endif()

endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different cuVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION          ${CUGRAPH_MIN_VERSION_cuvs}
                        FORK             rapidsai
                        PINNED_TAG       ${rapids-cmake-checkout-tag}

                        # When PINNED_TAG above doesn't match cugraph,
                        # force local cuVS clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_CUVS_STATIC  ${CUGRAPH_USE_CUVS_STATIC}
                        EXCLUDE_FROM_ALL ON
                        )

#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

set(CUGRAPH_MIN_VERSION_raft "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")

function(find_and_configure_raft)

    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN USE_RAFT_STATIC COMPILE_RAFT_LIB)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
        set(CPM_DOWNLOAD_raft ON)
    elseif(PKG_USE_RAFT_STATIC AND (NOT CPM_raft_SOURCE))
      message(STATUS "CUGRAPH: Cloning raft locally to build static libraries.")
      set(CPM_DOWNLOAD_raft ON)
    endif()

    if(PKG_COMPILE_RAFT_LIB)
      if(NOT PKG_USE_RAFT_STATIC)
        string(APPEND RAFT_COMPONENTS " compiled")
      else()
        string(APPEND RAFT_COMPONENTS " compiled_static")
      endif()
    endif()

    rapids_cpm_find(raft ${PKG_VERSION}
      GLOBAL_TARGETS      raft::raft
      BUILD_EXPORT_SET    cugraph-exports
      INSTALL_EXPORT_SET  cugraph-exports
      COMPONENTS ${RAFT_COMPONENTS}
        CPM_ARGS
            EXCLUDE_FROM_ALL TRUE
            GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
                "RAFT_COMPILE_LIBRARY ${PKG_COMPILE_RAFT_LIB}"
                "BUILD_TESTS OFF"
                "BUILD_PRIMS_BENCH OFF"
                "BUILD_CAGRA_HNSWLIB OFF"
    )

    if(raft_ADDED)
        message(VERBOSE "CUGRAPH: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "CUGRAPH: Using RAFT located in ${raft_DIR}")
    endif()

endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    ${CUGRAPH_MIN_VERSION_raft}
                        FORK       achirkin
                        PINNED_TAG enh-consistent-container-policy

                        # When PINNED_TAG above doesn't match cugraph,
                        # force local raft clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_RAFT_STATIC ${USE_RAFT_STATIC}
                        COMPILE_RAFT_LIB ${CUGRAPH_COMPILE_RAFT_LIB}
                        )

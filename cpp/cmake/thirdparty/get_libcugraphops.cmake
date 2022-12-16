#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
#=============================================================================

set(CUGRAPH_MIN_VERSION_cugraph_ops "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")

if(NOT DEFINED CUGRAPH_CUGRAPH_OPS_VERSION)
  set(CUGRAPH_CUGRAPH_OPS_VERSION "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}")
endif()

if(NOT DEFINED CUGRAPH_CUGRAPH_OPS_BRANCH)
  set(CUGRAPH_CUGRAPH_OPS_BRANCH "branch-${CUGRAPH_CUGRAPH_OPS_VERSION}")
endif()

if(NOT DEFINED CUGRAPH_CUGRAPH_OPS_REPOSITORY)
  set(CUGRAPH_CUGRAPH_OPS_REPOSITORY "git@github.com:rapidsai/cugraph-ops.git")
endif()

function(find_and_configure_cugraph_ops)

    set(oneValueArgs VERSION REPO PINNED_TAG BUILD_STATIC EXCLUDE_FROM_ALL ALLOW_CLONE_CUGRAPH_OPS)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    if(PKG_ALLOW_CLONE_CUGRAPH_OPS)
        if(NOT PKG_PINNED_TAG STREQUAL "branch-${CUGRAPH_CUGRAPH_OPS_VERSION}")
            message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning cugraph-ops locally.")
            set(CPM_DOWNLOAD_cugraph-ops ON)
        elseif(PKG_BUILD_STATIC AND (NOT CPM_cugraph-ops_SOURCE))
            message(STATUS "CUGRAPH: Cloning cugraph-ops locally to build static libraries.")
            set(CPM_DOWNLOAD_cugraph-ops ON)
        endif()
    endif()

    set(CUGRAPH_OPS_BUILD_SHARED_LIBS ON)
    if(PKG_BUILD_STATIC)
      set(CUGRAPH_OPS_BUILD_SHARED_LIBS OFF)
    endif()

    rapids_cpm_find(cugraph-ops ${PKG_VERSION} REQUIRED
      GLOBAL_TARGETS      cugraph-ops::cugraph-ops++
      BUILD_EXPORT_SET    cugraph-exports
      INSTALL_EXPORT_SET  cugraph-exports
      CPM_ARGS
        SOURCE_SUBDIR    cpp
        GIT_REPOSITORY   ${PKG_REPO}
        GIT_TAG          ${PKG_PINNED_TAG}
        EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
        OPTIONS
          "BUILD_CUGRAPH_OPS_CPP_TESTS OFF"
          "BUILD_SHARED_LIBS ${CUGRAPH_OPS_BUILD_SHARED_LIBS}"
    )
endfunction()

###
# Change pinned tag and fork here to test a commit in CI
#
# To use a locally-built cugraph-ops package, set the CMake variable
# `-D cugraph-ops_ROOT=/path/to/cugraph-ops/build`
###
find_and_configure_cugraph_ops(VERSION      ${CUGRAPH_MIN_VERSION_cugraph_ops}
                               REPO         ${CUGRAPH_CUGRAPH_OPS_REPOSITORY}
                               PINNED_TAG   ${CUGRAPH_CUGRAPH_OPS_BRANCH}
                               BUILD_STATIC     ${CUGRAPH_USE_CUGRAPH_OPS_STATIC}
                               EXCLUDE_FROM_ALL ${CUGRAPH_EXCLUDE_CUGRAPH_OPS_FROM_ALL}
                               # Allow cloning cugraph-ops in cases when we
                               # expect the local copy of cugraph-ops to not be
                               # the one that we want. Cases include static
                               # linking such that we need to compile a static
                               # lib or during development when using a
                               # different branch of cugraph-ops.
                               ALLOW_CLONE_CUGRAPH_OPS ${ALLOW_CLONE_CUGRAPH_OPS})

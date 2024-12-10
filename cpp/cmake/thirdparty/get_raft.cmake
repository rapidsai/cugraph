#=============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

set(CUGRAPH_MIN_VERSION_raft "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")
set(CUGRAPH_BRANCH_VERSION_raft "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}")

function(find_and_configure_raft)

    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN USE_RAFT_STATIC COMPILE_RAFT_LIB)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUGRAPH_BRANCH_VERSION_raft}")
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
      GLOBAL_TARGETS      raft raft_compiled_static
      BUILD_EXPORT_SET    cugraph-raft-exports
      INSTALL_EXPORT_SET  cugraph-raft-exports
      COMPONENTS ${RAFT_COMPONENTS}
        CPM_ARGS
            EXCLUDE_FROM_ALL TRUE
            GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
                "RAFT_COMPILE_LIBRARY ${PKG_COMPILE_RAFT_LIB}"
                "BUILD_TESTS OFF"
                "BUILD_BENCH OFF"
                "BUILD_CAGRA_HNSWLIB OFF"
                "BUILD_PRIMS_BENCH OFF"
    )

    # _raft_include_dirs:
    #  _raft_include_dirs-NOTFOUND
    # get_target_property(_raft_include_dirs raft INCLUDE_DIRECTORIES)
    # message(STATUS "_raft_include_dirs: ${_raft_include_dirs}")

    # _raft_interface_include_dirs:
    #   $<BUILD_INTERFACE:/tmp/cugraph/python/libcugraph/build/py3-none-linux_x86_64/_deps/raft-src/cpp/include>;$<INSTALL_INTERFACE:include>
    # get_target_property(_raft_interface_include_dirs raft INTERFACE_INCLUDE_DIRECTORIES)
    # message(FATAL_ERROR "_raft_interface_include_dirs: ${_raft_interface_include_dirs}")

    # TODO: move this under if(raft_ADDED)?
    # TODO: install just the headers
    install(
        TARGETS raft raft_lib_static
        COMPONENT raft
        DESTINATION include/raft-static/
        INCLUDES DESTINATION include/raft
        EXPORT cugraph-raft-exports
    )

    # TODO: maybe raft should be using target_sources instead?
    #  ref: https://cmake.org/cmake/help/latest/command/install.html#directory
    get_target_property(_raft_interface_include_dirs raft INTERFACE_INCLUDE_DIRECTORIES)
    install(
        DIRECTORY ${_raft_interface_include_dirs}
        DESTINATION include/raft-headers/
        # EXPORT cugraph-raft-exports
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
                        FORK       rapidsai
                        PINNED_TAG branch-${CUGRAPH_BRANCH_VERSION_raft}

                        # When PINNED_TAG above doesn't match cugraph,
                        # force local raft clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_RAFT_STATIC ${USE_RAFT_STATIC}
                        COMPILE_RAFT_LIB ${CUGRAPH_COMPILE_RAFT_LIB}
                        )

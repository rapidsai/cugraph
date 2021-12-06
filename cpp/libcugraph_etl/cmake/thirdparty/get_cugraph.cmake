#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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
set(CUGRAPH_ETL_BRANCH_VERSION_cugraph "${CUGRAPH_ETL_VERSION_MAJOR}.${CUGRAPH_ETL_VERSION_MINOR}")


# Change pinned tag and fork here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# RPM_cugraph_SOURCE=/path/to/local/cugraph
find_and_configure_cugraph(VERSION    ${CUGRAPH_ETL_MIN_VERSION_cugraph}
                           FORK       rapidsai
                           PINNED_TAG branch-${CUGRAPH_ETL_BRANCH_VERSION_cugraph}
                           )

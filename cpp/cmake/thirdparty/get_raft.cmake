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

function(find_and_configure_raft VERSION)

    rapids_cpm_find(raft ${VERSION}
      GLOBAL_TARGETS raft::raft
      BUILD_EXPORT_SET    cugraph-exports
      INSTALL_EXPORT_SET  cugraph-exports
        CPM_ARGS
            # GIT_REPOSITORY https://github.com/rapidsai/raft.git
            # GIT_TAG        branch-${VERSION}
            GIT_REPOSITORY https://github.com/robertmaynard/raft.git
            GIT_TAG        use_rapids_cmake
            SOURCE_SUBDIR  cpp
            OPTIONS "BUILD_TESTS OFF"

    )

endfunction()

set(CUGRAPH_MIN_VERSION_raft "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}")

find_and_configure_raft(${CUGRAPH_MIN_VERSION_raft})

#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

function(find_and_configure_cuco VERSION)

    rapids_cpm_find(cuco ${VERSION}
      GLOBAL_TARGETS      cuco::cuco
      BUILD_EXPORT_SET    cugraph-exports
      CPM_ARGS
        EXCLUDE_FROM_ALL TRUE
        GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
        GIT_TAG        55029034c3f82bca36148c9be29941b37492394d
        OPTIONS        "BUILD_TESTS OFF"
                       "BUILD_BENCHMARKS OFF"
                       "BUILD_EXAMPLES OFF"
    )

endfunction()

# cuCollections doesn't have a version yet
find_and_configure_cuco(0.0)

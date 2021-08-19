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

function(find_and_configure_cuhornet)

    # We are not using the cuhornet CMake targets, so no need to call `add_subdirectory()`,
    # or to use CPM
    FetchContent_Declare(
        cuhornet
        GIT_REPOSITORY    https://github.com/rapidsai/cuhornet.git
        GIT_TAG           4a1daa18405c0242370e16ce302dfa7eb5d9e857
        SOURCE_SUBDIR     hornet
    )
    FetchContent_GetProperties(cuhornet)

    if(NOT cuhornet_POPULATED)
        FetchContent_Populate(cuhornet)
    endif()

    if(NOT TARGET cugraph::cuHornet)
        add_library(cugraph::cuHornet IMPORTED INTERFACE GLOBAL)
        target_include_directories(cugraph::cuHornet INTERFACE
            "${cuhornet_SOURCE_DIR}/hornet/include"
            "${cuhornet_SOURCE_DIR}/hornetsnest/include"
            "${cuhornet_SOURCE_DIR}/xlib/include"
            "${cuhornet_SOURCE_DIR}/primitives"
            )
    endif()
endfunction()


find_and_configure_cuhornet()

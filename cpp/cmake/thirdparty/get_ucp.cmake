#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

function(find_and_configure_ucp)

    if(TARGET UCP::UCP)
        return()
    endif()

    rapids_find_generate_module(UCP
        HEADER_NAMES ucp.h
        LIBRARY_NAMES ucp
        INCLUDE_SUFFIXES ucp/api
    )

    # Currently UCP has no CMake build-system so we require
    # it built and installed on the machine already
    rapids_find_package(UCP REQUIRED)

endfunction()

find_and_configure_ucp()

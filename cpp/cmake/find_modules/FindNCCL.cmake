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

#[=======================================================================[.rst:
FindNCCL
--------

Find NCCL

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target(s):

``NCCL::NCCL``
  The NCCL library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``NCCL_FOUND``
  True if NCCL is found.
``NCCL_INCLUDE_DIRS``
  The include directories needed to use NCCL.
``NCCL_LIBRARIES``
  The libraries needed to useNCCL.
``NCCL_VERSION_STRING``
  The version of the NCCL library found. [OPTIONAL]

#]=======================================================================]

# Prefer using a Config module if it exists for this project
set(NCCL_NO_CONFIG FALSE)
if(NOT NCCL_NO_CONFIG)
  find_package(NCCL CONFIG QUIET)
  if(NCCL_FOUND)
    find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_CONFIG)
    return()
  endif()
endif()

find_path(NCCL_INCLUDE_DIR NAMES nccl.h )

set(NCCL_IS_HEADER_ONLY FALSE)
if(NOT NCCL_LIBRARY AND NOT NCCL_IS_HEADER_ONLY)
  find_library(NCCL_LIBRARY_RELEASE NAMES nccl NAMES_PER_DIR )
  find_library(NCCL_LIBRARY_DEBUG   NAMES nccld   NAMES_PER_DIR )

  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  select_library_configurations(NCCL)
  unset(NCCL_FOUND) #incorrectly set by select_library_configurations
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(NCCL_IS_HEADER_ONLY)
  find_package_handle_standard_args(NCCL
                                    FOUND_VAR NCCL_FOUND
                                    REQUIRED_VARS NCCL_INCLUDE_DIR
                                    VERSION_VAR )
else()
  find_package_handle_standard_args(NCCL
                                    FOUND_VAR NCCL_FOUND
                                    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
                                    VERSION_VAR )
endif()

if(NCCL_FOUND)
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

  if(NOT NCCL_LIBRARIES)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  endif()

  if(NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}")

    if(NCCL_LIBRARY_RELEASE)
      set_property(TARGET NCCL::NCCL APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(NCCL::NCCL PROPERTIES
        IMPORTED_LOCATION_RELEASE "${NCCL_LIBRARY_RELEASE}")
    endif()

    if(NCCL_LIBRARY_DEBUG)
      set_property(TARGET NCCL::NCCL APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(NCCL::NCCL PROPERTIES
        IMPORTED_LOCATION_DEBUG "${NCCL_LIBRARY_DEBUG}")
    endif()

    if(NOT NCCL_LIBRARY_RELEASE AND NOT NCCL_LIBRARY_DEBUG)
      set_property(TARGET NCCL::NCCL APPEND PROPERTY
        IMPORTED_LOCATION "${NCCL_LIBRARY}")
    endif()
  endif()
endif()

unset(NCCL_NO_CONFIG)
unset(NCCL_IS_HEADER_ONLY)

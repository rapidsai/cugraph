# Copyright (c) 2020, NVIDIA CORPORATION.
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
#

# Based on FindPNG.cmake from cmake 3.14.3

#[=======================================================================[.rst:
FindFAISS
--------

Template to generate FindPKG_NAME.cmake CMake modules

Find FAISS

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``FAISS::FAISS``
  The libFAISS library, if found.

Result variables
^^^^^^^^^^^^^^^^

This module will set the following variables in your project:

``FAISS_INCLUDE_DIRS``
  where to find FAISS.hpp , etc.
``FAISS_LIBRARIES``
  the libraries to link against to use libFAISS.
``FAISS_FOUND``
  If false, do not try to use FAISS.
``FAISS_VERSION_STRING``
  the version of the FAISS library found

#]=======================================================================]

find_path(FAISS_LOCATION faiss/IndexFlat.h
          HINTS ${FAISS_INSTALL_DIR}
          PATH_SUFFIXES include include/)

list(APPEND FAISS_NAMES faiss libfaiss)
set(_FAISS_VERSION_SUFFIXES )

foreach(v IN LISTS _FAISS_VERSION_SUFFIXES)
  list(APPEND FAISS_NAMES faiss${v} libfaiss${v})
  list(APPEND FAISS_NAMES faiss.${v} libfaiss.${v})
endforeach()
unset(_FAISS_VERSION_SUFFIXES)

find_library(FAISS_LIBRARY_RELEASE NAMES ${FAISS_NAMES}
             HINTS ${FAISS_INSTALL_DIR}
             PATH_SUFFIXES lib)

include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
select_library_configurations(FAISS)
mark_as_advanced(FAISS_LIBRARY_RELEASE)
unset(FAISS_NAMES)

# Set by select_library_configurations(), but we want the one from
# find_package_handle_standard_args() below.
unset(FAISS_FOUND)

if (FAISS_LIBRARY AND FAISS_LOCATION)
  set(FAISS_INCLUDE_DIRS ${FAISS_LOCATION} )
  set(FAISS_LIBRARY ${FAISS_LIBRARY})

  if(NOT TARGET FAISS::FAISS)
    add_library(FAISS::FAISS UNKNOWN IMPORTED)
    set_target_properties(FAISS::FAISS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIRS}")
    if(EXISTS "${FAISS_LIBRARY}")
      set_target_properties(FAISS::FAISS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${FAISS_LIBRARY}")
    endif()
  endif()
endif ()


include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(FAISS
                                  REQUIRED_VARS FAISS_LIBRARY FAISS_LOCATION
                                  VERSION_VAR FAISS_VERSION_STRING)

mark_as_advanced(FAISS_LOCATION FAISS_LIBRARY)

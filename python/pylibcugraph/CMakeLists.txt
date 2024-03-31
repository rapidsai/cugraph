# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../rapids_config.cmake)

# We always need CUDA for pylibcugraph because the raft dependency brings in a
# header-only cuco dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(pylibcugraph-python)

project(
  pylibcugraph-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# TODO: Temporarily not using cugraph-ops for wheels
option(USE_CUGRAPH_OPS "Enable all functions that call cugraph-ops" OFF)

if(NOT USE_CUGRAPH_OPS)
    message(STATUS "Disabling libcugraph functions that reference cugraph-ops")
    add_compile_definitions(NO_CUGRAPH_OPS)
endif()

# If the user requested it we attempt to find CUGRAPH.
find_package(cugraph "${RAPIDS_VERSION}" REQUIRED)

# Register a library that is installed to site-packages for rpath linking
function(rapids_cython_register_site_packages_library target)
  # TODO: Is this solution robust? Should we also check for other configs, or the default IMPORTED_LOCATION?
  get_property(target_library TARGET "${target}" PROPERTY IMPORTED_LOCATION_RELEASE)
  cmake_path(GET target_library PARENT_PATH library_dir)

  find_package(Python REQUIRED)
  # TODO: Should we support other install locations like user site?
  cmake_path(IS_PREFIX Python_SITELIB "${library_dir}" library_is_in_site)
  # We cannot support libraries that are not installed to site-packages
  # TODO: We do need to still allow building in such cases. For now the easiest
  # thing I can think of is just a status message here because a warning seems
  # pretty loud for standard uses cases like conda or devcontainers where this
  # won't link.
  if(NOT library_is_in_site)
      message(STATUS "Library ${target} is not installed to site-packages. This is not supported.")
      return()
  endif()

  cmake_path(RELATIVE_PATH library_dir BASE_DIRECTORY "${Python_SITELIB}"
      OUTPUT_VARIABLE library_relative_path)
  set_property(GLOBAL PROPERTY "rapids_cython_registered_libraries_${target}" "${library_relative_path}")
endfunction()

rapids_cython_register_site_packages_library(cugraph::cugraph)

function(rapids_cython_link_to_site_packages_library target linked_target)
  get_property(library_relative_path GLOBAL PROPERTY "rapids_cython_registered_libraries_${linked_target}")
  if(NOT library_relative_path)
    message(STATUS "Library ${linked_target} is not registered as a site-packages library, not updating RPATH of ${target}")
    return()
  endif()

  cmake_path(RELATIVE_PATH PROJECT_SOURCE_DIR BASE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
      OUTPUT_VARIABLE path_to_site)
  set_property(TARGET "${target}" APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/${path_to_site}/${library_relative_path}")
endfunction()

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(pylibcugraph)

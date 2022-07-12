# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
#include(cmake/thirdparty/get_libcudacxx.cmake)
set(cython_sources
    bfs.pyx
    eigenvector_centrality.pyx
    graph_properties.pyx
    graphs.pyx
    hits.pyx
    katz_centrality.pyx
    node2vec.pyx
    pagerank.pyx
    resource_handle.pyx
    sssp.pyx
    triangle_count.pyx
    uniform_neighbor_sample.pyx
    utils.pyx
)
set(linked_libraries cugraph::cugraph;cugraph_c)

#set(linked_libraries cugraph::cugraph)
#set(linked_libraries_c cugraph_c)
#LINKED_LIBRARIES ${linked_libraries} ${linked_libraries_c}

#set(linked_libraries_c cugraph_c)
#set(linked_libraries "/jnke/scikit-build-pr/cugraph/cpp")
#message(STATUS "linked library is ${linked_libraries}")

#${linked_libraries_c}
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES ${linked_libraries}
)

# TODO: Finding NumPy requires finding Development as well. Once this is fixed in CMake (no date
# yet) we can remove the extra component spec.
find_package(Python REQUIRED COMPONENTS Development NumPy)
set(targets_using_numpy pagerank sssp utils)
foreach(target IN LISTS targets_using_numpy)
  target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endforeach()

#message(STATUS "origin is ${HOME}")
foreach(cython_module IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  message(STATUS "cython module is ${cython_module}")
  set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../library")
endforeach()

#message(STATUS "Done adding path")
# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

add_subdirectory(components)
add_subdirectory(internal_types)
add_subdirectory(testing)

set(cython_sources
    bfs.pyx
    core_number.pyx
    egonet.pyx
    eigenvector_centrality.pyx
    graph_properties.pyx
    graphs.pyx
    hits.pyx
    k_core.pyx
    jaccard_coefficients.pyx
    sorensen_coefficients.pyx
    overlap_coefficients.pyx
    katz_centrality.pyx
    louvain.pyx
    node2vec.pyx
    pagerank.pyx
    personalized_pagerank.pyx
    random.pyx
    resource_handle.pyx
    sssp.pyx
    triangle_count.pyx
    two_hop_neighbors.pyx
    uniform_neighbor_sample.pyx
    uniform_random_walks.pyx
    utils.pyx
    weakly_connected_components.pyx
)
set(linked_libraries cugraph::cugraph;cugraph::cugraph_c)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES ${linked_libraries}
  ASSOCIATED_TARGETS cugraph
)

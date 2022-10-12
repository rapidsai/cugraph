/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <community/flatten_dendrogram.hpp>
#include <community/legacy/leiden.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<size_t, weight_t> leiden(raft::handle_t const& handle,
                                   legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                   vertex_t* clustering,
                                   size_t max_level,
                                   weight_t resolution)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr,
                  "Invalid input argument: leiden expects a weighted graph");
  CUGRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is null, should be a device pointer to "
                  "memory for storing the result");

  legacy::Leiden<legacy::GraphCSRView<vertex_t, edge_t, weight_t>> runner(handle, graph);
  weight_t wt = runner(max_level, resolution);

  rmm::device_uvector<vertex_t> vertex_ids_v(graph.number_of_vertices, handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               thrust::make_counting_iterator<vertex_t>(0),  // MNMG - base vertex id
               thrust::make_counting_iterator<vertex_t>(
                 graph.number_of_vertices),  // MNMG - base vertex id + number_of_vertices
               vertex_ids_v.begin());

  partition_at_level<vertex_t, false>(handle,
                                      runner.get_dendrogram(),
                                      vertex_ids_v.data(),
                                      clustering,
                                      runner.get_dendrogram().num_levels());

  // FIXME: Consider returning the Dendrogram at some point
  return std::make_pair(runner.get_dendrogram().num_levels(), wt);
}

// Explicit template instantations
template std::pair<size_t, float> leiden(raft::handle_t const&,
                                         legacy::GraphCSRView<int32_t, int32_t, float> const&,
                                         int32_t*,
                                         size_t,
                                         float);

template std::pair<size_t, double> leiden(raft::handle_t const&,
                                          legacy::GraphCSRView<int32_t, int32_t, double> const&,
                                          int32_t*,
                                          size_t,
                                          double);

}  // namespace cugraph

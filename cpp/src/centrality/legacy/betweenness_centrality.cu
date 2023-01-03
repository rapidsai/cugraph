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

#include <vector>

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/replace.h>
#include <thrust/transform.h>

#include <raft/util/cudart_utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_vector.hpp>

#include "betweenness_centrality.cuh"
#include "betweenness_centrality_kernels.cuh"
#include <raft/core/handle.hpp>

namespace cugraph {
namespace detail {
namespace {
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void betweenness_centrality_impl(raft::handle_t const& handle,
                                 legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                 result_t* result,
                                 bool normalize,
                                 bool endpoints,
                                 weight_t const* weight,
                                 vertex_t number_of_sources,
                                 vertex_t const* sources,
                                 vertex_t total_number_of_sources)
{
  // Current Implementation relies on BFS
  // FIXME: For SSSP version
  // Brandes Algorithm expects non negative weights for the accumulation
  bool is_edge_betweenness = false;
  verify_betweenness_centrality_input<vertex_t, edge_t, weight_t, result_t>(
    result, is_edge_betweenness, normalize, endpoints, weight, number_of_sources, sources);
  cugraph::detail::BC<vertex_t, edge_t, weight_t, result_t> bc(handle, graph);
  bc.configure(
    result, is_edge_betweenness, normalize, endpoints, weight, sources, number_of_sources);
  bc.compute();
  bc.rescale_by_total_sources_used(total_number_of_sources);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void edge_betweenness_centrality_impl(raft::handle_t const& handle,
                                      legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                      result_t* result,
                                      bool normalize,
                                      weight_t const* weight,
                                      vertex_t number_of_sources,
                                      vertex_t const* sources,
                                      vertex_t /* total_number_of_sources */)
{
  // Current Implementation relies on BFS
  // FIXME: For SSSP version
  // Brandes Algorithm expects non negative weights for the accumulation
  bool is_edge_betweenness = true;
  bool endpoints           = false;
  verify_betweenness_centrality_input<vertex_t, edge_t, weight_t, result_t>(
    result, is_edge_betweenness, normalize, endpoints, weight, number_of_sources, sources);
  cugraph::detail::BC<vertex_t, edge_t, weight_t, result_t> bc(handle, graph);
  bc.configure(
    result, is_edge_betweenness, normalize, endpoints, weight, sources, number_of_sources);
  bc.compute();
  // NOTE: As of 07/2020 NetworkX does not apply rescaling based on number
  // of sources
  // bc.rescale_by_total_sources_used(total_number_of_sources);
}
template <typename vertex_t>
vertex_t get_total_number_of_sources(raft::handle_t const& handle, vertex_t local_number_of_sources)
{
  vertex_t total_number_of_sources_used = local_number_of_sources;
  if (handle.comms_initialized()) {
    rmm::device_scalar<vertex_t> d_number_of_sources(local_number_of_sources, handle.get_stream());
    handle.get_comms().allreduce(d_number_of_sources.data(),
                                 d_number_of_sources.data(),
                                 1,
                                 raft::comms::op_t::SUM,
                                 handle.get_stream());
    total_number_of_sources_used = d_number_of_sources.value(handle.get_stream());
    // RAFT_CUDA_TRY(
    // cudaMemcpy(&total_number_of_sources_used, data, sizeof(vertex_t), cudaMemcpyDeviceToHost));
  }
  return total_number_of_sources_used;
}
}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void verify_betweenness_centrality_input(result_t* result,
                                         bool is_edge_betweenness,
                                         bool normalize,
                                         bool endpoints,
                                         weight_t const* weights,
                                         vertex_t const number_of_sources,
                                         vertex_t const* sources)
{
  static_assert(std::is_same<vertex_t, int>::value, "vertex_t should be int");
  static_assert(std::is_same<edge_t, int>::value, "edge_t should be int");
  static_assert(std::is_same<weight_t, float>::value || std::is_same<weight_t, double>::value,
                "weight_t should be float or double");
  static_assert(std::is_same<result_t, float>::value || std::is_same<result_t, double>::value,
                "result_t should be float or double");

  CUGRAPH_EXPECTS(result != nullptr, "Invalid input argument: betwenness pointer is NULL");
  CUGRAPH_EXPECTS(number_of_sources >= 0, "Number of sources must be positive or equal to 0.");
  if (number_of_sources != 0) {
    CUGRAPH_EXPECTS(sources != nullptr,
                    "Sources cannot be NULL if number_of_source is different from 0.");
  }
  if (is_edge_betweenness) {
    CUGRAPH_EXPECTS(!endpoints, "Endpoints is not supported for edge betweenness centrality.");
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::setup()
{
  number_of_vertices_ = graph_.number_of_vertices;
  number_of_edges_    = graph_.number_of_edges;
  offsets_ptr_        = graph_.offsets;
  indices_ptr_        = graph_.indices;
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::configure(result_t* betweenness,
                                                         bool is_edge_betweenness,
                                                         bool normalized,
                                                         bool endpoints,
                                                         weight_t const* weights,
                                                         vertex_t const* sources,
                                                         vertex_t number_of_sources)
{
  // --- Bind betweenness output vector to internal ---
  betweenness_         = betweenness;
  normalized_          = normalized;
  endpoints_           = endpoints;
  sources_             = sources;
  number_of_sources_   = number_of_sources;
  edge_weights_ptr_    = weights;
  is_edge_betweenness_ = is_edge_betweenness;

  // --- Working data allocation ---
  initialize_work_vectors();
  initialize_pointers_to_vectors();

  // --- Get Device Information ---
  initialize_device_information();

  // --- Confirm that configuration went through ---
  configured_ = true;
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::initialize_work_vectors()
{
  distances_vec_.resize(number_of_vertices_);
  predecessors_vec_.resize(number_of_vertices_);
  sp_counters_vec_.resize(number_of_vertices_);
  deltas_vec_.resize(number_of_vertices_);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::initialize_pointers_to_vectors()
{
  distances_    = distances_vec_.data().get();
  predecessors_ = predecessors_vec_.data().get();
  sp_counters_  = sp_counters_vec_.data().get();
  deltas_       = deltas_vec_.data().get();
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::initialize_device_information()
{
  max_grid_dim_1D_  = handle_.get_device_properties().maxGridSize[0];
  max_block_dim_1D_ = handle_.get_device_properties().maxThreadsDim[0];
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::compute()
{
  CUGRAPH_EXPECTS(configured_, "BC must be configured before computation");
  if (sources_) {
    for (vertex_t source_idx = 0; source_idx < number_of_sources_; ++source_idx) {
      vertex_t source_vertex = sources_[source_idx];
      compute_single_source(source_vertex);
    }
  } else {
    for (vertex_t source_vertex = 0; source_vertex < number_of_vertices_; ++source_vertex) {
      compute_single_source(source_vertex);
    }
  }
  rescale();
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::compute_single_source(vertex_t source_vertex)
{
  // Step 1) Singe-source shortest-path problem
  cugraph::bfs(handle_,
               graph_,
               distances_,
               predecessors_,
               sp_counters_,
               source_vertex,
               graph_.prop.directed,
               true);

  // FIXME: Remove that with a BC specific class to gather
  //        information during traversal

  // Numeric max value is replaced by -1 as we look for the maximal depth of
  // the traversal, this value is avalaible within the bfs implementation and
  // there could be a way to access it directly and avoid both replace and the
  // max
  thrust::replace(handle_.get_thrust_policy(),
                  distances_,
                  distances_ + number_of_vertices_,
                  std::numeric_limits<vertex_t>::max(),
                  static_cast<vertex_t>(-1));
  auto current_max_depth =
    thrust::max_element(handle_.get_thrust_policy(), distances_, distances_ + number_of_vertices_);
  vertex_t max_depth = 0;
  RAFT_CUDA_TRY(
    cudaMemcpy(&max_depth, current_max_depth, sizeof(vertex_t), cudaMemcpyDeviceToHost));
  // Step 2) Dependency accumulation
  accumulate(source_vertex, max_depth);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::accumulate(vertex_t source_vertex,
                                                          vertex_t max_depth)
{
  dim3 grid_configuration, block_configuration;
  block_configuration.x = max_block_dim_1D_;
  grid_configuration.x  = min(max_grid_dim_1D_, (number_of_edges_ / block_configuration.x + 1));

  initialize_dependencies();

  if (is_edge_betweenness_) {
    accumulate_edges(max_depth, grid_configuration, block_configuration);
  } else if (endpoints_) {
    accumulate_vertices_with_endpoints(
      source_vertex, max_depth, grid_configuration, block_configuration);
  } else {
    accumulate_vertices(max_depth, grid_configuration, block_configuration);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::initialize_dependencies()
{
  thrust::fill(
    handle_.get_thrust_policy(), deltas_, deltas_ + number_of_vertices_, static_cast<result_t>(0));
}
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::accumulate_edges(vertex_t max_depth,
                                                                dim3 grid_configuration,
                                                                dim3 block_configuration)
{
  for (vertex_t depth = max_depth; depth >= 0; --depth) {
    edges_accumulation_kernel<vertex_t, edge_t, weight_t, result_t>
      <<<grid_configuration, block_configuration, 0, handle_.get_stream()>>>(betweenness_,
                                                                             number_of_vertices_,
                                                                             graph_.indices,
                                                                             graph_.offsets,
                                                                             distances_,
                                                                             sp_counters_,
                                                                             deltas_,
                                                                             depth);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::accumulate_vertices_with_endpoints(
  vertex_t source_vertex, vertex_t max_depth, dim3 grid_configuration, dim3 block_configuration)
{
  for (vertex_t depth = max_depth; depth > 0; --depth) {
    endpoints_accumulation_kernel<vertex_t, edge_t, weight_t, result_t>
      <<<grid_configuration, block_configuration, 0, handle_.get_stream()>>>(betweenness_,
                                                                             number_of_vertices_,
                                                                             graph_.indices,
                                                                             graph_.offsets,
                                                                             distances_,
                                                                             sp_counters_,
                                                                             deltas_,
                                                                             depth);
  }
  add_reached_endpoints_to_source_betweenness(source_vertex);
  add_vertices_dependencies_to_betweenness();
}

// Distances should contain -1 for unreached nodes,

// FIXME: There might be a cleaner way to add a value to a single
//        score in the betweenness vector
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::add_reached_endpoints_to_source_betweenness(
  vertex_t source_vertex)
{
  vertex_t number_of_unvisited_vertices =
    thrust::count(handle_.get_thrust_policy(), distances_, distances_ + number_of_vertices_, -1);
  vertex_t number_of_visited_vertices_except_source =
    number_of_vertices_ - number_of_unvisited_vertices - 1;
  rmm::device_vector<vertex_t> buffer(1);
  buffer[0] = number_of_visited_vertices_except_source;
  thrust::transform(handle_.get_thrust_policy(),
                    buffer.begin(),
                    buffer.end(),
                    betweenness_ + source_vertex,
                    betweenness_ + source_vertex,
                    thrust::plus<result_t>());
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::add_vertices_dependencies_to_betweenness()
{
  thrust::transform(handle_.get_thrust_policy(),
                    deltas_,
                    deltas_ + number_of_vertices_,
                    betweenness_,
                    betweenness_,
                    thrust::plus<result_t>());
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::accumulate_vertices(vertex_t max_depth,
                                                                   dim3 grid_configuration,
                                                                   dim3 block_configuration)
{
  for (vertex_t depth = max_depth; depth > 0; --depth) {
    accumulation_kernel<vertex_t, edge_t, weight_t, result_t>
      <<<grid_configuration, block_configuration, 0, handle_.get_stream()>>>(betweenness_,
                                                                             number_of_vertices_,
                                                                             graph_.indices,
                                                                             graph_.offsets,
                                                                             distances_,
                                                                             sp_counters_,
                                                                             deltas_,
                                                                             depth);
  }
  add_vertices_dependencies_to_betweenness();
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::rescale()
{
  bool modified           = false;
  result_t rescale_factor = static_cast<result_t>(1);
  if (normalized_) {
    if (is_edge_betweenness_) {
      std::tie(rescale_factor, modified) =
        rescale_edges_betweenness_centrality(rescale_factor, modified);
    } else {
      std::tie(rescale_factor, modified) =
        rescale_vertices_betweenness_centrality(rescale_factor, modified);
    }
  } else {
    if (!graph_.prop.directed) {
      rescale_factor /= static_cast<result_t>(2);
      modified = true;
    }
  }
  apply_rescale_factor_to_betweenness(rescale_factor);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
std::tuple<result_t, bool>
BC<vertex_t, edge_t, weight_t, result_t>::rescale_edges_betweenness_centrality(
  result_t rescale_factor, bool modified)
{
  result_t casted_number_of_vertices_ = static_cast<result_t>(number_of_vertices_);
  if (number_of_vertices_ > 1) {
    rescale_factor /= ((casted_number_of_vertices_) * (casted_number_of_vertices_ - 1));
    modified = true;
  }
  return std::make_tuple(rescale_factor, modified);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
std::tuple<result_t, bool>
BC<vertex_t, edge_t, weight_t, result_t>::rescale_vertices_betweenness_centrality(
  result_t rescale_factor, bool modified)
{
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices_);
  if (number_of_vertices_ > 2) {
    if (endpoints_) {
      rescale_factor /= (casted_number_of_vertices * (casted_number_of_vertices - 1));
    } else {
      rescale_factor /= ((casted_number_of_vertices - 1) * (casted_number_of_vertices - 2));
    }
    modified = true;
  }
  return std::make_tuple(rescale_factor, modified);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::apply_rescale_factor_to_betweenness(
  result_t rescale_factor)
{
  size_t result_size = number_of_vertices_;
  if (is_edge_betweenness_) result_size = number_of_edges_;
  thrust::transform(handle_.get_thrust_policy(),
                    betweenness_,
                    betweenness_ + result_size,
                    thrust::make_constant_iterator(rescale_factor),
                    betweenness_,
                    thrust::multiplies<result_t>());
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void BC<vertex_t, edge_t, weight_t, result_t>::rescale_by_total_sources_used(
  vertex_t total_number_of_sources_used)
{
  result_t rescale_factor = static_cast<result_t>(1);
  result_t casted_total_number_of_sources_used =
    static_cast<result_t>(total_number_of_sources_used);
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices_);

  if (normalized_) {
    if (number_of_vertices_ > 2 && total_number_of_sources_used > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_total_number_of_sources_used);
    }
  } else if (!graph_.prop.directed) {
    if (number_of_vertices_ > 2 && total_number_of_sources_used > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_total_number_of_sources_used);
    }
  }
  apply_rescale_factor_to_betweenness(rescale_factor);
}
}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void betweenness_centrality(raft::handle_t const& handle,
                            legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                            result_t* result,
                            bool normalize,
                            bool endpoints,
                            weight_t const* weight,
                            vertex_t k,
                            vertex_t const* vertices)
{
  vertex_t total_number_of_sources_used = detail::get_total_number_of_sources<vertex_t>(handle, k);
  if (handle.comms_initialized()) {
    rmm::device_vector<result_t> betweenness(graph.number_of_vertices, 0);
    detail::betweenness_centrality_impl(handle,
                                        graph,
                                        betweenness.data().get(),
                                        normalize,
                                        endpoints,
                                        weight,
                                        k,
                                        vertices,
                                        total_number_of_sources_used);
    handle.get_comms().reduce(betweenness.data().get(),
                              result,
                              betweenness.size(),
                              raft::comms::op_t::SUM,
                              0,
                              handle.get_stream());
  } else {
    detail::betweenness_centrality_impl(handle,
                                        graph,
                                        result,
                                        normalize,
                                        endpoints,
                                        weight,
                                        k,
                                        vertices,
                                        total_number_of_sources_used);
  }
}

template void betweenness_centrality<int, int, float, float>(
  const raft::handle_t&,
  legacy::GraphCSRView<int, int, float> const&,
  float*,
  bool,
  bool,
  float const*,
  int,
  int const*);
template void betweenness_centrality<int, int, double, double>(
  const raft::handle_t&,
  legacy::GraphCSRView<int, int, double> const&,
  double*,
  bool,
  bool,
  double const*,
  int,
  int const*);

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void edge_betweenness_centrality(raft::handle_t const& handle,
                                 legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                 result_t* result,
                                 bool normalize,
                                 weight_t const* weight,
                                 vertex_t k,
                                 vertex_t const* vertices)
{
  vertex_t total_number_of_sources_used = detail::get_total_number_of_sources<vertex_t>(handle, k);
  if (handle.comms_initialized()) {
    rmm::device_vector<result_t> betweenness(graph.number_of_edges, 0);
    detail::edge_betweenness_centrality_impl(handle,
                                             graph,
                                             betweenness.data().get(),
                                             normalize,
                                             weight,
                                             k,
                                             vertices,
                                             total_number_of_sources_used);
    handle.get_comms().reduce(betweenness.data().get(),
                              result,
                              betweenness.size(),
                              raft::comms::op_t::SUM,
                              0,
                              handle.get_stream());
  } else {
    detail::edge_betweenness_centrality_impl(
      handle, graph, result, normalize, weight, k, vertices, total_number_of_sources_used);
  }
}

template void edge_betweenness_centrality<int, int, float, float>(
  const raft::handle_t&,
  legacy::GraphCSRView<int, int, float> const&,
  float*,
  bool,
  float const*,
  int,
  int const*);

template void edge_betweenness_centrality<int, int, double, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const&,
  double*,
  bool,
  double const*,
  int,
  int const*);
}  // namespace cugraph

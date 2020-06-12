/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/transform.h>

#include <algorithms.hpp>
#include <graph.hpp>

#include <utilities/error_utils.h>

#include "betweenness_centrality.cuh"
#include "betweenness_centrality_kernels.cuh"

namespace cugraph {
namespace detail {
namespace {
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality_impl(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 result_t *result,
                                 bool normalize,
                                 bool endpoints,
                                 WT const *weight,
                                 VT const number_of_sources,
                                 VT const *sources)
{
  // Current Implementation relies on BFS
  // FIXME: For SSSP version
  // Brandes Algorithm expects non negative weights for the accumulation
  bool is_edge_betweenness = false;
  verify_betweenness_centrality_input<VT, ET, WT, result_t>(
    result, is_edge_betweenness, normalize, endpoints, weight, number_of_sources, sources);
  cugraph::detail::BC<VT, ET, WT, result_t> bc(graph);
  bc.configure(
    result, is_edge_betweenness, normalize, endpoints, weight, sources, number_of_sources);
  bc.compute();
}

template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality_impl(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                      result_t *result,
                                      bool normalize,
                                      WT const *weight,
                                      VT const number_of_sources,
                                      VT const *sources)
{
  // Current Implementation relies on BFS
  // FIXME: For SSSP version
  // Brandes Algorithm expects non negative weights for the accumulation
  bool is_edge_betweenness = true;
  bool endpoints           = false;
  verify_betweenness_centrality_input<VT, ET, WT, result_t>(
    result, is_edge_betweenness, normalize, endpoints, weight, number_of_sources, sources);
  cugraph::detail::BC<VT, ET, WT, result_t> bc(graph);
  bc.configure(
    result, is_edge_betweenness, normalize, endpoints, weight, sources, number_of_sources);
  bc.compute();
}
}  // namespace

template <typename VT, typename ET, typename WT, typename result_t>
void verify_betweenness_centrality_input(result_t *result,
                                         bool is_edge_betweenness,
                                         bool normalize,
                                         bool endpoints,
                                         WT const *weights,
                                         VT const number_of_sources,
                                         VT const *sources)
{
  static_assert(std::is_same<VT, int>::value, "VT should be int");
  static_assert(std::is_same<ET, int>::value, "ET should be int");
  static_assert(std::is_same<WT, float>::value || std::is_same<WT, double>::value,
                "WT should be float or double");
  static_assert(std::is_same<result_t, float>::value || std::is_same<result_t, double>::value,
                "result_t should be float or double");

  CUGRAPH_EXPECTS(result != nullptr, "Invalid API parameter: betwenness pointer is NULL");
  CUGRAPH_EXPECTS(number_of_sources >= 0, "Number of sources must be positive or equal to 0.");
  if (number_of_sources != 0) {
    CUGRAPH_EXPECTS(sources != nullptr,
                    "Sources cannot be NULL if number_of_source is different from 0.");
  }
  if (is_edge_betweenness) {
    CUGRAPH_EXPECTS(!endpoints, "Endpoints is not supported for edge betweenness centrality.");
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::setup()
{
  number_of_vertices_ = graph_.number_of_vertices;
  number_of_edges_    = graph_.number_of_edges;
  offsets_ptr_        = graph_.offsets;
  indices_ptr_        = graph_.indices;
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::configure(result_t *betweenness,
                                         bool is_edge_betweenness,
                                         bool normalized,
                                         bool endpoints,
                                         WT const *weights,
                                         VT const *sources,
                                         VT number_of_sources)
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

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_work_vectors()
{
  distances_vec_.resize(number_of_vertices_);
  predecessors_vec_.resize(number_of_vertices_);
  sp_counters_vec_.resize(number_of_vertices_);
  deltas_vec_.resize(number_of_vertices_);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_pointers_to_vectors()
{
  distances_    = distances_vec_.data().get();
  predecessors_ = predecessors_vec_.data().get();
  sp_counters_  = sp_counters_vec_.data().get();
  deltas_       = deltas_vec_.data().get();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_device_information()
{
  CUDA_TRY(cudaGetDevice(&device_id_));
  CUDA_TRY(cudaDeviceGetAttribute(&max_grid_dim_1D_, cudaDevAttrMaxGridDimX, device_id_));
  CUDA_TRY(cudaDeviceGetAttribute(&max_block_dim_1D_, cudaDevAttrMaxBlockDimX, device_id_));
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute()
{
  CUGRAPH_EXPECTS(configured_, "BC must be configured before computation");
  if (sources_) {
    for (VT source_idx = 0; source_idx < number_of_sources_; ++source_idx) {
      VT source_vertex = sources_[source_idx];
      compute_single_source(source_vertex);
    }
  } else {
    for (VT source_vertex = 0; source_vertex < number_of_vertices_; ++source_vertex) {
      compute_single_source(source_vertex);
    }
  }
  rescale();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute_single_source(VT source_vertex)
{
  // Step 1) Singe-source shortest-path problem
  cugraph::bfs(
    graph_, distances_, predecessors_, sp_counters_, source_vertex, graph_.prop.directed);

  // FIXME: Remove that with a BC specific class to gather
  //        information during traversal

  // Numeric max value is replaced by -1 as we look for the maximal depth of
  // the traversal, this value is avalaible within the bfs implementation and
  // there could be a way to access it directly and avoid both replace and the
  // max
  thrust::replace(rmm::exec_policy(stream_)->on(stream_),
                  distances_,
                  distances_ + number_of_vertices_,
                  std::numeric_limits<VT>::max(),
                  static_cast<VT>(-1));
  auto current_max_depth = thrust::max_element(
    rmm::exec_policy(stream_)->on(stream_), distances_, distances_ + number_of_vertices_);
  VT max_depth = 0;
  CUDA_TRY(cudaMemcpy(&max_depth, current_max_depth, sizeof(VT), cudaMemcpyDeviceToHost));
  // Step 2) Dependency accumulation
  accumulate(source_vertex, max_depth);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate(VT source_vertex, VT max_depth)
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

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_dependencies()
{
  thrust::fill(rmm::exec_policy(stream_)->on(stream_),
               deltas_,
               deltas_ + number_of_vertices_,
               static_cast<result_t>(0));
}
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_edges(VT max_depth,
                                                dim3 grid_configuration,
                                                dim3 block_configuration)
{
  for (VT depth = max_depth; depth >= 0; --depth) {
    edges_accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream_>>>(betweenness_,
                                                                number_of_vertices_,
                                                                graph_.indices,
                                                                graph_.offsets,
                                                                distances_,
                                                                sp_counters_,
                                                                deltas_,
                                                                depth);
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_vertices_with_endpoints(VT source_vertex,
                                                                  VT max_depth,
                                                                  dim3 grid_configuration,
                                                                  dim3 block_configuration)
{
  for (VT depth = max_depth; depth > 0; --depth) {
    endpoints_accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream_>>>(betweenness_,
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
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::add_reached_endpoints_to_source_betweenness(VT source_vertex)
{
  VT number_of_unvisited_vertices = thrust::count(
    rmm::exec_policy(stream_)->on(stream_), distances_, distances_ + number_of_vertices_, -1);
  VT number_of_visited_vertices_except_source =
    number_of_vertices_ - number_of_unvisited_vertices - 1;
  rmm::device_vector<VT> buffer(1);
  buffer[0] = number_of_visited_vertices_except_source;
  thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                    buffer.begin(),
                    buffer.end(),
                    betweenness_ + source_vertex,
                    betweenness_ + source_vertex,
                    thrust::plus<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::add_vertices_dependencies_to_betweenness()
{
  thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                    deltas_,
                    deltas_ + number_of_vertices_,
                    betweenness_,
                    betweenness_,
                    thrust::plus<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_vertices(VT max_depth,
                                                   dim3 grid_configuration,
                                                   dim3 block_configuration)
{
  for (VT depth = max_depth; depth > 0; --depth) {
    accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream_>>>(betweenness_,
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

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale()
{
  bool modified                      = false;
  result_t rescale_factor            = static_cast<result_t>(1);
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices_);
  result_t casted_number_of_sources  = static_cast<result_t>(number_of_sources_);
  if (normalized_) {
    if (is_edge_betweenness_) {
      rescale_edges_betweenness_centrality(rescale_factor, modified);
    } else {
      rescale_vertices_betweenness_centrality(rescale_factor, modified);
    }
  } else {
    if (!graph_.prop.directed) {
      rescale_factor /= static_cast<result_t>(2);
      modified = true;
    }
  }
  if (modified && !is_edge_betweenness_) {
    if (number_of_sources_ > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_number_of_sources);
    }
  }
  apply_rescale_factor_to_betweenness(rescale_factor);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale_edges_betweenness_centrality(result_t &rescale_factor,
                                                                    bool &modified)
{
  result_t casted_number_of_vertices_ = static_cast<result_t>(number_of_vertices_);
  if (number_of_vertices_ > 1) {
    rescale_factor /= ((casted_number_of_vertices_) * (casted_number_of_vertices_ - 1));
    modified = true;
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale_vertices_betweenness_centrality(result_t &rescale_factor,
                                                                       bool &modified)
{
  result_t casted_number_of_vertices_ = static_cast<result_t>(number_of_vertices_);
  if (number_of_vertices_ > 2) {
    if (endpoints_) {
      rescale_factor /= (casted_number_of_vertices_ * (casted_number_of_vertices_ - 1));
    } else {
      rescale_factor /= ((casted_number_of_vertices_ - 1) * (casted_number_of_vertices_ - 2));
    }
    modified = true;
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::apply_rescale_factor_to_betweenness(result_t rescale_factor)
{
  size_t result_size = number_of_vertices_;
  if (is_edge_betweenness_) result_size = number_of_edges_;
  thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                    betweenness_,
                    betweenness_ + result_size,
                    thrust::make_constant_iterator(rescale_factor),
                    betweenness_,
                    thrust::multiplies<result_t>());
}
}  // namespace detail

template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                            result_t *result,
                            bool normalize,
                            bool endpoints,
                            WT const *weight,
                            VT k,
                            VT const *vertices)
{
  detail::betweenness_centrality_impl(graph, result, normalize, endpoints, weight, k, vertices);
}

template void betweenness_centrality<int, int, float, float>(
  experimental::GraphCSRView<int, int, float> const &,
  float *,
  bool,
  bool,
  float const *,
  int,
  int const *);
template void betweenness_centrality<int, int, double, double>(
  experimental::GraphCSRView<int, int, double> const &,
  double *,
  bool,
  bool,
  double const *,
  int,
  int const *);

template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 result_t *result,
                                 bool normalize,
                                 WT const *weight,
                                 VT k,
                                 VT const *vertices)
{
  detail::edge_betweenness_centrality_impl(graph, result, normalize, weight, k, vertices);
}

template void edge_betweenness_centrality<int, int, float, float>(
  experimental::GraphCSRView<int, int, float> const &,
  float *,
  bool,
  float const *,
  int,
  int const *);
template void edge_betweenness_centrality<int, int, double, double>(
  experimental::GraphCSRView<int, int, double> const &,
  double *,
  bool,
  double const *,
  int,
  int const *);
}  // namespace cugraph

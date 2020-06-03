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
/**
 * @param[out]  result          array<result_t>(number_of_vertices)
 * @param[in]   normalize       bool True -> Apply normalization
 * @param[in]   endpoints       bool Include endpoints
 * @param[in]   weights   (NIY) array<WT>(number_of_edges) Weights to use
 * @param[in]   k               Number of sources
 * @param[in]   vertices        array<VT>(k) Sources for traversal
 */
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                            result_t *result,
                            bool normalize,
                            bool endpoints,
                            WT const *weight,
                            VT k,
                            VT const *vertices)
{
  detail::betweenness_centrality(graph, result, normalize, endpoints, weight, k, vertices);
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

/**
 * @param[out]  result          array<result_t>(number_of_vertices)
 * @param[in]   normalize       bool True -> Apply normalization
 * @param[in]   weights   (NIY) array<WT>(number_of_edges) Weights to use
 * @param[in]   k               Number of sources
 * @param[in]   vertices        array<VT>(k) Sources for traversal
 */
template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 result_t *result,
                                 bool normalize,
                                 WT const *weight,
                                 VT k,
                                 VT const *vertices)
{
  detail::edge_betweenness_centrality(graph, result, normalize, weight, k, vertices);
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

namespace detail {
/**
 * ---------------------------------------------------------------------------*
 * @brief Native betweenness centrality
 *
 * @file betweenness_centrality.cu
 * --------------------------------------------------------------------------*/
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
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
void verify_betweenness_centrality_input(result_t *result,
                                         bool is_edge_betweenness,
                                         bool normalize,
                                         bool endpoints,
                                         WT const *weights,
                                         VT const number_of_sources,
                                         VT const *sources)
{
  CUGRAPH_EXPECTS(result != nullptr, "Invalid API parameter: output betwenness is nullptr");
  if (typeid(VT) != typeid(int)) {
    CUGRAPH_FAIL("Unsupported vertex id data type, please use int");
  }
  if (typeid(ET) != typeid(int)) { CUGRAPH_FAIL("Unsupported edge id data type, please use int"); }
  if (typeid(WT) != typeid(float) && typeid(WT) != typeid(double)) {
    CUGRAPH_FAIL("Unsupported weight data type, please use float or double");
  }
  if (typeid(result_t) != typeid(float) && typeid(result_t) != typeid(double)) {
    CUGRAPH_FAIL("Unsupported result data type, please use float or double");
  }
  if (number_of_sources < 0) {
    CUGRAPH_FAIL("Number of sources must be positive or equal to 0.");
  } else if (number_of_sources != 0) {
    CUGRAPH_EXPECTS(sources != nullptr,
                    "sources cannot be null if number_of_source is different from 0.");
  }
  if (is_edge_betweenness) {
    CUGRAPH_EXPECTS(!endpoints, "endpoints is not supported for edge betweenness centrality.");
  }
}

/**
 * ---------------------------------------------------------------------------*
 * @brief Native edge betweenness centrality
 *
 * @file betweenness_centrality.cu
 * --------------------------------------------------------------------------*/
template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
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

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::setup()
{
  number_of_vertices = graph.number_of_vertices;
  number_of_edges    = graph.number_of_edges;
  offsets_ptr        = graph.offsets;
  indices_ptr        = graph.indices;
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::configure(result_t *_betweenness,
                                         bool _is_edge_betweenness,
                                         bool _normalized,
                                         bool _endpoints,
                                         WT const *_weights,
                                         VT const *_sources,
                                         VT _number_of_sources)
{
  // --- Bind betweenness output vector to internal ---
  betweenness         = _betweenness;
  normalized          = _normalized;
  endpoints           = _endpoints;
  sources             = _sources;
  number_of_sources   = _number_of_sources;
  edge_weights_ptr    = _weights;
  is_edge_betweenness = _is_edge_betweenness;

  // --- Working data allocation ---
  initialize_work_vectors();
  initialize_pointers_to_vectors();

  // --- Get Device Information ---
  initialize_device_information();

  // --- Confirm that configuration went through ---
  configured = true;
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_work_vectors()
{
  distances_vec.resize(number_of_vertices);
  predecessors_vec.resize(number_of_vertices);
  sp_counters_vec.resize(number_of_vertices);
  deltas_vec.resize(number_of_vertices);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_pointers_to_vectors()
{
  distances    = distances_vec.data().get();
  predecessors = predecessors_vec.data().get();
  sp_counters  = sp_counters_vec.data().get();
  deltas       = deltas_vec.data().get();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_device_information()
{
  CUDA_TRY(cudaGetDevice(&device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&max_grid_dim_1D, cudaDevAttrMaxGridDimX, device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&max_block_dim_1D, cudaDevAttrMaxBlockDimX, device_id));
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute()
{
  CUGRAPH_EXPECTS(configured, "BC must be configured before computation");
  if (sources) {
    for (VT source_idx = 0; source_idx < number_of_sources; ++source_idx) {
      VT source_vertex = sources[source_idx];
      compute_single_source(source_vertex);
    }
  } else {
    for (VT source_vertex = 0; source_vertex < number_of_vertices; ++source_vertex) {
      compute_single_source(source_vertex);
    }
  }
  rescale();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::compute_single_source(VT source_vertex)
{
  // Step 1) Singe-source shortest-path problem
  cugraph::bfs(graph, distances, predecessors, sp_counters, source_vertex, graph.prop.directed);

  // FIXME: Remove that with a BC specific class to gather
  //        information during traversal

  // Numeric max value is replaced by -1 as we look for the maximal depth of
  // the traversal, this value is avalaible within the bfs implementation and
  // there could be a way to access it directly and avoid both replace and the
  // max
  thrust::replace(rmm::exec_policy(stream)->on(stream),
                  distances,
                  distances + number_of_vertices,
                  std::numeric_limits<VT>::max(),
                  static_cast<VT>(-1));
  auto current_max_depth = thrust::max_element(
    rmm::exec_policy(stream)->on(stream), distances, distances + number_of_vertices);
  VT max_depth = 0;
  CUDA_TRY(cudaMemcpy(&max_depth, current_max_depth, sizeof(VT), cudaMemcpyDeviceToHost));
  // Step 2) Dependency accumulation
  accumulate(source_vertex, max_depth);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate(VT source_vertex, VT max_depth)
{
  dim3 grid_configuration, block_configuration;
  block_configuration.x = max_block_dim_1D;
  grid_configuration.x  = min(max_grid_dim_1D, (number_of_edges / block_configuration.x + 1));

  initialize_dependencies();

  if (is_edge_betweenness) {
    accumulate_edges(max_depth, grid_configuration, block_configuration);
  } else if (endpoints) {
    accumulate_vertices_with_endpoints(
      source_vertex, max_depth, grid_configuration, block_configuration);
  } else {
    accumulate_vertices(max_depth, grid_configuration, block_configuration);
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::initialize_dependencies()
{
  thrust::fill(rmm::exec_policy(stream)->on(stream),
               deltas,
               deltas + number_of_vertices,
               static_cast<result_t>(0));
}
template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_edges(VT max_depth,
                                                dim3 grid_configuration,
                                                dim3 block_configuration)
{
  for (VT depth = max_depth; depth >= 0; --depth) {
    edges_accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream>>>(betweenness,
                                                               number_of_vertices,
                                                               graph.indices,
                                                               graph.offsets,
                                                               distances,
                                                               sp_counters,
                                                               deltas,
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
      <<<grid_configuration, block_configuration, 0, stream>>>(betweenness,
                                                               number_of_vertices,
                                                               graph.indices,
                                                               graph.offsets,
                                                               distances,
                                                               sp_counters,
                                                               deltas,
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
    rmm::exec_policy(stream)->on(stream), distances, distances + number_of_vertices, -1);
  VT number_of_visited_vertices_except_source =
    number_of_vertices - number_of_unvisited_vertices - 1;
  rmm::device_vector<VT> buffer(1);
  buffer[0] = {number_of_visited_vertices_except_source};
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    buffer.begin(),
                    buffer.end(),
                    betweenness + source_vertex,
                    betweenness + source_vertex,
                    thrust::plus<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::add_vertices_dependencies_to_betweenness()
{
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    deltas,
                    deltas + number_of_vertices,
                    betweenness,
                    betweenness,
                    thrust::plus<result_t>());
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::accumulate_vertices(VT max_depth,
                                                   dim3 grid_configuration,
                                                   dim3 block_configuration)
{
  for (VT depth = max_depth; depth > 0; --depth) {
    accumulation_kernel<VT, ET, WT, result_t>
      <<<grid_configuration, block_configuration, 0, stream>>>(betweenness,
                                                               number_of_vertices,
                                                               graph.indices,
                                                               graph.offsets,
                                                               distances,
                                                               sp_counters,
                                                               deltas,
                                                               depth);
  }
  add_vertices_dependencies_to_betweenness();
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale()
{
  bool modified                      = false;
  result_t rescale_factor            = static_cast<result_t>(1);
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices);
  result_t casted_number_of_sources  = static_cast<result_t>(number_of_sources);
  if (normalized) {
    if (is_edge_betweenness) {
      rescale_edges_betweenness_centrality(rescale_factor, modified);
    } else {
      rescale_vertices_betweenness_centrality(rescale_factor, endpoints, modified);
    }
  } else {
    if (!graph.prop.directed) {
      rescale_factor /= static_cast<result_t>(2);
      modified = true;
    }
  }
  if (modified && !is_edge_betweenness) {
    if (number_of_sources > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_number_of_sources);
    }
  }
  apply_rescale_factor_to_betweenness(rescale_factor);
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale_edges_betweenness_centrality(result_t &rescale_factor,
                                                                    bool &modified)
{
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices);
  if (number_of_vertices > 1) {
    rescale_factor /= ((casted_number_of_vertices) * (casted_number_of_vertices - 1));
    modified = true;
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::rescale_vertices_betweenness_centrality(result_t &rescale_factor,
                                                                       bool endpoints,
                                                                       bool &modified)
{
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices);
  if (number_of_vertices > 2) {
    if (endpoints) {
      rescale_factor /= (casted_number_of_vertices * (casted_number_of_vertices - 1));
    } else {
      rescale_factor /= ((casted_number_of_vertices - 1) * (casted_number_of_vertices - 2));
    }
    modified = true;
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void BC<VT, ET, WT, result_t>::apply_rescale_factor_to_betweenness(result_t rescale_factor)
{
  size_t result_size = number_of_vertices;
  if (is_edge_betweenness) result_size = number_of_edges;
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    betweenness,
                    betweenness + result_size,
                    thrust::make_constant_iterator(rescale_factor),
                    betweenness,
                    thrust::multiplies<result_t>());
}
}  // namespace detail
}  // namespace cugraph

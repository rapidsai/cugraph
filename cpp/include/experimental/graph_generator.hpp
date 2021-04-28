/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#pragma once

#include <experimental/graph.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <tuple>

namespace cugraph {
namespace experimental {

/**
 * @brief generate an edge list for an R-mat graph.
 *
 * This function allows multi-edges and self-loops similar to the Graph 500 reference
 * implementation.
 *
 * @p scramble_vertex_ids needs to be set to `true` to generate a graph conforming to the Graph 500
 * specification (note that scrambling does not affect cuGraph's graph construction performance, so
 * this is generally unnecessary). If `edge_factor` is given (e.g. Graph 500), set @p num_edges to
 * (size_t{1} << @p scale) * `edge_factor`. To generate an undirected graph, set @p b == @p c and @p
 * clip_and_flip = true. All the resulting edges will be placed in the lower triangular part
 * (inculding the diagonal) of the graph adjacency matrix.
 *
 * For multi-GPU generation with `P` GPUs, @p seed should be set to different values in different
 * GPUs to avoid every GPU generating the same set of edges. @p num_edges should be adjusted as
 * well; e.g. assuming `edge_factor` is given, set @p num_edges = (size_t{1} << @p scale) *
 * `edge_factor` / `P` + (rank < (((size_t{1} << @p scale) * `edge_factor`) % P) ? 1 : 0).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param scale Scale factor to set the number of verties in the graph. Vertex IDs have values in
 * [0, V), where V = 1 << @p scale.
 * @param num_edges Number of edges to generate.
 * @param a a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator (vist https://graph500.org
 * for additional details). a, b, c, d should be non-negative and a + b + c should be no larger
 * than 1.0.
 * @param b a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator (vist https://graph500.org
 * for additional details). a, b, c, d should be non-negative and a + b + c should be no larger
 * than 1.0.
 * @param c a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator (vist https://graph500.org
 * for additional details). a, b, c, d should be non-negative and a + b + c should be no larger
 * than 1.0.
 * @param seed Seed value for the random number generator.
 * @param clip_and_flip Flag controlling whether to generate edges only in the lower triangular part
 * (including the diagonal) of the graph adjacency matrix (if set to `true`) or not (if set to
 * `false`).
 * @param scramble_vertex_ids Flag controlling whether to scramble vertex ID bits (if set to `true`)
 * or not (if set to `false`); scrambling vertx ID bits breaks correlation between vertex ID values
 * and vertex degrees. The scramble code here follows the algorithm in the Graph 500 reference
 * implementation version 3.0.0.
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> generate_rmat_edgelist(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a                 = 0.57,
  double b                 = 0.19,
  double c                 = 0.19,
  uint64_t seed            = 0,
  bool clip_and_flip       = false,
  bool scramble_vertex_ids = false);

enum class generator_distribution_t { POWER_LAW = 0, UNIFORM };

/**
 * @brief generate multiple edge lists using the R-mat graph generator.
 *
 * This function allows multi-edges and self-loops similar to the Graph 500 reference
 * implementation.
 *
 * @p scramble_vertex_ids needs to be set to `true` to generate a graph conforming to the Graph 500
 * specification (note that scrambling does not affect cuGraph's graph construction performance, so
 * this is generally unnecessary). If `edge_factor` is given (e.g. Graph 500), set @p num_edges to
 * (size_t{1} << @p scale) * `edge_factor`. To generate an undirected graph, set @p b == @p c and @p
 * clip_and_flip = true. All the resulting edges will be placed in the lower triangular part
 * (inculding the diagonal) of the graph adjacency matrix.
 *
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param n_edgelists Number of edge lists (graphs) to generate
 * @param min_scale Scale factor to set the minimum number of verties in the graph.
 * @param max_scale Scale factor to set the maximum number of verties in the graph.
 * @param edge_factor Average number of edges per vertex to generate.
 * @param size_distribution Distribution of the graph sizes, impacts the scale parameter of the
 * R-MAT generator
 * @param edge_distribution Edges distribution for each graph, impacts how R-MAT parameters a,b,c,d,
 * are set.
 * @param seed Seed value for the random number generator.
 * @param clip_and_flip Flag controlling whether to generate edges only in the lower triangular part
 * (including the diagonal) of the graph adjacency matrix (if set to `true`) or not (if set to
 * `false`).
 * @param scramble_vertex_ids Flag controlling whether to scramble vertex ID bits (if set to `true`)
 * or not (if set to `false`); scrambling vertx ID bits breaks correlation between vertex ID values
 * and vertex degrees. The scramble code here follows the algorithm in the Graph 500 reference
 * implementation version 3.0.0.
 * @return A vector of std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> of
 *size @p n_edgelists, each vector element being a tuple of rmm::device_uvector objects for edge
 *source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::vector<std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>>
generate_rmat_edgelists(
  raft::handle_t const& handle,
  size_t n_edgelists,
  size_t min_scale,
  size_t max_scale,
  size_t edge_factor                         = 16,
  generator_distribution_t size_distribution = generator_distribution_t::POWER_LAW,
  generator_distribution_t edge_distribution = generator_distribution_t::POWER_LAW,
  uint64_t seed                              = 0,
  bool clip_and_flip                         = false,
  bool scramble_vertex_ids                   = false);

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
generate_graph_from_edgelist(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_vertex_span,
  rmm::device_uvector<vertex_t>&& edgelist_rows,
  rmm::device_uvector<vertex_t>&& edgelist_cols,
  rmm::device_uvector<weight_t>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber);

}  // namespace experimental
}  // namespace cugraph

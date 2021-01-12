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

#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>
#include <utilities/device_comm.cuh>
#include <utilities/error.hpp>
#include <utilities/host_scalar_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cuco/static_map.cuh>

#include <tuple>

namespace cugraph {
namespace experimental {
namespace detail {
}  // namespace detail

/**
 * @brief extract induced subgraph(s).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed
 * @tparam store_transposed Flag indicating whether to store the graph adjacency matrix as is or as
 * transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of, we extract induced subgraphs from @p graph_view.
 * @param subgraph_offsets Pointer to subgraph vertex offsets (size == @p num_subgraphs + 1).
 * @param subgraph_vertices Pointer to subgraph vertices (size == @p subgraph_offsets[@p
 * num_subgraphs]). @p subgraph_vertices for each subgraph should be sorted in ascending order.
 * @param num_subgraphs Number of induced subgraphs to extract.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>,
 * rmm::device_uvector<weight_t>, rmm::device_uvector<size_t>> Quadraplet of edge source vertices,
 * edge destination vertices, edge weights, and edge offsets for each induced subgraph.
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<size_t>>
extract_induced_subgraph(
  raft::handle_t const &handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transpsoed, multi_gpu> const &graph_view,
  size_t const *subgraph_offsets /* size == num_subgraphs + 1 */,
  vertex_t const *subgraph_vertices /* size == subgraph_offsets[num_subgraphs] */,
  size_t num_subgraphs,
  bool do_expensive_check = false)
{
  // FIXME: this code is inefficient for the vertices with their local degrees much larger than the
  // number of vertices in the subgraphs (in this case, searching that the subgraph vertices are
  // included in the local neighbors is more efficient than searching the local neighbors are
  // included in the subgraph vertices). We may later add additional code to handle such cases.
  // FIXME: we may consider the performance (speed & memory footprint, hash based approach uses
  // extra-memory) of hash table based and binary search based approaches

  // 1. check input arguments

  if (do_expensive_check) {
    size_t num_aggregate_subgraph_vertices{};
    raft::update_host(
      &num_aggregate_subgraph_vertices, subgraph_offsets + num_subgraphs, 1, handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    size_t should_be_zero{std::numeric_limits<size_t>::max()};
    raft::update_host(&should_be_zero, subgraph_offsets, 1, handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
    CUGRAPH_EXPECTS(should_be_zero == 0,
                    "Invalid input argument: subgraph_offsets[0] should be 0.");
    CUGRAPH_EXPECTS(
      thrust::is_sorted(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        subgraph_offsets,
                        subgraph_offsets + (num_subgraphs + 1)),
      "Invalid input argument: subgraph_offsets is not sorted.");
    vertex_partition_device_t<graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>
      vertex_partition(graph_view);
    CUGRAPH_EXPECTS(thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                     subgraph_vertices,
                                     subgraph_vertices + num_aggregate_subgraph_vertices,
                                     [vertex_partition] __device__(auto v) {
                                       return !vertex_partition.is_valid_vertex(v) ||
                                              !vertex_partition.is_local_vertex_nocheck(v);
                                     }),
                    "Invalid input argument: subgraph_vertices has invalid vertex IDs.");

    CUGRAPH_EXPECTS(
      thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(num_subgraphs),
                       [subgraph_offsets, subgraph_vertices] __device__(auto i) {
                         return !thrust::is_sorted(thrust::seq,
                                                   subgraph_vertices + subgraph_offsets[i],
                                                   subgraph_vertices + subgraph_offsets[i + 1]) ||
                                (thrust::unique(thrust::seq,
                                                subgraph_vertices + subgraph_offsets[i],
                                                subgraph_vertices + subgraph_offsets[i + 1]) !=
                                 subgraph_vertices + subgraph_offsets[i + 1]);
                       }) == 0,
      "Invalid input argument: subgraph_vertices for each subgraph idx should be sorted in "
      "ascending order and unique.");
  }

  // 2. extract induced subgraphs

  if (multi_gpu) {
    CUGRAPH_FAIL("Unimplemented.");
  } else {
    size_t num_aggregate_subgraph_vertices{};
    raft::update_host(
      &num_aggregate_subgraph_vertices, subgraph_offsets + num_subgraphs, 1, handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    rmm::device_uvector<size_t> subgraph_edge_offsets(num_aggregate_subgraph_vertices + 1,
                                                      handle.get_stream());

    matrix_partition<GraphViewType> matrix_partition(graph_view, 0);
    thrust::tabulate(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      subgraph_edge_offsets.begin(),
      subgraph_edge_offsets.end() - 1,
      [subgraph_offsets, subgraph_vertices, num_subgraphs, matrix_partition] __device__(auto i) {
        auto subgraph_idx = thrust::distance(
          subgraph_offsets + 1,
          thrust::lower_bound(
            thrust::seq, subgraph_offsets + 1, subgraph_offsets + num_subgraphs + 1, size_t{i}));
        vertex_t const *indices{nullptr};
        weight_t cosnt *weights{nullptr};
        edge_t local_degree{};
        auto major_offset =
          matrix_partition.get_major_offset_from_major_nocheck(subgraph_vertices[i]);
        thrust::tie(indices, weights, local_degree) =
          matrix_partition.get_local_edges(major_offset);
        return thrust::count_if(
          thrust::seq,
          indices,
          indices + local_degree,
          [vertex_first = subgraph_offsets + subgraph_idx,
           vertex_last  = subgraph_offsets + (subgraph_idx + 1)] __device__(auto nbr) {
            return thrust::binary_search(thrust::seq, vertex_first, vertex_last, nbr);
          });
      });
    thrust::exclusive_scan(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                           subgraph_edge_offsets,
                           subgraph_edge_offsets + num_aggregate_subgraph_vertices + 1,
                           subgraph_edge_offsets);

    size_t num_aggregate_edges{};
    raft::update_host(&num_aggregate_edges,
                      subgraph_edge_offsets + num_aggregate_subgraph_vertices,
                      1,
                      handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    rmm::device_uvector<vertex_t> edge_majors(num_aggregate_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> edge_minors(num_aggregate_edges, handle.get_stream());
    rmm::device_uvector<weight_t> edge_weights(graph_view.is_weighted() ? num_aggregate_edges : size_t{0},
                                              handle.get_stream());

    thrust::for_each(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_subgraphs),
      [subgraph_offsets, subgraph_vertices, num_subgraphs, matrix_partition, subgraph_edge_offsets = subgraph_edge_offsets.data()] __device__(auto i) {
        auto subgraph_idx = thrust::distance(
          subgraph_offsets + 1,
          thrust::lower_bound(
            thrust::seq, subgraph_offsets + 1, subgraph_offsets + num_subgraphs + 1, size_t{i}));
        vertex_t const *indices{nullptr};
        weight_t cosnt *weights{nullptr};
        edge_t local_degree{};
        auto major_offset =
          matrix_partition.get_major_offset_from_major_nocheck(subgraph_vertices[i]);
        thrust::tie(indices, weights, local_degree) =
          matrix_partition.get_local_edges(major_offset);
        thrust::copy_if(
          thrust::seq,
          thrust::make_zip_iterator(thrust::make_constant_iterator(subgraph_vertices[i]), indices, weights, ,
          indices + local_degree,
          [vertex_first = subgraph_offsets + subgraph_idx,
           vertex_last  = subgraph_offsets + (subgraph_idx + 1)] __device__(auto nbr) {
            return thrust::binary_search(thrust::seq, vertex_first, vertex_last, nbr);
          });
      });
  }

  return std::make_tuple(std::move(), std::move(), std::move(), std::move());
}

}  // namespace experimental
}  // namespace cugraph

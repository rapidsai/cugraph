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

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph.hpp>
#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <utilities/error.hpp>
#include <utilities/shuffle_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cuco/static_map.cuh>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace experimental {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  compressed_sparse_to_edgelist(edge_t const *compressed_sparse_offsets,
                                vertex_t const *compressed_sparse_indices,
                                weight_t const *compressed_sparse_weights,
                                vertex_t major_first,
                                vertex_t major_last,
                                cudaStream_t stream)
{
  edge_t number_of_edges{0};
  raft::update_host(
    &number_of_edges, compressed_sparse_offsets + (major_last - major_first), 1, stream);
  CUDA_TRY(cudaStreamSynchronize(stream));
  rmm::device_uvector<vertex_t> edgelist_major_vertices(number_of_edges, stream);
  rmm::device_uvector<vertex_t> edgelist_minor_vertices(number_of_edges, stream);
  rmm::device_uvector<weight_t> edgelist_weights(
    compressed_sparse_weights != nullptr ? number_of_edges : 0, stream);

  // FIXME: this is highly inefficient for very high-degree vertices, for better performance, we can
  // fill high-degree vertices using one CUDA block per vertex, mid-degree vertices using one CUDA
  // warp per vertex, and low-degree vertices using one CUDA thread per block
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator(major_first),
                   thrust::make_counting_iterator(major_last),
                   [compressed_sparse_offsets,
                    major_first,
                    p_majors = edgelist_major_vertices.begin()] __device__(auto v) {
                     auto first = compressed_sparse_offsets[v - major_first];
                     auto last  = compressed_sparse_offsets[v - major_first + 1];
                     thrust::fill(thrust::seq, p_majors + first, p_majors + last, v);
                   });
  thrust::copy(rmm::exec_policy(stream)->on(stream),
               compressed_sparse_indices,
               compressed_sparse_indices + number_of_edges,
               edgelist_minor_vertices.begin());
  if (compressed_sparse_weights != nullptr) {
    thrust::copy(rmm::exec_policy(stream)->on(stream),
                 compressed_sparse_weights,
                 compressed_sparse_weights + number_of_edges,
                 edgelist_weights.data());
  }

  return std::make_tuple(std::move(edgelist_major_vertices),
                         std::move(edgelist_minor_vertices),
                         std::move(edgelist_weights));
}

template <typename vertex_t, typename weight_t>
void sort_and_coarsen_edgelist(rmm::device_uvector<vertex_t> &edgelist_major_vertices /* [INOUT] */,
                               rmm::device_uvector<vertex_t> &edgelist_minor_vertices /* [INOUT] */,
                               rmm::device_uvector<weight_t> &edgelist_weights /* [INOUT] */,
                               cudaStream_t stream)
{
  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));

  size_t number_of_edges{0};
  if (edgelist_weights.size() > 0) {
    thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
                        pair_first,
                        pair_first + edgelist_major_vertices.size(),
                        edgelist_weights.begin());

    rmm::device_uvector<vertex_t> tmp_edgelist_major_vertices(edgelist_major_vertices.size(),
                                                              stream);
    rmm::device_uvector<vertex_t> tmp_edgelist_minor_vertices(tmp_edgelist_major_vertices.size(),
                                                              stream);
    rmm::device_uvector<weight_t> tmp_edgelist_weights(tmp_edgelist_major_vertices.size(), stream);
    auto it = thrust::reduce_by_key(
      rmm::exec_policy(stream)->on(stream),
      pair_first,
      pair_first + edgelist_major_vertices.size(),
      edgelist_weights.begin(),
      thrust::make_zip_iterator(thrust::make_tuple(tmp_edgelist_major_vertices.begin(),
                                                   tmp_edgelist_minor_vertices.begin())),
      tmp_edgelist_weights.begin());
    number_of_edges = thrust::distance(tmp_edgelist_weights.begin(), thrust::get<1>(it));

    CUDA_TRY(cudaStreamSynchronize(
      stream));  // memory blocks owned by edgelist_(major_vertices,minor_vertices,weights) will be
                 // freed after the assignments below

    edgelist_major_vertices = std::move(tmp_edgelist_major_vertices);
    edgelist_minor_vertices = std::move(tmp_edgelist_minor_vertices);
    edgelist_weights        = std::move(tmp_edgelist_weights);
  } else {
    thrust::sort(rmm::exec_policy(stream)->on(stream),
                 pair_first,
                 pair_first + edgelist_major_vertices.size());
    auto it         = thrust::unique(rmm::exec_policy(stream)->on(stream),
                             pair_first,
                             pair_first + edgelist_major_vertices.size());
    number_of_edges = thrust::distance(pair_first, it);
  }

  edgelist_major_vertices.resize(number_of_edges, stream);
  edgelist_minor_vertices.resize(number_of_edges, stream);
  edgelist_weights.resize(number_of_edges, stream);
  edgelist_major_vertices.shrink_to_fit(stream);
  edgelist_minor_vertices.shrink_to_fit(stream);
  edgelist_weights.shrink_to_fit(stream);
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  compressed_sparse_to_relabeled_and_sorted_and_coarsened_edgelist(
    edge_t const *compressed_sparse_offsets,
    vertex_t const *compressed_sparse_indices,
    weight_t const *compressed_sparse_weights,
    vertex_t const *p_major_labels,
    vertex_t const *p_minor_labels,
    vertex_t major_first,
    vertex_t major_last,
    vertex_t minor_first,
    vertex_t minor_last,
    cudaStream_t stream)
{
  // FIXME: it might be possible to directly create relabled & coarsened edgelist from the
  // compressed sparse format to save memory

  rmm::device_uvector<vertex_t> edgelist_major_vertices(0, stream);
  rmm::device_uvector<vertex_t> edgelist_minor_vertices(0, stream);
  rmm::device_uvector<weight_t> edgelist_weights(0, stream);
  std::tie(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights) =
    compressed_sparse_to_edgelist(compressed_sparse_offsets,
                                  compressed_sparse_indices,
                                  compressed_sparse_weights,
                                  major_first,
                                  major_last,
                                  stream);

  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));
  thrust::transform(
    rmm::exec_policy(stream)->on(stream),
    pair_first,
    pair_first + edgelist_major_vertices.size(),
    pair_first,
    [p_major_labels, p_minor_labels, major_first, minor_first] __device__(auto val) {
      return thrust::make_tuple(p_major_labels[thrust::get<0>(val) - major_first],
                                p_minor_labels[thrust::get<1>(val) - minor_first]);
    });

  sort_and_coarsen_edgelist(
    edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights, stream);

  return std::make_tuple(std::move(edgelist_major_vertices),
                         std::move(edgelist_minor_vertices),
                         std::move(edgelist_weights));
}

// multi-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
             rmm::device_uvector<vertex_t>>>
coarsen_graph(
  raft::handle_t const &handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const &graph_view,
  vertex_t const *labels)
{
  auto &comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto &row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto const row_comm_rank = row_comm.get_rank();
  auto &col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();
  auto const col_comm_rank = col_comm.get_rank();

  // 1. locally construct coarsened edge list

  // FIXME: we don't need adj_matrix_major_labels if we apply the same partitioning scheme
  // regardless of hypergraph partitioning is applied or not
  rmm::device_uvector<vertex_t> adj_matrix_major_labels(
    store_transposed ? graph_view.get_number_of_local_adj_matrix_partition_cols()
                     : graph_view.get_number_of_local_adj_matrix_partition_rows(),
    handle.get_stream());
  rmm::device_uvector<vertex_t> adj_matrix_minor_labels(
    store_transposed ? graph_view.get_number_of_local_adj_matrix_partition_rows()
                     : graph_view.get_number_of_local_adj_matrix_partition_cols(),
    handle.get_stream());
  if (store_transposed) {
    copy_to_adj_matrix_col(handle, graph_view, labels, adj_matrix_major_labels.data());
    copy_to_adj_matrix_row(handle, graph_view, labels, adj_matrix_minor_labels.data());
  } else {
    copy_to_adj_matrix_row(handle, graph_view, labels, adj_matrix_major_labels.data());
    copy_to_adj_matrix_col(handle, graph_view, labels, adj_matrix_minor_labels.data());
  }

  rmm::device_uvector<vertex_t> coarsened_edgelist_major_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> coarsened_edgelist_minor_vertices(0, handle.get_stream());
  rmm::device_uvector<weight_t> coarsened_edgelist_weights(0, handle.get_stream());
  // FIXME: we may compare performance/memory footprint with the hash_based approach especially when
  // cuco::dynamic_map becomes available (so we don't need to preallocate memory assuming the worst
  // case). We may be able to limit the memory requirement close to the final coarsened edgelist
  // with the hash based approach.
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    // get edge list

    rmm::device_uvector<vertex_t> edgelist_major_vertices(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_minor_vertices(0, handle.get_stream());
    rmm::device_uvector<weight_t> edgelist_weights(0, handle.get_stream());
    std::tie(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights) =
      compressed_sparse_to_relabeled_and_sorted_and_coarsened_edgelist(
        graph_view.offsets(i),
        graph_view.indices(i),
        graph_view.weights(i),
        adj_matrix_major_labels.begin() +
          (store_transposed ? graph_view.get_local_adj_matrix_partition_col_value_start_offset(i)
                            : graph_view.get_local_adj_matrix_partition_row_value_start_offset(i)),
        adj_matrix_minor_labels.begin(),
        store_transposed ? graph_view.get_local_adj_matrix_partition_col_first(i)
                         : graph_view.get_local_adj_matrix_partition_row_first(i),
        store_transposed ? graph_view.get_local_adj_matrix_partition_col_last(i)
                         : graph_view.get_local_adj_matrix_partition_row_last(i),
        store_transposed ? graph_view.get_local_adj_matrix_partition_row_first(i)
                         : graph_view.get_local_adj_matrix_partition_col_first(i),
        store_transposed ? graph_view.get_local_adj_matrix_partition_row_last(i)
                         : graph_view.get_local_adj_matrix_partition_col_last(i),
        handle.get_stream());

    auto cur_size = coarsened_edgelist_major_vertices.size();
    // FIXME: this can lead to frequent costly reallocation; we may be able to avoid this if we can
    // reserve address space to avoid expensive reallocation.
    // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management
    coarsened_edgelist_major_vertices.resize(cur_size + edgelist_major_vertices.size(),
                                             handle.get_stream());
    coarsened_edgelist_minor_vertices.resize(coarsened_edgelist_major_vertices.size(),
                                             handle.get_stream());
    coarsened_edgelist_weights.resize(
      graph_view.is_weighted() ? coarsened_edgelist_major_vertices.size() : 0, handle.get_stream());

    if (graph_view.is_weighted()) {
      auto src_edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_major_vertices.begin(),
                                                     edgelist_minor_vertices.begin(),
                                                     edgelist_weights.begin()));
      auto dst_edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_major_vertices.begin(),
                                                     coarsened_edgelist_minor_vertices.begin(),
                                                     coarsened_edgelist_weights.begin())) +
        cur_size;
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   src_edge_first,
                   src_edge_first + edgelist_major_vertices.size(),
                   dst_edge_first);
    } else {
      auto src_edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));
      auto dst_edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_major_vertices.begin(),
                                                     coarsened_edgelist_minor_vertices.begin())) +
        cur_size;
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   src_edge_first,
                   src_edge_first + edgelist_major_vertices.size(),
                   dst_edge_first);
    }

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // edgelist_(major_vertices,minor_vertices,weights)
                              // will become out-of-scope
  }

  sort_and_coarsen_edgelist(coarsened_edgelist_major_vertices,
                            coarsened_edgelist_minor_vertices,
                            coarsened_edgelist_weights,
                            handle.get_stream());

  // 2. globally shuffle edge list and re-coarsen

  {
    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_major_vertices.begin(),
                                                   coarsened_edgelist_minor_vertices.begin(),
                                                   coarsened_edgelist_weights.begin()));
    rmm::device_uvector<vertex_t> rx_edgelist_major_vertices(0, handle.get_stream());
    rmm::device_uvector<vertex_t> rx_edgelist_minor_vertices(0, handle.get_stream());
    rmm::device_uvector<weight_t> rx_edgelist_weights(0, handle.get_stream());
    std::forward_as_tuple(
      std::tie(rx_edgelist_major_vertices, rx_edgelist_minor_vertices, rx_edgelist_weights),
      std::ignore) =
      sort_and_shuffle_values(
        handle.get_comms(),
        edge_first,
        edge_first + coarsened_edgelist_major_vertices.size(),
        [key_func =
           detail::compute_gpu_id_from_edge_t<vertex_t, store_transposed>{
             graph_view.is_hypergraph_partitioned(),
             comm.get_size(),
             row_comm.get_size(),
             col_comm.get_size()}] __device__(auto val) {
          return store_transposed ? key_func(thrust::get<1>(val), thrust::get<0>(val))
                                  : key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());

    sort_and_coarsen_edgelist(rx_edgelist_major_vertices,
                              rx_edgelist_minor_vertices,
                              rx_edgelist_weights,
                              handle.get_stream());

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // memory blocks owned by
                              // coarsened_edgelist_(major_vertices,minor_vertices,weights)
                              // will be freed after the assignments below

    coarsened_edgelist_major_vertices = std::move(rx_edgelist_major_vertices);
    coarsened_edgelist_minor_vertices = std::move(rx_edgelist_minor_vertices);
    coarsened_edgelist_weights        = std::move(rx_edgelist_weights);
  }

  // 3. find unique labels for this GPU

  rmm::device_uvector<vertex_t> unique_labels(graph_view.get_number_of_local_vertices(),
                                              handle.get_stream());
  thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               labels,
               labels + unique_labels.size(),
               unique_labels.begin());
  thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               unique_labels.begin(),
               unique_labels.end());
  unique_labels.resize(
    thrust::distance(unique_labels.begin(),
                     thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    unique_labels.begin(),
                                    unique_labels.end())),
    handle.get_stream());

  rmm::device_uvector<vertex_t> rx_unique_labels(0, handle.get_stream());
  std::tie(rx_unique_labels, std::ignore) = sort_and_shuffle_values(
    handle.get_comms(),
    unique_labels.begin(),
    unique_labels.end(),
    [key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm.get_size()}] __device__(
      auto val) { return key_func(val); },
    handle.get_stream());

  unique_labels = std::move(rx_unique_labels);

  thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               unique_labels.begin(),
               unique_labels.end());
  unique_labels.resize(
    thrust::distance(unique_labels.begin(),
                     thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    unique_labels.begin(),
                                    unique_labels.end())),
    handle.get_stream());

  // 4. renumber

  rmm::device_uvector<vertex_t> renumber_map_labels(0, handle.get_stream());
  partition_t<vertex_t> partition(
    std::vector<vertex_t>{}, graph_view.is_hypergraph_partitioned(), 0, 0, 0, 0);
  vertex_t number_of_vertices{};
  edge_t number_of_edges{};
  std::tie(renumber_map_labels, partition, number_of_vertices, number_of_edges) =
    renumber_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                   unique_labels,
                                                   coarsened_edgelist_major_vertices,
                                                   coarsened_edgelist_minor_vertices,
                                                   graph_view.is_hypergraph_partitioned());

  // 5. build a graph

  std::vector<edgelist_t<vertex_t, edge_t, weight_t>> edgelists{};
  if (graph_view.is_hypergraph_partitioned()) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    edgelists.resize(1);
    edgelists[0].p_src_vertices  = store_transposed ? coarsened_edgelist_minor_vertices.data()
                                                    : coarsened_edgelist_major_vertices.data();
    edgelists[0].p_dst_vertices  = store_transposed ? coarsened_edgelist_major_vertices.data()
                                                    : coarsened_edgelist_minor_vertices.data();
    edgelists[0].p_edge_weights  = coarsened_edgelist_weights.data();
    edgelists[0].number_of_edges = static_cast<edge_t>(coarsened_edgelist_major_vertices.size());
  }

  return std::make_tuple(
    std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
      handle,
      edgelists,
      partition,
      number_of_vertices,
      number_of_edges,
      graph_properties_t{graph_view.is_symmetric(), false},
      true),
    std::move(renumber_map_labels));
}

// single-GPU version
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
             rmm::device_uvector<vertex_t>>>
coarsen_graph(
  raft::handle_t const &handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const &graph_view,
  vertex_t const *labels)
{
  rmm::device_uvector<vertex_t> coarsened_edgelist_major_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> coarsened_edgelist_minor_vertices(0, handle.get_stream());
  rmm::device_uvector<weight_t> coarsened_edgelist_weights(0, handle.get_stream());
  std::tie(coarsened_edgelist_major_vertices,
           coarsened_edgelist_minor_vertices,
           coarsened_edgelist_weights) =
    compressed_sparse_to_relabeled_and_sorted_and_coarsened_edgelist(
      graph_view.offsets(),
      graph_view.indices(),
      graph_view.weights(),
      labels,
      labels,
      vertex_t{0},
      graph_view.get_number_of_vertices(),
      vertex_t{0},
      graph_view.get_number_of_vertices(),
      handle.get_stream());

  rmm::device_uvector<vertex_t> unique_labels(graph_view.get_number_of_vertices(),
                                              handle.get_stream());
  thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               labels,
               labels + unique_labels.size(),
               unique_labels.begin());
  thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               unique_labels.begin(),
               unique_labels.end());
  unique_labels.resize(
    thrust::distance(unique_labels.begin(),
                     thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    unique_labels.begin(),
                                    unique_labels.end())),
    handle.get_stream());

  auto renumber_map_labels = renumber_edgelist<vertex_t, edge_t, multi_gpu>(
    handle, unique_labels, coarsened_edgelist_major_vertices, coarsened_edgelist_minor_vertices);

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{};
  edgelist.p_src_vertices  = store_transposed ? coarsened_edgelist_minor_vertices.data()
                                              : coarsened_edgelist_major_vertices.data();
  edgelist.p_dst_vertices  = store_transposed ? coarsened_edgelist_major_vertices.data()
                                              : coarsened_edgelist_minor_vertices.data();
  edgelist.p_edge_weights  = coarsened_edgelist_weights.data();
  edgelist.number_of_edges = static_cast<edge_t>(coarsened_edgelist_major_vertices.size());

  return std::make_tuple(
    std::make_unique<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>(
      handle,
      edgelist,
      static_cast<vertex_t>(renumber_map_labels.size()),
      graph_properties_t{graph_view.is_symmetric(), false},
      true),
    std::move(renumber_map_labels));
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
           rmm::device_uvector<vertex_t>>
coarsen_graph(
  raft::handle_t const &handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const &graph_view,
  vertex_t const *labels)
{
  return detail::coarsen_graph(handle, graph_view, labels);
}

// explicit instantiation

template std::tuple<std::unique_ptr<graph_t<int32_t, int32_t, float, true, true>>,
                    rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const &handle,
              graph_view_t<int32_t, int32_t, float, true, true> const &graph_view,
              int32_t const *labels);

template std::tuple<std::unique_ptr<graph_t<int32_t, int32_t, float, false, true>>,
                    rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const &handle,
              graph_view_t<int32_t, int32_t, float, false, true> const &graph_view,
              int32_t const *labels);

template std::tuple<std::unique_ptr<graph_t<int32_t, int32_t, float, true, false>>,
                    rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const &handle,
              graph_view_t<int32_t, int32_t, float, true, false> const &graph_view,
              int32_t const *labels);

template std::tuple<std::unique_ptr<graph_t<int32_t, int32_t, float, false, false>>,
                    rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const &handle,
              graph_view_t<int32_t, int32_t, float, false, false> const &graph_view,
              int32_t const *labels);

}  // namespace experimental
}  // namespace cugraph

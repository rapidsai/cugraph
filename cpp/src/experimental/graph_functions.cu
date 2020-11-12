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
#include <experimental/graph_view.hpp>
#include <utilities/comm_utils.cuh>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cuco/detail/hash_functions.cuh>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace experimental {

namespace {

template <typename vertex_t>
rmm::device_uvector<vertex_t> find_unique_labels(vertex_t const *labels,
                                                 vertex_t num_labels,
                                                 cudaStream_t stream)
{
  rmm::device_uvector<vertex_t> unique_labels(num_labels);
  thrust::copy(
    rmm::exec_policy(stream)->on(stream), labels, labels + num_labels, unique_labels.data());
  thrust::sort(rmm::exec_policy(stream)->on(stream), unique_labels.begin(), unique_labels.end());
  auto it = thrust::unique(
    rmm::exec_policy(stream)->on(stream), unique_labels.begin(), unique_labels.end());
  unique_labels.resize(thrust::distance(unique_labels.begin(), it));
  unique_labels.shrink_to_fit();

  return std::move(unique_labels);
}

template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
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
  rmm::device_uvector<vertex_t> edgelist_src_vertices(number_of_edges, stream);
  rmm::device_uvector<vertex_t> edgelist_dst_vertices(number_of_edges, stream);
  rmm::device_uvector<weight_t> edgelist_weights(number_of_edges, stream);

  auto p_majors = store_transposed ? edgelist_dst_vertices.data() : edgelist_src_vertices.data();
  auto p_minors = store_transposed ? edgelist_src_vertices.data() : edgelist_dst_vertices.data();

  // FIXME: this is highly inefficient for very high-degree vertices, for better performance, we can
  // fill high-degree vertices using one CUDA block per vertex, mid-degree vertices using one CUDA
  // warp per vertex, and low-degree vertices using one CUDA thread per block
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator(major_first),
                   thrust::make_counting_iterator(major_last),
                   [compressed_sparse_offsets, major_first, p_majors] __device__(auto v) {
                     auto first = compressed_sparse_offsets[v - major_first];
                     auto last  = compressed_sparse_offsets[v - major_first + 1];
                     thrust::fill(thrust::seq, p_majors + first, p_majors + last, v);
                   });
  thrust::copy(rmm::exec_policy(stream)->on(stream),
               compressed_sparse_indices,
               compressed_sparse_indices + number_of_edges,
               p_minors);
  thrust::copy(rmm::exec_policy(stream)->on(stream),
               compressed_sparse_weights,
               compressed_sparse_weights + number_of_edges,
               edgelist_weights.data());

  return std::make_tuple(std::move(edgelist_src_vertices),
                         std::move(edgelist_dst_vertices),
                         std::move(edgelist_weights));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  compute_coarsened_edgelist(
    raft::handle_t const &handle,
    graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const &graph_view,
    vertex_t const *labels)
{
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

  // FIXME: we may compare performance/memory footprint with the hash_based approach especially when
  // cuco::dynamic_map becomes available (so we don't need to preallocate memory assuming the worst
  // case). We may be able to limit the memory requirement close to the final coarsened edgelist
  // with the hash based approach.
  rmm::device_uvector<vertex_t> coarsened_edgelist_src_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> coarsened_edgelist_dst_vertices(0, handle.get_stream());
  rmm::device_uvector<weight_t> coarsened_edgelist_weights(0, handle.get_stream());
  for (size_t i = 0; i < graph_view.adj_matrix_partition_offsets_.size(); ++i) {
    rmm::device_uvector<vertex_t> edgelist_src_vertices(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dst_vertices(0, handle.get_stream());
    rmm::device_uvector<weight_t> edgelist_weights(0, handle.get_stream());
    std::tie(edgelist_src_vertices, edgelist_dst_vertices, edgelist_weights) =
      compressed_sparse_to_edgelist(
        handle,
        graph_view.adj_matrix_partition_offsets[i],
        graph_view.adj_matrix_partition_indices[i],
        graph_view.adj_matrix_partition_weights[i],
        store_transposed ? graph_view.get_local_adj_matrix_partition_col_first()
                         : graph_view.get_local_adj_matrix_partition_row_first(),
        store_transposed ? graph_view.get_local_adj_matrix_partition_col_last()
                         : graph_view.get_local_adj_matrix_partition_row_last());
    auto src_dst_pair_first =
      thrust::make_zip_iterator(edgelist_src_vertices.begin(), edgelist_dst_vertices.begin());
    thrust::transform(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      src_dst_pair_first,
      src_dst_pair_first + edgelist_src_vertices.size(),
      src_dst_pair_first,
      [p_adj_matrix_major_labels =
         adj_matrix_major_labels.data() +
         (store_transposed ? graph_view.get_local_adj_matrix_partition_col_value_start_offset(i)
                           : graph_view.get_local_adj_matrix_partition_row_value_start_offset(i)),
       p_adj_matrix_minor_labels = adj_matrix_minor_labels.data(),
       src_first                 = graph_view.get_local_adj_matrix_partition_row_first(),
       dst_first = graph_view.get_local_adj_matrix_partition_col_first()] __device__(auto val) {
        auto src = thrust::get<0>(val);
        auto dst = thrust::get<1>(val);
        return store_transposed ? thrust::make_tuple(p_adj_matrix_minor_labels[src - src_first],
                                                     p_adj_matrix_major_labels[dst - dst_first])
                                : thrust::make_tuple(p_adj_matrix_major_labels[src - src_first],
                                                     p_adj_matrix_minor_labels[dst - dst_first]);
      });
    thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        src_dst_pair_first,
                        src_dst_pair_first + edgelist_src_vertices.size(),
                        edgelist_weights);
    if (coarsened_edgelist_src_vertices.size() > 0) {
      rmm::device_uvector<vertex_t> tmp_src_vertices(
        coarsened_edgelist_src_vertices.size() + edgelist_src_vertices.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> tmp_dst_vertices(tmp_src_vertices.size(), handle.get_stream());
      rmm::device_uvector<weight_t> tmp_weights(tmp_src_vertices.size(), handle.get_stream());
      auto coarsened_src_dst_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        coarsened_edgelist_src_vertices.begin(), coarsened_edgelist_dst_vertices.begin()));
      thrust::merge_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                           coarsened_src_dst_pair_first,
                           coarsened_src_dst_pair_first + coarsened_edgelist_src_vertices.size(),
                           src_dst_pair_first,
                           src_dst_pair_first + edgelist_src_vertices.size(),
                           coarsened_edgelist_weights.begin(),
                           edgelist_weights.begin(),
                           thrust::make_zip_iterator(
                             thrust::make_tuple(tmp_src_vertices.begin(), tmp_dst_vertices.begin()),
                             tmp_weights.begin()));
      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // this is necessary as memory blocks in edge_list_(src_vertices,
                                // dst_vertices, weights) will be freed after the following move
                                // assignments.
      edgelist_src_vertices = std::move(tmp_src_vertices);
      edgelist_dst_vertices = std::move(tmp_dst_vertices);
      edgelist_weights      = std::move(tmp_weights);
      src_dst_pair_first =
        thrust::make_zip_iterator(edgelist_src_vertices.begin(), edgelist_dst_vertices.begin());
    }

    coarsened_edgelist_src_vertices.resize(edgelist_src_vertices.size(), handle.get_stream());
    coarsened_edgelist_dst_vertices.resize(coarsened_edgelist_src_vertices.size(),
                                           handle.get_stream());
    coarsened_edgelist_weights.resize(coarsened_edgelist_src_vertices.size(), handle.get_stream());
    auto it = thrust::reduce_by_key(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      src_dst_pair_first,
      src_dst_pair_first + edgelist_src_vertices.size(),
      edgelist_weights,
      thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_src_vertices.begin(),
                                                   coarsened_edgelist_dst_vertices.begin())),
      coarsened_edgelist_weights.begin());
    coarsened_edgelist_src_vertices.resize(thrust::distance(src_dst_pair_first, it),
                                           handle.get_stream());
    coarsened_edgelist_dst_vertices.resize(coarsened_edgelist_src_vertices.size(),
                                           handle.get_stream());
    coarsened_edgelist_weights.resize(coarsened_edgelist_src_vertices.size(), handle.get_stream());

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // this is necessary as edge_list_(src_vertices, dst_vertices,
                              // weights) will become out-of-scope.
  }

  coarsened_edgelist_src_vertices.shrink_to_fit(handle.get_stream());
  coarsened_edgelist_dst_vertices.shrink_to_fit(handle.get_stream());
  coarsened_edgelist_weights.shrink_to_fit(handle.get_stream());
  return std::make_tuple(std::move(coarsened_edgelist_src_vertices),
                         std::move(coarsened_edgelist_dst_vertices),
                         std::move(coarsened_edgelist_weights));
}

template <typename TxValueIterator>
auto shuffle_values(raft::handle_t const &handle,
                    TxValueIterator tx_value_first,
                    rmm::device_uvector<size_t> const &tx_value_counts)
{
  auto &comm = handle.get_comms();

  rmm::device_uvector<size_t> rx_value_counts(comm.get_size(), handle.get_stream());

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released.
  std::vector<size_t> tx_counts(comm.get_size(), size_t{1});
  std::vector<size_t> tx_offsets(comm.get_size());
  std::iota(tx_offsets.begin(), tx_offsets.end(), size_t{0});
  std::vector<int> tx_dst_ranks(comm.get_size());
  std::iota(tx_dst_ranks.begin(), tx_dst_ranks.end(), int{0});
  std::vector<size_t> rx_counts(comm.get_size(), size_t{1});
  std::vector<size_t> rx_offsets(comm.get_size());
  std::iota(rx_offsets.begin(), rx_offsets.end(), size_t{0});
  std::vector<int> rx_src_ranks(comm.get_size());
  std::iota(rx_src_ranks.begin(), rx_src_ranks.end(), int{0});
  device_multicast_sendrecv(comm,
                            tx_value_counts.data(),
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            rx_value_counts.data(),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            handle.get_stream());

  raft::update_host(tx_counts.data(), tx_value_counts.data(), comm.get_size(), handle.get_stream());
  std::partial_sum(tx_counts.begin(), tx_counts.end() - 1, tx_offsets.begin() + 1);
  raft::update_host(rx_counts.data(), rx_value_counts.data(), comm.get_size(), handle.get_stream());
  std::partial_sum(rx_counts.begin(), rx_counts.end() - 1, rx_offsets.begin() + 1);

  auto rx_value_buffer =
    allocate_comm_buffer<typename std::iterator_traits<TxValueIterator>::value_type>(
      rx_offsets.back(), handle.get_stream());
  auto rx_value_first =
    get_comm_buffer_begin<typename std::iterator_traits<TxValueIterator>::value_type>(
      rx_value_buffer);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released.
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            rx_value_first,
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            handle.get_stream());

  return std::move(rx_value_buffer);
}

}  // namespace

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
  // 1. locally construct coarsened edge list

  rmm::device_uvector<vertex_t> coarsened_edgelist_src_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> coarsened_edgelist_dst_vertices(0, handle.get_stream());
  rmm::device_uvector<weight_t> coarsened_edgelist_weights(0, handle.get_stream());
  std::tie(
    coarsened_edgelist_src_vertices, coarsened_edgelist_dst_vertices, coarsened_edgelist_weights) =
    compute_coarsened_edgelist(handle, graph_view, labels);

  // 2. globally shuffle edge list

  {
    auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_src_vertices.begin(),
                                                   coarsened_edgelist_dst_vertices.begin(),
                                                   coarsened_edgelist_weights.begin()));
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 edge_first,
                 edge_first + coarsened_edgelist_src_vertices.size(),
                 [key_func = detail::compute_gpu_id_from_edge_t<vertex_t, store_transposed>{
                    graph_view.is_hypergraph_partitioned(),
                    handle.get_comms().get_size(),
                    row_comm.get_size(),
                    col_comm.get_size()}] __device__(auto lhs, auto rhs) {
                   return key_func(thrust::get<0>(lhs), thrust::get<1>(lhs)) <
                          key_func(thrust::get<0>(rhs), thrust::get<1>(rhs));
                 });

    auto key_first = thrust::make_transform_iterator(
      edge_first,
      [key_func = detail::compute_gpu_id_from_edge_t<vertex_t, store_transposed>{
        graph_view.is_hypergraph_partitioned(),
        handle.get_comms().get_size(),
        row_comm.get_size(),
        col_comm.get_size()}] __device__(auto val) { return key_func(val); });
    rmm::device_uvector<vertex_t> tx_value_counts(comm.get_size(), handle.get_stream());
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          edge_first,
                          edge_first + coarsened_edgelist_src_vertices.size(),
                          thrust::make_constant_iterator(vertex_t{1}),
                          thrust::make_discard_iterator(),
                          tx_value_counts.begin());

    auto = shuffle_values(edge_first, tx_value_counts);

    std::tie(coarsened_edgelist_src_vertices,
             coarsened_edgelist_dst_vertices,
             coarsened_edgelist_weights) = compute_coarsened_edgelist(handle, graph_view, labels);
  }

  // 3. find unique labels assigned to each GPU

  rmm::device_uvector<vertex_t> unique_labels(0, handle.get_stream());
  {
    auto tx_unique_labels =
      find_unique_labels(labels, graph_view.get_number_of_local_vertices(), handle.get_stream());

    auto &comm = handle.get_comms();
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 tx_unique_labels.begin(),
                 tx_unique_labels.end(),
                 [hash_func = cuco::detail::MurmurHash3_32<vertex_t>(),
                  comm_size = comm.get_size()] __device__(auto lhs, auto rhs) {
                   return (hash_func(lhs) % comm_size) < (hash_func(rhs) % comm_size);
                 });
    auto key_first = thrust::make_transform_iterator(
      tx_unique_label_keys.begin(),
      [hash_func = cuco::detail::MurmurHash3_32<vertex_t>(),
       comm_size = comm.get_size()] __device__(auto label) { return hash(label) % comm_size; });
    rmm::device_uvector<vertex_t> tx_num_unique_labels(comm.get_size(), handle.get_stream());
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          key_first,
                          key_first + tx_unique_labels.size(),
                          thrust::make_constant_iterator(vertex_t{1}),
                          thrust::make_discard_iterator(),
                          tx_num_unique_labels.begin());
    
    auto rx_unique_labels = shuffle_values(tx_unique_labels, tx_num_unique_labels);

    unique_labels =
      find_unique_labels(rx_unique_labels.data(), rx_unique_labels.size(), handle.get_stream());

    // FIXME: should I cudaStreamSynchronize()?
  }

  // 4. acquire unique labels for the major range

  // 5. locally compute (label, count) pairs and globally reduce

  // 6. sort (label, count) pairs and compute label to vertex ID map

  // 7. acquire (label, vertex ID) pairs for the major & minor ranges.

  // 8. renumber edgelists.

  // 9. create a coarsened graph.
}

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
  CUGRAPH_FAIL("unimplemented.");
}

// explicit instantiation

}  // namespace experimental
}  // namespace cugraph

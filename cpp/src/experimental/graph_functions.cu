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
#include <patterns/copy_to_adj_matrix_row_col.cuh>
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
#include <cuco/static_map.cuh>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace experimental {

namespace {

// FIXME: better move this elsewhere for reusability
template <typename TxValueIterator>
auto shuffle_values(raft::handle_t const &handle,
                    TxValueIterator tx_value_first,
                    rmm::device_uvector<size_t> const &tx_value_counts)
{
  auto &comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> rx_value_counts(comm_size(), handle.get_stream());

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released.
  std::vector<size_t> tx_counts(comm_size(), size_t{1});
  std::vector<size_t> tx_offsets(comm_size());
  std::iota(tx_offsets.begin(), tx_offsets.end(), size_t{0});
  std::vector<int> tx_dst_ranks(comm_size());
  std::iota(tx_dst_ranks.begin(), tx_dst_ranks.end(), int{0});
  std::vector<size_t> rx_counts(comm_size(), size_t{1});
  std::vector<size_t> rx_offsets(comm_size);
  std::iota(rx_offsets.begin(), rx_offsets.end(), size_t{0});
  std::vector<int> rx_src_ranks(comm_size());
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

  raft::update_host(tx_counts.data(), tx_value_counts.data(), comm_size(), handle.get_stream());
  std::partial_sum(tx_counts.begin(), tx_counts.end() - 1, tx_offsets.begin() + 1);
  raft::update_host(rx_counts.data(), rx_value_counts.data(), comm_size(), handle.get_stream());
  std::partial_sum(rx_counts.begin(), rx_counts.end() - 1, rx_offsets.begin() + 1);

  auto rx_value_buffer =
    allocate_comm_buffer<typename std::iterator_traits<TxValueIterator>::value_type>(
      rx_offsets.back(), handle.get_stream());
  auto rx_value_first =
    get_comm_buffer_begin<typename std::iterator_traits<TxValueIterator>::value_type>(
      rx_value_buffer);

  int num_tx_dst_ranks{0};
  int num_rx_src_ranks{0};
  for (int i = 0; i < comm_size; ++i) {
    if (tx_counts[i] != 0) {
      tx_counts[num_tx_dst_ranks]    = tx_counts[i];
      tx_offsets[num_tx_dst_ranks]   = tx_offsets[i];
      tx_dst_ranks[num_tx_dst_ranks] = tx_dst_ranks[i];
      ++num_tx_dst_ranks;
    }
    if (rx_counts[i] != 0) {
      rx_counts[num_rx_src_ranks]    = rx_counts[i];
      rx_offsets[num_rx_src_ranks]   = rx_offsets[i];
      rx_src_ranks[num_rx_src_ranks] = rx_src_ranks[i];
    }
  }
  tx_counts.resize(num_tx_dst_ranks);
  tx_offsets.resize(num_tx_dst_ranks);
  tx_dst_ranks.resize(num_tx_dst_ranks);
  rx_counts.resize(num_rx_dst_ranks);
  rx_offsets.resize(num_rx_dst_ranks);
  rx_src_ranks.resize(num_rx_src_ranks);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
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

  return std::make_tuple(std::move(edgelist_src_vertices),
                         std::move(edgelist_dst_vertices),
                         std::move(edgelist_weights));
}

template <typename vertex_t, typename weight_t>
void sort_and_coarsen_edgelist(raft::handle_t const &handle,
                               rmm::device_uvector<vertex_t> &edgelist_major_vertices /* [INOUT] */,
                               rmm::device_uvector<vertex_t> &edgelist_minor_vertices /* [INOUT] */,
                               rmm::device_uvector<weight_t> &edgelist_weights /* [INOUT] */)
{
  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));

  size_t number_of_edges{0};
  if (edgelist_weights.size() > 0) {
    thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        pair_first,
                        pair_first + edgelist_major_vertices.begin(),
                        edgelist_weights.begin());

    rmm::device_uvector<vertex_t> tmp_edgelist_major_vertices(edgelist_major_vertices.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_edgelist_minor_vertices(tmp_edgelist_major_vertices.size(), handle.get_stream());
    rmm::device_uvector<weight_t> tmp_edgelist_weights(tmp_edgelist_major_vertices.size(), handle.get_stream());
    auto it = thrust::reduce_by_key(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      pair_first,
      pair_first + edgelist_major_vertices.begin(),
      edgelist_weights.begin(),
      thrust::make_zip_iterator(thrust::make_tuple(tmp_edgelist_major_vertices.begin(),
                                                   tmp_edgeilst_minor_vertices.begin())),
      tmp_edgelist_weights.begin());
    number_of_edges = thrust::distance(tmp_edgelist_weights.begin(), thrust::get<1>(it));

    CUDA_TRY(cudaStreamSynchronize(
      handle
        .get_stream()));  // memory blocks owned by edgelist_(major_vertices,minor_vertices,weights)
                          // will be freed after the assignments below

    edgelist_major_vertices = std::move(tmp_edgelist_major_vertices);
    edgelist_minor_vertices = std::move(tmp_edgelist_minor_vertices);
    edgelist_weights        = std::move(tmp_edgelist_weights);
  } else {
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 pair_first,
                 pair_first + edgelist_major_vertices.begin());
    auto it         = thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                             pair_first,
                             pair_first + edgelist_major_vertices.size());
    number_of_edges = thrust::distance(pair_first, it);
  }

  edgelist_major_vertices.resize(number_of_edges, handle.get_stream());
  edgelist_minor_vertices.resize(number_of_edges, handle.get_stream());
  edgelist_weights.resize(number_of_edges, handle.get_stream());
  edgelist_major_vertices.shrink_to_fit(handle.get_stream());
  edgelist_minor_vertices.shrink_to_fit(handle.get_stream());
  edgelist_weights.shrink_to_fit(handle.get_stream());

  return;
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
    vertex_t minor_first,
    cudaStream_t stream)
{
  // FIXME: it might be possible to directly create relabled & coarsened edgelist from the
  // compressed sparse format to save memory

  rmm::device_uvector<vertex_t> edgelist_major_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minor_vertices(0, handle.get_stream());
  rmm::device_uvector<weight_t> edgelist_weights(0, handle.get_stream());
  std::tie(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights) =
    compressed_sparse_to_edgelist(handle,
                                  compressed_sparse_offsets,
                                  compressed_sparse_indices,
                                  compressed_sparse_weights,
                                  major_first,
                                  major_last,
                                  stream);

  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));
  thrust::transform(
    rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
    pair_first,
    pair_first + edgelist_major_vertices.size(),
    pair_first,
    [p_major_labels, p_minor_labels, major_first, minor_first] __device__(auto val) {
      return thrust::make_tuple(p_major_labels[thrust::get<0>(val) - major_first],
                                p_minor_labels[thrust::get<1>(val) - minor_first]);
    });

  sort_and_coarsen_edgelist(
    handle, edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights);

  return std::make_tuple(std::move(edgelist_major_vertices),
                         std::move(edgelist_minor_vertices),
                         std::move(edgelist_weights));
}

#if 0
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
void compute_coarsened_edgelist(raft::handle_t const &handle,
                                rmm::device_uvector<vertex_t> &coarsened_edgelist_major_vertices,
                                rmm::device_uvector<vertex_t> &coarsened_edgelist_minor_vertices,
                                rmm::device_uvector<weight_t> &coarsened_edgelist_weights,
                                edge_t const *uncoarsened_edgelist_major_vertices,
                                vertex_t const *uncoarsened_edgelist_minor_vertices,
                                weight_t const *uncoarsened_edgelist_weights,
                                edge_t number_of_uncoarsened_edges,
                                vertex_t uncoarsened_major_first,
                                vertex_t uncoarsened_minor_first,
                                vertex_t major_labels,
                                vertex_t minor_labels)
{
  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(uncoarsened_edgelist_major_vertices, uncoarsened_edgelist_minor_vertices));
  if (uncoarsened_edgelist_eights != nullptr) {
    thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        pair_first,
                        pair_first + number_of_uncoarsened_edges,
                        uncoarsened_edgelist_weights);
  } else {
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 pair_first,
                 pair_first + number_of_uncoarsened_edges);
  }
  if (coarsened_edgelist_src_vertices.size() > 0) {
    rmm::device_uvector<vertex_t> tmp_src_vertices(
      coarsened_edgelist_src_vertices.size() + number_of_uncoarsened_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_dst_vertices(tmp_src_vertices.size(), handle.get_stream());
    rmm::device_uvector<weight_t> tmp_weights(
      graph_view.is_weighted() ? tmp_src_vertices.size() : 0, handle.get_stream());
    auto coarsened_src_dst_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      coarsened_edgelist_src_vertices.begin(), coarsened_edgelist_dst_vertices.begin()));
    if (graph_view.is_weighted()) {
      thrust::merge_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                           coarsened_src_dst_pair_first,
                           coarsened_src_dst_pair_first + coarsened_edgelist_src_vertices.size(),
                           src_dst_pair_first,
                           src_dst_pair_first + edgelist_src_vertices.size(),
                           coarsened_edgelist_weights.begin(),
                           edgelist_weights.begin(),
                           thrust::make_zip_iterator(thrust::make_tuple(tmp_src_vertices.begin(),
                                                                        tmp_dst_vertices.begin())),
                           tmp_weights.begin());
    } else {
      thrust::merge(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    coarsened_src_dst_pair_first,
                    coarsened_src_dst_pair_first + coarsened_edgelist_src_vertices.size(),
                    src_dst_pair_first,
                    src_dst_pair_first + edgelist_src_vertices.size(),
                    thrust::make_zip_iterator(
                      thrust::make_tuple(tmp_src_vertices.begin(), tmp_dst_vertices.begin())));
    }
    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // this is necessary as memory blocks in edgelist_(src_vertices,
                              // dst_vertices, weights) will be freed after the following move
                              // assignments.
    edgelist_src_vertices = std::move(tmp_src_vertices);
    edgelist_dst_vertices = std::move(tmp_dst_vertices);
    edgelist_weights      = std::move(tmp_weights);
    src_dst_pair_first    = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_src_vertices.begin(), edgelist_dst_vertices.begin()));
  }

  edge_t num_edges{0};
  if (graph_view.is_weighted()) {
    coarsened_edgelist_src_vertices.resize(edgelist_src_vertices.size(), handle.get_stream());
    coarsened_edgelist_dst_vertices.resize(coarsened_edgelist_src_vertices.size(),
                                           handle.get_stream());
    coarsened_edgelist_weights.resize(
      graph_view.is_weighted() ? coarsened_edgelist_src_vertices.size() : 0, handle.get_stream());
    auto it = thrust::reduce_by_key(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      src_dst_pair_first,
      src_dst_pair_first + edgelist_src_vertices.size(),
      edgelist_weights.begin(),
      thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_src_vertices.begin(),
                                                   coarsened_edgelist_dst_vertices.begin())),
      coarsened_edgelist_weights.begin());
    num_edges = static_cast<edge_t>(thrust::distance(src_dst_pair_first, it));
  } else {
    auto it = thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                             src_dst_pair_first,
                             src_dst_pair_first + edgelist_src_vertices.size());
    coarsened_edgelist_src_vertices = std::move(edgelist_src_vertices);
    coarsened_edgelist_dst_vertices = std::move(edgelist_dst_vertices);
    coarsened_edgelist_weights      = std::move(edgelist_weights);
    num_edges                       = static_cast<edge_t>(thrust::distance(src_dst_pair_first, it));
  }
  coarsened_edgelist_src_vertices.resize(num_edges, handle.get_stream());
  coarsened_edgelist_dst_vertices.resize(num_edges, handle.get_stream());
  coarsened_edgelist_weights.resize(graph_view.is_weighted() ? num_edges : 0, handle.get_stream());

  CUDA_TRY(cudaStreamSynchronize(
    handle.get_stream()));  // this is necessary as edgelist_(src_vertices, dst_vertices,
                            // weights) will become out-of-scope.

  coarsened_edgelist_src_vertices.shrink_to_fit(handle.get_stream());
  coarsened_edgelist_dst_vertices.shrink_to_fit(handle.get_stream());
  coarsened_edgelist_weights.shrink_to_fit(handle.get_stream());
  return std::make_tuple(std::move(coarsened_edgelist_src_vertices),
                         std::move(coarsened_edgelist_dst_vertices),
                         std::move(coarsened_edgelist_weights));
}
#endif

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_renumber_map(
  raft::handle_t const &handle,
  rmm::device_uvector<vertex_t> const &edgelist_major_vertices,
  rmm::device_uvector<vertex_t> const &edgelist_minor_vertices)
{
  // FIXME: compare this sort based approach with hash based approach in both speed and memory
  // footprint

  // 1. acquire (unique major label, count) pairs

  rmm::device_uvector<vertex_t> tmp_labels = edgelist_major_vertices;
  thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               tmp_labels.begin(),
               tmp_labels.end());
  rmm::device_uvector<vertex_t> major_labels(tmp_labels.size(), handle.get_stream());
  rmm::device_uvector<edge_t> major_counts(major_labels.size(), handle.get_stream());
  auto major_pair_it =
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          tmp_labels.begin(),
                          tmp_labels.end(),
                          thrust::make_constant_iterator(edge_t{1}),
                          major_labels.begin(),
                          major_counts.begin());
  tmp_labels.resize(0, handle.get_stream());
  tmp_labels.shrink_to_fit(handle.get_stream());
  major_labels.resize(thrust::distance(major_labels.begin(), thrust::get<0>(major_pair_it)),
                      handle.get_stream());
  major_counts.resize(major_labels.size(), handle.get_stream());
  major_labels.shrink_to_fit(handle.get_stream());
  major_counts.shrink_to_fit(handle.get_stream());

  // 2. acquire unique minor labels

  rmm::device_uvector<vertex_t> minor_labels = edgelist_minor_vertices;
  thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               minor_labels.begin(),
               minor_labels.end());
  auto minor_label_it =
    thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   minor_labels.begin(),
                   minor_labels.end());
  minor_labels.resize(thrust::distance(minor_labels.begin(), minor_label_it));
  minor_labels.shrink_to_fit(handle.get_stream());

  // 3. merge major and minor labels

  rmm::device_uvector<vertex_t> merged_labels(major_labels.size() + minor_labels.size(),
                                              handle.get_stream());
  rmm::device_uvector<edge_t> merged_counts(merged_labels.size(), handle.get_stream());
  thrust::merge_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       major_labels.begin(),
                       major_labels.end(),
                       minor_labels.begin(),
                       minor_labels.end(),
                       major_counts.begin(),
                       thrust::make_constant_iterator(edge_t{0}),
                       merged_labels.begin(),
                       merged_counts.begin());
  major_labels.resize(0, handle.get_stream());
  major_counts.resize(0, handle.get_stream());
  minor_labels.resize(0, handle.get_stream());
  major_labels.shrink_to_fit(handle.get_stream());
  major_counts.shrink_to_fit(handle.get_stream());
  minor_labels.shrink_to_fit(handle.get_stream());
  rmm::device_uvector<vertex_t> labels(merged_labels.size(), handle.get_stream());
  rmm::device_uvector<edge_t> counts(labels.size(), handle.get_stream());
  auto pair_it =
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          merged_labels.begin(),
                          merged_labels.end(),
                          merged_counts.begin(),
                          labels.begin(),
                          counts.begin());
  merged_labels.resize(0, handle.get_stream());
  merged_counts.resize(0, handle.get_stream());
  merged_labels.shrink_to_fit();
  merged_counts.shrink_to_fit();
  labels.resize(thrust::distance(labels.begin(), thrust::get<0>(pair_it)), handle.get_stream());
  counts.resize(labels.size(), handle.get_stream());
  labels.shrink_to_fit(handle.get_stream());
  counts.shrink_to_fit(handle.get_stream());

  // 4. if multi-GPU, shuffle and reduce (label, count) pairs

  if (multi_gpu) {
    auto &comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(labels.begin(), counts.begin()));
    auto key_func   = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size};
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 pair_first,
                 pair_first + labels.size(),
                 [key_func] __device__(auto lhs, auto rhs) {
                   return key_func(thrust::get<0>(lhs)) < key_func(thrust::get<0>(rhs));
                 });
    auto key_first = thrust::make_transform_iterator(
      labels.begin(), [key_func] __device__(auto val) { return key_func(val); });
    rmm::device_uvector<size_t> tx_value_counts(comm_size, handle.get_stream());
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          key_first,
                          key_first + labels.size(),
                          thrust::make_constant_iterator(size_t{1}),
                          thrust::make_discard_iterator(),
                          tx_value_counts.begin());

    rmm::device_uvector<vertex_t> rx_labels(0, handle.get_stream());
    rmm::device_uvector<edge_t> rx_counts(0, handle.get_stream());
    std::tie(rx_labels, rx_counts) = shuffle_values(handle, pair_first, tx_value_counts);

    labels.resize(rx_labels.size(), handle.get_stream());
    counts.resize(labels.size(), handle.get_stream());
    thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        rx_labels.begin(),
                        rx_labels.end(),
                        rx_counts.begin());
    pair_it = thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    rx_labels.begin(),
                                    rx_labels.end(),
                                    rx_counts.begin(),
                                    labels.begin(),
                                    counts.begin());
    rx_labels.resize(0, handle.get_stream());
    rx_counts.resize(0, handle.get_stream());
    rx_labels.shrink_to_fit(handle.get_stream());
    rx_counts.shrink_to_fit(handle.get_stream());
    labels.resize(thrust::distance(labels.begin(), thrust::get<0>(pair_it)), handle.get_stream());
    counts.resize(labels.size(), handle.get_stream());
    labels.shrink_to_fit(handle.get_stream());
    labels.shrink_to_fit(handle.get_stream());
  }

  // 5. sort by degree

  thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      counts.begin(),
                      counts.end(),
                      labels.begin(),
                      thrust::greater<edge_t>());

  CUDA_TRY(
    cudaStreamSynchronize(handle.get_stream()));  // temporary rmm::devicec_uvector objects become
                                                  // out-of-scope once this function returns.

  return std::move(labels);
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::enable_if_t<multi_gpu, std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>> renumber_edgelist(
  raft::handle_t const &handle,
  rmm::device_uvector<vertex_t> &edgelist_major_vertices /* [INOUT] */,
  rmm::device_uvector<vertex_t> &edgelist_minor_vertices /* [INOUT] */,
  rmm::device_uvector<weight_t> &edgelist_weights /* [INOUT] */)
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

  // 1. compute renumber map

  auto renumber_map_labels = compute_renumber_map<vertex_t, edge_t, multi_gpu>(
    handle, edgelist_major_vertices, edgelist_minor_vertices);

  // 2. initialize partition_t object, number_of_vertices, and number_of_edges for the coarsened
  // graph

  auto vertex_partition_counts = host_scalar_allgather(
    comm, static_cast<vertex_t>(renumber_map_labels.size()), handle.get_stream());
  std::vector<vertex_t> vertex_partition_offsets(comm_size + 1, 0);
  std::partial_sum(vertex_partition_counts.begin(),
                   vertex_partition_counts.end(),
                   vertex_partition_offsets.begin() + 1);

  partition_t<vertex_t> partition(vertex_partition_offsets,
                                  graph_view.is_hypergraph_partitioned(),
                                  row_comm_size,
                                  col_comm_size,
                                  row_comm_rank,
                                  col_comm_rank);

  auto number_of_vertices = vertex_partition_offsets.back();
  auto number_of_edges    = host_scalar_allreduce(
    comm, static_cast<edge_t>(coarsened_edgelist_src_vertices.size()), handle.get_stream());

  // 3. renumber edges

  if (graph_view.is_hypergraph_partitioned()) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    double constexpr load_factor = 0.7;

    // FIXME: compare this hash based approach with a binary search based approach in both memory
    // footprint and execution time

    {
      vertex_t major_first{};
      vertex_t major_last{};
      std::tie(major_first, major_last) = partition.get_matrix_partition_major_range(0);
      rmm::device_uvector<vertex_t> renumber_map_major_labels(major_last - major_first,
                                                              handle.get_stream());
      std::vector<size_t> recvcounts(row_comm_size);
      for (int i = 0; i < row_comm_size; ++i) {
        recvcounts[i] = partition.get_vertex_partition_size(row_comm_rank * row_comm_size + i);
      }
      std::vector<size_t> displacements(row_comm_size, 0);
      std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
      device_allgatherv(row_comm,
                        renumber_map_labels.begin(),
                        renumber_map_major_labels.begin(),
                        recvcounts,
                        displacements,
                        handle.get_stream());

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // cuco::static_map currently does not take stream

      cuco::static_map<vertex_t, vertex_t> renumber_map{
        static_cast<size_t>(static_cast<double>(renumber_map_major_labels.size()) / load_factor),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value};
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        renumber_map_major_labels.begin(), thrust::make_counting_iterator(major_first)));
      renumber_map.insert(pair_first, pair_first + renumber_map_major_labels.size());
      renumber_map.find(edgelist_major_vertices.begin(),
                        edgelist_major_vertices.end(),
                        edgelist_major_vertices.begin());
    }

    {
      vertex_t minor_first{};
      vertex_t minor_last{};
      std::tie(minor_first, minor_last) = partition.get_matrix_partition_minor_range();
      rmm::device_uvector<vertex_t> renumber_map_minor_labels(minor_last - minor_first,
                                                              handle.get_stream());

      // FIXME: this P2P is unnecessary if we apply the partitioning scheme used with hypergraph
      // partitioning
      auto comm_src_rank = row_comm_rank * col_comm_size + col_comm_rank;
      auto comm_dst_rank = (comm_rank % col_comm_size) * row_comm_size + comm_rank / col_comm_size;
      // FIXME: this branch may be no longer necessary with NCCL backend
      if (comm_src_rank == comm_rank) {
        assert(comm_dst_rank == comm_rank);
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     renumber_map_labels.begin(),
                     renumber_map_labels.end(),
                     renumber_map_minor_labels.begin() +
                       (partition.get_vertex_partition_first(comm_src_rank) -
                        partition.get_vertex_partition_first(row_comm_rank * col_comm_size)));
      } else {
        device_sendrecv(comm,
                        renumber_map_labels.begin(),
                        renumber_map_labels.size(),
                        comm_dst_rank,
                        renumber_map_minor_labels.begin() +
                          (partition.get_vertex_partition_first(comm_src_rank) -
                           partition.get_vertex_partition_first(row_comm_rank * col_comm_size)),
                        static_cast<size_t>(partition.get_vertex_partition_size(comm_src_rank)),
                        comm_src_rank,
                        handle.get_stream());
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      for (int i = 0; i < col_comm_size; ++i) {
        auto offset = partition.get_vertex_partition_first(row_comm_rank * col_comm_size + i) -
                      partition.get_vertex_partition_first(row_comm_rank * col_comm_size);
        auto count = partition.get_vertex_partition_size(row_comm_rank * col_comm_size + i);
        device_bcast(col_comm,
                     renumber_map_minor_labels.begin() + offset,
                     renumber_map_minor_labels.begin() + offset,
                     count,
                     i,
                     handle.get_stream());
      }

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // cuco::static_map currently does not take stream

      cuco::static_map<vertex_t, vertex_t> renumber_map{
        static_cast<size_t>(static_cast<double>(renumber_map_minor_labels.size()) / load_factor),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value};
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        renumber_map_minor_labels.begin(), thrust::make_counting_iterator(minor_first)));
      renumber_map.insert(pair_first, pair_first + renumber_map_minor_labels.size());
      renumber_map.find(coarsened_edgelist_minor_vertices.begin(),
                        coarsened_edgelist_minor_vertices.end(),
                        coarsened_edgelist_minor_vertices.begin());
    }
  }

  return std::make_tuple(std::move(renumber_map_labels), partition, number_of_vertices, number_of_edges);
}

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
  for (size_t i = 0; i < graph_view.adj_matrix_partition_offsets_.size(); ++i) {
    // get edge list

    rmm::device_uvector<vertex_t> edgelist_major_vertices(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_minor_vertices(0, handle.get_stream());
    rmm::device_uvector<weight_t> edgelist_weights(0, handle.get_stream());
    std::tie(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights) =
      compressed_sparse_to_relabeled_and_sorted_and_coarsened_edgelist(
        handle,
        graph_view.adj_matrix_partition_offsets_[i],
        graph_view.adj_matrix_partition_indices_[i],
        graph_view.is_weighted() ? graph_view.adj_matrix_partition_weights_[i] : nullptr,
        adj_matrix_major_labels.begin() +
          (store_transposed ? graph_view.get_local_adj_matrix_partition_col_vaule_start_offset(i)
                            : graph_view.get_local_adj_matrix_partition_row_vaule_start_offset(i)),
        adj_matrix_minor_labels.begin(),
        store_transposed ? graph_view.get_local_adj_matrix_partition_col_first(i)
                         : graph_view.get_local_adj_matrix_partition_row_first(i),
        store_transposed ? graph_view.get_local_adj_matrix_partition_row_first(i)
                         : graph_view.get_local_adj_matrix_partition_col_first(i));

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
      auto src_edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin(), edgelist_weights.begin()));
      auto dst_edge_first = thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_major_vertices.begin(),
                                                      coarsened_edgelist_minor_vertices.begin(),
                                                      coarsened_edgelist_weights.begin())) +
                            cur_size;
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   src_edge_first,
                   src_edge_first + edgelist_major_vertices.size(),
                   dst_edge_first);
    } else {
      auto src_edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_major_vertices.begin(), edgelist_minor_vertices.begin()));
      auto dst_edge_first = thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_major_vertices.begin(),
                                                      coarsened_edgelist_minor_vertices.begin())) +
                            cur_size;
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   src_edge_first,
                   src_edge_first + edgelist_major_vertices.size(),
                   dst_edge_first);
    }

    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));  // edgelist_(major_vertices,minor_vertices,weights)
                                                           // will become out-of-scope
  }

  sort_and_coarsen_edgelist(handle,
                            coarsened_edgelist_major_vertices,
                            coarsened_edgelist_minor_vertices,
                            coarsened_edgelist_weights);

  // 2. globally shuffle edge list and re-coarsen

  {
    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(coarsened_edgelist_major_vertices.begin(),
                                                   coarsened_edgelist_minor_vertices.begin(),
                                                   coarsened_edgelist_weights.begin()));
    auto key_func = detail::compute_gpu_id_from_edge_t<vertex_t, store_transposed>{
      graph_view.is_hypergraph_partitioned(),
      comm.get_size(),
      row_comm.get_size(),
      col_comm.get_size()};
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 edge_first,
                 edge_first + coarsened_edgelist_major_vertices.size(),
                 [key_func] __device__(auto lhs, auto rhs) {
                   return store_transposed ? (key_func(thrust::get<1>(lhs), thrust::get<0>(lhs)) <
                                              key_func(thrust::get<1>(rhs), thrust::get<0>(rhs)))
                                           : (key_func(thrust::get<0>(lhs), thrust::get<1>(lhs)) <
                                              key_func(thrust::get<0>(rhs), thrust::get<1>(rhs)));
                 });
    auto key_first = thrust::make_transform_iterator(edge_first, [key_func] __device__(auto val) {
      return store_transposed ? key_func(thrust::get<1>(val), thrust::get<0>(val))
                              : key_func(thrust::get<0>(val), thrust::get<1>(val));
    });
    rmm::device_uvector<vertex_t> tx_value_counts(comm.get_size(), handle.get_stream());
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          key_first,
                          key_first + coarsened_edgelist_major_vertices.size(),
                          thrust::make_constant_iterator(vertex_t{1}),
                          thrust::make_discard_iterator(),
                          tx_value_counts.begin());

    rmm::device_uvector<vertex_t> rx_edgelist_major_vertices(0, handle.get_stream());
    rmm::device_uvector<vertex_t> rx_edgelist_minor_vertices(0, handle.get_stream());
    rmm::device_uvector<weight_t> rx_edgelist_weights(0, handle.get_stream());
    std::tie(rx_edgelist_major_vertices, rx_edgelist_minor_vertices, rx_edgelist_weights) =
      shuffle_values(handle, edge_first, tx_value_counts);

    sort_and_coarsen_edgelist(
      handle, rx_edgelist_major_vertices, rx_edgelist_minor_vertices, rx_edgelist_weights);

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // memory blocks owned by
                              // coarsened_edgelist_(major_vertices,minor_vertices,weights)
                              // will be freed after the assignments below

    coarsened_edgelist_major_vertices = std::move(rx_edgelist_major_vertices);
    coarsened_edgelist_minor_vertices = std::move(rx_edgelist_minor_vertices);
    coarsened_edgelist_weights        = std::move(rx_edgelist_weights);
  }

  rmm::device_uvector<vertex_t> renumber_map_labels{};
  partition_t<vertex_t> partition{};
  vertex_t number_of_vertices{};
  edge_t number_of_edges{};
  std::tie(renumber_map_labels, partition, number_of_vertices, number_of_edges) =
    renumber_edgelist<vertex_t, edge_t, weight_t, multi_gpu>(handle,
                                                             coarsened_edgelist_major_vertices,
                                                             coarsened_edgelist_minor_vertices,
                                                             coarsened_edgelist_weights);

  // 4. build a graph

  std::vector<edgelist_t<vertex_t, edge_t, weight_t>> edgelists{};
  if (graph_view.is_hypergraph_partitioned()) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    edgelists.resize(1);
    edgelists[0].p_src_vertices = store_transposed ? coarsened_edgelist_minor_vertices.data()
                                                   : coarsened_edgelist_major_vertices.data();
    edgelists[0].p_dst_vertices = store_transposed ? coarsened_edgelist_major_vertices.data()
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

template std::tuple<std::unique_ptr<graph_t<int32_t, int32_t, float, false, true>>,
                    rmm::device_uvector<int32_t>>
coarsen_graph(raft::handle_t const &handle,
              graph_view_t<int32_t, int32_t, float, false, true> const &graph_view,
              int32_t const *labels);

}  // namespace experimental
}  // namespace cugraph

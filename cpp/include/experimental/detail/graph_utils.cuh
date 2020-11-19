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
#pragma once

#include <experimental/graph_view.hpp>
#include <partition_manager.hpp>
#include <utilities/comm_utils.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/sort.h>
#include <thrust/transform.h>
#include <cuco/detail/hash_functions.cuh>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {
namespace experimental {
namespace detail {

// compute the numbers of nonzeros in rows (of the graph adjacency matrix, if store_transposed =
// false) or columns (of the graph adjacency matrix, if store_transposed = true)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degree(
  raft::handle_t const &handle,
  std::vector<edge_t const *> const &adj_matrix_partition_offsets,
  partition_t<vertex_t> const &partition)
{
  auto &row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();
  auto &col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();

  rmm::device_uvector<edge_t> local_degrees(0, handle.get_stream());
  rmm::device_uvector<edge_t> degrees(0, handle.get_stream());

  vertex_t max_num_local_degrees{0};
  for (int i = 0; i < (partition.is_hypergraph_partitioned() ? col_comm_size : row_comm_size);
       ++i) {
    auto vertex_partition_idx = partition.is_hypergraph_partitioned()
                                  ? static_cast<size_t>(i * row_comm_size + row_comm_rank)
                                  : static_cast<size_t>(col_comm_rank * row_comm_size + i);
    auto vertex_partition_size = partition.get_vertex_partition_size(vertex_partition_idx);
    max_num_local_degrees      = std::max(max_num_local_degrees, vertex_partition_size);
    if (i == (partition.is_hypergraph_partitioned() ? col_comm_rank : row_comm_rank)) {
      degrees.resize(vertex_partition_size, handle.get_stream());
    }
  }
  local_degrees.resize(max_num_local_degrees, handle.get_stream());
  for (int i = 0; i < (partition.is_hypergraph_partitioned() ? col_comm_size : row_comm_size);
       ++i) {
    auto vertex_partition_idx = partition.is_hypergraph_partitioned()
                                  ? static_cast<size_t>(i * row_comm_size + row_comm_rank)
                                  : static_cast<size_t>(col_comm_rank * row_comm_size + i);
    vertex_t major_first{};
    vertex_t major_last{};
    std::tie(major_first, major_last) = partition.get_vertex_partition_range(vertex_partition_idx);
    auto p_offsets =
      partition.is_hypergraph_partitioned()
        ? adj_matrix_partition_offsets[i]
        : adj_matrix_partition_offsets[0] +
            (major_first - partition.get_vertex_partition_first(col_comm_rank * row_comm_size));
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(major_last - major_first),
                      local_degrees.data(),
                      [p_offsets] __device__(auto i) { return p_offsets[i + 1] - p_offsets[i]; });
    if (partition.is_hypergraph_partitioned()) {
      col_comm.reduce(local_degrees.data(),
                      i == col_comm_rank ? degrees.data() : static_cast<edge_t *>(nullptr),
                      static_cast<size_t>(major_last - major_first),
                      raft::comms::op_t::SUM,
                      i,
                      handle.get_stream());
    } else {
      row_comm.reduce(local_degrees.data(),
                      i == row_comm_rank ? degrees.data() : static_cast<edge_t *>(nullptr),
                      static_cast<size_t>(major_last - major_first),
                      raft::comms::op_t::SUM,
                      i,
                      handle.get_stream());
    }
  }

  raft::comms::status_t status{};
  if (partition.is_hypergraph_partitioned()) {
    status =
      col_comm.sync_stream(handle.get_stream());  // this is neessary as local_degrees will become
                                                  // out-of-scope once this function returns.
  } else {
    status =
      row_comm.sync_stream(handle.get_stream());  // this is neessary as local_degrees will become
                                                  // out-of-scope once this function returns.
  }
  CUGRAPH_EXPECTS(status == raft::comms::status_t::SUCCESS, "sync_stream() failure.");

  return degrees;
}

// compute the numbers of nonzeros in rows (of the graph adjacency matrix, if store_transposed =
// false) or columns (of the graph adjacency matrix, if store_transposed = true)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degree(
  raft::handle_t const &handle,
  std::vector<rmm::device_uvector<edge_t>> const &adj_matrix_partition_offsets,
  partition_t<vertex_t> const &partition)
{
  // we can avoid creating this temporary with "if constexpr" supported from C++17
  std::vector<edge_t const *> tmp_offsets(adj_matrix_partition_offsets.size(), nullptr);
  std::transform(adj_matrix_partition_offsets.begin(),
                 adj_matrix_partition_offsets.end(),
                 tmp_offsets.begin(),
                 [](auto const &offsets) { return offsets.data(); });
  return compute_major_degree(handle, tmp_offsets, partition);
}

template <typename TxValueIterator>
auto shuffle_values(raft::handle_t const &handle,
                    TxValueIterator tx_value_first,
                    rmm::device_uvector<size_t> const &tx_value_counts)
{
  auto &comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> rx_value_counts(comm_size, handle.get_stream());

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released.
  std::vector<size_t> tx_counts(comm_size, size_t{1});
  std::vector<size_t> tx_offsets(comm_size);
  std::iota(tx_offsets.begin(), tx_offsets.end(), size_t{0});
  std::vector<int> tx_dst_ranks(comm_size);
  std::iota(tx_dst_ranks.begin(), tx_dst_ranks.end(), int{0});
  std::vector<size_t> rx_counts(comm_size, size_t{1});
  std::vector<size_t> rx_offsets(comm_size);
  std::iota(rx_offsets.begin(), rx_offsets.end(), size_t{0});
  std::vector<int> rx_src_ranks(comm_size);
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

  raft::update_host(tx_counts.data(), tx_value_counts.data(), comm_size, handle.get_stream());
  std::partial_sum(tx_counts.begin(), tx_counts.end() - 1, tx_offsets.begin() + 1);
  raft::update_host(rx_counts.data(), rx_value_counts.data(), comm_size, handle.get_stream());
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
  rx_counts.resize(num_rx_src_ranks);
  rx_offsets.resize(num_rx_src_ranks);
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

template <typename vertex_t, typename edge_t>
struct degree_from_offsets_t {
  edge_t const *offsets{nullptr};

  __device__ edge_t operator()(vertex_t v) { return offsets[v + 1] - offsets[v]; }
};

template <typename vertex_t>
struct compute_gpu_id_from_vertex_t {
  int comm_size{0};

  __device__ int operator()(vertex_t v) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    return hash_func(v) % comm_size;
  }
};

template <typename vertex_t, bool store_transposed>
struct compute_gpu_id_from_edge_t {
  bool hypergraph_partitioned{false};
  int comm_size{0};
  int row_comm_size{0};
  int col_comm_size{0};

  __device__ int operator()(vertex_t src, vertex_t dst) const
  {
    cuco::detail::MurmurHash3_32<vertex_t> hash_func{};
    auto major_comm_rank = static_cast<int>(hash_func(store_transposed ? dst : src) % comm_size);
    auto minor_comm_rank = static_cast<int>(hash_func(store_transposed ? src : dst) % comm_size);
    if (hypergraph_partitioned) {
      return (minor_comm_rank / col_comm_size) * row_comm_size + (major_comm_rank % row_comm_size);
    } else {
      return (major_comm_rank - (major_comm_rank % row_comm_size)) +
             (minor_comm_rank / col_comm_size);
    }
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

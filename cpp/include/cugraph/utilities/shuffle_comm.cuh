/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/groupby_and_count.cuh>
#include <cugraph/utilities/thrust_wrappers.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/unique.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

constexpr size_t cache_line_size = 128;

// inline to suppress a complaint about ODR violation
inline std::tuple<std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>,
                  std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>>
compute_tx_rx_counts_displs_ranks(raft::comms::comms_t const& comm,
                                  raft::device_span<size_t const> d_tx_value_counts,
                                  bool drop_empty_ranks,
                                  rmm::cuda_stream_view stream_view)
{
  auto const comm_size = comm.get_size();

  std::vector<size_t> tx_counts(comm_size, size_t{1});
  std::vector<size_t> tx_displs(comm_size);
  std::iota(tx_displs.begin(), tx_displs.end(), size_t{0});
  std::vector<int> tx_dst_ranks(comm_size);
  std::iota(tx_dst_ranks.begin(), tx_dst_ranks.end(), int{0});
  std::vector<size_t> rx_counts(comm_size, size_t{1});
  std::vector<size_t> rx_displs(comm_size);
  std::iota(rx_displs.begin(), rx_displs.end(), size_t{0});
  std::vector<int> rx_src_ranks(comm_size);
  std::iota(rx_src_ranks.begin(), rx_src_ranks.end(), int{0});

  rmm::device_uvector<size_t> d_rx_value_counts(comm_size, stream_view);
  device_alltoall(comm, d_tx_value_counts.data(), d_rx_value_counts.data(), size_t{1}, stream_view);

  raft::update_host(tx_counts.data(), d_tx_value_counts.data(), comm_size, stream_view.value());
  raft::update_host(rx_counts.data(), d_rx_value_counts.data(), comm_size, stream_view.value());
  stream_view.synchronize();

  std::partial_sum(tx_counts.begin(), tx_counts.end() - 1, tx_displs.begin() + 1);
  std::partial_sum(rx_counts.begin(), rx_counts.end() - 1, rx_displs.begin() + 1);

  if (drop_empty_ranks) {
    int num_tx_dst_ranks{0};
    int num_rx_src_ranks{0};
    for (int i = 0; i < comm_size; ++i) {
      if (tx_counts[i] != 0) {
        tx_counts[num_tx_dst_ranks]    = tx_counts[i];
        tx_displs[num_tx_dst_ranks]    = tx_displs[i];
        tx_dst_ranks[num_tx_dst_ranks] = tx_dst_ranks[i];
        ++num_tx_dst_ranks;
      }
      if (rx_counts[i] != 0) {
        rx_counts[num_rx_src_ranks]    = rx_counts[i];
        rx_displs[num_rx_src_ranks]    = rx_displs[i];
        rx_src_ranks[num_rx_src_ranks] = rx_src_ranks[i];
        ++num_rx_src_ranks;
      }
    }
    tx_counts.resize(num_tx_dst_ranks);
    tx_displs.resize(num_tx_dst_ranks);
    tx_dst_ranks.resize(num_tx_dst_ranks);
    rx_counts.resize(num_rx_src_ranks);
    rx_displs.resize(num_rx_src_ranks);
    rx_src_ranks.resize(num_rx_src_ranks);
  }

  return std::make_tuple(tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks);
}

template <typename key_type, typename KeyToGroupIdOp>
struct key_group_id_less_t {
  KeyToGroupIdOp key_to_group_id_op;
  int pivot{};
  __device__ bool operator()(key_type k) const { return key_to_group_id_op(k) < pivot; }
};

template <typename value_type, typename ValueToGroupIdOp>
struct value_group_id_greater_equal_t {
  ValueToGroupIdOp value_to_group_id_op;
  int pivot{};
  __device__ bool operator()(value_type v) const { return value_to_group_id_op(v) >= pivot; }
};

template <typename key_type, typename value_type, typename KeyToGroupIdOp>
struct kv_pair_group_id_greater_equal_t {
  KeyToGroupIdOp key_to_group_id_op;
  int pivot{};
  __device__ bool operator()(cuda::std::tuple<key_type, value_type> t) const
  {
    return key_to_group_id_op(cuda::std::get<0>(t)) >= pivot;
  }
};

}  // namespace detail

template <typename TxValueIterator>
auto shuffle_values(raft::comms::comms_t const& comm,
                    TxValueIterator tx_value_first,
                    raft::device_span<size_t const> d_tx_value_counts,
                    rmm::cuda_stream_view stream_view,
                    std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  auto const comm_size = comm.get_size();

  CUGRAPH_EXPECTS(
    static_cast<int>(d_tx_value_counts.size()) == comm_size,
    "Invalid input argument: d_tx_value_countsw.size() should coincide with comm.get_size()");
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto [tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks] =
    detail::compute_tx_rx_counts_displs_ranks(comm, d_tx_value_counts, false, stream_view);

  auto rx_buffer_size = rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0};
  auto rx_value_buffer =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<value_t>(rx_buffer_size, stream_view)
      : allocate_dataframe_buffer<value_t>(rx_buffer_size, stream_view);

  device_multicast_sendrecv(comm,
                            tx_value_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            get_dataframe_buffer_begin(rx_value_buffer),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                            stream_view);

  if (rx_counts.size() < static_cast<size_t>(comm_size)) {
    std::vector<size_t> tmp_rx_counts(comm_size, size_t{0});
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      assert(rx_src_ranks[i] < comm_size);
      tmp_rx_counts[rx_src_ranks[i]] = rx_counts[i];
    }
    rx_counts = std::move(tmp_rx_counts);
  }

  return std::make_tuple(std::move(rx_value_buffer), rx_counts);
}

template <typename TxValueIterator>
auto shuffle_values(raft::comms::comms_t const& comm,
                    TxValueIterator tx_value_first,
                    raft::host_span<size_t const> tx_value_counts,
                    rmm::cuda_stream_view stream_view,
                    std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> d_tx_value_counts(comm_size, stream_view);
  raft::update_device(
    d_tx_value_counts.data(), tx_value_counts.data(), comm_size, stream_view.value());

  return shuffle_values(
    comm,
    tx_value_first,
    raft::device_span<size_t const>{d_tx_value_counts.data(), d_tx_value_counts.size()},
    stream_view,
    large_buffer_type);
}

// Add gaps in the receive buffer to enforce that the sent data offset and the received data offset
// have the same alignment for every rank. This is faster assuming that @p alignment ensures cache
// line alignment in both send & receive buffer (tested with NCCL 2.23.4)
template <typename TxValueIterator>
auto shuffle_values(
  raft::comms::comms_t const& comm,
  TxValueIterator tx_value_first,
  raft::host_span<size_t const> tx_value_counts,
  size_t alignment,  // # elements
  std::optional<typename thrust::iterator_traits<TxValueIterator>::value_type> fill_value,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto const comm_size = comm.get_size();

  std::vector<size_t> tx_value_displacements(tx_value_counts.size());
  std::exclusive_scan(
    tx_value_counts.begin(), tx_value_counts.end(), tx_value_displacements.begin(), size_t{0});

  std::vector<size_t> tx_unaligned_counts(comm_size);
  std::vector<size_t> tx_displacements(comm_size);
  std::vector<size_t> tx_aligned_counts(comm_size);
  std::vector<size_t> tx_aligned_displacements(comm_size);
  std::vector<size_t> rx_unaligned_counts(comm_size);
  std::vector<size_t> rx_displacements(comm_size);
  std::vector<size_t> rx_aligned_counts(comm_size);
  std::vector<size_t> rx_aligned_displacements(comm_size);
  std::vector<int> tx_ranks(comm_size);
  std::iota(tx_ranks.begin(), tx_ranks.end(), int{0});
  auto rx_ranks = tx_ranks;
  for (size_t i = 0; i < tx_value_counts.size(); ++i) {
    tx_unaligned_counts[i] = 0;
    if (tx_value_displacements[i] % alignment != 0) {
      tx_unaligned_counts[i] =
        std::min(alignment - (tx_value_displacements[i] % alignment), tx_value_counts[i]);
    }
    tx_displacements[i]         = tx_value_displacements[i];
    tx_aligned_counts[i]        = tx_value_counts[i] - tx_unaligned_counts[i];
    tx_aligned_displacements[i] = tx_value_displacements[i] + tx_unaligned_counts[i];
  }

  rmm::device_uvector<size_t> d_tx_unaligned_counts(tx_unaligned_counts.size(), stream_view);
  rmm::device_uvector<size_t> d_tx_aligned_counts(tx_aligned_counts.size(), stream_view);
  rmm::device_uvector<size_t> d_rx_unaligned_counts(rx_unaligned_counts.size(), stream_view);
  rmm::device_uvector<size_t> d_rx_aligned_counts(rx_aligned_counts.size(), stream_view);
  raft::update_device(d_tx_unaligned_counts.data(),
                      tx_unaligned_counts.data(),
                      tx_unaligned_counts.size(),
                      stream_view);
  raft::update_device(
    d_tx_aligned_counts.data(), tx_aligned_counts.data(), tx_aligned_counts.size(), stream_view);
  std::vector<size_t> tx_counts(comm_size, size_t{1});
  std::vector<size_t> tx_displs(comm_size);
  std::iota(tx_displs.begin(), tx_displs.end(), size_t{0});
  auto rx_counts = tx_counts;
  auto rx_displs = tx_displs;
  cugraph::device_multicast_sendrecv(
    comm,
    d_tx_unaligned_counts.data(),
    raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
    raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    d_rx_unaligned_counts.data(),
    raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
    raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);
  cugraph::device_multicast_sendrecv(
    comm,
    d_tx_aligned_counts.data(),
    raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
    raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    d_rx_aligned_counts.data(),
    raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
    raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);
  raft::update_host(rx_unaligned_counts.data(),
                    d_rx_unaligned_counts.data(),
                    d_rx_unaligned_counts.size(),
                    stream_view);
  raft::update_host(
    rx_aligned_counts.data(), d_rx_aligned_counts.data(), d_rx_aligned_counts.size(), stream_view);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view));
  size_t offset{0};
  for (size_t i = 0; i < rx_counts.size(); ++i) {
    auto target_alignment = (alignment - rx_unaligned_counts[i]) % alignment;
    auto cur_alignment    = offset % alignment;
    if (target_alignment >= cur_alignment) {
      offset += target_alignment - cur_alignment;
    } else {
      offset += (target_alignment + alignment) - cur_alignment;
    }
    rx_displacements[i]         = offset;
    rx_aligned_displacements[i] = rx_displacements[i] + rx_unaligned_counts[i];
    offset                      = rx_aligned_displacements[i] + rx_aligned_counts[i];
  }

  auto rx_buffer_size = rx_aligned_displacements.back() + rx_aligned_counts.back();
  auto rx_values =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<value_t>(rx_buffer_size, stream_view)
      : allocate_dataframe_buffer<value_t>(rx_buffer_size, stream_view);
  if (fill_value) {
    cugraph::fill(rmm::exec_policy_nosync(stream_view),
                  get_dataframe_buffer_begin(rx_values),
                  get_dataframe_buffer_end(rx_values),
                  *fill_value);
  }
  cugraph::device_multicast_sendrecv(
    comm,
    tx_value_first,
    raft::host_span<size_t const>(tx_unaligned_counts.data(), tx_unaligned_counts.size()),
    raft::host_span<size_t const>(tx_displacements.data(), tx_displacements.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    get_dataframe_buffer_begin(rx_values),
    raft::host_span<size_t const>(rx_unaligned_counts.data(), rx_unaligned_counts.size()),
    raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);
  cugraph::device_multicast_sendrecv(
    comm,
    tx_value_first,
    raft::host_span<size_t const>(tx_aligned_counts.data(), tx_aligned_counts.size()),
    raft::host_span<size_t const>(tx_aligned_displacements.data(), tx_aligned_displacements.size()),
    raft::host_span<int const>(tx_ranks.data(), tx_ranks.size()),
    get_dataframe_buffer_begin(rx_values),
    raft::host_span<size_t const>(rx_aligned_counts.data(), rx_aligned_counts.size()),
    raft::host_span<size_t const>(rx_aligned_displacements.data(), rx_aligned_displacements.size()),
    raft::host_span<int const>(rx_ranks.data(), rx_ranks.size()),
    stream_view);

  return std::make_tuple(std::move(rx_values),
                         tx_unaligned_counts,
                         tx_aligned_counts,
                         tx_displacements,
                         rx_unaligned_counts,
                         rx_aligned_counts,
                         rx_displacements);
}

// this uses less memory than calling shuffle_values then sort & unique but requires comm.get_size()
// - 1 communication steps
template <typename TxValueIterator>
auto shuffle_and_unique_segment_sorted_values(
  raft::comms::comms_t const& comm,
  TxValueIterator
    segment_sorted_tx_value_first,  // sorted within each segment (segment sizes:
                                    // tx_value_counts[i], where i = [0, comm_size); and bettter be
                                    // unique to reduce communication volume
  raft::host_span<size_t const> tx_value_counts,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  auto sorted_unique_values =
    large_buffer_type ? large_buffer_manager::allocate_memory_buffer<value_t>(0, stream_view)
                      : allocate_dataframe_buffer<value_t>(0, stream_view);
  if (comm_size == 1) {
    resize_dataframe_buffer(sorted_unique_values, tx_value_counts[comm_rank], stream_view);
    thrust::copy(rmm::exec_policy_nosync(stream_view),
                 segment_sorted_tx_value_first,
                 segment_sorted_tx_value_first + tx_value_counts[comm_rank],
                 get_dataframe_buffer_begin(sorted_unique_values));
    resize_dataframe_buffer(
      sorted_unique_values,
      cuda::std::distance(get_dataframe_buffer_begin(sorted_unique_values),
                          cugraph::unique(rmm::exec_policy_nosync(stream_view),
                                          get_dataframe_buffer_begin(sorted_unique_values),
                                          get_dataframe_buffer_end(sorted_unique_values))),
      stream_view);
  } else {
    rmm::device_uvector<size_t> d_tx_value_counts(comm_size, stream_view);
    raft::update_device(
      d_tx_value_counts.data(), tx_value_counts.data(), comm_size, stream_view.value());

    std::vector<size_t> tx_counts{};
    std::vector<size_t> tx_displs{};
    std::vector<size_t> rx_counts{};
    std::vector<size_t> rx_displs{};
    std::tie(tx_counts, tx_displs, std::ignore, rx_counts, rx_displs, std::ignore) =
      detail::compute_tx_rx_counts_displs_ranks(
        comm,
        raft::device_span<size_t const>{d_tx_value_counts.data(), d_tx_value_counts.size()},
        false,
        stream_view);

    d_tx_value_counts.resize(0, stream_view);
    d_tx_value_counts.shrink_to_fit(stream_view);

    for (int i = 1; i < comm_size; ++i) {
      auto dst = (comm_rank + i) % comm_size;
      auto src =
        static_cast<int>((static_cast<size_t>(comm_rank) + static_cast<size_t>(comm_size - i)) %
                         static_cast<size_t>(comm_size));
      auto rx_sorted_values =
        large_buffer_type
          ? large_buffer_manager::allocate_memory_buffer<value_t>(rx_counts[src], stream_view)
          : allocate_dataframe_buffer<value_t>(rx_counts[src], stream_view);
      device_sendrecv(comm,
                      segment_sorted_tx_value_first + tx_displs[dst],
                      tx_counts[dst],
                      dst,
                      get_dataframe_buffer_begin(rx_sorted_values),
                      rx_counts[src],
                      src,
                      stream_view);
      auto merged_size =
        (i == 1 ? tx_counts[comm_rank] : size_dataframe_buffer(sorted_unique_values)) +
        rx_counts[src];
      auto merged_sorted_values =
        large_buffer_type
          ? large_buffer_manager::allocate_memory_buffer<value_t>(merged_size, stream_view)
          : allocate_dataframe_buffer<value_t>(merged_size, stream_view);
      if (i == 1) {
        thrust::merge(rmm::exec_policy_nosync(stream_view),
                      segment_sorted_tx_value_first + tx_displs[comm_rank],
                      segment_sorted_tx_value_first + (tx_displs[comm_rank] + tx_counts[comm_rank]),
                      get_dataframe_buffer_begin(rx_sorted_values),
                      get_dataframe_buffer_end(rx_sorted_values),
                      get_dataframe_buffer_begin(merged_sorted_values));
      } else {
        thrust::merge(rmm::exec_policy_nosync(stream_view),
                      get_dataframe_buffer_begin(sorted_unique_values),
                      get_dataframe_buffer_end(sorted_unique_values),
                      get_dataframe_buffer_begin(rx_sorted_values),
                      get_dataframe_buffer_end(rx_sorted_values),
                      get_dataframe_buffer_begin(merged_sorted_values));
      }
      resize_dataframe_buffer(
        merged_sorted_values,
        cuda::std::distance(get_dataframe_buffer_begin(merged_sorted_values),
                            cugraph::unique(rmm::exec_policy_nosync(stream_view),
                                            get_dataframe_buffer_begin(merged_sorted_values),
                                            get_dataframe_buffer_end(merged_sorted_values))),
        stream_view);
      sorted_unique_values = std::move(merged_sorted_values);
    }
  }
  shrink_to_fit_dataframe_buffer(sorted_unique_values, stream_view);
  return sorted_unique_values;
}

template <typename ValueIterator, typename ValueToGPUIdOp>
auto groupby_gpu_id_and_shuffle_values(
  raft::comms::comms_t const& comm,
  ValueIterator tx_value_first /* [INOUT */,
  ValueIterator tx_value_last /* [INOUT */,
  ValueToGPUIdOp value_to_gpu_id_op,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto const comm_size = comm.get_size();

  auto d_tx_value_counts = groupby_and_count(tx_value_first,
                                             tx_value_last,
                                             value_to_gpu_id_op,
                                             comm.get_size(),
                                             std::numeric_limits<size_t>::max(),
                                             stream_view,
                                             large_buffer_type);

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_displs{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_displs{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks) =
    detail::compute_tx_rx_counts_displs_ranks(
      comm,
      raft::device_span<size_t const>{d_tx_value_counts.data(), d_tx_value_counts.size()},
      false,
      stream_view);

  auto rx_buffer_size = rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0};
  auto rx_value_buffer =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<value_t>(rx_buffer_size, stream_view)
      : allocate_dataframe_buffer<value_t>(rx_buffer_size, stream_view);

  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            get_dataframe_buffer_begin(rx_value_buffer),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                            stream_view);

  if (rx_counts.size() < static_cast<size_t>(comm_size)) {
    std::vector<size_t> tmp_rx_counts(comm_size, size_t{0});
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      tmp_rx_counts[rx_src_ranks[i]] = rx_counts[i];
    }
    rx_counts = std::move(tmp_rx_counts);
  }

  return std::make_tuple(std::move(rx_value_buffer), rx_counts);
}

template <typename KeyIterator, typename ValueIterator, typename KeyToGPUIdOp>
auto groupby_gpu_id_and_shuffle_kv_pairs(
  raft::comms::comms_t const& comm,
  KeyIterator tx_key_first /* [INOUT */,
  KeyIterator tx_key_last /* [INOUT */,
  ValueIterator tx_value_first /* [INOUT */,
  KeyToGPUIdOp key_to_gpu_id_op,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using key_t   = typename thrust::iterator_traits<KeyIterator>::value_type;
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto const comm_size = comm.get_size();

  auto d_tx_value_counts = groupby_and_count(tx_key_first,
                                             tx_key_last,
                                             tx_value_first,
                                             key_to_gpu_id_op,
                                             comm.get_size(),
                                             std::numeric_limits<size_t>::max(),
                                             stream_view,
                                             large_buffer_type);

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_displs{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_displs{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks) =
    detail::compute_tx_rx_counts_displs_ranks(
      comm,
      raft::device_span<size_t const>{d_tx_value_counts.data(), d_tx_value_counts.size()},
      false,
      stream_view);

  auto rx_buffer_size = rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0};
  auto rx_keys        = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<key_t>(
                                       rx_buffer_size, stream_view)
                                          : rmm::device_uvector<key_t>(rx_buffer_size, stream_view);
  auto rx_value_buffer =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<value_t>(rx_buffer_size, stream_view)
      : allocate_dataframe_buffer<value_t>(rx_buffer_size, stream_view);

  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_key_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            rx_keys.begin(),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                            stream_view);

  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_value_first,
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            get_dataframe_buffer_begin(rx_value_buffer),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                            stream_view);

  if (rx_counts.size() < static_cast<size_t>(comm_size)) {
    std::vector<size_t> tmp_rx_counts(comm_size, size_t{0});
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      assert(rx_src_ranks[i] < comm_size);
      tmp_rx_counts[rx_src_ranks[i]] = rx_counts[i];
    }
    rx_counts = std::move(tmp_rx_counts);
  }

  return std::make_tuple(std::move(rx_keys), std::move(rx_value_buffer), rx_counts);
}

}  // namespace CUGRAPH_EXPORT cugraph

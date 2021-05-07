/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace cugraph {
namespace experimental {

namespace detail {

// inline to suppress a complaint about ODR violation
inline std::tuple<std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>,
                  std::vector<size_t>,
                  std::vector<size_t>,
                  std::vector<int>>
compute_tx_rx_counts_offsets_ranks(raft::comms::comms_t const &comm,
                                   rmm::device_uvector<size_t> const &d_tx_value_counts,
                                   cudaStream_t stream)
{
  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> d_rx_value_counts(comm_size, stream);

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
                            d_tx_value_counts.data(),
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            d_rx_value_counts.data(),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            stream);
  // FIXME: temporary unverified work-around for a NCCL (2.9.6) bug that causes a hang on DGX1 (due
  // to remote memory allocation), this synchronization is unnecessary otherwise but seems like
  // suppress the hange issue. Need to be revisited once NCCL 2.10 is released.
  CUDA_TRY(cudaDeviceSynchronize());

  raft::update_host(tx_counts.data(), d_tx_value_counts.data(), comm_size, stream);
  raft::update_host(rx_counts.data(), d_rx_value_counts.data(), comm_size, stream);

  CUDA_TRY(cudaStreamSynchronize(stream));  // rx_counts should be up-to-date

  std::partial_sum(tx_counts.begin(), tx_counts.end() - 1, tx_offsets.begin() + 1);
  std::partial_sum(rx_counts.begin(), rx_counts.end() - 1, rx_offsets.begin() + 1);

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
      ++num_rx_src_ranks;
    }
  }
  tx_counts.resize(num_tx_dst_ranks);
  tx_offsets.resize(num_tx_dst_ranks);
  tx_dst_ranks.resize(num_tx_dst_ranks);
  rx_counts.resize(num_rx_src_ranks);
  rx_offsets.resize(num_rx_src_ranks);
  rx_src_ranks.resize(num_rx_src_ranks);

  return std::make_tuple(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks);
}

}  // namespace detail

template <typename ValueIterator, typename ValueToGPUIdOp>
rmm::device_uvector<size_t> groupby_and_count(ValueIterator tx_value_first /* [INOUT */,
                                              ValueIterator tx_value_last /* [INOUT */,
                                              ValueToGPUIdOp value_to_group_id_op,
                                              int num_groups,
                                              cudaStream_t stream)
{
  thrust::sort(rmm::exec_policy(stream)->on(stream),
               tx_value_first,
               tx_value_last,
               [value_to_group_id_op] __device__(auto lhs, auto rhs) {
                 return value_to_group_id_op(lhs) < value_to_group_id_op(rhs);
               });

  auto group_id_first = thrust::make_transform_iterator(
    tx_value_first,
    [value_to_group_id_op] __device__(auto value) { return value_to_group_id_op(value); });
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream);
  auto last =
    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          group_id_first,
                          group_id_first + thrust::distance(tx_value_first, tx_value_last),
                          thrust::make_constant_iterator(size_t{1}),
                          d_tx_dst_ranks.begin(),
                          d_tx_value_counts.begin());
  if (thrust::distance(d_tx_dst_ranks.begin(), thrust::get<0>(last)) < num_groups) {
    rmm::device_uvector<size_t> d_counts(num_groups, stream);
    thrust::fill(rmm::exec_policy(stream)->on(stream), d_counts.begin(), d_counts.end(), size_t{0});
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    d_tx_value_counts.begin(),
                    thrust::get<1>(last),
                    d_tx_dst_ranks.begin(),
                    d_counts.begin());
    d_tx_value_counts = std::move(d_counts);
  }

  return d_tx_value_counts;
}

template <typename VertexIterator, typename ValueIterator, typename KeyToGPUIdOp>
rmm::device_uvector<size_t> groupby_and_count(VertexIterator tx_key_first /* [INOUT */,
                                              VertexIterator tx_key_last /* [INOUT */,
                                              ValueIterator tx_value_first /* [INOUT */,
                                              KeyToGPUIdOp key_to_group_id_op,
                                              int num_groups,
                                              cudaStream_t stream)
{
  thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
                      tx_key_first,
                      tx_key_last,
                      tx_value_first,
                      [key_to_group_id_op] __device__(auto lhs, auto rhs) {
                        return key_to_group_id_op(lhs) < key_to_group_id_op(rhs);
                      });

  auto group_id_first = thrust::make_transform_iterator(
    tx_key_first, [key_to_group_id_op] __device__(auto key) { return key_to_group_id_op(key); });
  rmm::device_uvector<int> d_tx_dst_ranks(num_groups, stream);
  rmm::device_uvector<size_t> d_tx_value_counts(d_tx_dst_ranks.size(), stream);
  auto last = thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                                    group_id_first,
                                    group_id_first + thrust::distance(tx_key_first, tx_key_last),
                                    thrust::make_constant_iterator(size_t{1}),
                                    d_tx_dst_ranks.begin(),
                                    d_tx_value_counts.begin());
  if (thrust::distance(d_tx_dst_ranks.begin(), thrust::get<0>(last)) < num_groups) {
    rmm::device_uvector<size_t> d_counts(num_groups, stream);
    thrust::fill(rmm::exec_policy(stream)->on(stream), d_counts.begin(), d_counts.end(), size_t{0});
    thrust::scatter(rmm::exec_policy(stream)->on(stream),
                    d_tx_value_counts.begin(),
                    thrust::get<1>(last),
                    d_tx_dst_ranks.begin(),
                    d_counts.begin());
    d_tx_value_counts = std::move(d_counts);
  }

  return d_tx_value_counts;
}

template <typename TxValueIterator>
auto shuffle_values(raft::comms::comms_t const &comm,
                    TxValueIterator tx_value_first,
                    std::vector<size_t> const &tx_value_counts,
                    cudaStream_t stream)
{
  auto const comm_size = comm.get_size();

  rmm::device_uvector<size_t> d_tx_value_counts(comm_size, stream);
  raft::update_device(d_tx_value_counts.data(), tx_value_counts.data(), comm_size, stream);

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_offsets{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_offsets{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks) =
    detail::compute_tx_rx_counts_offsets_ranks(comm, d_tx_value_counts, stream);

  auto rx_value_buffer =
    allocate_dataframe_buffer<typename std::iterator_traits<TxValueIterator>::value_type>(
      rx_offsets.size() > 0 ? rx_offsets.back() + rx_counts.back() : size_t{0}, stream);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(
    comm,
    tx_value_first,
    tx_counts,
    tx_offsets,
    tx_dst_ranks,
    get_dataframe_buffer_begin<typename std::iterator_traits<TxValueIterator>::value_type>(
      rx_value_buffer),
    rx_counts,
    rx_offsets,
    rx_src_ranks,
    stream);

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

template <typename ValueIterator, typename ValueToGPUIdOp>
auto groupby_gpuid_and_shuffle_values(raft::comms::comms_t const &comm,
                                      ValueIterator tx_value_first /* [INOUT */,
                                      ValueIterator tx_value_last /* [INOUT */,
                                      ValueToGPUIdOp value_to_gpu_id_op,
                                      cudaStream_t stream)
{
  auto const comm_size = comm.get_size();

  auto d_tx_value_counts =
    groupby_and_count(tx_value_first, tx_value_last, value_to_gpu_id_op, comm.get_size(), stream);

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_offsets{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_offsets{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks) =
    detail::compute_tx_rx_counts_offsets_ranks(comm, d_tx_value_counts, stream);

  auto rx_value_buffer =
    allocate_dataframe_buffer<typename std::iterator_traits<ValueIterator>::value_type>(
      rx_offsets.size() > 0 ? rx_offsets.back() + rx_counts.back() : size_t{0}, stream);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(
    comm,
    tx_value_first,
    tx_counts,
    tx_offsets,
    tx_dst_ranks,
    get_dataframe_buffer_begin<typename std::iterator_traits<ValueIterator>::value_type>(
      rx_value_buffer),
    rx_counts,
    rx_offsets,
    rx_src_ranks,
    stream);

  if (rx_counts.size() < static_cast<size_t>(comm_size)) {
    std::vector<size_t> tmp_rx_counts(comm_size, size_t{0});
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      tmp_rx_counts[rx_src_ranks[i]] = rx_counts[i];
    }
    rx_counts = std::move(tmp_rx_counts);
  }

  return std::make_tuple(std::move(rx_value_buffer), rx_counts);
}

template <typename VertexIterator, typename ValueIterator, typename KeyToGPUIdOp>
auto groupby_gpuid_and_shuffle_kv_pairs(raft::comms::comms_t const &comm,
                                        VertexIterator tx_key_first /* [INOUT */,
                                        VertexIterator tx_key_last /* [INOUT */,
                                        ValueIterator tx_value_first /* [INOUT */,
                                        KeyToGPUIdOp key_to_gpu_id_op,
                                        cudaStream_t stream)
{
  auto const comm_size = comm.get_size();

  auto d_tx_value_counts = groupby_and_count(
    tx_key_first, tx_key_last, tx_value_first, key_to_gpu_id_op, comm.get_size(), stream);

  std::vector<size_t> tx_counts{};
  std::vector<size_t> tx_offsets{};
  std::vector<int> tx_dst_ranks{};
  std::vector<size_t> rx_counts{};
  std::vector<size_t> rx_offsets{};
  std::vector<int> rx_src_ranks{};
  std::tie(tx_counts, tx_offsets, tx_dst_ranks, rx_counts, rx_offsets, rx_src_ranks) =
    detail::compute_tx_rx_counts_offsets_ranks(comm, d_tx_value_counts, stream);

  rmm::device_uvector<typename std::iterator_traits<VertexIterator>::value_type> rx_keys(
    rx_offsets.size() > 0 ? rx_offsets.back() + rx_counts.back() : size_t{0}, stream);
  auto rx_value_buffer =
    allocate_dataframe_buffer<typename std::iterator_traits<ValueIterator>::value_type>(
      rx_keys.size(), stream);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(comm,
                            tx_key_first,
                            tx_counts,
                            tx_offsets,
                            tx_dst_ranks,
                            rx_keys.begin(),
                            rx_counts,
                            rx_offsets,
                            rx_src_ranks,
                            stream);

  // FIXME: this needs to be replaced with AlltoAll once NCCL 2.8 is released
  // (if num_tx_dst_ranks == num_rx_src_ranks == comm_size).
  device_multicast_sendrecv(
    comm,
    tx_value_first,
    tx_counts,
    tx_offsets,
    tx_dst_ranks,
    get_dataframe_buffer_begin<typename std::iterator_traits<ValueIterator>::value_type>(
      rx_value_buffer),
    rx_counts,
    rx_offsets,
    rx_src_ranks,
    stream);

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

}  // namespace experimental
}  // namespace cugraph

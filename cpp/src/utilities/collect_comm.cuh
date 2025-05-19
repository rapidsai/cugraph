/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/kv_store.cuh"

#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuda/std/iterator>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <iterator>
#include <memory>
#include <vector>

namespace cugraph {

// for the keys in kv_store_view, key_to_comm_rank_op(key) should coincide with comm.get_rank()
template <typename KVStoreViewType, typename KeyIterator, typename KeyToCommRankOp>
dataframe_buffer_type_t<typename KVStoreViewType::value_type> collect_values_for_keys(
  raft::comms::comms_t const& comm,
  KVStoreViewType kv_store_view,
  KeyIterator collect_key_first,
  KeyIterator collect_key_last,
  KeyToCommRankOp key_to_comm_rank_op,
  rmm::cuda_stream_view stream_view)
{
  using key_t = typename KVStoreViewType::key_type;
  static_assert(std::is_same_v<typename thrust::iterator_traits<KeyIterator>::value_type, key_t>);
  using value_t = typename KVStoreViewType::value_type;

  // 1. collect values for the unique keys in [collect_key_first, collect_key_last)

  rmm::device_uvector<key_t> unique_keys(cuda::std::distance(collect_key_first, collect_key_last),
                                         stream_view);
  thrust::copy(
    rmm::exec_policy_nosync(stream_view), collect_key_first, collect_key_last, unique_keys.begin());
  thrust::sort(rmm::exec_policy_nosync(stream_view), unique_keys.begin(), unique_keys.end());
  unique_keys.resize(
    cuda::std::distance(
      unique_keys.begin(),
      thrust::unique(rmm::exec_policy(stream_view), unique_keys.begin(), unique_keys.end())),
    stream_view);

  auto values_for_unique_keys = allocate_dataframe_buffer<value_t>(0, stream_view);
  {
    rmm::device_uvector<key_t> rx_unique_keys(0, stream_view);
    std::vector<size_t> rx_value_counts{};
    std::tie(rx_unique_keys, rx_value_counts) = groupby_gpu_id_and_shuffle_values(
      comm,
      unique_keys.begin(),
      unique_keys.end(),
      [key_to_comm_rank_op] __device__(auto val) { return key_to_comm_rank_op(val); },
      stream_view);

    auto values_for_rx_unique_keys =
      allocate_dataframe_buffer<value_t>(rx_unique_keys.size(), stream_view);

    kv_store_view.find(rx_unique_keys.begin(),
                       rx_unique_keys.end(),
                       get_dataframe_buffer_begin(values_for_rx_unique_keys),
                       stream_view);

    auto rx_values_for_unique_keys = allocate_dataframe_buffer<value_t>(0, stream_view);
    std::tie(rx_values_for_unique_keys, std::ignore) =
      shuffle_values(comm,
                     get_dataframe_buffer_begin(values_for_rx_unique_keys),
                     raft::host_span<size_t const>(rx_value_counts.data(), rx_value_counts.size()),
                     stream_view);

    values_for_unique_keys = std::move(rx_values_for_unique_keys);
  }

  // 2. build a kv_store_t object for the k, v pairs in unique_keys, values_for_unique_keys.

  kv_store_t<key_t, value_t, KVStoreViewType::binary_search> unique_key_value_store(stream_view);
  if constexpr (KVStoreViewType::binary_search) {
    unique_key_value_store = kv_store_t<key_t, value_t, true>(std::move(unique_keys),
                                                              std::move(values_for_unique_keys),
                                                              kv_store_view.invalid_value(),
                                                              false /* key_sorted */,
                                                              stream_view);
  } else {
    auto kv_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(unique_keys.begin(), get_dataframe_buffer_begin(values_for_unique_keys)));
    auto valid_kv_pair_last =
      thrust::remove_if(rmm::exec_policy(stream_view),
                        kv_pair_first,
                        kv_pair_first + unique_keys.size(),
                        [invalid_value = kv_store_view.invalid_value()] __device__(auto pair) {
                          return thrust::get<1>(pair) == invalid_value;
                        });  // remove (k,v) pairs with unmatched keys (it is invalid to insert a
                             // (k,v) pair with v = empty_key_sentinel)
    auto num_valid_pairs =
      static_cast<size_t>(cuda::std::distance(kv_pair_first, valid_kv_pair_last));
    unique_key_value_store =
      kv_store_t<key_t, value_t, false>(unique_keys.begin(),
                                        unique_keys.begin() + num_valid_pairs,
                                        get_dataframe_buffer_begin(values_for_unique_keys),
                                        kv_store_view.invalid_key(),
                                        kv_store_view.invalid_value(),
                                        stream_view);

    unique_keys.resize(0, stream_view);
    resize_dataframe_buffer(values_for_unique_keys, 0, stream_view);
    unique_keys.shrink_to_fit(stream_view);
    shrink_to_fit_dataframe_buffer(values_for_unique_keys, stream_view);
  }
  auto unique_key_value_store_view = unique_key_value_store.view();

  // 3. find values for [collect_key_first, collect_key_last)

  auto value_buffer = allocate_dataframe_buffer<value_t>(
    cuda::std::distance(collect_key_first, collect_key_last), stream_view);
  unique_key_value_store_view.find(
    collect_key_first, collect_key_last, get_dataframe_buffer_begin(value_buffer), stream_view);

  return value_buffer;
}

// for the keys in kv_store_view, key_to_comm_rank_op(key) should coincide with comm.get_rank()
template <typename KVStoreViewType, typename KeyToCommRankOp>
std::tuple<rmm::device_uvector<typename KVStoreViewType::key_type>,
           dataframe_buffer_type_t<typename KVStoreViewType::value_type>>
collect_values_for_unique_keys(
  raft::comms::comms_t const& comm,
  KVStoreViewType kv_store_view,
  rmm::device_uvector<typename KVStoreViewType::key_type>&& collect_unique_keys,
  KeyToCommRankOp key_to_comm_rank_op,
  rmm::cuda_stream_view stream_view)
{
  using key_t   = typename KVStoreViewType::key_type;
  using value_t = typename KVStoreViewType::value_type;

  auto values_for_collect_unique_keys = allocate_dataframe_buffer<value_t>(0, stream_view);
  {
    auto [rx_unique_keys, rx_value_counts] = groupby_gpu_id_and_shuffle_values(
      comm,
      collect_unique_keys.begin(),
      collect_unique_keys.end(),
      [key_to_comm_rank_op] __device__(auto val) { return key_to_comm_rank_op(val); },
      stream_view);
    auto values_for_rx_unique_keys =
      allocate_dataframe_buffer<value_t>(rx_unique_keys.size(), stream_view);
    kv_store_view.find(rx_unique_keys.begin(),
                       rx_unique_keys.end(),
                       get_dataframe_buffer_begin(values_for_rx_unique_keys),
                       stream_view);

    std::tie(values_for_collect_unique_keys, std::ignore) =
      shuffle_values(comm,
                     get_dataframe_buffer_begin(values_for_rx_unique_keys),
                     raft::host_span<size_t const>(rx_value_counts.data(), rx_value_counts.size()),
                     stream_view);
  }

  return std::make_tuple(std::move(collect_unique_keys), std::move(values_for_collect_unique_keys));
}

template <typename vertex_t, typename ValueIterator>
dataframe_buffer_type_t<typename thrust::iterator_traits<ValueIterator>::value_type>
collect_values_for_sorted_unique_int_vertices(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> collect_sorted_unique_int_vertices,
  ValueIterator local_value_first,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  vertex_t local_vertex_partition_range_first)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  auto& comm                 = handle.get_comms();
  auto const comm_rank       = comm.get_rank();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_rank = major_comm.get_rank();
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_rank = minor_comm.get_rank();
  auto const minor_comm_size = minor_comm.get_size();

  std::vector<int> tx_dst_ranks(comm_size);
  std::vector<int> rx_src_ranks(comm_size);
  std::iota(tx_dst_ranks.begin(), tx_dst_ranks.end(), int{0});
  std::iota(rx_src_ranks.begin(), rx_src_ranks.end(), int{0});

  // 1.find tx_counts

  std::vector<size_t> tx_counts(comm_size);
  std::vector<size_t> tx_displs(tx_counts.size());
  {
    rmm::device_uvector<vertex_t> d_range_lasts(vertex_partition_range_lasts.size(),
                                                handle.get_stream());
    raft::update_device(d_range_lasts.data(),
                        vertex_partition_range_lasts.data(),
                        vertex_partition_range_lasts.size(),
                        handle.get_stream());

    rmm::device_uvector<size_t> d_offsets(d_range_lasts.size() - 1, handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        collect_sorted_unique_int_vertices.begin(),
                        collect_sorted_unique_int_vertices.end(),
                        d_range_lasts.begin(),
                        d_range_lasts.begin() + (d_range_lasts.size() - 1),
                        d_offsets.begin());

    std::vector<size_t> h_offsets(d_offsets.size() + 2);
    raft::update_host(
      h_offsets.data() + 1, d_offsets.data(), d_offsets.size(), handle.get_stream());
    h_offsets[0]     = 0;
    h_offsets.back() = collect_sorted_unique_int_vertices.size();
    handle.sync_stream();

    for (size_t i = 0; i < vertex_partition_range_lasts.size(); ++i) {
      auto comm_rank_for_vertex_partition_id =
        partition_manager::compute_global_comm_rank_from_vertex_partition_id(
          major_comm_size, minor_comm_size, static_cast<int>(i));
      tx_counts[comm_rank_for_vertex_partition_id] = h_offsets[i + 1] - h_offsets[i];
      tx_displs[comm_rank_for_vertex_partition_id] = h_offsets[i];
    }
  }

  // 2. shuffle sorted unique internal vertices to the owning ranks

  std::vector<size_t> rx_counts(comm_size);
  std::vector<size_t> rx_displs(rx_counts.size());
  rmm::device_uvector<vertex_t> rx_int_vertices(0, handle.get_stream());
  {
    rmm::device_uvector<size_t> d_tx_counts(tx_counts.size(), handle.get_stream());
    raft::update_device(
      d_tx_counts.data(), tx_counts.data(), tx_counts.size(), handle.get_stream());
    rmm::device_uvector<size_t> d_rx_counts(0, handle.get_stream());
    std::vector<size_t> ones(comm_size, size_t{1});
    std::tie(d_rx_counts, std::ignore) =
      shuffle_values(comm,
                     d_tx_counts.begin(),
                     raft::host_span<size_t const>(ones.data(), ones.size()),
                     handle.get_stream());
    raft::update_host(
      rx_counts.data(), d_rx_counts.data(), d_rx_counts.size(), handle.get_stream());
    handle.sync_stream();
    std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_displs.begin(), size_t{0});
    rx_int_vertices.resize(rx_displs.back() + rx_counts.back(), handle.get_stream());
    device_multicast_sendrecv(comm,
                              collect_sorted_unique_int_vertices.begin(),
                              raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                              raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                              raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                              rx_int_vertices.begin(),
                              raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                              raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                              raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                              handle.get_stream());
  }

  // 3.Lookup return values

  auto value_buffer =
    allocate_dataframe_buffer<value_t>(rx_int_vertices.size(), handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    rx_int_vertices.begin(),
                    rx_int_vertices.end(),
                    get_dataframe_buffer_begin(value_buffer),
                    [local_value_first, local_vertex_partition_range_first] __device__(auto v) {
                      return local_value_first[v - local_vertex_partition_range_first];
                    });
  rx_int_vertices.resize(0, handle.get_stream());
  rx_int_vertices.shrink_to_fit(handle.get_stream());

  // 4. Shuffle results back to the original ranks

  auto rx_value_buffer =
    allocate_dataframe_buffer<value_t>(tx_displs.back() + tx_counts.back(), handle.get_stream());
  device_multicast_sendrecv(comm,
                            get_dataframe_buffer_begin(value_buffer),
                            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                            raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
                            get_dataframe_buffer_begin(rx_value_buffer),
                            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                            raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
                            handle.get_stream());

  return rx_value_buffer;
}

template <typename VertexIterator, typename ValueIterator>
dataframe_buffer_type_t<typename thrust::iterator_traits<ValueIterator>::value_type>
collect_values_for_int_vertices(
  raft::handle_t const& handle,
  VertexIterator collect_vertex_first,
  VertexIterator collect_vertex_last,
  ValueIterator local_value_first,
  raft::host_span<typename thrust::iterator_traits<VertexIterator>::value_type const>
    vertex_partition_range_lasts,
  typename thrust::iterator_traits<VertexIterator>::value_type local_vertex_partition_range_first)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;
  using value_t  = typename thrust::iterator_traits<ValueIterator>::value_type;

  size_t input_size = cuda::std::distance(collect_vertex_first, collect_vertex_last);

  rmm::device_uvector<vertex_t> sorted_unique_int_vertices(input_size, handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               collect_vertex_first,
               collect_vertex_last,
               sorted_unique_int_vertices.begin());
  thrust::sort(handle.get_thrust_policy(),
               sorted_unique_int_vertices.begin(),
               sorted_unique_int_vertices.end());
  auto last = thrust::unique(handle.get_thrust_policy(),
                             sorted_unique_int_vertices.begin(),
                             sorted_unique_int_vertices.end());
  sorted_unique_int_vertices.resize(cuda::std::distance(sorted_unique_int_vertices.begin(), last),
                                    handle.get_stream());

  auto tmp_value_buffer = collect_values_for_sorted_unique_int_vertices(
    handle,
    raft::device_span<vertex_t const>(sorted_unique_int_vertices.data(),
                                      sorted_unique_int_vertices.size()),
    local_value_first,
    vertex_partition_range_lasts,
    local_vertex_partition_range_first);

  kv_store_t<vertex_t, value_t, true> kv_map(std::move(sorted_unique_int_vertices),
                                             std::move(tmp_value_buffer),
                                             invalid_vertex_id<vertex_t>::value,
                                             true /* key_sorted */,
                                             handle.get_stream());
  auto kv_map_view  = kv_map.view();
  auto value_buffer = allocate_dataframe_buffer<value_t>(input_size, handle.get_stream());
  kv_map_view.find(collect_vertex_first,
                   collect_vertex_last,
                   get_dataframe_buffer_begin(value_buffer),
                   handle.get_stream());

  return value_buffer;
}

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::comms::comms_t const& comms,
                                         raft::device_span<T const> d_input)
{
  auto rx_sizes = cugraph::host_scalar_allgather(comms, d_input.size(), handle.get_stream());
  std::vector<size_t> rx_displs(static_cast<size_t>(comms.get_size()));
  std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);

  rmm::device_uvector<T> gathered_v(std::reduce(rx_sizes.begin(), rx_sizes.end()),
                                    handle.get_stream());

  cugraph::device_allgatherv(comms,
                             d_input.data(),
                             gathered_v.data(),
                             raft::host_span<size_t const>(rx_sizes.data(), rx_sizes.size()),
                             raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                             handle.get_stream());

  return gathered_v;
}

}  // namespace cugraph

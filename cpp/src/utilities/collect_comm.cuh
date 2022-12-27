/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <prims/kv_store.cuh>

#include <cugraph/graph.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
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

// for the keys in kv_store_view, key_to_gpu_id_op(key) should coincide with comm.get_rank()
template <typename KVStoreViewType, typename KeyIterator, typename KeyToGPUIdOp>
decltype(allocate_dataframe_buffer<typename KVStoreViewType::value_type>(0,
                                                                         rmm::cuda_stream_view{}))
collect_values_for_keys(raft::comms::comms_t const& comm,
                        KVStoreViewType kv_store_view,
                        KeyIterator collect_key_first,
                        KeyIterator collect_key_last,
                        KeyToGPUIdOp key_to_gpu_id_op,
                        rmm::cuda_stream_view stream_view)
{
  using key_t = typename KVStoreViewType::key_type;
  static_assert(std::is_same_v<typename thrust::iterator_traits<KeyIterator>::value_type, key_t>);
  using value_t = typename KVStoreViewType::value_type;

  // 1. collect values for the unique keys in [collect_key_first, collect_key_last)

  rmm::device_uvector<key_t> unique_keys(thrust::distance(collect_key_first, collect_key_last),
                                         stream_view);
  thrust::copy(
    rmm::exec_policy(stream_view), collect_key_first, collect_key_last, unique_keys.begin());
  thrust::sort(rmm::exec_policy(stream_view), unique_keys.begin(), unique_keys.end());
  unique_keys.resize(
    thrust::distance(
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
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      stream_view);

    auto values_for_rx_unique_keys =
      allocate_dataframe_buffer<value_t>(rx_unique_keys.size(), stream_view);

    kv_store_view.find(rx_unique_keys.begin(),
                       rx_unique_keys.end(),
                       get_dataframe_buffer_begin(values_for_rx_unique_keys),
                       stream_view);

    auto rx_values_for_unique_keys = allocate_dataframe_buffer<value_t>(0, stream_view);
    std::tie(rx_values_for_unique_keys, std::ignore) =
      shuffle_values(comm, values_for_rx_unique_keys.begin(), rx_value_counts, stream_view);

    values_for_unique_keys = std::move(rx_values_for_unique_keys);
  }

  // 2. build a kv_store_t object for the k, v pairs in unique_keys, values_for_unique_keys.

  kv_store_t<key_t, value_t, KVStoreViewType::binary_search> unique_key_value_store(stream_view);
  if constexpr (KVStoreViewType::binary_search) {
    unique_key_value_store = kv_store_t<key_t, value_t, true>(std::move(unique_keys),
                                                              std::move(values_for_unique_keys),
                                                              kv_store_view.invalid_value,
                                                              false,
                                                              stream_view);
  } else {
    unique_key_value_store =
      kv_store_t<key_t, value_t, false>(unique_keys.begin(),
                                        unique_keys.begin() + unique_keys.size(),
                                        get_dataframe_buffer_begin(values_for_unique_keys),
                                        kv_store_view.cuco_store->get_empty_key_sentinel(),
                                        kv_store_view.cuco_store->get_empty_value_sentinel(),
                                        stream_view);
  }
  auto unique_key_value_store_view = unique_key_value_store.view();

  // 3. find values for [collect_key_first, collect_key_last)

  auto value_buffer = allocate_dataframe_buffer<value_t>(
    thrust::distance(collect_key_first, collect_key_last), stream_view);
  unique_key_value_store_view.find(
    collect_key_first, collect_key_last, get_dataframe_buffer_begin(value_buffer), stream_view);

  return value_buffer;
}

// for the keys in kv_store_view, key_to_gpu_id_op(key) should coincide with comm.get_rank()
template <typename KVStoreViewType, typename KeyToGPUIdOp>
std::tuple<rmm::device_uvector<typename KVStoreViewType::key_type>,
           decltype(allocate_dataframe_buffer<typename KVStoreViewType::value_type>(
             0, cudaStream_t{nullptr}))>
collect_values_for_unique_keys(
  raft::comms::comms_t const& comm,
  KVStoreViewType kv_store_view,
  rmm::device_uvector<typename KVStoreViewType::key_type>&& collect_unique_keys,
  KeyToGPUIdOp key_to_gpu_id_op,
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
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      stream_view);
    auto values_for_rx_unique_keys =
      allocate_dataframe_buffer<value_t>(rx_unique_keys.size(), stream_view);
    kv_store_view.find(rx_unique_keys.begin(),
                       rx_unique_keys.end(),
                       get_dataframe_buffer_begin(values_for_rx_unique_keys),
                       stream_view);

    std::tie(values_for_collect_unique_keys, std::ignore) = shuffle_values(
      comm, get_dataframe_buffer_begin(values_for_rx_unique_keys), rx_value_counts, stream_view);
  }

  return std::make_tuple(std::move(collect_unique_keys), std::move(values_for_collect_unique_keys));
}

template <typename VertexIterator, typename ValueIterator>
decltype(allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
  0, cudaStream_t{nullptr}))
collect_values_for_sorted_unique_int_vertices(
  raft::comms::comms_t const& comm,
  VertexIterator collect_sorted_unique_vertex_first,
  VertexIterator collect_sorted_unique_vertex_last,
  ValueIterator local_value_first,
  std::vector<typename thrust::iterator_traits<VertexIterator>::value_type> const&
    vertex_partition_range_lasts,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;
  using value_t  = typename thrust::iterator_traits<ValueIterator>::value_type;

  // 1: Compute bounds of values

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               stream_view);
  rmm::device_uvector<size_t> d_local_counts(vertex_partition_range_lasts.size(), stream_view);

  raft::copy(d_vertex_partition_range_lasts.data(),
             vertex_partition_range_lasts.data(),
             vertex_partition_range_lasts.size(),
             stream_view);

  thrust::lower_bound(rmm::exec_policy(stream_view),
                      collect_sorted_unique_vertex_first,
                      collect_sorted_unique_vertex_last,
                      d_vertex_partition_range_lasts.begin(),
                      d_vertex_partition_range_lasts.end(),
                      d_local_counts.begin());

  thrust::adjacent_difference(rmm::exec_policy(stream_view),
                              d_local_counts.begin(),
                              d_local_counts.end(),
                              d_local_counts.begin());

  std::vector<size_t> h_local_counts(d_local_counts.size());

  raft::update_host(
    h_local_counts.data(), d_local_counts.data(), d_local_counts.size(), stream_view);

  // 2: Shuffle data

  auto [shuffled_vertices, shuffled_counts] =
    shuffle_values(comm, collect_sorted_unique_vertex_first, h_local_counts, stream_view);

  auto value_buffer = allocate_dataframe_buffer<value_t>(shuffled_vertices.size(), stream_view);

  // 3: Lookup return values

  thrust::transform(rmm::exec_policy(stream_view),
                    shuffled_vertices.begin(),
                    shuffled_vertices.end(),
                    value_buffer.begin(),
                    [local_value_first,
                     vertex_local_first =
                       (comm.get_rank() == 0
                          ? vertex_t{0}
                          : vertex_partition_range_lasts[comm.get_rank() - 1])] __device__(auto v) {
                      return local_value_first[v - vertex_local_first];
                    });

  // 4: Shuffle results back to original GPU

  std::tie(value_buffer, std::ignore) =
    shuffle_values(comm, value_buffer.begin(), shuffled_counts, stream_view);

  return value_buffer;
}

template <typename VertexIterator, typename ValueIterator>
decltype(allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
  0, cudaStream_t{nullptr}))
collect_values_for_int_vertices(
  raft::comms::comms_t const& comm,
  VertexIterator collect_vertex_first,
  VertexIterator collect_vertex_last,
  ValueIterator local_value_first,
  std::vector<typename thrust::iterator_traits<VertexIterator>::value_type> const&
    vertex_partition_range_lasts,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;
  using value_t  = typename thrust::iterator_traits<ValueIterator>::value_type;

  size_t input_size = thrust::distance(collect_vertex_first, collect_vertex_last);

  rmm::device_uvector<vertex_t> sorted_unique_vertices(input_size, stream_view);

  raft::copy(sorted_unique_vertices.data(), collect_vertex_first, input_size, stream_view);

  // FIXME:  It's possible that the input data might already be sorted and unique in
  //         which case we could skip these steps.
  thrust::sort(
    rmm::exec_policy(stream_view), sorted_unique_vertices.begin(), sorted_unique_vertices.end());
  auto last = thrust::unique(
    rmm::exec_policy(stream_view), sorted_unique_vertices.begin(), sorted_unique_vertices.end());
  sorted_unique_vertices.resize(thrust::distance(sorted_unique_vertices.begin(), last),
                                stream_view);

  auto tmp_value_buffer =
    collect_values_for_sorted_unique_int_vertices(comm,
                                                  sorted_unique_vertices.begin(),
                                                  sorted_unique_vertices.end(),
                                                  local_value_first,
                                                  vertex_partition_range_lasts,
                                                  stream_view);

  kv_store_t<vertex_t, value_t, true> kv_map(std::move(sorted_unique_vertices),
                                             std::move(tmp_value_buffer),
                                             invalid_vertex_id<vertex_t>::value,
                                             false,
                                             stream_view);
  auto device_view = detail::kv_binary_search_store_device_view_t(kv_map.view());

  auto value_buffer = allocate_dataframe_buffer<value_t>(input_size, stream_view);
  thrust::transform(rmm::exec_policy(stream_view),
                    collect_vertex_first,
                    collect_vertex_last,
                    value_buffer.begin(),
                    [device_view] __device__(auto v) { return device_view.find(v); });

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

  cugraph::device_allgatherv(
    comms, d_input.data(), gathered_v.data(), rx_sizes, rx_displs, handle.get_stream());

  return gathered_v;
}

}  // namespace cugraph

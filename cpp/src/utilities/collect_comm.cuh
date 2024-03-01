/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
collect_values_for_keys(raft::handle_t const& handle,
                        KVStoreViewType kv_store_view,
                        KeyIterator collect_key_first,
                        KeyIterator collect_key_last,
                        KeyToGPUIdOp key_to_gpu_id_op)
{
  using key_t = typename KVStoreViewType::key_type;
  static_assert(std::is_same_v<typename thrust::iterator_traits<KeyIterator>::value_type, key_t>);
  using value_t = typename KVStoreViewType::value_type;

  auto& comm = handle.get_comms();

  // 1. collect values for the unique keys in [collect_key_first, collect_key_last)

  rmm::device_uvector<key_t> unique_keys(thrust::distance(collect_key_first, collect_key_last),
                                         handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), collect_key_first, collect_key_last, unique_keys.begin());
  thrust::sort(handle.get_thrust_policy(), unique_keys.begin(), unique_keys.end());
  unique_keys.resize(
    thrust::distance(
      unique_keys.begin(),
      thrust::unique(handle.get_thrust_policy(), unique_keys.begin(), unique_keys.end())),
    handle.get_stream());

  auto values_for_unique_keys = allocate_dataframe_buffer<value_t>(0, handle.get_stream());
  {
    rmm::device_uvector<key_t> rx_unique_keys(0, handle.get_stream());
    std::vector<size_t> rx_value_counts{};
    std::tie(rx_unique_keys, rx_value_counts) = groupby_gpu_id_and_shuffle_values(
      comm,
      unique_keys.begin(),
      unique_keys.end(),
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      handle.get_stream());

    auto values_for_rx_unique_keys =
      allocate_dataframe_buffer<value_t>(rx_unique_keys.size(), handle.get_stream());

    kv_store_view.find(rx_unique_keys.begin(),
                       rx_unique_keys.end(),
                       get_dataframe_buffer_begin(values_for_rx_unique_keys),
                       handle.get_stream());

    auto rx_values_for_unique_keys = allocate_dataframe_buffer<value_t>(0, handle.get_stream());
    std::tie(rx_values_for_unique_keys, std::ignore) =
      shuffle_values(comm, values_for_rx_unique_keys.begin(), rx_value_counts, handle.get_stream());

    values_for_unique_keys = std::move(rx_values_for_unique_keys);
  }

  // 2. build a kv_store_t object for the k, v pairs in unique_keys, values_for_unique_keys.

  kv_store_t<key_t, value_t, KVStoreViewType::binary_search> unique_key_value_store(
    handle.get_stream());
  if constexpr (KVStoreViewType::binary_search) {
    unique_key_value_store = kv_store_t<key_t, value_t, true>(std::move(unique_keys),
                                                              std::move(values_for_unique_keys),
                                                              kv_store_view.invalid_value(),
                                                              false,
                                                              handle.get_stream());
  } else {
    auto kv_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(unique_keys.begin(), get_dataframe_buffer_begin(values_for_unique_keys)));
    auto valid_kv_pair_last =
      thrust::remove_if(handle.get_thrust_policy(),
                        kv_pair_first,
                        kv_pair_first + unique_keys.size(),
                        [invalid_value = kv_store_view.invalid_value()] __device__(auto pair) {
                          return thrust::get<1>(pair) == invalid_value;
                        });  // remove (k,v) pairs with unmatched keys (it is invalid to insert a
                             // (k,v) pair with v = empty_key_sentinel)
    auto num_valid_pairs = static_cast<size_t>(thrust::distance(kv_pair_first, valid_kv_pair_last));
    unique_key_value_store =
      kv_store_t<key_t, value_t, false>(unique_keys.begin(),
                                        unique_keys.begin() + num_valid_pairs,
                                        get_dataframe_buffer_begin(values_for_unique_keys),
                                        kv_store_view.invalid_key(),
                                        kv_store_view.invalid_value(),
                                        handle.get_stream());

    unique_keys.resize(0, handle.get_stream());
    values_for_unique_keys.resize(0, handle.get_stream());
    unique_keys.shrink_to_fit(handle.get_stream());
    values_for_unique_keys.shrink_to_fit(handle.get_stream());
  }
  auto unique_key_value_store_view = unique_key_value_store.view();

  // 3. find values for [collect_key_first, collect_key_last)

  auto value_buffer = allocate_dataframe_buffer<value_t>(
    thrust::distance(collect_key_first, collect_key_last), handle.get_stream());
  unique_key_value_store_view.find(collect_key_first,
                                   collect_key_last,
                                   get_dataframe_buffer_begin(value_buffer),
                                   handle.get_stream());

  return value_buffer;
}

// for the keys in kv_store_view, key_to_gpu_id_op(key) should coincide with comm.get_rank()
template <typename KVStoreViewType, typename KeyToGPUIdOp>
std::tuple<rmm::device_uvector<typename KVStoreViewType::key_type>,
           decltype(allocate_dataframe_buffer<typename KVStoreViewType::value_type>(
             0, cudaStream_t{nullptr}))>
collect_values_for_unique_keys(
  raft::handle_t const& handle,
  KVStoreViewType kv_store_view,
  rmm::device_uvector<typename KVStoreViewType::key_type>&& collect_unique_keys,
  KeyToGPUIdOp key_to_gpu_id_op)
{
  using key_t   = typename KVStoreViewType::key_type;
  using value_t = typename KVStoreViewType::value_type;

  auto& comm = handle.get_comms();

  auto values_for_collect_unique_keys = allocate_dataframe_buffer<value_t>(0, handle.get_stream());
  {
    auto [rx_unique_keys, rx_value_counts] = groupby_gpu_id_and_shuffle_values(
      comm,
      collect_unique_keys.begin(),
      collect_unique_keys.end(),
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      handle.get_stream());
    auto values_for_rx_unique_keys =
      allocate_dataframe_buffer<value_t>(rx_unique_keys.size(), handle.get_stream());
    kv_store_view.find(rx_unique_keys.begin(),
                       rx_unique_keys.end(),
                       get_dataframe_buffer_begin(values_for_rx_unique_keys),
                       handle.get_stream());

    std::tie(values_for_collect_unique_keys, std::ignore) =
      shuffle_values(comm,
                     get_dataframe_buffer_begin(values_for_rx_unique_keys),
                     rx_value_counts,
                     handle.get_stream());
  }

  return std::make_tuple(std::move(collect_unique_keys), std::move(values_for_collect_unique_keys));
}

template <typename vertex_t, typename ValueIterator>
std::tuple<
  rmm::device_uvector<vertex_t>,
  decltype(allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
    0, cudaStream_t{nullptr}))>
collect_values_for_unique_int_vertices(raft::handle_t const& handle,
                                       rmm::device_uvector<vertex_t>&& collect_unique_int_vertices,
                                       ValueIterator local_value_first,
                                       std::vector<vertex_t> const& vertex_partition_range_lasts)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  auto& comm                 = handle.get_comms();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto const major_comm_rank = major_comm.get_rank();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();
  auto const minor_comm_rank = minor_comm.get_rank();

  // 1. groupby and shuffle internal vertices

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      handle.get_stream());

  auto [rx_int_vertices, rx_int_vertex_counts] = groupby_gpu_id_and_shuffle_values(
    comm,
    collect_unique_int_vertices.begin(),
    collect_unique_int_vertices.end(),
    detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size},
    handle.get_stream());

  // 2: Lookup return values

  auto vertex_partition_id =
    partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
      major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);
  auto local_int_vertex_first =
    vertex_partition_id == 0 ? vertex_t{0} : vertex_partition_range_lasts[vertex_partition_id - 1];

  auto value_buffer =
    allocate_dataframe_buffer<value_t>(rx_int_vertices.size(), handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    rx_int_vertices.begin(),
                    rx_int_vertices.end(),
                    value_buffer.begin(),
                    [local_value_first, local_int_vertex_first] __device__(auto v) {
                      return local_value_first[v - local_int_vertex_first];
                    });

  // 3: Shuffle results back to original GPU

  std::tie(value_buffer, std::ignore) =
    shuffle_values(comm, value_buffer.begin(), rx_int_vertex_counts, handle.get_stream());

  return std::make_tuple(std::move(collect_unique_int_vertices), std::move(value_buffer));
}

template <typename VertexIterator, typename ValueIterator>
decltype(allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
  0, cudaStream_t{nullptr}))
collect_values_for_int_vertices(
  raft::handle_t const& handle,
  VertexIterator collect_vertex_first,
  VertexIterator collect_vertex_last,
  ValueIterator local_value_first,
  std::vector<typename thrust::iterator_traits<VertexIterator>::value_type> const&
    vertex_partition_range_lasts)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;
  using value_t  = typename thrust::iterator_traits<ValueIterator>::value_type;

  size_t input_size = thrust::distance(collect_vertex_first, collect_vertex_last);

  rmm::device_uvector<vertex_t> sorted_unique_int_vertices(input_size, handle.get_stream());

  raft::copy(
    sorted_unique_int_vertices.data(), collect_vertex_first, input_size, handle.get_stream());

  thrust::sort(handle.get_thrust_policy(),
               sorted_unique_int_vertices.begin(),
               sorted_unique_int_vertices.end());
  auto last = thrust::unique(handle.get_thrust_policy(),
                             sorted_unique_int_vertices.begin(),
                             sorted_unique_int_vertices.end());
  sorted_unique_int_vertices.resize(thrust::distance(sorted_unique_int_vertices.begin(), last),
                                    handle.get_stream());

  auto [unique_int_vertices, tmp_value_buffer] = collect_values_for_unique_int_vertices(
    handle, std::move(sorted_unique_int_vertices), local_value_first, vertex_partition_range_lasts);

  kv_store_t<vertex_t, value_t, true> kv_map(std::move(unique_int_vertices),
                                             std::move(tmp_value_buffer),
                                             invalid_vertex_id<vertex_t>::value,
                                             false,
                                             handle.get_stream());
  auto device_view = detail::kv_binary_search_store_device_view_t(kv_map.view());

  auto value_buffer = allocate_dataframe_buffer<value_t>(input_size, handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
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

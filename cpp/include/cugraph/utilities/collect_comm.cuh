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

#include <cugraph/graph.hpp>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuco/static_map.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>

#include <iterator>
#include <memory>
#include <vector>

namespace cugraph {

// for key = [map_key_first, map_key_last), key_to_gpu_id_op(key) should be coincide with
// comm.get_rank()
template <typename VertexIterator0,
          typename VertexIterator1,
          typename ValueIterator,
          typename KeyToGPUIdOp>
decltype(allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
  0, cudaStream_t{nullptr}))
collect_values_for_keys(raft::comms::comms_t const& comm,
                        VertexIterator0 map_key_first,
                        VertexIterator0 map_key_last,
                        ValueIterator map_value_first,
                        VertexIterator1 collect_key_first,
                        VertexIterator1 collect_key_last,
                        KeyToGPUIdOp key_to_gpu_id_op,
                        rmm::cuda_stream_view stream_view)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator0>::value_type;
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<VertexIterator1>::value_type, vertex_t>);
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  double constexpr load_factor = 0.7;

  // FIXME: we may compare the performance & memory footprint of this hash based approach vs binary
  // search based approach (especially when thrust::distance(collect_key_first, collect_key_last) <<
  // thrust::distance(map_key_first, map_key_last)

  // 1. build a cuco::static_map object for the map k, v pairs.

  auto poly_alloc = rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
  auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(poly_alloc, stream_view);
  auto kv_map_ptr     = std::make_unique<
    cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>>(
    // cuco::static_map requires at least one empty slot
    std::max(static_cast<size_t>(
               static_cast<double>(thrust::distance(map_key_first, map_key_last)) / load_factor),
             static_cast<size_t>(thrust::distance(map_key_first, map_key_last)) + 1),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value,
    stream_adapter,
    stream_view);
  {
    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(map_key_first, map_value_first));
    kv_map_ptr->insert(pair_first,
                       pair_first + thrust::distance(map_key_first, map_key_last),
                       cuco::detail::MurmurHash3_32<vertex_t>{},
                       thrust::equal_to<vertex_t>{},
                       stream_view);
  }

  // 2. collect values for the unique keys in [collect_key_first, collect_key_last)

  rmm::device_uvector<vertex_t> unique_keys(thrust::distance(collect_key_first, collect_key_last),
                                            stream_view);
  thrust::copy(
    rmm::exec_policy(stream_view), collect_key_first, collect_key_last, unique_keys.begin());
  thrust::sort(rmm::exec_policy(stream_view), unique_keys.begin(), unique_keys.end());
  unique_keys.resize(
    thrust::distance(
      unique_keys.begin(),
      thrust::unique(rmm::exec_policy(stream_view), unique_keys.begin(), unique_keys.end())),
    stream_view);

  rmm::device_uvector<value_t> values_for_unique_keys(0, stream_view);
  {
    rmm::device_uvector<vertex_t> rx_unique_keys(0, stream_view);
    std::vector<size_t> rx_value_counts{};
    std::tie(rx_unique_keys, rx_value_counts) = groupby_gpu_id_and_shuffle_values(
      comm,
      unique_keys.begin(),
      unique_keys.end(),
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      stream_view);

    rmm::device_uvector<value_t> values_for_rx_unique_keys(rx_unique_keys.size(), stream_view);

    kv_map_ptr->find(rx_unique_keys.begin(),
                     rx_unique_keys.end(),
                     values_for_rx_unique_keys.begin(),
                     cuco::detail::MurmurHash3_32<vertex_t>{},
                     thrust::equal_to<vertex_t>{},
                     stream_view);

    rmm::device_uvector<value_t> rx_values_for_unique_keys(0, stream_view);
    std::tie(rx_values_for_unique_keys, std::ignore) =
      shuffle_values(comm, values_for_rx_unique_keys.begin(), rx_value_counts, stream_view);

    values_for_unique_keys = std::move(rx_values_for_unique_keys);
  }

  // 3. re-build a cuco::static_map object for the k, v pairs in unique_keys,
  // values_for_unique_keys.

  kv_map_ptr.reset();

  kv_map_ptr = std::make_unique<
    cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>>(
    // cuco::static_map requires at least one empty slot
    std::max(static_cast<size_t>(static_cast<double>(unique_keys.size()) / load_factor),
             unique_keys.size() + 1),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value,
    stream_adapter,
    stream_view);
  {
    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(unique_keys.begin(), values_for_unique_keys.begin()));
    kv_map_ptr->insert(pair_first,
                       pair_first + unique_keys.size(),
                       cuco::detail::MurmurHash3_32<vertex_t>{},
                       thrust::equal_to<vertex_t>{},
                       stream_view);
  }

  // 4. find values for [collect_key_first, collect_key_last)

  auto value_buffer = allocate_dataframe_buffer<value_t>(
    thrust::distance(collect_key_first, collect_key_last), stream_view);
  kv_map_ptr->find(collect_key_first,
                   collect_key_last,
                   get_dataframe_buffer_begin(value_buffer),
                   cuco::detail::MurmurHash3_32<vertex_t>{},
                   thrust::equal_to<vertex_t>{},
                   stream_view);

  return value_buffer;
}

// for the keys stored in kv_map, key_to_gpu_id_op(key) should be coincide with comm.get_rank()
template <typename vertex_t, typename value_t, typename MapAllocator, typename KeyToGPUIdOp>
std::tuple<rmm::device_uvector<vertex_t>,
           decltype(allocate_dataframe_buffer<value_t>(0, cudaStream_t{nullptr}))>
collect_values_for_unique_keys(
  raft::comms::comms_t const& comm,
  cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, MapAllocator>& kv_map,
  rmm::device_uvector<vertex_t>&& collect_unique_keys,
  KeyToGPUIdOp key_to_gpu_id_op,
  rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<value_t> values_for_collect_unique_keys(0, stream_view);
  {
    auto [rx_unique_keys, rx_value_counts] = groupby_gpu_id_and_shuffle_values(
      comm,
      collect_unique_keys.begin(),
      collect_unique_keys.end(),
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      stream_view);
    rmm::device_uvector<value_t> values_for_rx_unique_keys(rx_unique_keys.size(), stream_view);
    kv_map.find(rx_unique_keys.begin(),
                rx_unique_keys.end(),
                values_for_rx_unique_keys.begin(),
                cuco::detail::MurmurHash3_32<vertex_t>{},
                thrust::equal_to<vertex_t>{},
                stream_view);

    std::tie(values_for_collect_unique_keys, std::ignore) =
      shuffle_values(comm, values_for_rx_unique_keys.begin(), rx_value_counts, stream_view);
  }

  return std::make_tuple(std::move(collect_unique_keys), std::move(values_for_collect_unique_keys));
}

// for key = [map_key_first, map_key_last), key_to_gpu_id_op(key) should be coincide with
// comm.get_rank()
template <typename VertexIterator, typename ValueIterator, typename KeyToGPUIdOp>
std::tuple<
  rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type>,
  decltype(allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
    0, cudaStream_t{nullptr}))>
collect_values_for_unique_keys(
  raft::comms::comms_t const& comm,
  VertexIterator map_key_first,
  VertexIterator map_key_last,
  ValueIterator map_value_first,
  rmm::device_uvector<typename thrust::iterator_traits<VertexIterator>::value_type>&&
    collect_unique_keys,
  KeyToGPUIdOp key_to_gpu_id_op,
  rmm::cuda_stream_view stream_view)
{
  using vertex_t = typename thrust::iterator_traits<VertexIterator>::value_type;
  using value_t  = typename thrust::iterator_traits<ValueIterator>::value_type;

  double constexpr load_factor = 0.7;

  // FIXME: we may compare the performance & memory footprint of this hash based approach vs binary
  // search based approach (especially when thrust::distance(collect_unique_key_first,
  // collect_unique_key_last) << thrust::distance(map_key_first, map_key_last)

  // 1. build a cuco::static_map object for the map k, v pairs.

  auto poly_alloc = rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
  auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(poly_alloc, stream_view);
  cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)> kv_map(
    // cuco::static_map requires at least one empty slot
    std::max(static_cast<size_t>(
               static_cast<double>(thrust::distance(map_key_first, map_key_last)) / load_factor),
             static_cast<size_t>(thrust::distance(map_key_first, map_key_last)) + 1),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value,
    stream_adapter,
    stream_view);
  {
    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(map_key_first, map_value_first));
    kv_map.insert(pair_first,
                  pair_first + thrust::distance(map_key_first, map_key_last),
                  cuco::detail::MurmurHash3_32<vertex_t>{},
                  thrust::equal_to<vertex_t>{},
                  stream_view);
  }

  // 2. collect values for the unique keys in collect_unique_keys (elements in collect_unique_keys
  // will be shuffled while colledting the values)

  return collect_values_for_unique_keys<vertex_t, value_t, decltype(stream_adapter), KeyToGPUIdOp>(
    comm, kv_map, std::move(collect_unique_keys), key_to_gpu_id_op, stream_view);
}

template <typename vertex_t, typename ValueIterator>
decltype(allocate_dataframe_buffer<typename thrust::iterator_traits<ValueIterator>::value_type>(
  0, cudaStream_t{nullptr}))
collect_values_for_sorted_unique_vertices(raft::comms::comms_t const& comm,
                                          vertex_t const* collect_unique_vertex_first,
                                          vertex_t num_vertices,
                                          ValueIterator local_value_first,
                                          std::vector<vertex_t> const& vertex_partition_range_lasts,
                                          rmm::cuda_stream_view stream_view)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  // 1: Compute bounds of values
  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               stream_view);
  rmm::device_uvector<size_t> d_local_counts(vertex_partition_range_lasts.size(), stream_view);

  raft::copy(d_vertex_partition_range_lasts.data(),
             vertex_partition_range_lasts.data(),
             vertex_partition_range_lasts.size(),
             stream_view);

  thrust::lower_bound(rmm::exec_policy(stream_view),
                      collect_unique_vertex_first,
                      collect_unique_vertex_first + num_vertices,
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
    shuffle_values(comm, collect_unique_vertex_first, h_local_counts, stream_view);

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
collect_values_for_vertices(
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

  rmm::device_uvector<vertex_t> input_vertices(input_size, stream_view);
  auto value_buffer = allocate_dataframe_buffer<value_t>(input_size, stream_view);

  raft::copy(input_vertices.data(), collect_vertex_first, input_size, stream_view);

  // FIXME:  It's possible that the input data might already be sorted and unique in
  //         which case we could skip these steps.
  thrust::sort(rmm::exec_policy(stream_view), input_vertices.begin(), input_vertices.end());
  auto input_end =
    thrust::unique(rmm::exec_policy(stream_view), input_vertices.begin(), input_vertices.end());
  input_vertices.resize(thrust::distance(input_vertices.begin(), input_end), stream_view);

  auto tmp_value_buffer =
    collect_values_for_sorted_unique_vertices(comm,
                                              input_vertices.data(),
                                              static_cast<vertex_t>(input_vertices.size()),
                                              local_value_first,
                                              vertex_partition_range_lasts,
                                              stream_view);

  thrust::transform(rmm::exec_policy(stream_view),
                    collect_vertex_first,
                    collect_vertex_last,
                    value_buffer.begin(),
                    [num_vertices = input_vertices.size(),
                     d_vertices   = input_vertices.data(),
                     d_values     = tmp_value_buffer.data()] __device__(auto v) {
                      auto pos =
                        thrust::find(thrust::seq, d_vertices, d_vertices + num_vertices, v);
                      auto offset = thrust::distance(d_vertices, pos);

                      return d_values[offset];
                    });

  return value_buffer;
}

}  // namespace cugraph

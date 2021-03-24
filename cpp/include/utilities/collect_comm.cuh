/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <experimental/graph.hpp>
#include <utilities/dataframe_buffer.cuh>
#include <utilities/shuffle_comm.cuh>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <cuco/static_map.cuh>

#include <iterator>
#include <memory>
#include <vector>

namespace cugraph {
namespace experimental {

// for key = [map_key_first, map_key_last), key_to_gpu_id_op(key) should be coincide with
// comm.get_rank()
template <typename VertexIterator0,
          typename VertexIterator1,
          typename ValueIterator,
          typename KeyToGPUIdOp>
decltype(allocate_dataframe_buffer<typename std::iterator_traits<ValueIterator>::value_type>(
  0, cudaStream_t{nullptr}))
collect_values_for_keys(raft::comms::comms_t const &comm,
                        VertexIterator0 map_key_first,
                        VertexIterator0 map_key_last,
                        ValueIterator map_value_first,
                        VertexIterator1 collect_key_first,
                        VertexIterator1 collect_key_last,
                        KeyToGPUIdOp key_to_gpu_id_op,
                        cudaStream_t stream)
{
  using vertex_t = typename std::iterator_traits<VertexIterator0>::value_type;
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator1>::value_type, vertex_t>::value);
  using value_t = typename std::iterator_traits<ValueIterator>::value_type;

  double constexpr load_factor = 0.7;

  // FIXME: we may compare the performance & memory footprint of this hash based approach vs binary
  // search based approach (especially when thrust::distance(collect_key_first, collect_key_last) <<
  // thrust::distance(map_key_first, map_key_last)

  // 1. build a cuco::static_map object for the map k, v pairs.

  auto kv_map_ptr = std::make_unique<cuco::static_map<vertex_t, value_t>>(
    // FIXME: std::max(..., size_t{1}) as a temporary workaround for
    // https://github.com/NVIDIA/cuCollections/issues/72
    std::max(static_cast<size_t>(
               static_cast<double>(thrust::distance(map_key_first, map_key_last)) / load_factor),
             size_t{1}),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value);
  {
    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(map_key_first, map_value_first)),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });
    kv_map_ptr->insert(pair_first, pair_first + thrust::distance(map_key_first, map_key_last));
  }

  // 2. collect values for the unique keys in [collect_key_first, collect_key_last)

  rmm::device_uvector<vertex_t> unique_keys(thrust::distance(collect_key_first, collect_key_last),
                                            stream);
  thrust::copy(
    rmm::exec_policy(stream)->on(stream), collect_key_first, collect_key_last, unique_keys.begin());
  thrust::sort(rmm::exec_policy(stream)->on(stream), unique_keys.begin(), unique_keys.end());
  unique_keys.resize(
    thrust::distance(
      unique_keys.begin(),
      thrust::unique(rmm::exec_policy(stream)->on(stream), unique_keys.begin(), unique_keys.end())),
    stream);

  rmm::device_uvector<value_t> values_for_unique_keys(0, stream);
  {
    rmm::device_uvector<vertex_t> rx_unique_keys(0, stream);
    std::vector<size_t> rx_value_counts{};
    std::tie(rx_unique_keys, rx_value_counts) = groupby_gpuid_and_shuffle_values(
      comm,
      unique_keys.begin(),
      unique_keys.end(),
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      stream);

    rmm::device_uvector<value_t> values_for_rx_unique_keys(rx_unique_keys.size(), stream);

    CUDA_TRY(cudaStreamSynchronize(stream));  // cuco::static_map currently does not take stream

    kv_map_ptr->find(
      rx_unique_keys.begin(), rx_unique_keys.end(), values_for_rx_unique_keys.begin());

    rmm::device_uvector<value_t> rx_values_for_unique_keys(0, stream);
    std::tie(rx_values_for_unique_keys, std::ignore) =
      shuffle_values(comm, values_for_rx_unique_keys.begin(), rx_value_counts, stream);

    values_for_unique_keys = std::move(rx_values_for_unique_keys);
  }

  // 3. re-build a cuco::static_map object for the k, v pairs in unique_keys,
  // values_for_unique_keys.

  CUDA_TRY(cudaStreamSynchronize(stream));  // cuco::static_map currently does not take stream

  kv_map_ptr.reset();

  kv_map_ptr = std::make_unique<cuco::static_map<vertex_t, value_t>>(
    // FIXME: std::max(..., size_t{1}) as a temporary workaround for
    // https://github.com/NVIDIA/cuCollections/issues/72
    std::max(static_cast<size_t>(static_cast<double>(unique_keys.size()) / load_factor), size_t{1}),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value);
  {
    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(unique_keys.begin(), values_for_unique_keys.begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });

    kv_map_ptr->insert(pair_first, pair_first + unique_keys.size());
  }

  // 4. find values for [collect_key_first, collect_key_last)

  auto value_buffer = allocate_dataframe_buffer<value_t>(
    thrust::distance(collect_key_first, collect_key_last), stream);
  kv_map_ptr->find(
    collect_key_first, collect_key_last, get_dataframe_buffer_begin<value_t>(value_buffer));

  return value_buffer;
}

// for key = [map_key_first, map_key_last), key_to_gpu_id_op(key) should be coincide with
// comm.get_rank()
template <typename VertexIterator0,
          typename VertexIterator1,
          typename ValueIterator,
          typename KeyToGPUIdOp>
decltype(allocate_dataframe_buffer<typename std::iterator_traits<ValueIterator>::value_type>(
  0, cudaStream_t{nullptr}))
collect_values_for_unique_keys(raft::comms::comms_t const &comm,
                               VertexIterator0 map_key_first,
                               VertexIterator0 map_key_last,
                               ValueIterator map_value_first,
                               VertexIterator1 collect_unique_key_first,
                               VertexIterator1 collect_unique_key_last,
                               KeyToGPUIdOp key_to_gpu_id_op,
                               cudaStream_t stream)
{
  using vertex_t = typename std::iterator_traits<VertexIterator0>::value_type;
  static_assert(
    std::is_same<typename std::iterator_traits<VertexIterator1>::value_type, vertex_t>::value);
  using value_t = typename std::iterator_traits<ValueIterator>::value_type;

  double constexpr load_factor = 0.7;

  // FIXME: we may compare the performance & memory footprint of this hash based approach vs binary
  // search based approach (especially when thrust::distance(collect_unique_key_first,
  // collect_unique_key_last) << thrust::distance(map_key_first, map_key_last)

  // 1. build a cuco::static_map object for the map k, v pairs.

  auto kv_map_ptr = std::make_unique<cuco::static_map<vertex_t, value_t>>(
    // FIXME: std::max(..., size_t{1}) as a temporary workaround for
    // https://github.com/NVIDIA/cuCollections/issues/72
    std::max(static_cast<size_t>(
               static_cast<double>(thrust::distance(map_key_first, map_key_last)) / load_factor),
             size_t{1}),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value);
  {
    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(map_key_first, map_value_first)),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });
    kv_map_ptr->insert(pair_first, pair_first + thrust::distance(map_key_first, map_key_last));
  }

  // 2. collect values for the unique keys in [collect_unique_key_first, collect_unique_key_last)

  rmm::device_uvector<vertex_t> unique_keys(
    thrust::distance(collect_unique_key_first, collect_unique_key_last), stream);
  thrust::copy(rmm::exec_policy(stream)->on(stream),
               collect_unique_key_first,
               collect_unique_key_last,
               unique_keys.begin());

  rmm::device_uvector<value_t> values_for_unique_keys(0, stream);
  {
    rmm::device_uvector<vertex_t> rx_unique_keys(0, stream);
    std::vector<size_t> rx_value_counts{};
    std::tie(rx_unique_keys, rx_value_counts) = groupby_gpuid_and_shuffle_values(
      comm,
      unique_keys.begin(),
      unique_keys.end(),
      [key_to_gpu_id_op] __device__(auto val) { return key_to_gpu_id_op(val); },
      stream);

    rmm::device_uvector<value_t> values_for_rx_unique_keys(rx_unique_keys.size(), stream);

    CUDA_TRY(cudaStreamSynchronize(stream));  // cuco::static_map currently does not take stream

    kv_map_ptr->find(
      rx_unique_keys.begin(), rx_unique_keys.end(), values_for_rx_unique_keys.begin());

    rmm::device_uvector<value_t> rx_values_for_unique_keys(0, stream);
    std::tie(rx_values_for_unique_keys, std::ignore) =
      shuffle_values(comm, values_for_rx_unique_keys.begin(), rx_value_counts, stream);

    values_for_unique_keys = std::move(rx_values_for_unique_keys);
  }

  // 3. re-build a cuco::static_map object for the k, v pairs in unique_keys,
  // values_for_unique_keys.

  CUDA_TRY(cudaStreamSynchronize(stream));  // cuco::static_map currently does not take stream

  kv_map_ptr.reset();

  kv_map_ptr = std::make_unique<cuco::static_map<vertex_t, value_t>>(
    // FIXME: std::max(..., size_t{1}) as a temporary workaround for
    // https://github.com/NVIDIA/cuCollections/issues/72
    std::max(static_cast<size_t>(static_cast<double>(unique_keys.size()) / load_factor), size_t{1}),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value);
  {
    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(unique_keys.begin(), values_for_unique_keys.begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });

    kv_map_ptr->insert(pair_first, pair_first + unique_keys.size());
  }

  // 4. find values for [collect_unique_key_first, collect_unique_key_last)

  auto value_buffer = allocate_dataframe_buffer<value_t>(
    thrust::distance(collect_unique_key_first, collect_unique_key_last), stream);
  kv_map_ptr->find(collect_unique_key_first,
                   collect_unique_key_last,
                   get_dataframe_buffer_begin<value_t>(value_buffer));

  return value_buffer;
}

}  // namespace experimental
}  // namespace cugraph

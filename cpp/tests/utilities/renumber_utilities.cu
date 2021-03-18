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

#include <utilities/renumber_utilities.hpp>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/sort.h>

namespace cugraph {
namespace test {

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>> unrenumber_kv_pairs(
  raft::handle_t const& handle,
  vertex_t const* keys /* 0 <= keys[] < renumber_map_size */,
  value_t const* values,
  size_t num_pairs,
  vertex_t const* renumber_map_labels,
  size_t renumber_map_size)
{
  rmm::device_uvector<vertex_t> unrenumbered_keys(num_pairs, handle.get_stream_view());
  rmm::device_uvector<value_t> values_for_unrenumbered_keys(num_pairs, handle.get_stream_view());

  auto unrenumbered_key_first = thrust::make_transform_iterator(
    keys, [renumber_map_labels] __device__(auto v) { return renumber_map_labels[v]; });
  thrust::copy(rmm::exec_policy(handle.get_stream_view()),
               unrenumbered_key_first,
               unrenumbered_key_first + num_pairs,
               unrenumbered_keys.begin());
  thrust::copy(rmm::exec_policy(handle.get_stream_view()),
               values,
               values + num_pairs,
               values_for_unrenumbered_keys.begin());

  thrust::sort_by_key(rmm::exec_policy(handle.get_stream_view()),
                      unrenumbered_keys.begin(),
                      unrenumbered_keys.end(),
                      values_for_unrenumbered_keys.begin());

  return std::make_tuple(std::move(unrenumbered_keys), std::move(values_for_unrenumbered_keys));
}

template <typename vertex_t, typename value_t>
rmm::device_uvector<value_t> sort_values_by_key(raft::handle_t const& handle,
                                                vertex_t const* keys,
                                                value_t const* values,
                                                size_t num_pairs)
{
  rmm::device_uvector<vertex_t> sorted_keys(num_pairs, handle.get_stream_view());
  rmm::device_uvector<value_t> sorted_values(num_pairs, handle.get_stream_view());

  thrust::copy(
    rmm::exec_policy(handle.get_stream_view()), keys, keys + num_pairs, sorted_keys.begin());
  thrust::copy(
    rmm::exec_policy(handle.get_stream_view()), values, values + num_pairs, sorted_values.begin());

  thrust::sort_by_key(rmm::exec_policy(handle.get_stream_view()),
                      sorted_keys.begin(),
                      sorted_keys.end(),
                      sorted_values.begin());

  return sorted_values;
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
unrenumber_kv_pairs<int32_t, float>(raft::handle_t const& handle,
                                    int32_t const* keys,
                                    float const* values,
                                    size_t num_pairs,
                                    int32_t const* renumber_map_labels,
                                    size_t renumber_map_size);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
unrenumber_kv_pairs<int32_t, double>(raft::handle_t const& handle,
                                     int32_t const* keys,
                                     double const* values,
                                     size_t num_pairs,
                                     int32_t const* renumber_map_labels,
                                     size_t renumber_map_size);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
unrenumber_kv_pairs<int64_t, float>(raft::handle_t const& handle,
                                    int64_t const* keys,
                                    float const* values,
                                    size_t num_pairs,
                                    int64_t const* renumber_map_labels,
                                    size_t renumber_map_size);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
unrenumber_kv_pairs<int64_t, double>(raft::handle_t const& handle,
                                     int64_t const* keys,
                                     double const* values,
                                     size_t num_pairs,
                                     int64_t const* renumber_map_labels,
                                     size_t renumber_map_size);

template rmm::device_uvector<float> sort_values_by_key<int32_t, float>(raft::handle_t const& handle,
                                                                       int32_t const* keys,
                                                                       float const* values,
                                                                       size_t num_pairs);

template rmm::device_uvector<double> sort_values_by_key<int32_t, double>(
  raft::handle_t const& handle, int32_t const* keys, double const* values, size_t num_pairs);

template rmm::device_uvector<int32_t> sort_values_by_key<int32_t, int32_t>(
  raft::handle_t const& handle, int32_t const* keys, int32_t const* values, size_t num_pairs);

template rmm::device_uvector<float> sort_values_by_key<int64_t, float>(raft::handle_t const& handle,
                                                                       int64_t const* keys,
                                                                       float const* values,
                                                                       size_t num_pairs);

template rmm::device_uvector<double> sort_values_by_key<int64_t, double>(
  raft::handle_t const& handle, int64_t const* keys, double const* values, size_t num_pairs);

template rmm::device_uvector<int64_t> sort_values_by_key<int64_t, int64_t>(
  raft::handle_t const& handle, int64_t const* keys, int64_t const* values, size_t num_pairs);

}  // namespace test
}  // namespace cugraph

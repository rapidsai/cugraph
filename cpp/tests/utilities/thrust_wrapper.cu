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

#include "utilities/thrust_wrapper.hpp"

#include <cugraph/utilities/dataframe_buffer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace cugraph {
namespace test {

template <typename value_buffer_type>
value_buffer_type sort(raft::handle_t const& handle, value_buffer_type const& values)
{
  auto sorted_values =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<value_buffer_type>>(
      values.size(), handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               cugraph::get_dataframe_buffer_begin(values),
               cugraph::get_dataframe_buffer_end(values),
               cugraph::get_dataframe_buffer_begin(sorted_values));

  thrust::sort(handle.get_thrust_policy(),
               cugraph::get_dataframe_buffer_begin(sorted_values),
               cugraph::get_dataframe_buffer_end(sorted_values));

  return sorted_values;
}

template <typename value_buffer_type>
std::tuple<value_buffer_type, value_buffer_type> sort(raft::handle_t const& handle,
                                                      value_buffer_type const& first,
                                                      value_buffer_type const& second)
{
  auto sorted_first =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<value_buffer_type>>(
      first.size(), handle.get_stream());
  auto sorted_second =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<value_buffer_type>>(
      first.size(), handle.get_stream());

  auto execution_policy = handle.get_thrust_policy();
  thrust::copy(execution_policy,
               cugraph::get_dataframe_buffer_begin(first),
               cugraph::get_dataframe_buffer_end(first),
               cugraph::get_dataframe_buffer_begin(sorted_first));
  thrust::copy(execution_policy,
               cugraph::get_dataframe_buffer_begin(second),
               cugraph::get_dataframe_buffer_end(second),
               cugraph::get_dataframe_buffer_begin(sorted_second));
  thrust::sort(execution_policy,
               thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_first), cugraph::get_dataframe_buffer_begin(sorted_second)),
               thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_first) + first.size(), cugraph::get_dataframe_buffer_begin(sorted_second) + first.size()));


  return std::make_tuple(std::move(sorted_first), std::move(sorted_second));
}

template rmm::device_uvector<int32_t> sort(raft::handle_t const& handle,
                                           rmm::device_uvector<int32_t> const& values);

template rmm::device_uvector<int64_t> sort(raft::handle_t const& handle,
                                           rmm::device_uvector<int64_t> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> sort(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& first,
  rmm::device_uvector<int32_t> const& second);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> sort(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& first,
  rmm::device_uvector<int64_t> const& second);

template <typename key_buffer_type, typename value_buffer_type>
std::tuple<key_buffer_type, value_buffer_type> sort_by_key(raft::handle_t const& handle,
                                                           key_buffer_type const& keys,
                                                           value_buffer_type const& values)
{
  auto sorted_keys =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<key_buffer_type>>(
      keys.size(), handle.get_stream());
  auto sorted_values =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<value_buffer_type>>(
      keys.size(), handle.get_stream());

  auto execution_policy = handle.get_thrust_policy();
  thrust::copy(execution_policy,
               cugraph::get_dataframe_buffer_begin(keys),
               cugraph::get_dataframe_buffer_end(keys),
               cugraph::get_dataframe_buffer_begin(sorted_keys));
  thrust::copy(execution_policy,
               cugraph::get_dataframe_buffer_begin(values),
               cugraph::get_dataframe_buffer_end(values),
               cugraph::get_dataframe_buffer_begin(sorted_values));

  thrust::sort_by_key(execution_policy,
                      cugraph::get_dataframe_buffer_begin(sorted_keys),
                      cugraph::get_dataframe_buffer_end(sorted_keys),
                      cugraph::get_dataframe_buffer_begin(sorted_values));

  return std::make_tuple(std::move(sorted_keys), std::move(sorted_values));
}

template std::tuple<rmm::device_uvector<float>, rmm::device_uvector<int32_t>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<float> const& keys,
  rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<double>, rmm::device_uvector<int32_t>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<double> const& keys,
  rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& keys,
  rmm::device_uvector<float> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& keys,
  rmm::device_uvector<double> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& keys,
  rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int64_t>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& keys,
  rmm::device_uvector<int64_t> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& keys,
  rmm::device_uvector<float> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& keys,
  rmm::device_uvector<double> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& keys,
  rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& keys,
  rmm::device_uvector<int64_t> const& values);

template <typename key_buffer_type, typename value_buffer_type>
std::tuple<key_buffer_type, key_buffer_type, value_buffer_type> sort_by_key(raft::handle_t const& handle,
                                                           key_buffer_type const& first,
                                                           key_buffer_type const& second,
                                                           value_buffer_type const& values)
{
  auto sorted_first =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<key_buffer_type>>(
      first.size(), handle.get_stream());
  auto sorted_second =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<key_buffer_type>>(
      first.size(), handle.get_stream());
  auto sorted_values =
    cugraph::allocate_dataframe_buffer<cugraph::dataframe_element_t<value_buffer_type>>(
      first.size(), handle.get_stream());

  auto execution_policy = handle.get_thrust_policy();
  thrust::copy(execution_policy,
               cugraph::get_dataframe_buffer_begin(first),
               cugraph::get_dataframe_buffer_end(first),
               cugraph::get_dataframe_buffer_begin(sorted_first));
  thrust::copy(execution_policy,
               cugraph::get_dataframe_buffer_begin(second),
               cugraph::get_dataframe_buffer_end(second),
               cugraph::get_dataframe_buffer_begin(sorted_second));
  thrust::copy(execution_policy,
               cugraph::get_dataframe_buffer_begin(values),
               cugraph::get_dataframe_buffer_end(values),
               cugraph::get_dataframe_buffer_begin(sorted_values));
  thrust::sort_by_key(execution_policy,
               thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_first), cugraph::get_dataframe_buffer_begin(sorted_second)),
               thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_first) + first.size(), cugraph::get_dataframe_buffer_begin(sorted_second) + first.size()),
               cugraph::get_dataframe_buffer_begin(sorted_values));

  return std::make_tuple(std::move(sorted_first), std::move(sorted_second), std::move(sorted_values));
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<float>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& first,
  rmm::device_uvector<int32_t> const& second,
  rmm::device_uvector<float> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<float>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& first,
  rmm::device_uvector<int64_t> const& second,
  rmm::device_uvector<float> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<double>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& first,
  rmm::device_uvector<int32_t> const& second,
  rmm::device_uvector<double> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<double>> sort_by_key(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& first,
  rmm::device_uvector<int64_t> const& second,
  rmm::device_uvector<double> const& values);

template std::tuple<rmm::device_uvector<int32_t>,
                    std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>>
sort_by_key(raft::handle_t const& handle,
            rmm::device_uvector<int32_t> const& keys,
            std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>> const& values);

template std::tuple<rmm::device_uvector<int64_t>,
                    std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>>
sort_by_key(raft::handle_t const& handle,
            rmm::device_uvector<int64_t> const& keys,
            std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>> const& values);

template <typename vertex_t>
vertex_t max_element(raft::handle_t const& handle, raft::device_span<vertex_t const> vertices)
{
  auto ptr = thrust::max_element(
    handle.get_thrust_policy(), vertices.data(), vertices.data() + vertices.size());
  vertex_t ret{};
  raft::update_host(&ret, ptr, size_t{1}, handle.get_stream());
  handle.sync_stream();
  return ret;
}

template int32_t max_element(raft::handle_t const& handle,
                             raft::device_span<int32_t const> vertices);
template int64_t max_element(raft::handle_t const& handle,
                             raft::device_span<int64_t const> vertices);

template <typename vertex_t>
void translate_vertex_ids(raft::handle_t const& handle,
                          rmm::device_uvector<vertex_t>& vertices,
                          vertex_t vertex_id_offset)
{
  thrust::transform(handle.get_thrust_policy(),
                    vertices.begin(),
                    vertices.end(),
                    vertices.begin(),
                    [offset = vertex_id_offset] __device__(vertex_t v) { return offset + v; });
}

template <typename vertex_t>
void populate_vertex_ids(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& d_vertices_v,
                         vertex_t vertex_id_offset)
{
  thrust::sequence(
    handle.get_thrust_policy(), d_vertices_v.begin(), d_vertices_v.end(), vertex_id_offset);
}

template void translate_vertex_ids(raft::handle_t const& handle,
                                   rmm::device_uvector<int32_t>& vertices,
                                   int32_t vertex_id_offset);

template void translate_vertex_ids(raft::handle_t const& handle,
                                   rmm::device_uvector<int64_t>& vertices,
                                   int64_t vertex_id_offset);

template void populate_vertex_ids(raft::handle_t const& handle,
                                  rmm::device_uvector<int32_t>& d_vertices_v,
                                  int32_t vertex_id_offset);

template void populate_vertex_ids(raft::handle_t const& handle,
                                  rmm::device_uvector<int64_t>& d_vertices_v,
                                  int64_t vertex_id_offset);

}  // namespace test
}  // namespace cugraph

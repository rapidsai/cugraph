/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <utilities/thrust_wrapper.hpp>

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

template <typename T>
rmm::device_uvector<T> randomly_select(raft::handle_t const& handle,
                                       rmm::device_uvector<T> const& input,
                                       size_t count,
                                       bool sort_results)
{
  thrust::default_random_engine random_engine;

  rmm::device_uvector<T> tmp(input.size(), handle.get_stream());

  thrust::copy(handle.get_thrust_policy(), input.begin(), input.end(), tmp.begin());
  thrust::shuffle(handle.get_thrust_policy(), tmp.begin(), tmp.end(), random_engine);

  tmp.resize(std::min(count, tmp.size()), handle.get_stream());
  tmp.shrink_to_fit(handle.get_stream());

  if (sort_results) thrust::sort(handle.get_thrust_policy(), tmp.begin(), tmp.end());

  return tmp;
}

template rmm::device_uvector<int32_t> randomly_select(raft::handle_t const& handle,
                                                      rmm::device_uvector<int32_t> const& input,
                                                      size_t count,
                                                      bool sort_results);
template rmm::device_uvector<int64_t> randomly_select(raft::handle_t const& handle,
                                                      rmm::device_uvector<int64_t> const& input,
                                                      size_t count,
                                                      bool sort_results);

template <typename vertex_t, typename weight_t>
void remove_self_loops(raft::handle_t const& handle,
                       rmm::device_uvector<vertex_t>& d_src_v /* [INOUT] */,
                       rmm::device_uvector<vertex_t>& d_dst_v /* [INOUT] */,
                       std::optional<rmm::device_uvector<weight_t>>& d_weight_v /* [INOUT] */)
{
  if (d_weight_v) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(d_src_v.begin(), d_dst_v.begin(), (*d_weight_v).begin()));
    d_src_v.resize(
      thrust::distance(edge_first,
                       thrust::remove_if(
                         handle.get_thrust_policy(),
                         edge_first,
                         edge_first + d_src_v.size(),
                         [] __device__(auto e) { return thrust::get<0>(e) == thrust::get<1>(e); })),
      handle.get_stream());
    d_dst_v.resize(d_src_v.size(), handle.get_stream());
    (*d_weight_v).resize(d_src_v.size(), handle.get_stream());
  } else {
    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));
    d_src_v.resize(
      thrust::distance(edge_first,
                       thrust::remove_if(
                         handle.get_thrust_policy(),
                         edge_first,
                         edge_first + d_src_v.size(),
                         [] __device__(auto e) { return thrust::get<0>(e) == thrust::get<1>(e); })),
      handle.get_stream());
    d_dst_v.resize(d_src_v.size(), handle.get_stream());
  }

  d_src_v.shrink_to_fit(handle.get_stream());
  d_dst_v.shrink_to_fit(handle.get_stream());
  if (d_weight_v) { (*d_weight_v).shrink_to_fit(handle.get_stream()); }
}

template void remove_self_loops(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int32_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<float>>& d_weight_v /* [INOUT] */);

template void remove_self_loops(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int32_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<double>>& d_weight_v /* [INOUT] */);

template void remove_self_loops(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int64_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<float>>& d_weight_v /* [INOUT] */);

template void remove_self_loops(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int64_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<double>>& d_weight_v /* [INOUT] */);

template <typename vertex_t, typename weight_t>
void sort_and_remove_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<vertex_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<weight_t>>& d_weight_v /* [INOUT] */)
{
  if (d_weight_v) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(d_src_v.begin(), d_dst_v.begin(), (*d_weight_v).begin()));
    thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + d_src_v.size());
    d_src_v.resize(
      thrust::distance(edge_first,
                       thrust::unique(handle.get_thrust_policy(),
                                      edge_first,
                                      edge_first + d_src_v.size(),
                                      [] __device__(auto lhs, auto rhs) {
                                        return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) &&
                                               (thrust::get<1>(lhs) == thrust::get<1>(rhs));
                                      })),
      handle.get_stream());
    d_dst_v.resize(d_src_v.size(), handle.get_stream());
    (*d_weight_v).resize(d_src_v.size(), handle.get_stream());
  } else {
    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(d_src_v.begin(), d_dst_v.begin()));
    thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + d_src_v.size());
    d_src_v.resize(
      thrust::distance(
        edge_first,
        thrust::unique(handle.get_thrust_policy(), edge_first, edge_first + d_src_v.size())),
      handle.get_stream());
    d_dst_v.resize(d_src_v.size(), handle.get_stream());
  }

  d_src_v.shrink_to_fit(handle.get_stream());
  d_dst_v.shrink_to_fit(handle.get_stream());
  if (d_weight_v) { (*d_weight_v).shrink_to_fit(handle.get_stream()); }
}

template void sort_and_remove_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int32_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<float>>& d_weight_v /* [INOUT] */);

template void sort_and_remove_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int32_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<double>>& d_weight_v /* [INOUT] */);

template void sort_and_remove_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int64_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<float>>& d_weight_v /* [INOUT] */);

template void sort_and_remove_multi_edges(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_src_v /* [INOUT] */,
  rmm::device_uvector<int64_t>& d_dst_v /* [INOUT] */,
  std::optional<rmm::device_uvector<double>>& d_weight_v /* [INOUT] */);

}  // namespace test
}  // namespace cugraph

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

#include "utilities/thrust_wrapper.hpp"

#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/misc_utils.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace cugraph {
namespace test {

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> sort(

  raft::handle_t const& handle, cugraph::dataframe_buffer_type_t<value_t> const& values)
{
  auto sorted_values = cugraph::allocate_dataframe_buffer<value_t>(
    cugraph::size_dataframe_buffer(values), handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               cugraph::get_dataframe_buffer_begin(values),
               cugraph::get_dataframe_buffer_end(values),
               cugraph::get_dataframe_buffer_begin(sorted_values));

  thrust::sort(handle.get_thrust_policy(),
               cugraph::get_dataframe_buffer_begin(sorted_values),
               cugraph::get_dataframe_buffer_end(sorted_values));

  return sorted_values;
}

template rmm::device_uvector<int32_t> sort<int32_t>(raft::handle_t const& handle,
                                                    rmm::device_uvector<int32_t> const& values);

template rmm::device_uvector<int64_t> sort<int64_t>(raft::handle_t const& handle,
                                                    rmm::device_uvector<int64_t> const& values);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> sort(raft::handle_t const& handle,
                                               cugraph::dataframe_buffer_type_t<value_t>&& values)
{
  auto sorted_values = std::move(values);

  thrust::sort(handle.get_thrust_policy(),
               cugraph::get_dataframe_buffer_begin(sorted_values),
               cugraph::get_dataframe_buffer_end(sorted_values));

  return sorted_values;
}

template rmm::device_uvector<int32_t> sort<int32_t>(raft::handle_t const& handle,
                                                    rmm::device_uvector<int32_t>&& values);

template rmm::device_uvector<int64_t> sort<int64_t>(raft::handle_t const& handle,
                                                    rmm::device_uvector<int64_t>&& values);

template <typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<value_t>, cugraph::dataframe_buffer_type_t<value_t>>
sort(raft::handle_t const& handle,
     cugraph::dataframe_buffer_type_t<value_t> const& first,
     cugraph::dataframe_buffer_type_t<value_t> const& second)
{
  auto sorted_first =
    cugraph::allocate_dataframe_buffer<value_t>(size_dataframe_buffer(first), handle.get_stream());
  auto sorted_second =
    cugraph::allocate_dataframe_buffer<value_t>(size_dataframe_buffer(first), handle.get_stream());

  auto input_first  = thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(first),
                                               cugraph::get_dataframe_buffer_begin(second));
  auto output_first = thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_first),
                                                cugraph::get_dataframe_buffer_begin(sorted_second));
  thrust::copy(handle.get_thrust_policy(),
               input_first,
               input_first + size_dataframe_buffer(first),
               output_first);
  thrust::sort(
    handle.get_thrust_policy(), output_first, output_first + size_dataframe_buffer(sorted_first));

  return std::make_tuple(std::move(sorted_first), std::move(sorted_second));
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> sort<int32_t>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& first,
  rmm::device_uvector<int32_t> const& second);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> sort<int64_t>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& first,
  rmm::device_uvector<int64_t> const& second);

template <typename key_t, typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<key_t>, cugraph::dataframe_buffer_type_t<value_t>>
sort_by_key(raft::handle_t const& handle,
            cugraph::dataframe_buffer_type_t<key_t> const& keys,
            cugraph::dataframe_buffer_type_t<value_t> const& values)
{
  auto sorted_keys =
    cugraph::allocate_dataframe_buffer<key_t>(size_dataframe_buffer(keys), handle.get_stream());
  auto sorted_values =
    cugraph::allocate_dataframe_buffer<value_t>(size_dataframe_buffer(keys), handle.get_stream());

  auto input_first = thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(keys),
                                               cugraph::get_dataframe_buffer_begin(values));
  thrust::copy(handle.get_thrust_policy(),
               input_first,
               input_first + size_dataframe_buffer(keys),
               thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_keys),
                                         cugraph::get_dataframe_buffer_begin(sorted_values)));
  thrust::sort_by_key(handle.get_thrust_policy(),
                      cugraph::get_dataframe_buffer_begin(sorted_keys),
                      cugraph::get_dataframe_buffer_end(sorted_keys),
                      cugraph::get_dataframe_buffer_begin(sorted_values));

  return std::make_tuple(std::move(sorted_keys), std::move(sorted_values));
}

template std::tuple<rmm::device_uvector<float>, rmm::device_uvector<int32_t>>
sort_by_key<float, int32_t>(raft::handle_t const& handle,
                            rmm::device_uvector<float> const& keys,
                            rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<double>, rmm::device_uvector<int32_t>>
sort_by_key<double, int32_t>(raft::handle_t const& handle,
                             rmm::device_uvector<double> const& keys,
                             rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
sort_by_key<int32_t, float>(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t> const& keys,
                            rmm::device_uvector<float> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
sort_by_key<int32_t, double>(raft::handle_t const& handle,
                             rmm::device_uvector<int32_t> const& keys,
                             rmm::device_uvector<double> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
sort_by_key<int32_t, int32_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int32_t> const& keys,
                              rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int64_t>>
sort_by_key<int32_t, int64_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int32_t> const& keys,
                              rmm::device_uvector<int64_t> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
sort_by_key<int64_t, float>(raft::handle_t const& handle,
                            rmm::device_uvector<int64_t> const& keys,
                            rmm::device_uvector<float> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
sort_by_key<int64_t, double>(raft::handle_t const& handle,
                             rmm::device_uvector<int64_t> const& keys,
                             rmm::device_uvector<double> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
sort_by_key<int64_t, int32_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int64_t> const& keys,
                              rmm::device_uvector<int32_t> const& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
sort_by_key<int64_t, int64_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int64_t> const& keys,
                              rmm::device_uvector<int64_t> const& values);

template std::tuple<rmm::device_uvector<int32_t>,
                    std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>>
sort_by_key<int32_t, thrust::tuple<int32_t, float>>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t> const& keys,
  std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>> const& values);

template std::tuple<rmm::device_uvector<int64_t>,
                    std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>>
sort_by_key<int64_t, thrust::tuple<int32_t, float>>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t> const& keys,
  std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>> const& values);

template <typename key_t, typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<key_t>, cugraph::dataframe_buffer_type_t<value_t>>
sort_by_key(raft::handle_t const& handle,
            cugraph::dataframe_buffer_type_t<key_t>&& keys,
            cugraph::dataframe_buffer_type_t<value_t>&& values)
{
  auto sorted_keys   = std::move(keys);
  auto sorted_values = std::move(values);

  thrust::sort_by_key(handle.get_thrust_policy(),
                      cugraph::get_dataframe_buffer_begin(sorted_keys),
                      cugraph::get_dataframe_buffer_end(sorted_keys),
                      cugraph::get_dataframe_buffer_begin(sorted_values));

  return std::make_tuple(std::move(sorted_keys), std::move(sorted_values));
}

template std::tuple<rmm::device_uvector<float>, rmm::device_uvector<int32_t>>
sort_by_key<float, int32_t>(raft::handle_t const& handle,
                            rmm::device_uvector<float>&& keys,
                            rmm::device_uvector<int32_t>&& values);

template std::tuple<rmm::device_uvector<float>, rmm::device_uvector<int64_t>>
sort_by_key<float, int64_t>(raft::handle_t const& handle,
                            rmm::device_uvector<float>&& keys,
                            rmm::device_uvector<int64_t>&& values);

template std::tuple<rmm::device_uvector<double>, rmm::device_uvector<int32_t>>
sort_by_key<double, int32_t>(raft::handle_t const& handle,
                             rmm::device_uvector<double>&& keys,
                             rmm::device_uvector<int32_t>&& values);

template std::tuple<rmm::device_uvector<double>, rmm::device_uvector<int64_t>>
sort_by_key<double, int64_t>(raft::handle_t const& handle,
                             rmm::device_uvector<double>&& keys,
                             rmm::device_uvector<int64_t>&& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
sort_by_key<int32_t, float>(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& keys,
                            rmm::device_uvector<float>&& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
sort_by_key<int32_t, double>(raft::handle_t const& handle,
                             rmm::device_uvector<int32_t>&& keys,
                             rmm::device_uvector<double>&& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
sort_by_key<int32_t, int32_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int32_t>&& keys,
                              rmm::device_uvector<int32_t>&& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int64_t>>
sort_by_key<int32_t, int64_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int32_t>&& keys,
                              rmm::device_uvector<int64_t>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
sort_by_key<int64_t, float>(raft::handle_t const& handle,
                            rmm::device_uvector<int64_t>&& keys,
                            rmm::device_uvector<float>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
sort_by_key<int64_t, double>(raft::handle_t const& handle,
                             rmm::device_uvector<int64_t>&& keys,
                             rmm::device_uvector<double>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
sort_by_key<int64_t, int32_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int64_t>&& keys,
                              rmm::device_uvector<int32_t>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
sort_by_key<int64_t, int64_t>(raft::handle_t const& handle,
                              rmm::device_uvector<int64_t>&& keys,
                              rmm::device_uvector<int64_t>&& values);

template <typename key_t, typename value_t>
std::tuple<cugraph::dataframe_buffer_type_t<key_t>,
           cugraph::dataframe_buffer_type_t<key_t>,
           cugraph::dataframe_buffer_type_t<value_t>>
sort_by_key(raft::handle_t const& handle,
            cugraph::dataframe_buffer_type_t<key_t> const& first,
            cugraph::dataframe_buffer_type_t<key_t> const& second,
            cugraph::dataframe_buffer_type_t<value_t> const& values)
{
  auto sorted_first = cugraph::allocate_dataframe_buffer<key_t>(
    cugraph::size_dataframe_buffer(first), handle.get_stream());
  auto sorted_second = cugraph::allocate_dataframe_buffer<key_t>(
    cugraph::size_dataframe_buffer(first), handle.get_stream());
  auto sorted_values = cugraph::allocate_dataframe_buffer<value_t>(
    cugraph::size_dataframe_buffer(first), handle.get_stream());

  auto input_first = thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(first),
                                               cugraph::get_dataframe_buffer_begin(second),
                                               cugraph::get_dataframe_buffer_begin(values));
  thrust::copy(handle.get_thrust_policy(),
               input_first,
               input_first + size_dataframe_buffer(first),
               thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_first),
                                         cugraph::get_dataframe_buffer_begin(sorted_second),
                                         cugraph::get_dataframe_buffer_begin(sorted_values)));
  auto sorted_key_first =
    thrust::make_zip_iterator(cugraph::get_dataframe_buffer_begin(sorted_first),
                              cugraph::get_dataframe_buffer_begin(sorted_second));
  thrust::sort_by_key(handle.get_thrust_policy(),
                      sorted_key_first,
                      sorted_key_first + cugraph::size_dataframe_buffer(sorted_first),
                      cugraph::get_dataframe_buffer_begin(sorted_values));

  return std::make_tuple(
    std::move(sorted_first), std::move(sorted_second), std::move(sorted_values));
}

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
  sort_by_key<int32_t, float>(raft::handle_t const& handle,
                              rmm::device_uvector<int32_t> const& first,
                              rmm::device_uvector<int32_t> const& second,
                              rmm::device_uvector<float> const& values);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
  sort_by_key<int64_t, float>(raft::handle_t const& handle,
                              rmm::device_uvector<int64_t> const& first,
                              rmm::device_uvector<int64_t> const& second,
                              rmm::device_uvector<float> const& values);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
  sort_by_key<int32_t, double>(raft::handle_t const& handle,
                               rmm::device_uvector<int32_t> const& first,
                               rmm::device_uvector<int32_t> const& second,
                               rmm::device_uvector<double> const& values);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
  sort_by_key<int64_t, double>(raft::handle_t const& handle,
                               rmm::device_uvector<int64_t> const& first,
                               rmm::device_uvector<int64_t> const& second,
                               rmm::device_uvector<double> const& values);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> unique(raft::handle_t const& handle,
                                                 cugraph::dataframe_buffer_type_t<value_t>&& values)
{
  auto last = thrust::unique(handle.get_thrust_policy(),
                             cugraph::get_dataframe_buffer_begin(values),
                             cugraph::get_dataframe_buffer_end(values));
  cugraph::resize_dataframe_buffer(
    values,
    cuda::std::distance(cugraph::get_dataframe_buffer_begin(values), last),
    handle.get_stream());
  cugraph::shrink_to_fit_dataframe_buffer(values, handle.get_stream());

  return std::move(values);
}

template rmm::device_uvector<int32_t> unique<int32_t>(raft::handle_t const& handle,
                                                      rmm::device_uvector<int32_t>&& values);

template rmm::device_uvector<int64_t> unique<int64_t>(raft::handle_t const& handle,
                                                      rmm::device_uvector<int64_t>&& values);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> sequence(raft::handle_t const& handle,
                                                   size_t length,
                                                   size_t repeat_count,
                                                   value_t init)
{
  auto values = cugraph::allocate_dataframe_buffer<value_t>(length, handle.get_stream());
  if (repeat_count == 1) {
    thrust::sequence(handle.get_thrust_policy(), values.begin(), values.end(), init);
  } else {
    thrust::tabulate(handle.get_thrust_policy(),
                     values.begin(),
                     values.end(),
                     [repeat_count, init] __device__(size_t i) {
                       return init + static_cast<value_t>(i / repeat_count);
                     });
  }

  return values;
}

template rmm::device_uvector<int32_t> sequence(raft::handle_t const& handle,
                                               size_t length,
                                               size_t repeat_count,
                                               int32_t init);

template rmm::device_uvector<int64_t> sequence(raft::handle_t const& handle,
                                               size_t length,
                                               size_t repeat_count,
                                               int64_t init);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> scalar_fill(raft::handle_t const& handle,
                                                      size_t length,
                                                      value_t value)
{
  auto values = cugraph::allocate_dataframe_buffer<value_t>(length, handle.get_stream());

  thrust::tabulate(
    handle.get_thrust_policy(), values.begin(), values.end(), [value] __device__(size_t i) {
      return value;
    });

  return values;
}

template rmm::device_uvector<int32_t> scalar_fill(raft::handle_t const& handle,
                                                  size_t length,
                                                  int32_t value);

template <typename value_t>
cugraph::dataframe_buffer_type_t<value_t> modulo_sequence(raft::handle_t const& handle,
                                                          size_t length,
                                                          value_t modulo,
                                                          value_t init)
{
  auto values = cugraph::allocate_dataframe_buffer<value_t>(length, handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(), values.begin(), values.end(), [modulo, init] __device__(size_t i) {
      return static_cast<value_t>((init + i) % modulo);
    });

  return values;
}

template rmm::device_uvector<int32_t> modulo_sequence(raft::handle_t const& handle,
                                                      size_t length,
                                                      int32_t modulo,
                                                      int32_t init);

template rmm::device_uvector<int64_t> modulo_sequence(raft::handle_t const& handle,
                                                      size_t length,
                                                      int64_t modulo,
                                                      int64_t init);

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

template <typename idx_t, typename offset_t>
void expand_sparse_offsets(raft::handle_t const& handle,
                           raft::device_span<offset_t const> offsets,
                           raft::device_span<idx_t> indices,
                           offset_t base_offset,
                           idx_t base_idx)
{
  rmm::device_uvector<offset_t> tmp_offsets(offsets.size(), handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    offsets.begin(),
                    offsets.end(),
                    tmp_offsets.begin(),
                    cugraph::detail::shift_left_t<offset_t>{base_offset});
  auto tmp = cugraph::detail::expand_sparse_offsets(
    raft::device_span<offset_t const>(tmp_offsets.data(), tmp_offsets.size()),
    base_idx,
    handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), tmp.begin(), tmp.end(), indices.begin());
}

template void expand_sparse_offsets(raft::handle_t const& handle,
                                    raft::device_span<size_t const> offsets,
                                    raft::device_span<int32_t> indices,
                                    size_t base_offset,
                                    int32_t base_idx);

template void expand_sparse_offsets(raft::handle_t const& handle,
                                    raft::device_span<size_t const> offsets,
                                    raft::device_span<int64_t> indices,
                                    size_t base_offset,
                                    int64_t base_idx);

template <typename idx_t, typename offset_t>
void expand_hypersparse_offsets(raft::handle_t const& handle,
                                raft::device_span<offset_t const> offsets,
                                raft::device_span<idx_t const> nzd_indices,
                                raft::device_span<idx_t> indices,
                                offset_t base_offset)
{
  rmm::device_uvector<offset_t> tmp_offsets(offsets.size(), handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    offsets.begin(),
                    offsets.end(),
                    tmp_offsets.begin(),
                    cugraph::detail::shift_left_t<offset_t>{base_offset});
  auto tmp = cugraph::detail::expand_sparse_offsets(
    raft::device_span<offset_t const>(tmp_offsets.data(), tmp_offsets.size()),
    idx_t{0},
    handle.get_stream());
  thrust::gather(
    handle.get_thrust_policy(), tmp.begin(), tmp.end(), nzd_indices.begin(), indices.begin());
}

template void expand_hypersparse_offsets(raft::handle_t const& handle,
                                         raft::device_span<size_t const> offsets,
                                         raft::device_span<int32_t const> nzd_indices,
                                         raft::device_span<int32_t> indices,
                                         size_t base_offset);

template void expand_hypersparse_offsets(raft::handle_t const& handle,
                                         raft::device_span<size_t const> offsets,
                                         raft::device_span<int64_t const> nzd_indices,
                                         raft::device_span<int64_t> indices,
                                         size_t base_offset);

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> remove_self_loops(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& v1,
  rmm::device_uvector<vertex_t>&& v2)
{
  auto new_size = cuda::std::distance(
    thrust::make_zip_iterator(v1.begin(), v2.begin()),
    thrust::remove_if(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(v1.begin(), v2.begin()),
      thrust::make_zip_iterator(v1.end(), v2.end()),
      [] __device__(auto tuple) { return thrust::get<0>(tuple) == thrust::get<1>(tuple); }));

  v1.resize(new_size, handle.get_stream());
  v2.resize(new_size, handle.get_stream());

  return std::make_tuple(std::move(v1), std::move(v2));
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> remove_self_loops(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& v1,
  rmm::device_uvector<int32_t>&& v2);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> remove_self_loops(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& v1,
  rmm::device_uvector<int64_t>&& v2);

template <typename T>
bool device_spans_equal(raft::handle_t const& handle,
                        raft::device_span<T const> array_left,
                        raft::device_span<T const> array_right)
{
  return thrust::equal(
    handle.get_thrust_policy(), array_left.begin(), array_left.end(), array_right.begin());
}

template bool device_spans_equal(raft::handle_t const& handle,
                                 raft::device_span<int32_t const> array_left,
                                 raft::device_span<int32_t const> array_right);
template bool device_spans_equal(raft::handle_t const& handle,
                                 raft::device_span<int64_t const> array_left,
                                 raft::device_span<int64_t const> array_right);
template bool device_spans_equal(raft::handle_t const& handle,
                                 raft::device_span<float const> array_left,
                                 raft::device_span<float const> array_right);
template bool device_spans_equal(raft::handle_t const& handle,
                                 raft::device_span<double const> array_left,
                                 raft::device_span<double const> array_right);

}  // namespace test
}  // namespace cugraph

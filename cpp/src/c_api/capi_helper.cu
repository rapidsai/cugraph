/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "c_api/capi_helper.hpp"
#include "structure/detail/structure_utils.cuh"

#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <cuda/std/iterator>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

namespace cugraph {
namespace c_api {
namespace detail {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>>
shuffle_vertex_ids_and_offsets(raft::handle_t const& handle,
                               rmm::device_uvector<vertex_t>&& vertices,
                               raft::device_span<size_t const> offsets)
{
  auto ids = cugraph::detail::expand_sparse_offsets(offsets, vertex_t{0}, handle.get_stream());

  std::tie(vertices, ids) =
    cugraph::shuffle_ext_vertex_value_pairs(handle, std::move(vertices), std::move(ids));

  thrust::sort(handle.get_thrust_policy(),
               thrust::make_zip_iterator(ids.begin(), vertices.begin()),
               thrust::make_zip_iterator(ids.end(), vertices.end()));

  auto return_offsets = cugraph::detail::compute_sparse_offsets<size_t>(
    ids.begin(), ids.end(), size_t{0}, size_t{offsets.size() - 1}, true, handle.get_stream());

  return std::make_tuple(std::move(vertices), std::move(return_offsets));
}

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<size_t>>
shuffle_vertex_ids_and_offsets(raft::handle_t const& handle,
                               rmm::device_uvector<int32_t>&& vertices,
                               raft::device_span<size_t const> offsets);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<size_t>>
shuffle_vertex_ids_and_offsets(raft::handle_t const& handle,
                               rmm::device_uvector<int64_t>&& vertices,
                               raft::device_span<size_t const> offsets);

template <typename key_t, typename value_t>
void sort_by_key(raft::handle_t const& handle,
                 raft::device_span<key_t> keys,
                 raft::device_span<value_t> values)
{
  thrust::sort_by_key(handle.get_thrust_policy(), keys.begin(), keys.end(), values.begin());
}

template <typename key_t, typename value_t>
void sort_tuple_by_key(raft::handle_t const& handle,
                       raft::device_span<key_t> keys,
                       std::tuple<raft::device_span<value_t>, raft::device_span<value_t>> values)
{
  thrust::sort_by_key(
    handle.get_thrust_policy(),
    keys.begin(),
    keys.end(),
    thrust::make_zip_iterator(std::get<0>(values).begin(), std::get<1>(values).begin()));
}

template void sort_by_key(raft::handle_t const& handle,
                          raft::device_span<int32_t> keys,
                          raft::device_span<int32_t> values);
template void sort_by_key(raft::handle_t const& handle,
                          raft::device_span<int64_t> keys,
                          raft::device_span<int64_t> values);

template void sort_by_key(raft::handle_t const& handle,
                          raft::device_span<int32_t> keys,
                          raft::device_span<float> values);
template void sort_by_key(raft::handle_t const& handle,
                          raft::device_span<int64_t> keys,
                          raft::device_span<float> values);

template void sort_by_key(raft::handle_t const& handle,
                          raft::device_span<int32_t> keys,
                          raft::device_span<double> values);
template void sort_by_key(raft::handle_t const& handle,
                          raft::device_span<int64_t> keys,
                          raft::device_span<double> values);

template void sort_tuple_by_key(
  raft::handle_t const& handle,
  raft::device_span<int32_t> keys,
  std::tuple<raft::device_span<float>, raft::device_span<float>> values);

template void sort_tuple_by_key(
  raft::handle_t const& handle,
  raft::device_span<int32_t> keys,
  std::tuple<raft::device_span<double>, raft::device_span<double>> values);

template void sort_tuple_by_key(
  raft::handle_t const& handle,
  raft::device_span<int64_t> keys,
  std::tuple<raft::device_span<float>, raft::device_span<float>> values);

template void sort_tuple_by_key(
  raft::handle_t const& handle,
  raft::device_span<int64_t> keys,
  std::tuple<raft::device_span<double>, raft::device_span<double>> values);

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<size_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
reorder_extracted_egonets(raft::handle_t const& handle,
                          rmm::device_uvector<size_t>&& source_indices,
                          rmm::device_uvector<size_t>&& offsets,
                          rmm::device_uvector<vertex_t>&& edge_srcs,
                          rmm::device_uvector<vertex_t>&& edge_dsts,
                          std::optional<rmm::device_uvector<weight_t>>&& edge_weights)
{
  rmm::device_uvector<size_t> sort_indices(edge_srcs.size(), handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    sort_indices.begin(),
    sort_indices.end(),
    [offset_lasts   = raft::device_span<size_t const>(offsets.begin() + 1, offsets.end()),
     source_indices = raft::device_span<size_t const>(source_indices.data(),
                                                      source_indices.size())] __device__(size_t i) {
      auto idx = static_cast<size_t>(cuda::std::distance(
        offset_lasts.begin(),
        thrust::upper_bound(thrust::seq, offset_lasts.begin(), offset_lasts.end(), i)));
      return source_indices[idx];
    });
  source_indices.resize(0, handle.get_stream());
  source_indices.shrink_to_fit(handle.get_stream());

  auto triplet_first =
    thrust::make_zip_iterator(sort_indices.begin(), edge_srcs.begin(), edge_dsts.begin());
  if (edge_weights) {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        triplet_first,
                        triplet_first + sort_indices.size(),
                        (*edge_weights).begin());
  } else {
    thrust::sort(handle.get_thrust_policy(), triplet_first, triplet_first + sort_indices.size());
  }

  thrust::tabulate(
    handle.get_thrust_policy(),
    offsets.begin() + 1,
    offsets.end(),
    [sort_indices = raft::device_span<size_t const>(sort_indices.data(),
                                                    sort_indices.size())] __device__(size_t i) {
      return static_cast<size_t>(cuda::std::distance(
        sort_indices.begin(),
        thrust::upper_bound(thrust::seq, sort_indices.begin(), sort_indices.end(), i)));
    });

  return std::make_tuple(
    std::move(offsets), std::move(edge_srcs), std::move(edge_dsts), std::move(edge_weights));
}

template std::tuple<rmm::device_uvector<size_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
reorder_extracted_egonets(raft::handle_t const& handle,
                          rmm::device_uvector<size_t>&& source_indices,
                          rmm::device_uvector<size_t>&& offsets,
                          rmm::device_uvector<int32_t>&& edge_srcs,
                          rmm::device_uvector<int32_t>&& edge_dsts,
                          std::optional<rmm::device_uvector<float>>&& edge_weights);

template std::tuple<rmm::device_uvector<size_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
reorder_extracted_egonets(raft::handle_t const& handle,
                          rmm::device_uvector<size_t>&& source_indices,
                          rmm::device_uvector<size_t>&& offsets,
                          rmm::device_uvector<int32_t>&& edge_srcs,
                          rmm::device_uvector<int32_t>&& edge_dsts,
                          std::optional<rmm::device_uvector<double>>&& edge_weights);

template std::tuple<rmm::device_uvector<size_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
reorder_extracted_egonets(raft::handle_t const& handle,
                          rmm::device_uvector<size_t>&& source_indices,
                          rmm::device_uvector<size_t>&& offsets,
                          rmm::device_uvector<int64_t>&& edge_srcs,
                          rmm::device_uvector<int64_t>&& edge_dsts,
                          std::optional<rmm::device_uvector<float>>&& edge_weights);

template std::tuple<rmm::device_uvector<size_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
reorder_extracted_egonets(raft::handle_t const& handle,
                          rmm::device_uvector<size_t>&& source_indices,
                          rmm::device_uvector<size_t>&& offsets,
                          rmm::device_uvector<int64_t>&& edge_srcs,
                          rmm::device_uvector<int64_t>&& edge_dsts,
                          std::optional<rmm::device_uvector<double>>&& edge_weights);

}  // namespace detail
}  // namespace c_api
}  // namespace cugraph

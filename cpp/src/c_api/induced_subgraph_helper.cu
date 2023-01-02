/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <c_api/induced_subgraph_helper.hpp>
#include <structure/detail/structure_utils.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>

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
    cugraph::detail::shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
      handle, std::move(vertices), std::move(ids));

  thrust::sort(handle.get_thrust_policy(),
               thrust::make_zip_iterator(ids.begin(), vertices.begin()),
               thrust::make_zip_iterator(ids.end(), vertices.end()));

  auto return_offsets = cugraph::detail::compute_sparse_offsets<size_t>(
    ids.begin(), ids.end(), size_t{0}, size_t{offsets.size() - 1}, handle.get_stream());

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

}  // namespace detail
}  // namespace c_api
}  // namespace cugraph

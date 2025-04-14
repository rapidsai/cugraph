/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/mtmg/vertex_result_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/gather.h>

namespace cugraph {
namespace mtmg {

template <typename result_t>
template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<result_t> vertex_result_view_t<result_t>::gather(
  handle_t const& handle,
  raft::device_span<vertex_t const> vertices,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  vertex_partition_view_t<vertex_t, multi_gpu> vertex_partition_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<vertex_t>>& renumber_map_view,
  result_t default_value)
{
  auto stream = handle.raft_handle().get_stream();

  rmm::device_uvector<vertex_t> local_vertices(vertices.size(), stream);
  rmm::device_uvector<int> vertex_gpu_ids(multi_gpu ? vertices.size() : 0, stream);
  rmm::device_uvector<size_t> vertex_pos(multi_gpu ? vertices.size() : 0, stream);

  raft::copy(local_vertices.data(), vertices.data(), vertices.size(), stream);

  if constexpr (multi_gpu) {
    cugraph::detail::scalar_fill(
      stream, vertex_gpu_ids.data(), vertex_gpu_ids.size(), handle.get_rank());
    cugraph::detail::sequence_fill(stream, vertex_pos.data(), vertex_pos.size(), size_t{0});

    auto const comm_size = handle.raft_handle().get_comms().get_size();
    auto const major_comm_size =
      handle.raft_handle().get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
    auto const minor_comm_size =
      handle.raft_handle().get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();

    std::forward_as_tuple(local_vertices, std::tie(vertex_gpu_ids, vertex_pos), std::ignore) =
      groupby_gpu_id_and_shuffle_kv_pairs(
        handle.raft_handle().get_comms(),
        local_vertices.begin(),
        local_vertices.end(),
        thrust::make_zip_iterator(vertex_gpu_ids.begin(), vertex_pos.begin()),
        cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
          comm_size, major_comm_size, minor_comm_size},
        stream);
  }

  if (renumber_map_view) {
    cugraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
      handle.raft_handle(),
      local_vertices.data(),
      local_vertices.size(),
      renumber_map_view->get(handle).data(),
      vertex_partition_view.local_vertex_partition_range_first(),
      vertex_partition_view.local_vertex_partition_range_last());

    size_t new_size = cuda::std::distance(
      thrust::make_zip_iterator(local_vertices.begin(), vertex_gpu_ids.begin(), vertex_pos.begin()),
      thrust::remove_if(
        rmm::exec_policy(stream),
        thrust::make_zip_iterator(
          local_vertices.begin(), vertex_gpu_ids.begin(), vertex_pos.begin()),
        thrust::make_zip_iterator(local_vertices.end(), vertex_gpu_ids.end(), vertex_pos.end()),
        [check = cugraph::detail::check_out_of_range_t<vertex_t>{
           vertex_partition_view.local_vertex_partition_range_first(),
           vertex_partition_view.local_vertex_partition_range_last()}] __device__(auto tuple) {
          return check(thrust::get<0>(tuple));
        }));

    local_vertices.resize(new_size, stream);
    vertex_gpu_ids.resize(new_size, stream);
    vertex_pos.resize(new_size, stream);
  }

  //
  //  Now gather
  //
  rmm::device_uvector<result_t> result(local_vertices.size(), stream);
  cugraph::detail::scalar_fill(stream, result.data(), result.size(), default_value);

  auto& wrapped = this->get(handle);

  auto vertex_partition =
    vertex_partition_device_view_t<vertex_t, multi_gpu>(vertex_partition_view);

  auto iter = thrust::make_transform_iterator(
    local_vertices.begin(),
    cuda::proclaim_return_type<vertex_t>([vertex_partition] __device__(auto v) {
      return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
    }));

  thrust::gather(
    rmm::exec_policy(stream), iter, iter + local_vertices.size(), wrapped.begin(), result.begin());

  if constexpr (multi_gpu) {
    rmm::device_uvector<result_t> tmp_result(0, stream);

    //
    // Shuffle back
    //
    std::forward_as_tuple(std::ignore, std::tie(std::ignore, vertex_pos, tmp_result), std::ignore) =
      groupby_gpu_id_and_shuffle_kv_pairs(
        handle.raft_handle().get_comms(),
        vertex_gpu_ids.begin(),
        vertex_gpu_ids.end(),
        thrust::make_zip_iterator(local_vertices.begin(), vertex_pos.begin(), result.begin()),
        cuda::std::identity{},
        stream);

    //
    // Finally, reorder result
    //
    result.resize(tmp_result.size(), stream);
    cugraph::detail::scalar_fill(stream, result.data(), result.size(), default_value);

    thrust::scatter(rmm::exec_policy(stream),
                    tmp_result.begin(),
                    tmp_result.end(),
                    vertex_pos.begin(),
                    result.begin());
  }

  return result;
}

}  // namespace mtmg
}  // namespace cugraph

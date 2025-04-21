/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <cugraph/mtmg/vertex_pair_result_view.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <cuda/std/iterator>
#include <thrust/functional.h>
#include <thrust/gather.h>

namespace cugraph {
namespace mtmg {

template <typename vertex_t, typename result_t>
template <bool multi_gpu>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<result_t>>
  vertex_pair_result_view_t<vertex_t, result_t>::gather(
    handle_t const& handle,
    raft::device_span<vertex_t const> vertices,
    raft::host_span<vertex_t const> vertex_partition_range_lasts,
    vertex_partition_view_t<vertex_t, multi_gpu> vertex_partition_view,
    std::optional<cugraph::mtmg::renumber_map_view_t<vertex_t>>& renumber_map_view)
{
  // FIXME: Should this handle the case of multiple local host threads?
  //        It currently does not.  If vertices were a raft::host_span
  //        We could have the host threads copy the data to a device_uvector
  //        and then have rank 0 execute this logic, and we could copy to
  //        host at the end.
  auto stream = handle.raft_handle().get_stream();

  rmm::device_uvector<vertex_t> local_vertices(vertices.size(), stream);
  rmm::device_uvector<int> vertex_gpu_ids(vertices.size(), stream);

  raft::copy(local_vertices.data(), vertices.data(), vertices.size(), stream);
  cugraph::detail::scalar_fill(
    stream, vertex_gpu_ids.data(), vertex_gpu_ids.size(), handle.get_rank());

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               stream);
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      stream);

  if (renumber_map_view) {
    cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
      handle.raft_handle(),
      local_vertices.data(),
      local_vertices.size(),
      renumber_map_view->get(handle).data(),
      vertex_partition_view.local_vertex_partition_range_first(),
      vertex_partition_view.local_vertex_partition_range_last());
  }

  auto const major_comm_size =
    handle.raft_handle().get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
  auto const minor_comm_size =
    handle.raft_handle().get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();

  std::tie(local_vertices, vertex_gpu_ids, std::ignore) = groupby_gpu_id_and_shuffle_kv_pairs(
    handle.raft_handle().get_comms(),
    local_vertices.begin(),
    local_vertices.end(),
    vertex_gpu_ids.begin(),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size},
    stream);

  //
  // LOOK AT THIS...
  //    I think the above shuffle is correct...
  //    This will give us vertex/gpu_id tuples on the GPU that vertex is assigned
  //       to.  I need to take this and filter the device vector tuples based on the desired
  //       vertex (v1).
  //

  //
  //  Now gather
  //
  auto& wrapped = this->get(handle);

  rmm::device_uvector<vertex_t> v1(std::get<0>(wrapped).size(), stream);
  rmm::device_uvector<vertex_t> v2(std::get<0>(wrapped).size(), stream);
  rmm::device_uvector<result_t> result(std::get<0>(wrapped).size(), stream);

  thrust::copy(
    rmm::exec_policy(stream),
    thrust::make_zip_iterator(
      std::get<0>(wrapped).begin(), std::get<1>(wrapped).begin(), std::get<2>(wrapped).begin()),
    thrust::make_zip_iterator(
      std::get<0>(wrapped).end(), std::get<1>(wrapped).end(), std::get<2>(wrapped).end()),
    thrust::make_zip_iterator(v1.begin(), v2.begin(), result.begin()));

  thrust::sort_by_key(
    rmm::exec_policy(stream), local_vertices.begin(), local_vertices.end(), vertex_gpu_ids.begin());

  auto new_end =
    thrust::remove_if(rmm::exec_policy(stream),
                      thrust::make_zip_iterator(v1.begin(), v2.begin(), result.begin()),
                      thrust::make_zip_iterator(v1.end(), v2.end(), result.end()),
                      [v1_check = raft::device_span<vertex_t const>{
                         local_vertices.data(), local_vertices.size()}] __device__(auto tuple) {
                        return thrust::binary_search(
                          thrust::seq, v1_check.begin(), v1_check.end(), thrust::get<0>(tuple));
                      });

  v1.resize(
    cuda::std::distance(thrust::make_zip_iterator(v1.begin(), v2.begin(), result.begin()), new_end),
    stream);
  v2.resize(v1.size(), stream);
  result.resize(v1.size(), stream);

  //
  // Shuffle back
  //
  std::forward_as_tuple(std::ignore, std::tie(v1, v2, result), std::ignore) =
    groupby_gpu_id_and_shuffle_kv_pairs(
      handle.raft_handle().get_comms(),
      v1.begin(),
      v1.end(),
      thrust::make_zip_iterator(v1.begin(), v2.begin(), result.begin()),
      cuda::proclaim_return_type<int>(
        [local_v = raft::device_span<vertex_t const>{local_vertices.data(), local_vertices.size()},
         gpu     = raft::device_span<int const>{vertex_gpu_ids.data(),
                                                vertex_gpu_ids.size()}] __device__(auto v1) {
          return gpu[cuda::std::distance(
            local_v.begin(), thrust::lower_bound(thrust::seq, local_v.begin(), local_v.end(), v1))];
        }),
      stream);

  if (renumber_map_view) {
    cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle.raft_handle(),
                                                          v1.data(),
                                                          v1.size(),
                                                          renumber_map_view->get(handle).data(),
                                                          vertex_partition_range_lasts);

    cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(handle.raft_handle(),
                                                          v2.data(),
                                                          v2.size(),
                                                          renumber_map_view->get(handle).data(),
                                                          vertex_partition_range_lasts);
  }

  return std::make_tuple(std::move(v1), std::move(v2), std::move(result));
}

}  // namespace mtmg
}  // namespace cugraph

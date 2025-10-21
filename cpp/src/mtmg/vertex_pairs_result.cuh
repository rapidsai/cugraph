/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/graph_partition_utils.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/mtmg/vertex_pair_result_view.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <cuda/std/iterator>
#include <cuda/std/tuple>
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

  if (renumber_map_view) {
    cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
      handle.raft_handle(),
      local_vertices.data(),
      local_vertices.size(),
      renumber_map_view->get(handle).data(),
      vertex_partition_view.local_vertex_partition_range_first(),
      vertex_partition_view.local_vertex_partition_range_last());
  }

  // allgather the vertices and vertex_gpu_ids.
  // FIXME: This is a potential scaling problem.  This will create an all_vertices array that could
  // be as large as O(n)
  //   on each GPU.  We should explore other options for this.  We could do it in batches, or we
  //   could ensure that the vertex pairs are partitioned a certain way and share the source vertces
  //   (v1) on the appropriate subset of GPUs.
  auto all_vertices = cugraph::device_allgatherv(
    handle.raft_handle(),
    handle.raft_handle().get_comms(),
    raft::device_span<vertex_t const>{local_vertices.data(), local_vertices.size()});
  auto all_vertex_gpu_ids = cugraph::device_allgatherv(
    handle.raft_handle(),
    handle.raft_handle().get_comms(),
    raft::device_span<int const>{vertex_gpu_ids.data(), vertex_gpu_ids.size()});

  auto const major_comm_size =
    handle.raft_handle().get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
  auto const minor_comm_size =
    handle.raft_handle().get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();

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
    rmm::exec_policy(stream), all_vertices.begin(), all_vertices.end(), all_vertex_gpu_ids.begin());

  auto new_end =
    thrust::remove_if(rmm::exec_policy(stream),
                      thrust::make_zip_iterator(v1.begin(), v2.begin(), result.begin()),
                      thrust::make_zip_iterator(v1.end(), v2.end(), result.end()),
                      [v1_check = raft::device_span<vertex_t const>{
                         all_vertices.data(), all_vertices.size()}] __device__(auto tuple) {
                        return thrust::binary_search(
                          thrust::seq, v1_check.begin(), v1_check.end(), cuda::std::get<0>(tuple));
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

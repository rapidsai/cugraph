/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/mtmg/vertex_result_view.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <detail/graph_partition_utils.cuh>

#include <thrust/gather.h>

namespace cugraph {
namespace mtmg {

template <typename result_t>
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<result_t> vertex_result_view_t<result_t>::gather(
  handle_t const& handle,
  raft::device_span<vertex_t const> vertices,
  cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<vertex_t>>& renumber_map_view)
{
  auto this_gpu_graph_view = graph_view.get(handle);

  rmm::device_uvector<vertex_t> local_vertices(vertices.size(), handle.get_stream());
  rmm::device_uvector<int> vertex_gpu_ids(vertices.size(), handle.get_stream());
  rmm::device_uvector<size_t> vertex_pos(vertices.size(), handle.get_stream());
  rmm::device_uvector<result_t> result(vertices.size(), handle.get_stream());

  raft::copy(local_vertices.data(), vertices.data(), vertices.size(), handle.get_stream());
  cugraph::detail::scalar_fill(
    handle.get_stream(), vertex_gpu_ids.data(), vertex_gpu_ids.size(), handle.get_rank());
  cugraph::detail::sequence_fill(
    handle.get_stream(), vertex_pos.data(), vertex_pos.size(), size_t{0});

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
    this_gpu_graph_view.vertex_partition_range_lasts().size(), handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      this_gpu_graph_view.vertex_partition_range_lasts().data(),
                      this_gpu_graph_view.vertex_partition_range_lasts().size(),
                      handle.get_stream());

  if (renumber_map_view) {
    cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
      handle.raft_handle(),
      local_vertices.data(),
      local_vertices.size(),
      renumber_map_view->get(handle).data(),
      this_gpu_graph_view.local_vertex_partition_range_first(),
      this_gpu_graph_view.local_vertex_partition_range_last());
  }

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
      cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size},
      handle.get_stream());

  //
  //  Now gather
  //
  rmm::device_uvector<result_t> tmp_result(local_vertices.size(), handle.get_stream());

  auto& wrapped = this->get(handle);

  auto vertex_partition = vertex_partition_device_view_t<vertex_t, multi_gpu>(
    this_gpu_graph_view.local_vertex_partition_view());

  auto iter =
    thrust::make_transform_iterator(local_vertices.begin(), [vertex_partition] __device__(auto v) {
      return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
    });

  thrust::gather(handle.get_thrust_policy(),
                 iter,
                 iter + local_vertices.size(),
                 wrapped.begin(),
                 tmp_result.begin());

  //
  // Shuffle back
  //
  std::forward_as_tuple(std::ignore, std::tie(std::ignore, vertex_pos, tmp_result), std::ignore) =
    groupby_gpu_id_and_shuffle_kv_pairs(
      handle.raft_handle().get_comms(),
      vertex_gpu_ids.begin(),
      vertex_gpu_ids.end(),
      thrust::make_zip_iterator(local_vertices.begin(), vertex_pos.begin(), tmp_result.begin()),
      [] __device__(int gpu) { return gpu; },
      handle.get_stream());

  //
  // Finally, reorder result
  //
  thrust::scatter(handle.get_thrust_policy(),
                  tmp_result.begin(),
                  tmp_result.end(),
                  vertex_pos.begin(),
                  result.begin());

  return result;
}

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int64_t> vertex_result_view_t<int64_t>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int64_t> vertex_result_view_t<int64_t>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int64_t> vertex_result_view_t<int64_t>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int32_t> vertex_result_view_t<int32_t>::gather(
  handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::mtmg::graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int32_t>>& renumber_map_view);

template rmm::device_uvector<int64_t> vertex_result_view_t<int64_t>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::mtmg::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

}  // namespace mtmg
}  // namespace cugraph

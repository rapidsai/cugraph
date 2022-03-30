/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <sampling/detail/gather_utils_impl.cuh>

namespace cugraph {

namespace detail {

template rmm::device_uvector<int32_t> compute_local_major_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, float, false, true> const& graph_view);

template rmm::device_uvector<int32_t> compute_local_major_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, true> const& graph_view);

template rmm::device_uvector<int64_t> compute_local_major_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, float, false, true> const& graph_view);

template rmm::device_uvector<int64_t> compute_local_major_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, true> const& graph_view);

template rmm::device_uvector<int64_t> compute_local_major_degrees(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, float, false, true> const& graph_view);

template rmm::device_uvector<int64_t> compute_local_major_degrees(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, true> const& graph_view);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
get_global_degree_information(raft::handle_t const& handle,
                              graph_view_t<int32_t, int32_t, float, false, true> const& graph_view);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, true> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(raft::handle_t const& handle,
                              graph_view_t<int32_t, int64_t, float, false, true> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, true> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(raft::handle_t const& handle,
                              graph_view_t<int64_t, int64_t, float, false, true> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, true> const& graph_view);

template rmm::device_uvector<int32_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
  rmm::device_uvector<int32_t> const& global_degree_offsets,
  rmm::device_uvector<int32_t> const& global_out_degrees);

template rmm::device_uvector<int32_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
  rmm::device_uvector<int32_t> const& global_degree_offsets,
  rmm::device_uvector<int32_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
gather_active_majors(raft::handle_t const& handle,
                     graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
                     int32_t const* vertex_input_first,
                     int32_t const* vertex_input_last,
                     int32_t const* gpu_id_first);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
gather_active_majors(raft::handle_t const& handle,
                     graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
                     int32_t const* vertex_input_first,
                     int32_t const* vertex_input_last,
                     int32_t const* gpu_id_first);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
gather_active_majors(raft::handle_t const& handle,
                     graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
                     int32_t const* vertex_input_first,
                     int32_t const* vertex_input_last,
                     int32_t const* gpu_id_first);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
gather_active_majors(raft::handle_t const& handle,
                     graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
                     int32_t const* vertex_input_first,
                     int32_t const* vertex_input_last,
                     int32_t const* gpu_id_first);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
gather_active_majors(raft::handle_t const& handle,
                     graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
                     int64_t const* vertex_input_first,
                     int64_t const* vertex_input_last,
                     int32_t const* gpu_id_first);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
gather_active_majors(raft::handle_t const& handle,
                     graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
                     int64_t const* vertex_input_first,
                     int64_t const* vertex_input_last,
                     int32_t const* gpu_id_first);

template rmm::device_uvector<int32_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int32_t>& global_out_degrees);

template rmm::device_uvector<int32_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int32_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
  const rmm::device_uvector<int64_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
  const rmm::device_uvector<int64_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template std::tuple<
  rmm::device_uvector<edge_partition_device_view_t<int32_t, int32_t, float, true>>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>>
partition_information(raft::handle_t const& handle,
                      graph_view_t<int32_t, int32_t, float, false, true> const& graph_view);

template std::tuple<
  rmm::device_uvector<edge_partition_device_view_t<int32_t, int32_t, double, true>>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>>
partition_information(raft::handle_t const& handle,
                      graph_view_t<int32_t, int32_t, double, false, true> const& graph_view);

template std::tuple<
  rmm::device_uvector<edge_partition_device_view_t<int32_t, int64_t, float, true>>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>>
partition_information(raft::handle_t const& handle,
                      graph_view_t<int32_t, int64_t, float, false, true> const& graph_view);

template std::tuple<
  rmm::device_uvector<edge_partition_device_view_t<int32_t, int64_t, double, true>>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>>
partition_information(raft::handle_t const& handle,
                      graph_view_t<int32_t, int64_t, double, false, true> const& graph_view);

template std::tuple<
  rmm::device_uvector<edge_partition_device_view_t<int64_t, int64_t, float, true>>,
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>>
partition_information(raft::handle_t const& handle,
                      graph_view_t<int64_t, int64_t, float, false, true> const& graph_view);

template std::tuple<
  rmm::device_uvector<edge_partition_device_view_t<int64_t, int64_t, double, true>>,
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>>
partition_information(raft::handle_t const& handle,
                      graph_view_t<int64_t, int64_t, double, false, true> const& graph_view);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                   rmm::device_uvector<int32_t>&& minor_map,
                   int32_t indices_per_major,
                   const rmm::device_uvector<int32_t>& global_degree_offsets,
                   const rmm::device_uvector<int32_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                   rmm::device_uvector<int32_t>&& minor_map,
                   int32_t indices_per_major,
                   const rmm::device_uvector<int32_t>& global_degree_offsets,
                   const rmm::device_uvector<int32_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
                   const rmm::device_uvector<int64_t>& active_majors,
                   const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
                   const rmm::device_uvector<int64_t>& active_majors,
                   const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int32_t, float, false, true> const& graph_view,
                        const rmm::device_uvector<int32_t>& active_majors,
                        const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                        const rmm::device_uvector<int32_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int32_t, double, false, true> const& graph_view,
                        const rmm::device_uvector<int32_t>& active_majors,
                        const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                        const rmm::device_uvector<int32_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int64_t, float, false, true> const& graph_view,
                        const rmm::device_uvector<int32_t>& active_majors,
                        const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                        const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int64_t, double, false, true> const& graph_view,
                        const rmm::device_uvector<int32_t>& active_majors,
                        const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                        const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int64_t, int64_t, float, false, true> const& graph_view,
                        const rmm::device_uvector<int64_t>& active_majors,
                        const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                        const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int64_t>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int64_t, int64_t, double, false, true> const& graph_view,
                        const rmm::device_uvector<int64_t>& active_majors,
                        const rmm::device_uvector<int32_t>& active_major_gpu_ids,
                        const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

}  // namespace detail

}  // namespace cugraph

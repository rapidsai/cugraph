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

#include <sampling/detail/sampling_utils_impl.cuh>

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, float, false, false> const& graph_view);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, false> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, float, false, false> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, false> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, float, false, false> const& graph_view);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
get_global_degree_information(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, false> const& graph_view);

template rmm::device_uvector<int32_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
  rmm::device_uvector<int32_t> const& global_degree_offsets,
  rmm::device_uvector<int32_t> const& global_out_degrees);

template rmm::device_uvector<int32_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
  rmm::device_uvector<int32_t> const& global_degree_offsets,
  rmm::device_uvector<int32_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template rmm::device_uvector<int64_t> get_global_adjacency_offset(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
  rmm::device_uvector<int64_t> const& global_degree_offsets,
  rmm::device_uvector<int64_t> const& global_out_degrees);

template rmm::device_uvector<int32_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int32_t>& global_out_degrees);

template rmm::device_uvector<int32_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int32_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
  const rmm::device_uvector<int32_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
  const rmm::device_uvector<int64_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template rmm::device_uvector<int64_t> get_active_major_global_degrees(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
  const rmm::device_uvector<int64_t>& active_majors,
  const rmm::device_uvector<int64_t>& global_out_degrees);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   rmm::device_uvector<int32_t>&& minor_map,
                   int32_t indices_per_major,
                   const rmm::device_uvector<int32_t>& global_degree_offsets,
                   const rmm::device_uvector<int32_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
                   const rmm::device_uvector<int64_t>& active_majors,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   rmm::device_uvector<int32_t>&& minor_map,
                   int32_t indices_per_major,
                   const rmm::device_uvector<int32_t>& global_degree_offsets,
                   const rmm::device_uvector<int32_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
                   const rmm::device_uvector<int32_t>& active_majors,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_local_edges(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
                   const rmm::device_uvector<int64_t>& active_majors,
                   rmm::device_uvector<int64_t>&& minor_map,
                   int64_t indices_per_major,
                   const rmm::device_uvector<int64_t>& global_degree_offsets,
                   const rmm::device_uvector<int64_t>& global_adjacency_list_offsets);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
                        rmm::device_uvector<int64_t> const& active_majors);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
                        rmm::device_uvector<int32_t> const& active_majors);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
                        rmm::device_uvector<int64_t> const& active_majors);

//  Only need to build once, not separately for SG/MG
template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<int32_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<float>&& wgt);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<float>&& wgt);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<float>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int64_t>&& src,
                            rmm::device_uvector<int64_t>&& dst,
                            rmm::device_uvector<float>&& wgt);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<int32_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<double>&& wgt);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int32_t>&& src,
                            rmm::device_uvector<int32_t>&& dst,
                            rmm::device_uvector<double>&& wgt);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    rmm::device_uvector<double>,
                    rmm::device_uvector<int64_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<int64_t>&& src,
                            rmm::device_uvector<int64_t>&& dst,
                            rmm::device_uvector<double>&& wgt);

}  // namespace detail
}  // namespace cugraph

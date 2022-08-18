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
#include <community/detail/common_methods.cuh>

namespace cugraph {
namespace detail {

template float compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int32_t, float, false, false>,
                                int32_t> const& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, float, false, false>,
                                int32_t> const& dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<float> const& cluster_weights,
  float total_edge_weight,
  float resolution);

template float compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int64_t, float, false, false>,
                                int32_t> const& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, float, false, false>,
                                int32_t> const& dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<float> const& cluster_weights,
  float total_edge_weight,
  float resolution);

template float compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
  edge_partition_src_property_t<cugraph::graph_view_t<int64_t, int64_t, float, false, false>,
                                int64_t> const& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, float, false, false>,
                                int64_t> const& dst_clusters_cache,
  rmm::device_uvector<int64_t> const& next_clusters,
  rmm::device_uvector<float> const& cluster_weights,
  float total_edge_weight,
  float resolution);

template double compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int32_t, double, false, false>,
                                int32_t> const& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, double, false, false>,
                                int32_t> const& dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<double> const& cluster_weights,
  double total_edge_weight,
  double resolution);

template double compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int64_t, double, false, false>,
                                int32_t> const& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, double, false, false>,
                                int32_t> const& dst_clusters_cache,
  rmm::device_uvector<int32_t> const& next_clusters,
  rmm::device_uvector<double> const& cluster_weights,
  double total_edge_weight,
  double resolution);

template double compute_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
  edge_partition_src_property_t<cugraph::graph_view_t<int64_t, int64_t, double, false, false>,
                                int64_t> const& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, double, false, false>,
                                int64_t> const& dst_clusters_cache,
  rmm::device_uvector<int64_t> const& next_clusters,
  rmm::device_uvector<double> const& cluster_weights,
  double total_edge_weight,
  double resolution);

template cugraph::graph_t<int32_t, int32_t, float, false, false> graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
  raft::device_span<int32_t> labels);

template cugraph::graph_t<int32_t, int64_t, float, false, false> graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
  raft::device_span<int32_t> labels);

template cugraph::graph_t<int64_t, int64_t, float, false, false> graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
  raft::device_span<int64_t> labels);

template cugraph::graph_t<int32_t, int32_t, double, false, false> graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
  raft::device_span<int32_t> labels);

template cugraph::graph_t<int32_t, int64_t, double, false, false> graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
  raft::device_span<int32_t> labels);

template cugraph::graph_t<int64_t, int64_t, double, false, false> graph_contraction(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
  raft::device_span<int64_t> labels);

template void update_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, float, false, false> const& graph_view,
  float total_edge_weight,
  float resolution,
  rmm::device_uvector<float>& vertex_weights_v,
  rmm::device_uvector<int32_t>& cluster_keys_v,
  rmm::device_uvector<float>& cluster_weights_v,
  rmm::device_uvector<int32_t>& next_clusters_v,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int32_t, float, false, false>,
                                float> const& src_vertex_weights_cache,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int32_t, float, false, false>,
                                int32_t>& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, float, false, false>,
                                int32_t>& dst_clusters_cache,
  bool up_down);

template void update_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
  float total_edge_weight,
  float resolution,
  rmm::device_uvector<float>& vertex_weights_v,
  rmm::device_uvector<int32_t>& cluster_keys_v,
  rmm::device_uvector<float>& cluster_weights_v,
  rmm::device_uvector<int32_t>& next_clusters_v,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int64_t, float, false, false>,
                                float> const& src_vertex_weights_cache,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int64_t, float, false, false>,
                                int32_t>& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, float, false, false>,
                                int32_t>& dst_clusters_cache,
  bool up_down);

template void update_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
  float total_edge_weight,
  float resolution,
  rmm::device_uvector<float>& vertex_weights_v,
  rmm::device_uvector<int64_t>& cluster_keys_v,
  rmm::device_uvector<float>& cluster_weights_v,
  rmm::device_uvector<int64_t>& next_clusters_v,
  edge_partition_src_property_t<cugraph::graph_view_t<int64_t, int64_t, float, false, false>,
                                float> const& src_vertex_weights_cache,
  edge_partition_src_property_t<cugraph::graph_view_t<int64_t, int64_t, float, false, false>,
                                int64_t>& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, float, false, false>,
                                int64_t>& dst_clusters_cache,
  bool up_down);

template void update_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
  double total_edge_weight,
  double resolution,
  rmm::device_uvector<double>& vertex_weights_v,
  rmm::device_uvector<int32_t>& cluster_keys_v,
  rmm::device_uvector<double>& cluster_weights_v,
  rmm::device_uvector<int32_t>& next_clusters_v,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int32_t, double, false, false>,
                                double> const& src_vertex_weights_cache,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int32_t, double, false, false>,
                                int32_t>& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int32_t, double, false, false>,
                                int32_t>& dst_clusters_cache,
  bool up_down);

template void update_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
  double total_edge_weight,
  double resolution,
  rmm::device_uvector<double>& vertex_weights_v,
  rmm::device_uvector<int32_t>& cluster_keys_v,
  rmm::device_uvector<double>& cluster_weights_v,
  rmm::device_uvector<int32_t>& next_clusters_v,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int64_t, double, false, false>,
                                double> const& src_vertex_weights_cache,
  edge_partition_src_property_t<cugraph::graph_view_t<int32_t, int64_t, double, false, false>,
                                int32_t>& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int32_t, int64_t, double, false, false>,
                                int32_t>& dst_clusters_cache,
  bool up_down);

template void update_by_delta_modularity(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
  double total_edge_weight,
  double resolution,
  rmm::device_uvector<double>& vertex_weights_v,
  rmm::device_uvector<int64_t>& cluster_keys_v,
  rmm::device_uvector<double>& cluster_weights_v,
  rmm::device_uvector<int64_t>& next_clusters_v,
  edge_partition_src_property_t<cugraph::graph_view_t<int64_t, int64_t, double, false, false>,
                                double> const& src_vertex_weights_cache,
  edge_partition_src_property_t<cugraph::graph_view_t<int64_t, int64_t, double, false, false>,
                                int64_t>& src_clusters_cache,
  edge_partition_dst_property_t<cugraph::graph_view_t<int64_t, int64_t, double, false, false>,
                                int64_t>& dst_clusters_cache,
  bool up_down);

}  // namespace detail
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utilities/conversion_utilities_impl.cuh"

namespace cugraph {
namespace test {

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

}  // namespace test
}  // namespace cugraph

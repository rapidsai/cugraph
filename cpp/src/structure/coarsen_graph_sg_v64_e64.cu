/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/coarsen_graph_impl.cuh"

namespace cugraph {

// SG instantiation

template std::tuple<graph_t<int64_t, int64_t, true, false>,
                    std::optional<edge_property_t<int64_t, float>>,
                    std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, true, false> const& graph_view,
              std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<graph_t<int64_t, int64_t, false, false>,
                    std::optional<edge_property_t<int64_t, float>>,
                    std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, false, false> const& graph_view,
              std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<graph_t<int64_t, int64_t, true, false>,
                    std::optional<edge_property_t<int64_t, double>>,
                    std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, true, false> const& graph_view,
              std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<graph_t<int64_t, int64_t, false, false>,
                    std::optional<edge_property_t<int64_t, double>>,
                    std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, false, false> const& graph_view,
              std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

}  // namespace cugraph

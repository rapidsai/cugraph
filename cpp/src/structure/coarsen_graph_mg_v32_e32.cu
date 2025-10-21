/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/coarsen_graph_impl.cuh"

namespace cugraph {

// MG instantiation

template std::tuple<graph_t<int32_t, int32_t, true, true>,
                    std::optional<edge_property_t<int32_t, float>>,
                    std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, false, true>,
                    std::optional<edge_property_t<int32_t, float>>,
                    std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, true, true>,
                    std::optional<edge_property_t<int32_t, double>>,
                    std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, false, true>,
                    std::optional<edge_property_t<int32_t, double>>,
                    std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

}  // namespace cugraph

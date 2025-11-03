/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/symmetrize_graph_impl.cuh"

namespace cugraph {

// MG instantiation

template std::tuple<graph_t<int32_t, int32_t, true, true>,
                    std::optional<edge_property_t<int32_t, float>>,
                    std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(raft::handle_t const& handle,
                 graph_t<int32_t, int32_t, true, true>&& graph,
                 std::optional<edge_property_t<int32_t, float>>&& edge_weights,
                 std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
                 bool reciprocal,
                 bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, false, true>,
                    std::optional<edge_property_t<int32_t, float>>,
                    std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(raft::handle_t const& handle,
                 graph_t<int32_t, int32_t, false, true>&& graph,
                 std::optional<edge_property_t<int32_t, float>>&& edge_weights,
                 std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
                 bool reciprocal,
                 bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, true, true>,
                    std::optional<edge_property_t<int32_t, double>>,
                    std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(raft::handle_t const& handle,
                 graph_t<int32_t, int32_t, true, true>&& graph,
                 std::optional<edge_property_t<int32_t, double>>&& edge_weights,
                 std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
                 bool reciprocal,
                 bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, false, true>,
                    std::optional<edge_property_t<int32_t, double>>,
                    std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(raft::handle_t const& handle,
                 graph_t<int32_t, int32_t, false, true>&& graph,
                 std::optional<edge_property_t<int32_t, double>>&& edge_weights,
                 std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
                 bool reciprocal,
                 bool do_expensive_check);

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/transpose_graph_storage_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT std::tuple<graph_t<int64_t, int64_t, false, true>,
                                   std::optional<edge_property_t<int64_t, float>>,
                                   std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(raft::handle_t const& handle,
                        graph_t<int64_t, int64_t, true, true>&& graph,
                        std::optional<edge_property_t<int64_t, float>>&& edge_weights,
                        std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
                        bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<graph_t<int64_t, int64_t, true, true>,
                                   std::optional<edge_property_t<int64_t, float>>,
                                   std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(raft::handle_t const& handle,
                        graph_t<int64_t, int64_t, false, true>&& graph,
                        std::optional<edge_property_t<int64_t, float>>&& edge_weights,
                        std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
                        bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<graph_t<int64_t, int64_t, false, true>,
                                   std::optional<edge_property_t<int64_t, double>>,
                                   std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(raft::handle_t const& handle,
                        graph_t<int64_t, int64_t, true, true>&& graph,
                        std::optional<edge_property_t<int64_t, double>>&& edge_weights,
                        std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
                        bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<graph_t<int64_t, int64_t, true, true>,
                                   std::optional<edge_property_t<int64_t, double>>,
                                   std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(raft::handle_t const& handle,
                        graph_t<int64_t, int64_t, false, true>&& graph,
                        std::optional<edge_property_t<int64_t, double>>&& edge_weights,
                        std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
                        bool do_expensive_check);

}  // namespace cugraph

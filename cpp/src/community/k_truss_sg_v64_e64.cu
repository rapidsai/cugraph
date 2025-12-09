/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/k_truss_impl.cuh"

namespace cugraph {

// SG instantiation

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
k_truss(raft::handle_t const& handle,
        graph_view_t<int64_t, int64_t, false, false> const& graph_view,
        std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
        int64_t k,
        bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
k_truss(raft::handle_t const& handle,
        graph_view_t<int64_t, int64_t, false, false> const& graph_view,
        std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
        int64_t k,
        bool do_expensive_check);

}  // namespace cugraph

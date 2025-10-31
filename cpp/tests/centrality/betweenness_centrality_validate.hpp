/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>

namespace cugraph {
namespace test {

template <typename weight_t>
void betweenness_centrality_validate(raft::handle_t const& handle,
                                     rmm::device_uvector<weight_t>& d_cugraph_results,
                                     rmm::device_uvector<weight_t>& d_reference_results);

template <typename vertex_t, typename weight_t>
void edge_betweenness_centrality_validate(raft::handle_t const& handle,
                                          rmm::device_uvector<vertex_t>& d_cugraph_src_vertex_ids,
                                          rmm::device_uvector<vertex_t>& d_cugraph_dst_vertex_ids,
                                          rmm::device_uvector<weight_t>& d_cugraph_results,
                                          rmm::device_uvector<vertex_t>& d_reference_src_vertex_ids,
                                          rmm::device_uvector<vertex_t>& d_reference_dst_vertex_ids,
                                          rmm::device_uvector<weight_t>& d_reference_results);

}  // namespace test
}  // namespace cugraph

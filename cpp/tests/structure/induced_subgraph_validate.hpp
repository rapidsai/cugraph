/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>

template <typename vertex_t, typename weight_t>
void induced_subgraph_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_cugraph_subgraph_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_cugraph_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_cugraph_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_cugraph_subgraph_edge_offsets,
  rmm::device_uvector<vertex_t>& d_reference_subgraph_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_reference_subgraph_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_reference_subgraph_edgelist_weights,
  rmm::device_uvector<size_t>& d_reference_subgraph_edge_offsets);

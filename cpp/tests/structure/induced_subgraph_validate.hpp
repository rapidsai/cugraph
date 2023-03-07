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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
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

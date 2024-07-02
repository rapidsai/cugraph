/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#pragma once

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>

// utilities for testing / verification of Nbr Sampling functionality:
//
namespace cugraph {
namespace test {

template <typename vertex_t, typename weight_t>
bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t> const& src,
  rmm::device_uvector<vertex_t> const& dst,
  std::optional<rmm::device_uvector<weight_t>> const& wgt,
  rmm::device_uvector<vertex_t> const& subgraph_src,
  rmm::device_uvector<vertex_t> const& subgraph_dst,
  std::optional<rmm::device_uvector<weight_t>> const& subgraph_wgt);

template <typename vertex_t, typename weight_t>
bool validate_sampling_depth(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t>&& d_src,
                             rmm::device_uvector<vertex_t>&& d_dst,
                             std::optional<rmm::device_uvector<weight_t>>&& d_wgt,
                             rmm::device_uvector<vertex_t>&& d_source_vertices,
                             int max_depth);

}  // namespace test
}  // namespace cugraph

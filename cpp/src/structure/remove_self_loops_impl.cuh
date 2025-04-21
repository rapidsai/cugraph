/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "structure/detail/structure_utils.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <optional>

namespace cugraph {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>>
remove_self_loops(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edgelist_srcs,
                  rmm::device_uvector<vertex_t>&& edgelist_dsts,
                  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                  std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
                  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_start_times,
                  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_end_times)
{
  auto [keep_count, keep_flags] =
    detail::mark_entries(handle,
                         edgelist_srcs.size(),
                         [d_srcs = edgelist_srcs.data(), d_dsts = edgelist_dsts.data()] __device__(
                           size_t i) { return d_srcs[i] != d_dsts[i]; });

  if (keep_count < edgelist_srcs.size()) {
    edgelist_srcs = detail::keep_flagged_elements(
      handle,
      std::move(edgelist_srcs),
      raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
      keep_count);
    edgelist_dsts = detail::keep_flagged_elements(
      handle,
      std::move(edgelist_dsts),
      raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
      keep_count);

    if (edgelist_weights)
      edgelist_weights = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_weights),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);

    if (edgelist_edge_ids)
      edgelist_edge_ids = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_edge_ids),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);

    if (edgelist_edge_types)
      edgelist_edge_types = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_edge_types),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);

    if (edgelist_edge_start_times)
      edgelist_edge_start_times = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_edge_start_times),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);

    if (edgelist_edge_end_times)
      edgelist_edge_end_times = detail::keep_flagged_elements(
        handle,
        std::move(*edgelist_edge_end_times),
        raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
        keep_count);
  }

  return std::make_tuple(std::move(edgelist_srcs),
                         std::move(edgelist_dsts),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_edge_start_times),
                         std::move(edgelist_edge_end_times));
}

}  // namespace cugraph

/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "prims/detail/nbr_intersection.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <tuple>

namespace cugraph {

/**
 * @brief Iterate over each input vertex pair and returns the common destination neighbor list
 * pair in a CSR-like format
 *
 * Iterate over every vertex pair; intersect destination neighbor lists of the two vertices in the
 * pair and store the result in a CSR-like format
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexPairIterator Type of the iterator for input vertex pairs.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_pair_first Iterator pointing to the first (inclusive) input vertex pair.
 * @param vertex_pair_last Iterator pointing to the last (exclusive) input vertex pair.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of intersection offsets and indices.
 */
template <typename GraphViewType, typename VertexPairIterator>
std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<typename GraphViewType::vertex_type>>
per_v_pair_dst_nbr_intersection(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexPairIterator vertex_pair_first,
                                VertexPairIterator vertex_pair_last,
                                bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed);

  return detail::nbr_intersection(handle,
                                  graph_view,
                                  cugraph::edge_dummy_property_t{}.view(),
                                  vertex_pair_first,
                                  vertex_pair_last,
                                  std::array<bool, 2>{true, true},
                                  do_expensive_check);
}

}  // namespace cugraph

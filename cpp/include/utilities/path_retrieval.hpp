/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>

namespace cugraph {

/**
 * @brief Takes the results of BFS or SSSP function call and sums the given
 * weights along the path to the starting vertex.
 *
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms. Must have at least one worker stream.
 * @param vertices Pointer to vertex ids.
 * @param preds Pointer to predecessors.
 * @param info_weights Secondary weights along the edge from predecessor to vertex.
 * @param out Contains for each index the sum of weights along the path unfolding.
 * @param num_vertices Number of vertices.
 **/
template <typename vertex_t, typename weight_t>
void get_traversed_cost(raft::handle_t const &handle,
                        vertex_t const *vertices,
                        vertex_t const *preds,
                        weight_t const *info_weights,
                        weight_t *out,
                        vertex_t num_vertices);
}  // namespace cugraph

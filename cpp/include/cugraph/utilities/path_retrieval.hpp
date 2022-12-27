/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

namespace cugraph {

/**
 * @brief Takes the results of BFS or SSSP function call and sums the given
 * weights along the path to the starting vertex.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
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
void get_traversed_cost(raft::handle_t const& handle,
                        vertex_t const* vertices,
                        vertex_t const* preds,
                        weight_t const* info_weights,
                        weight_t* out,
                        vertex_t stop_vertex,
                        vertex_t num_vertices);

/**
 * @brief returns the COO format (src_vector, dst_vector) from the random walks (RW)
 * paths.
 *
 * @tparam vertex_t Type of vertex indices.
 * @tparam index_t Type used to store indexing and sizes.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param coalesced_sz_v coalesced vertex vector size.
 * @param num_paths number of paths.
 * @param d_coalesced_v coalesced vertex buffer.
 * @param d_sizes paths size buffer.
 * @return tuple of (src_vertex_vector, dst_Vertex_vector, path_offsets), where
 * path_offsets are the offsets where the COO set of each path starts.
 */
template <typename vertex_t, typename index_t>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<index_t>>
  convert_paths_to_coo(raft::handle_t const& handle,
                       index_t coalesced_sz_v,
                       index_t num_paths,
                       rmm::device_buffer&& d_coalesced_v,
                       rmm::device_buffer&& d_sizes);

/**
 * @brief returns additional RW information on vertex paths offsets and weight path sizes and
 * offsets, for the coalesced case (the padded case does not need or provide this information)
 *
 * @tparam index_t Type used to store indexing and sizes.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param num_paths number of paths.
 * @param ptr_d_sizes sizes of vertex paths.
 * @return tuple of (vertex_path_offsets, weight_path_sizes, weight_path_offsets), where offsets are
 * exclusive scan of corresponding sizes.
 */
template <typename index_t>
std::tuple<rmm::device_uvector<index_t>, rmm::device_uvector<index_t>, rmm::device_uvector<index_t>>
query_rw_sizes_offsets(raft::handle_t const& handle, index_t num_paths, index_t const* ptr_d_sizes);

}  // namespace cugraph

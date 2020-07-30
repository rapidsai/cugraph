/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/device_buffer.hpp>

#include <graph.hpp>

namespace cugraph {

/**
 * @brief    Convert COO to CSR
 *
 * Takes a list of edges in COOrdinate format and generates a CSR format.
 *
 * @throws                    cugraph::logic_error when an error occurs.
 *
 * @tparam VT                 type of vertex index
 * @tparam ET                 type of edge index
 * @tparam WT                 type of the edge weight
 *
 * @param[in]  graph          cuGraph graph in coordinate format
 * @param[in]  mr             Memory resource used to allocate the returned graph
 *
 * @return                    Unique pointer to generate Compressed Sparse Row graph
 *
 */
template <typename VT, typename ET, typename WT>
std::unique_ptr<GraphCSR<VT, ET, WT>> coo_to_csr(
  GraphCOOView<VT, ET, WT> const &graph,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief    Renumber source and destination indices
 *
 * Renumber source and destination indexes to be a dense numbering,
 * using contiguous values between 0 and number of vertices minus 1.
 *
 * @throws                    cugraph::logic_error when an error occurs.
 *
 * @tparam VT_IN              type of vertex index input
 * @tparam VT_OUT             type of vertex index output
 * @tparam ET                 type of edge index
 *
 * @param[in]  number_of_edges number of edges in the graph
 * @param[in]  src            Pointer to device memory containing source vertex ids
 * @param[in]  dst            Pointer to device memory containing destination vertex ids
 * @param[out] src_renumbered Pointer to device memory containing the output source vertices.
 * @param[out] dst_renumbered Pointer to device memory containing the output destination vertices.
 * @param[out] map_size       Pointer to local memory containing the number of elements in the
 * renumbering map
 * @param[in]  mr             Memory resource used to allocate the returned graph
 *
 * @return                    Unique pointer to renumbering map
 *
 */
template <typename VT_IN, typename VT_OUT, typename ET>
std::unique_ptr<rmm::device_buffer> renumber_vertices(
  ET number_of_edges,
  VT_IN const *src,
  VT_IN const *dst,
  VT_OUT *src_renumbered,
  VT_OUT *dst_renumbered,
  ET *map_size,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief    Broadcast using handle communicator
 *
 * Use handle's communicator to operate broadcasting.
 *
 * @throws                    cugraph::logic_error when an error occurs.
 *
 * @tparam value_t            Type of the data to broadcast
 *
 * @param[out] value          Point to the data
 * @param[in]  count          Number of elements to broadcast
 *
 */

template <typename value_t>
void comms_bcast(const raft::handle_t &handle, value_t *value, size_t count)
{
  handle.get_comms().bcast(value, count, 0, handle.get_stream());
}
}  // namespace cugraph

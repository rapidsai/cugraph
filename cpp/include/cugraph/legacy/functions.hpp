/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <rmm/device_buffer.hpp>

#include <cugraph/legacy/graph.hpp>

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
std::unique_ptr<legacy::GraphCSR<VT, ET, WT>> coo_to_csr(
  legacy::GraphCOOView<VT, ET, WT> const& graph,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cugraph

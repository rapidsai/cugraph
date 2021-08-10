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
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

/**
 * @brief    Shuffle edgelist using the edge key function which returns the target GPU ID.
 *
 * NOTE:  d_edgelist_rows, d_edgelist_cols and d_edgelist_weights
 *        are modified within this function (data is sorted)
 *        But the actual output is returned. The exact contents
 *        of d_edgelist_rows, d_edgelist_cols and d_edgelist_weights
 *        after the function is undefined.
 *
 * @tparam         vertex_t             vertex type
 * @tparam         weight_t             weight type
 *
 * @param[in]      handle               raft handle
 * @param[in/out]  d_edgelist_majors    vertex IDs for rows (if the graph adjacency matrix is stored
 * as is) or columns (if the graph adjacency matrix is stored transposed)
 * @param[in/out]  d_edgelist_minors    vertex IDs for columns (if the graph adjacency matrix is
 * stored as is) or rows (if the graph adjacency matrix is stored transposed)
 * @param[in/out]  d_edgelist_weights   optional edge weights
 *
 * @return tuple of shuffled major vertices, minor vertices and optional weights
 */
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<vertex_t>& d_edgelist_majors,
                           rmm::device_uvector<vertex_t>& d_edgelist_minors,
                           std::optional<rmm::device_uvector<weight_t>>& d_edgelist_weights);

/**
 * @brief    Shuffle vertices using the vertex key function which returns the target GPU ID.
 *
 * NOTE:  d_value is modified within this function
 *        (data is sorted).  But the actual output is returned.
 *        The exact contents of d_value after the function is
 *        undefined.
 *
 * @tparam         vertex_t      vertex type
 *
 * @param[in]      handle        raft handle
 * @param[in/out]  d_vertices    vertex ids to shuffle
 *
 * @return device vector of shuffled vertices
 */
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_vertices_by_gpu_id(raft::handle_t const& handle,
                                                         rmm::device_uvector<vertex_t>& d_vertices);

/**
 * @brief    Groupby and count edgelist using the edge key function which returns the target local
 * partition ID.
 *
 * NOTE:  d_edgelist_rows, d_edgelist_cols and d_edgelist_weights
 *        are modified within this function (data is sorted)
 *        But the actual output is returned. The exact contents
 *        of d_edgelist_rows, d_edgelist_cols and d_edgelist_weights
 *        after the function is undefined.
 *
 * @tparam         vertex_t             vertex type
 * @tparam         weight_t             weight type
 *
 * @param[in]      handle               raft handle
 * @param[in/out]  d_edgelist_majors    vertex IDs for rows (if the graph adjacency matrix is stored
 * as is) or columns (if the graph adjacency matrix is stored transposed)
 * @param[in/out]  d_edgelist_minors    vertex IDs for columns (if the graph adjacency matrix is
 * stored as is) or rows (if the graph adjacency matrix is stored transposed)
 * @param[in/out]  d_edgelist_weights   optional edge weights
 *
 * @return tuple of shuffled rows, columns and optional weights
 */
template <typename vertex_t, typename weight_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_edgelist_weights,
  size_t number_of_local_adj_matrix_partitions);

}  // namespace detail
}  // namespace cugraph

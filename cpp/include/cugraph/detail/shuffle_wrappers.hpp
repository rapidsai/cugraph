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
 * @brief    Shuffle edgelist using the edge key function
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
 * @param[in/out]  d_edgelist_rows      vertex ids for row
 * @param[in/out]  d_edgelist_cols      vertex ids for column
 * @param[in/out]  d_edgelist_weights   optional edge weights
 * @param[in]      store_transposed     true if operating on
 *                                      transposed matrix
 *
 * @return tuple of shuffled rows, columns and optional weights
 */
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
shuffle_edgelist_by_edge(raft::handle_t const &handle,
                         rmm::device_uvector<vertex_t> &d_edgelist_rows,
                         rmm::device_uvector<vertex_t> &d_edgelist_cols,
                         std::optional<rmm::device_uvector<weight_t>> &d_edgelist_weights,
                         bool store_transposed);

/**
 * @brief    Shuffle vertices using the vertex key function
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
rmm::device_uvector<vertex_t> shuffle_vertices(raft::handle_t const &handle,
                                               rmm::device_uvector<vertex_t> &d_vertices);

}  // namespace detail
}  // namespace cugraph

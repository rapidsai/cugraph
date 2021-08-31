/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cugraph/graph_view.hpp>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/prims/transform_reduce_e.cuh>

#include <raft/handle.hpp>

#include <cstdint>

namespace cugraph {

/**
 * @brief Count the number of edges that satisfies the given predicate.
 *
 * This function is inspired by thrust::count_if().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColValueInputWrapper Type of the wrapper for graph adjacency matrix column input
 * properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input Device-copyable wrapper used to access row input properties
 * (for the rows assigned to this process in multi-GPU). Use either
 * cugraph::row_properties_t::device_view() (if @p e_op needs to access row properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access row properties). Use
 * copy_to_adj_matrix_row to fill the wrapper.
 * @param adj_matrix_col_value_input Device-copyable wrapper used to access column input properties
 * (for the columns assigned to this process in multi-GPU). Use either
 * cugraph::col_properties_t::device_view() (if @p e_op needs to access column properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access column properties). Use
 * copy_to_adj_matrix_col to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), properties for the row (i.e. source), and properties for the column  (i.e. destination)
 * and returns true if this edge should be included in the returned count.
 * @return GraphViewType::edge_type Number of times @p e_op returned true.
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp>
typename GraphViewType::edge_type count_if_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  return transform_reduce_e(handle,
                            graph_view,
                            adj_matrix_row_value_input,
                            adj_matrix_col_value_input,
                            cast_edge_op_bool_to_integer<GraphViewType,
                                                         vertex_t,
                                                         AdjMatrixRowValueInputWrapper,
                                                         AdjMatrixColValueInputWrapper,
                                                         EdgeOp,
                                                         edge_t>{e_op},
                            edge_t{0});
}

}  // namespace cugraph

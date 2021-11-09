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
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>

#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

namespace detail {
}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and return an edge list with the edges with @p
 * edge_op evaluated to be true.
 *
 * This function is inspired by thrust::copy_if().
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
 * and returns a boolean value to designate whether to include this in the returned edge list (if
 * true is returned) or not (if false is returned).
 * @return std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
 * rmm::device_uvector<typename GraphViewType::vertex_type>,
 * std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>> Tuple storing an edge
 * list (sources, destinations, and optional weights).
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>>
copy_if_e(raft::handle_t const& handle,
          GraphViewType const& graph_view,
          AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
          AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
          EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights(0, handle.get_stream());

  CUGRAPH_FAIL("unimplemented.");

  return std::make_tuple(std::move(srcs), std::move(dsts), std::move(weights));
}

}  // namespace cugraph

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

#include <cugraph/detail/decompress_matrix_partition.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/prims/extract_if_e.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/prims/row_col_properties.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/error.hpp>

#include <raft/handle.hpp>

#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp>
struct call_e_op_t {
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu>
    matrix_partition{};
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input{};
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input{};
  EdgeOp e_op{};

  template <typename Edge>
  __device__ bool operator()(Edge e) const
  {
    static_assert((thrust::tuple_size<Edge>::value == 2) || (thrust::tuple_size<Edge>::value == 3));

    using vertex_t = typename GraphViewType::vertex_type;
    using weight_t = typename GraphViewType::weight_type;

    auto major = thrust::get<0>(e);
    auto minor = thrust::get<1>(e);
    weight_t weight{1.0};
    if constexpr (thrust::tuple_size<Edge>::value == 3) { weight = thrust::get<2>(e); }
    auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(major);
    auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
    auto row          = GraphViewType::is_adj_matrix_transposed ? minor : major;
    auto col          = GraphViewType::is_adj_matrix_transposed ? major : minor;
    auto row_offset   = GraphViewType::is_adj_matrix_transposed ? minor_offset : major_offset;
    auto col_offset   = GraphViewType::is_adj_matrix_transposed ? major_offset : minor_offset;
    return !evaluate_edge_op<GraphViewType,
                             vertex_t,
                             AdjMatrixRowValueInputWrapper,
                             AdjMatrixColValueInputWrapper,
                             EdgeOp>()
              .compute(row,
                       col,
                       weight,
                       adj_matrix_row_value_input.get(row_offset),
                       adj_matrix_col_value_input.get(col_offset),
                       e_op);
  }
};

}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and return an edge list with the edges with @p
 * edge_op evaluated to be true.
 *
 * This function is inspired by thrust::copy_if & thrust::remove_if().
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
 * and returns a boolean value to designate whether to include this edge in the returned edge list
 * (if true is returned) or not (if false is returned).
 * @return std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
 * rmm::device_uvector<typename GraphViewType::vertex_type>,
 * std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>> Tuple storing an
 * extracted edge list (sources, destinations, and optional weights).
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>>
extract_if_e(raft::handle_t const& handle,
             GraphViewType const& graph_view,
             AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
             AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
             EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  std::vector<size_t> edgelist_edge_counts(graph_view.get_number_of_local_adj_matrix_partitions(),
                                           size_t{0});
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    edgelist_edge_counts[i] =
      static_cast<size_t>(graph_view.get_number_of_local_adj_matrix_partition_edges(i));
  }
  auto number_of_local_edges =
    std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end());

  rmm::device_uvector<vertex_t> edgelist_majors(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
  auto edgelist_weights = graph_view.is_weighted()
                            ? std::make_optional<rmm::device_uvector<weight_t>>(
                                edgelist_majors.size(), handle.get_stream())
                            : std::nullopt;

  size_t cur_size{0};
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));

    auto matrix_partition_row_value_input = adj_matrix_row_value_input;
    auto matrix_partition_col_value_input = adj_matrix_col_value_input;
    if constexpr (GraphViewType::is_adj_matrix_transposed) {
      matrix_partition_col_value_input.set_local_adj_matrix_partition_idx(i);
    } else {
      matrix_partition_row_value_input.set_local_adj_matrix_partition_idx(i);
    }

    detail::decompress_matrix_partition_to_edgelist(
      handle,
      matrix_partition,
      edgelist_majors.data() + cur_size,
      edgelist_minors.data() + cur_size,
      edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data() + cur_size}
                       : std::nullopt,
      graph_view.get_local_adj_matrix_partition_segment_offsets(i));
    if (edgelist_weights) {
      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        edgelist_majors.begin(), edgelist_minors.begin(), (*edgelist_weights).begin()));
      cur_size += static_cast<size_t>(thrust::distance(
        edge_first + cur_size,
        thrust::remove_if(handle.get_thrust_policy(),
                          edge_first + cur_size,
                          edge_first + cur_size + edgelist_edge_counts[i],
                          detail::call_e_op_t<GraphViewType,
                                              AdjMatrixRowValueInputWrapper,
                                              AdjMatrixColValueInputWrapper,
                                              EdgeOp>{matrix_partition,
                                                      matrix_partition_row_value_input,
                                                      matrix_partition_col_value_input,
                                                      e_op})));
    } else {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
      cur_size += static_cast<size_t>(thrust::distance(
        edge_first + cur_size,
        thrust::remove_if(
          handle.get_thrust_policy(),
          edge_first + cur_size,
          edge_first + cur_size + edgelist_edge_counts[i],
          detail::call_e_op_t<GraphViewType,
                              AdjMatrixRowValueInputWrapper,
                              AdjMatrixColValueInputWrapper,
                              EdgeOp>{
            matrix_partition, adj_matrix_row_value_input, adj_matrix_col_value_input, e_op})));
    }
  }

  edgelist_majors.resize(cur_size, handle.get_stream());
  edgelist_minors.resize(edgelist_majors.size(), handle.get_stream());
  edgelist_majors.shrink_to_fit(handle.get_stream());
  edgelist_minors.shrink_to_fit(handle.get_stream());
  if (edgelist_weights) {
    (*edgelist_weights).resize(edgelist_majors.size(), handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(
    std::move(GraphViewType::is_adj_matrix_transposed ? edgelist_minors : edgelist_majors),
    std::move(GraphViewType::is_adj_matrix_transposed ? edgelist_majors : edgelist_minors),
    std::move(edgelist_weights));
}

}  // namespace cugraph

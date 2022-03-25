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
 * @tparam EdgePartitionSrcValueInputWrapper Type of the wrapper for edge partition source property
 * values.
 * @tparam EdgePartitionDstValueInputWrapper Type of the wrapper for edge partition destination
 * property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_partition_src_value_input Device-copyable wrapper used to access source input
 * property values (for the edge sources assigned to this process in multi-GPU). Use either
 * cugraph::edge_partition_src_property_t::device_view() (if @p e_op needs to access source property
 * values) or cugraph::dummy_property_t::device_view() (if @p e_op does not access source property
 * values). Use update_edge_partition_src_property to fill the wrapper.
 * @param edge_partition_dst_value_input Device-copyable wrapper used to access destination input
 * property values (for the edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_partition_dst_property_t::device_view() (if @p e_op needs to access destination
 * property values) or cugraph::dummy_property_t::device_view() (if @p e_op does not access
 * destination property values). Use update_edge_partition_dst_property to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), property values for the source, and property values for the destination and returns if
 * this edge should be included in the returned count.
 * @return GraphViewType::edge_type Number of times @p e_op returned true.
 */
template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgeOp>
typename GraphViewType::edge_type count_if_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  return transform_reduce_e(handle,
                            graph_view,
                            edge_partition_src_value_input,
                            edge_partition_dst_value_input,
                            cast_edge_op_bool_to_integer<GraphViewType,
                                                         vertex_t,
                                                         EdgePartitionSrcValueInputWrapper,
                                                         EdgePartitionDstValueInputWrapper,
                                                         EdgeOp,
                                                         edge_t>{e_op},
                            edge_t{0});
}

}  // namespace cugraph

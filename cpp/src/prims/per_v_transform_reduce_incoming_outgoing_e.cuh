/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "prims/detail/per_v_transform_reduce_e.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <numeric>
#include <utility>

namespace cugraph {

/**
 * @brief Iterate over every vertex's incoming edges to update vertex properties.
 *
 * This function is inspired by thrust::transform_reduce.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to
 * fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 outgoing edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_incoming_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = true;

  detail::per_v_transform_reduce_e<incoming>(
    handle,
    graph_view,
    static_cast<void*>(nullptr),
    static_cast<void*>(nullptr),
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    init,
    reduce_op,
    detail::const_true_e_op_t<typename GraphViewType::vertex_type,
                              typename GraphViewType::vertex_type,
                              typename EdgeSrcValueInputWrapper::value_type,
                              typename EdgeDstValueInputWrapper::value_type,
                              typename EdgeValueInputWrapper::value_type,
                              GraphViewType::is_storage_transposed>{},
    vertex_value_output_first);
}

/**
 * @brief For each (tagged-)vertex in the input (tagged-)vertex list, iterate over the incoming
 * edges to update (tagged-)vertex properties.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to update
 * (tagged-)vertex properties.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be reduced with the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 incoming edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the (tagged-)vertex property variables for
 * the first (inclusive) (tagged-)vertex in @p key_list. `vertex_value_output_last` (exclusive) is
 * deduced as @p vertex_value_output_first + @p key_list.size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_incoming_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       KeyBucketType const& key_list,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  static_assert(GraphViewType::is_storage_transposed);

  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = true;

  detail::per_v_transform_reduce_e<incoming>(
    handle,
    graph_view,
    key_list.begin(),
    key_list.end(),
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    init,
    reduce_op,
    detail::const_true_e_op_t<typename KeyBucketType::key_type,
                              typename GraphViewType::vertex_type,
                              typename EdgeSrcValueInputWrapper::value_type,
                              typename EdgeDstValueInputWrapper::value_type,
                              typename EdgeValueInputWrapper::value_type,
                              GraphViewType::is_storage_transposed>{},
    vertex_value_output_first);
}

/**
 * @brief Iterate over every vertex's outgoing edges to update vertex properties.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 outgoing edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_outgoing_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = false;

  detail::per_v_transform_reduce_e<incoming>(
    handle,
    graph_view,
    static_cast<void*>(nullptr),
    static_cast<void*>(nullptr),
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    init,
    reduce_op,
    detail::const_true_e_op_t<typename GraphViewType::vertex_type,
                              typename GraphViewType::vertex_type,
                              typename EdgeSrcValueInputWrapper::value_type,
                              typename EdgeDstValueInputWrapper::value_type,
                              typename EdgeValueInputWrapper::value_type,
                              GraphViewType::is_storage_transposed>{},
    vertex_value_output_first);
}

/**
 * @brief For each (tagged-)vertex in the input (tagged-)vertex list, iterate over the outgoing
 * edges to update (tagged-)vertex properties.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to update
 * (tagged-)vertex properties.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be reduced with the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 outgoing edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the (tagged-)vertex property variables for
 * the first (inclusive) (tagged-)vertex in @p key_list. `vertex_value_output_last` (exclusive) is
 * deduced as @p vertex_value_output_first + @p key_list.size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_outgoing_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       KeyBucketType const& key_list,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed);
  static_assert(KeyBucketType::is_sorted_unique);

  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = false;

  detail::per_v_transform_reduce_e<incoming>(
    handle,
    graph_view,
    key_list.begin(),
    key_list.end(),
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    init,
    reduce_op,
    detail::const_true_e_op_t<typename KeyBucketType::key_type,
                              typename GraphViewType::vertex_type,
                              typename EdgeSrcValueInputWrapper::value_type,
                              typename EdgeDstValueInputWrapper::value_type,
                              typename EdgeValueInputWrapper::value_type,
                              GraphViewType::is_storage_transposed>{},
    vertex_value_output_first);
}

}  // namespace cugraph

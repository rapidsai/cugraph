/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/detail/optional_dataframe_buffer.hpp"
#include "prims/detail/prim_functors.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>

#include <raft/core/handle.hpp>

#include <tuple>
#include <type_traits>

namespace cugraph {

/**
 * @brief Iterate over outgoing edges from the current vertex frontier and reduce all edge functor
 * outputs by (tagged-)destination ID.
 *
 * Vertices are assumed to be tagged if KeyBucketType::key_type is a tuple of a vertex type and a
 * tag type (KeyBucketType::key_type is identical to a vertex type otherwise).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the vertex frontier bucket class which abstracts the
 * current (tagged-)vertex frontier.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier KeyBucketType class object for the current vertex frontier.
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
 * @param e_op Quinary operator takes edge (tagged-)source, edge destination, property values for
 * the source, destination, and edge and 1) just returns (return value = void, if vertices are not
 * tagged and ReduceOp::value_type is void, in this case, @p e_op is dummy and won't be called); 2)
 * returns a value to be reduced (if vertices are not tagged and ReduceOp::value_type is not void);
 * 3) returns a tag (if vertices are tagged and ReduceOp::value_type is void); or 4) returns a tuple
 * of a tag and a value to be reduced (if vertices are tagged and ReduceOp::value_type is not void).
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @return Tuple of key values and payload values (if ReduceOp::value_type is not void) or just key
 * values (if ReduceOp::value_type is void). Keys in the return values are sorted in ascending order
 * using a vertex ID as the primary key and a tag (if relevant) as the secondary key.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp>
std::conditional_t<
  !std::is_same_v<typename ReduceOp::value_type, void>,
  std::tuple<dataframe_buffer_type_t<typename KeyBucketType::key_type>,
             detail::optional_dataframe_buffer_type_t<typename ReduceOp::value_type>>,
  dataframe_buffer_type_t<typename KeyBucketType::key_type>>
transform_reduce_v_frontier_outgoing_e_by_dst(raft::handle_t const& handle,
                                              GraphViewType const& graph_view,
                                              KeyBucketType const& frontier,
                                              EdgeSrcValueInputWrapper edge_src_value_input,
                                              EdgeDstValueInputWrapper edge_dst_value_input,
                                              EdgeValueInputWrapper edge_value_input,
                                              EdgeOp e_op,
                                              ReduceOp reduce_op,
                                              bool do_expensive_check = false)
{
  return detail::transform_reduce_if_v_frontier_outgoing_e_by_dst(
    handle,
    graph_view,
    frontier,
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    reduce_op,
    detail::const_true_e_op_t<typename KeyBucketType::key_type,
                              typename GraphViewType::vertex_type,
                              typename EdgeSrcValueInputWrapper::value_type,
                              typename EdgeDstValueInputWrapper::value_type,
                              typename EdgeValueInputWrapper::value_type,
                              GraphViewType::is_storage_transposed>{},
    do_expensive_check);
}

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/prims/detail/per_v_select_transform_e.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>

#include <raft/random/rng.cuh>

#include <optional>
#include <tuple>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Select the top-K highest bias outgoing edges of the input (tagged-)vertices and transform
 * the selected outgoing edges.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam BiasEdgeSrcValueInputWrapper Type of the wrapper for edge source property values (for
 * BiasEdgeOp).
 * @tparam BiasEdgeDstValueInputWrapper Type of the wrapper for edge destination property values
 * (for BiasEdgeOp).
 * @tparam BiasEdgeValueInputWrapper Type of the wrapper for edge property values  (for BiasEdgeOp).
 * @tparam BiasEdgeOp Type of the quinary edge operator to set-up selection bias
 * values.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to sample outgoing
 * edges.
 * @param bias_edge_src_value_input Wrapper used to access source input property values (for the
 * edge sources assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p bias_e_op. Use either cugraph::edge_src_property_t::view() (if @p
 * e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p
 * e_op does not access source property values). Use update_edge_src_property to fill the wrapper.
 * @param bias_edge_dst_value_input Wrapper used to access destination input property values (for
 * the edge destinations assigned to this process in multi-GPU). This parameter is used to pass an
 * edge source property value to @p bias_e_op. Use either cugraph::edge_dst_property_t::view() (if
 * @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param bias_edge_value_input Wrapper used to access edge input property values (for the edges
 * assigned to this process in multi-GPU). This parameter is used to pass an edge source property
 * value to @p bias_e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access
 * edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge
 * property values).
 * @param bias_e_op Quinary operator takes (tagged-)edge source, edge destination, property values
 * for the source, destination, and edge and returns a floating point bias value to be used in
 * top-K selection (highest bias values are selected). The return value should be non-negative. The
 * bias value of 0 indicates that the corresponding edge cannot be selected.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). This parameter is used to pass an edge source
 * property value to @p e_op. Use either cugraph::edge_src_property_t::view() (if @p e_op needs to
 * access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p e_op does not
 * access source property values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p e_op. Use either cugraph::edge_dst_property_t::view() (if @p e_op
 * needs to access destination property values) or cugraph::edge_dst_dummy_property_t::view() (if @p
 * e_op does not access destination property values). Use update_edge_dst_property to fill the
 * wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). This parameter is used to pass an edge source property value to @p
 * e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access edge property
 * values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge property
 * values).
 * @param e_op Quinary operator takes (tagged-)edge source, edge destination, property values for
 * the source, destination, and edge and returns a value to be collected in the output. This
 * function is called only for the selected edges.
 * @param K Number of outgoing edges to select per (tagged-)vertex.
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the vertices with their out-degrees smaller than @p K. If @p
 * invalid_value.has_value() is false, fewer than @p K values can be returned for the vertices with
 * fewer than @p K selected edges. See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p key_list.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p key_list.size() * @p K elements). If
 * @p invalid_value.has_value() is true, @p K values are returned for each key in @p key_list. Among
 * the @p K values, valid values proceed the invalid values; ordering of the valid values can be
 * arbitrary.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename BiasEdgeSrcValueInputWrapper,
          typename BiasEdgeDstValueInputWrapper,
          typename BiasEdgeValueInputWrapper,
          typename BiasEdgeOp,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
per_v_top_k_select_transform_outgoing_e(raft::handle_t const& handle,
                                        GraphViewType const& graph_view,
                                        KeyBucketType const& key_list,
                                        BiasEdgeSrcValueInputWrapper bias_edge_src_value_input,
                                        BiasEdgeDstValueInputWrapper bias_edge_dst_value_input,
                                        BiasEdgeValueInputWrapper bias_edge_value_input,
                                        BiasEdgeOp bias_e_op,
                                        EdgeSrcValueInputWrapper edge_src_value_input,
                                        EdgeDstValueInputWrapper edge_dst_value_input,
                                        EdgeValueInputWrapper edge_value_input,
                                        EdgeOp e_op,
                                        size_t K,
                                        std::optional<T> invalid_value,
                                        bool do_expensive_check = false)
{
  return detail::per_v_select_transform_e<false>(handle,
                                                 graph_view,
                                                 key_list,
                                                 bias_edge_src_value_input,
                                                 bias_edge_dst_value_input,
                                                 bias_edge_value_input,
                                                 bias_e_op,
                                                 edge_src_value_input,
                                                 edge_dst_value_input,
                                                 edge_value_input,
                                                 e_op,
                                                 edge_dummy_property_view_t{},
                                                 static_cast<raft::random::RngState*>(nullptr),
                                                 raft::host_span<size_t const>(&K, size_t{1}),
                                                 false /* with_replacement */,
                                                 invalid_value,
                                                 do_expensive_check);
}

/**
 * @brief Select the top-K highest bias outgoing edges of the input (tagged-)vertices (per edge
 * type) and transform the selected outgoing edges.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam BiasEdgeSrcValueInputWrapper Type of the wrapper for edge source property values (for
 * BiasEdgeOp).
 * @tparam BiasEdgeDstValueInputWrapper Type of the wrapper for edge destination property values
 * (for BiasEdgeOp).
 * @tparam BiasEdgeValueInputWrapper Type of the wrapper for edge property values  (for BiasEdgeOp).
 * @tparam BiasEdgeOp Type of the quinary edge operator to set-up selection bias
 * values.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam EdgeTypeInputWrapper Type of the wrapper for edge type values.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to sample outgoing
 * edges.
 * @param bias_edge_src_value_input Wrapper used to access source input property values (for the
 * edge sources assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p bias_e_op. Use either cugraph::edge_src_property_t::view() (if @p
 * e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p
 * e_op does not access source property values). Use update_edge_src_property to fill the wrapper.
 * @param bias_edge_dst_value_input Wrapper used to access destination input property values (for
 * the edge destinations assigned to this process in multi-GPU). This parameter is used to pass an
 * edge source property value to @p bias_e_op. Use either cugraph::edge_dst_property_t::view() (if
 * @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param bias_edge_value_input Wrapper used to access edge input property values (for the edges
 * assigned to this process in multi-GPU). This parameter is used to pass an edge source property
 * value to @p bias_e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access
 * edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge
 * property values).
 * @param bias_e_op Quinary operator takes (tagged-)edge source, edge destination, property values
 * for the source, destination, and edge and returns a floating point bias value to be used in
 * top-K selection (highest bias values are selected). The return value should be non-negative. The
 * bias value of 0 indicates that the corresponding edge cannot be selected.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). This parameter is used to pass an edge source
 * property value to @p e_op. Use either cugraph::edge_src_property_t::view() (if @p e_op needs to
 * access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p e_op does not
 * access source property values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p e_op. Use either cugraph::edge_dst_property_t::view() (if @p e_op
 * needs to access destination property values) or cugraph::edge_dst_dummy_property_t::view() (if @p
 * e_op does not access destination property values). Use update_edge_dst_property to fill the
 * wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). This parameter is used to pass an edge source property value to @p
 * e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access edge property
 * values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge property
 * values).
 * @param e_op Quinary operator takes (tagged-)edge source, edge destination, property values for
 * the source, destination, and edge and returns a value to be collected in the output. This
 * function is called only for the selected edges.
 * @param edge_type_input Wrapper used to access edge type value (for the edges assigned to this
 * process in multi-GPU). This parameter is used in per-type (heterogeneous) selection. Use
 * cugraph::edge_property_t::view().
 * @param Ks Number of outgoing edges to select per (tagged-)vertex for each edge type (size = #
 * edge types).
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the vertices with their out-degrees (per edge type) smaller than the
 * corresponding value in @p Ks. If @p invalid_value.has_value() is false, fewer than K_sum values
 * can be returned for the vertices with fewer than K_sum selected edges, where K_sum =
 * std::reduce(@p Ks.begin(), @p Ks.end()). See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p key_list.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p key_list.size() * K_sum elements),
 * where K_sum = std::reduce(@p Ks.begin(), @p Ks.end()). If @p invalid_value.has_value() is true,
 * K_sum values are returned for each key in @p key_list. Among the K_sum values, valid values
 * proceed the invalid values; ordering of the valid values can be arbitrary.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename BiasEdgeSrcValueInputWrapper,
          typename BiasEdgeDstValueInputWrapper,
          typename BiasEdgeValueInputWrapper,
          typename BiasEdgeOp,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeTypeInputWrapper,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
per_v_top_k_select_transform_outgoing_e(raft::handle_t const& handle,
                                        GraphViewType const& graph_view,
                                        KeyBucketType const& key_list,
                                        BiasEdgeSrcValueInputWrapper bias_edge_src_value_input,
                                        BiasEdgeDstValueInputWrapper bias_edge_dst_value_input,
                                        BiasEdgeValueInputWrapper bias_edge_value_input,
                                        BiasEdgeOp bias_e_op,
                                        EdgeSrcValueInputWrapper edge_src_value_input,
                                        EdgeDstValueInputWrapper edge_dst_value_input,
                                        EdgeValueInputWrapper edge_value_input,
                                        EdgeOp e_op,
                                        EdgeTypeInputWrapper edge_type_input,
                                        raft::host_span<size_t const> Ks,
                                        std::optional<T> invalid_value,
                                        bool do_expensive_check = false)
{
  return detail::per_v_select_transform_e<false>(handle,
                                                 graph_view,
                                                 key_list,
                                                 bias_edge_src_value_input,
                                                 bias_edge_dst_value_input,
                                                 bias_edge_value_input,
                                                 bias_e_op,
                                                 edge_src_value_input,
                                                 edge_dst_value_input,
                                                 edge_value_input,
                                                 e_op,
                                                 edge_type_input,
                                                 static_cast<raft::random::RngState*>(nullptr),
                                                 Ks,
                                                 false /* with_replacement */,
                                                 invalid_value,
                                                 do_expensive_check);
}

}  // namespace CUGRAPH_EXPORT cugraph

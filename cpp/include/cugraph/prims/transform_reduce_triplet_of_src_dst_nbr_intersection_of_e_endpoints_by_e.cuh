/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>

#include <tuple>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename IntersectionOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = float;  // dummy

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeDstValueInputWrapper::value_iterator,
      typename EdgeDstValueInputWrapper::value_type>>;

  if (do_expensive_check) {
    // currently, nothing to do.
  }

  rmm::device_uvector<vertex_t> result_srcs(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(size_t{0}, handle.get_stream());
  auto result_values = allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream());

  auto edge_mask_view = graph_view.edge_mask_view();

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : std::nullopt;

    edge_partition_src_input_device_view_t edge_partition_src_value_input{};
    edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_src_value_input = edge_partition_src_input_device_view_t(edge_src_value_input);
      edge_partition_dst_value_input =
        edge_partition_dst_input_device_view_t(edge_dst_value_input, i);
    } else {
      edge_partition_src_value_input =
        edge_partition_src_input_device_view_t(edge_src_value_input, i);
      edge_partition_dst_value_input = edge_partition_dst_input_device_view_t(edge_dst_value_input);
    }

    rmm::device_uvector<vertex_t> majors(
      edge_partition_e_mask
        ? detail::count_set_bits(
            handle, (*edge_partition_e_mask).value_first(), edge_partition.number_of_edges())
        : static_cast<size_t>(edge_partition.number_of_edges()),
      handle.get_stream());
    rmm::device_uvector<vertex_t> minors(majors.size(), handle.get_stream());

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    detail::decompress_edge_partition_to_edgelist<vertex_t,
                                                  edge_t,
                                                  weight_t,
                                                  int32_t,
                                                  GraphViewType::is_multi_gpu>(
      handle,
      edge_partition,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      edge_partition_e_mask,
      raft::device_span<vertex_t>(majors.data(), majors.size()),
      raft::device_span<vertex_t>(minors.data(), minors.size()),
      std::nullopt,
      std::nullopt,
      std::nullopt,
      segment_offsets);

    auto vertex_pair_first = thrust::make_zip_iterator(majors.begin(), minors.begin());

    // Per-edge accumulator: one slot per edge in this partition's CSR, indexed by edge offset.
    // The intersection kernel atomically adds the operator's emission values into pq / pr / qr
    // offsets, so the accumulator must span the full CSR edge range (including masked-out edges,
    // since offsets index the full indices[] array).
    auto edge_accumulator = allocate_dataframe_buffer<T>(
      static_cast<size_t>(edge_partition.number_of_edges()), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(edge_accumulator),
                 get_dataframe_buffer_end(edge_accumulator),
                 init);

    // === STEP 3: non-materializing intersection kernel (atomicAdd emission tuple per (edge, r)) ===
    // === STEP 4: stream-compact accumulator into (src, dst, value) triplets, append to result ===
  }

  return std::make_tuple(
    std::move(result_srcs), std::move(result_dsts), std::move(result_values));
}

}  // namespace detail

/**
 * @brief Iterate over each edge and apply a functor to each vertex in the common source neighbor
 * list of the edge endpoints, reduce the functor output values per-edge.
 *
 * Iterate over every edge; intersect source neighbor lists of source vertex & destination vertex;
 * invoke a user-provided functor once per vertex r in the intersection (i.e. once per
 * (edge, intersection vertex) triplet), and reduce the functor output values (cuda::std::tuple of
 * two values having the same type: one for the edge (src, dst), and one for each supporting edge
 * (the (src, r) & (dst, r) edges)) per-edge. The functor is invoked once per vertex r in the
 * intersection, so it can emit a different value for each (edge, r) triplet. We may add a per-edge
 * variant (transform_reduce_src_nbr_intersection_of_e_endpoints_by_e) in the future that invokes
 * the functor once per edge with the full intersection list, for callers whose emitted value does
 * not vary per intersection vertex. This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam IntersectionOp Type of the quinary per (edge, intersection vertex) operator.
 * @tparam T Type of the per-edge reduction value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p intersection_op needs to access source property values) or
 * cugraph::edge_src_dummy_property_t::view() (if @p intersection_op does not access source property
 * values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p intersection_op needs to access destination property
 * values) or cugraph::edge_dst_dummy_property_t::view() (if @p intersection_op does not access
 * destination property values). Use update_edge_dst_property to fill the wrapper.
 * @param intersection_op quinary operator takes edge source, edge destination, property values for
 * the source, property values for the destination, and one vertex r in the intersection of edge
 * source & destination vertices' source neighbors and returns a cuda::std::tuple of two values:
 * one value for the edge (src, dst) and one value for each supporting edge (src, r) and (dst, r).
 * @param init Initial value to be added to the reduced @p intersection_op return values for each
 * edge.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of three device vectors (srcs, dsts, values): for each edge with a non-init reduced
 * value, its source vertex, destination vertex, and reduced value.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename IntersectionOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_src_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  static_assert(GraphViewType::is_storage_transposed);

  return detail::transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
    handle,
    graph_view,
    edge_src_value_input,
    edge_dst_value_input,
    intersection_op,
    init,
    do_expensive_check);
}

/**
 * @brief Iterate over each edge and apply a functor to each vertex in the common destination
 * neighbor list of the edge endpoints, reduce the functor output values per-edge.
 *
 * Iterate over every edge; intersect destination neighbor lists of source vertex & destination
 * vertex; invoke a user-provided functor once per vertex r in the intersection (i.e. once per
 * (edge, intersection vertex) triplet), and reduce the functor output values (cuda::std::tuple of
 * two values having the same type: one for the edge (src, dst), and one for each supporting edge
 * (the (src, r) & (dst, r) edges)) per-edge. The functor is invoked once per vertex r in the
 * intersection, so it can emit a different value for each (edge, r) triplet. We may add a per-edge
 * variant (transform_reduce_dst_nbr_intersection_of_e_endpoints_by_e) in the future that invokes
 * the functor once per edge with the full intersection list, for callers whose emitted value does
 * not vary per intersection vertex. This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam IntersectionOp Type of the quinary per (edge, intersection vertex) operator.
 * @tparam T Type of the per-edge reduction value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p intersection_op needs to access source property values) or
 * cugraph::edge_src_dummy_property_t::view() (if @p intersection_op does not access source property
 * values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p intersection_op needs to access destination property
 * values) or cugraph::edge_dst_dummy_property_t::view() (if @p intersection_op does not access
 * destination property values). Use update_edge_dst_property to fill the wrapper.
 * @param intersection_op quinary operator takes edge source, edge destination, property values for
 * the source, property values for the destination, and one vertex r in the intersection of edge
 * source & destination vertices' destination neighbors and returns a cuda::std::tuple of two
 * values: one value for the edge (src, dst) and one value for each supporting edge (src, r) and
 * (dst, r).
 * @param init Initial value to be added to the reduced @p intersection_op return values for each
 * edge.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of three device vectors (srcs, dsts, values): for each edge with a non-init reduced
 * value, its source vertex, destination vertex, and reduced value.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename IntersectionOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           dataframe_buffer_type_t<T>>
transform_reduce_triplet_of_dst_nbr_intersection_of_e_endpoints_by_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  IntersectionOp intersection_op,
  T init,
  bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed);

  return detail::transform_reduce_triplet_of_minor_nbr_intersection_of_e_endpoints_by_e(
    handle,
    graph_view,
    edge_src_value_input,
    edge_dst_value_input,
    intersection_op,
    init,
    do_expensive_check);
}

}  // namespace CUGRAPH_EXPORT cugraph

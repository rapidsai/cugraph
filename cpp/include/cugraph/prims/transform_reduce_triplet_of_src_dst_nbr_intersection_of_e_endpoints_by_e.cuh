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
#include <cugraph/prims/detail/nbr_intersection.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <tuple>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

// Called by detail::nbr_intersection once per common neighbor w of a pair (v0, v1). Evaluates the
// user operator and adds its returned values to the per-edge accumulator at the (v0, v1), (v0, w)
// and (v1, w) edge offsets. The (v0, v1) and (v0, w) edges are local; (v1, w) is local in
// single-GPU and may be remote in multi-GPU.
template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename IntersectionOp,
          typename T,
          typename AccumulatorIterator>
struct accumulate_triplet_op_t {
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu> edge_partition{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input{};
  IntersectionOp intersection_op{};
  AccumulatorIterator accumulator{};
  T identity{};

  __device__ void operator()(vertex_t v0,
                             vertex_t v1,
                             vertex_t w,
                             edge_t v0_v1_edge_offset,
                             edge_t v0_w_edge_offset,
                             edge_t v1_w_edge_offset) const
  {
    auto major_offset = edge_partition.major_offset_from_major_nocheck(v0);
    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(v1);
    auto src          = GraphViewType::is_storage_transposed ? v1 : v0;
    auto dst          = GraphViewType::is_storage_transposed ? v0 : v1;
    auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;

    auto result = intersection_op(src,
                                  dst,
                                  edge_partition_src_value_input.get(src_offset),
                                  edge_partition_dst_value_input.get(dst_offset),
                                  w);
    auto edge_value       = cuda::std::get<0>(result);
    auto supporting_value = cuda::std::get<1>(result);

    cuda::atomic_ref<T, cuda::thread_scope_device> v0_v1_ref(accumulator[v0_v1_edge_offset]);
    v0_v1_ref.fetch_add(edge_value, cuda::memory_order_relaxed);
    // Skip the supporting contribution when it is the additive identity: it is a no-op locally and
    // (in multi-GPU) avoids buffering/shuffling an identity contribution for no effect.
    if (supporting_value != identity) {
      cuda::atomic_ref<T, cuda::thread_scope_device> v0_w_ref(accumulator[v0_w_edge_offset]);
      v0_w_ref.fetch_add(supporting_value, cuda::memory_order_relaxed);
      cuda::atomic_ref<T, cuda::thread_scope_device> v1_w_ref(accumulator[v1_w_edge_offset]);
      v1_w_ref.fetch_add(supporting_value, cuda::memory_order_relaxed);
    }
  }
};

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

    // Per-edge accumulator, indexed by CSR edge offset (sized over the full edge range).
    auto edge_accumulator = allocate_dataframe_buffer<T>(
      static_cast<size_t>(edge_partition.number_of_edges()), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(edge_accumulator),
                 get_dataframe_buffer_end(edge_accumulator),
                 init);

    uint32_t const* edge_mask =
      edge_partition_e_mask ? (*edge_partition_e_mask).value_first() : nullptr;

    // Compute the intersection without materializing it, accumulating the operator's values per
    // edge.
    auto accumulator_first = get_dataframe_buffer_begin(edge_accumulator);
    detail::nbr_intersection(
      handle,
      graph_view,
      edge_partition,
      vertex_pair_first,
      vertex_pair_first + majors.size(),
      accumulate_triplet_op_t<GraphViewType,
                              edge_partition_src_input_device_view_t,
                              edge_partition_dst_input_device_view_t,
                              IntersectionOp,
                              T,
                              decltype(accumulator_first)>{edge_partition,
                                                           edge_partition_src_value_input,
                                                           edge_partition_dst_value_input,
                                                           intersection_op,
                                                           accumulator_first,
                                                           init},
      edge_mask);

    // Stream-compact the accumulator to (src, dst, value) triplets for the updated edges. The
    // accumulator is indexed by CSR edge offset, so dst(e) = indices[e] and src(e) is the major
    // whose offset range contains e (found by upper_bound on offsets[]). Only edges whose value
    // changed from init are kept.
    auto num_partition_edges = static_cast<size_t>(edge_partition.number_of_edges());
    auto offsets_ptr         = edge_partition.offsets();
    auto indices_ptr         = edge_partition.indices();
    auto num_majors          = edge_partition.major_range_size();
    auto major_range_first   = edge_partition.major_range_first();

    auto num_updated = static_cast<size_t>(thrust::count_if(
      handle.get_thrust_policy(),
      accumulator_first,
      accumulator_first + num_partition_edges,
      cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; })));

    rmm::device_uvector<vertex_t> chunk_srcs(num_updated, handle.get_stream());
    rmm::device_uvector<vertex_t> chunk_dsts(num_updated, handle.get_stream());
    auto chunk_vals = allocate_dataframe_buffer<T>(num_updated, handle.get_stream());

    auto src_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(edge_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [offsets_ptr, num_majors, major_range_first] __device__(edge_t e) {
          auto it = thrust::upper_bound(thrust::seq, offsets_ptr, offsets_ptr + num_majors + 1, e);
          return static_cast<vertex_t>(major_range_first +
                                       (thrust::distance(offsets_ptr, it) - edge_t{1}));
        }));
    auto dst_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(edge_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [indices_ptr] __device__(edge_t e) { return indices_ptr[e]; }));
    auto input_first  = thrust::make_zip_iterator(src_first, dst_first, accumulator_first);
    auto output_first = thrust::make_zip_iterator(
      chunk_srcs.begin(), chunk_dsts.begin(), get_dataframe_buffer_begin(chunk_vals));

    thrust::copy_if(handle.get_thrust_policy(),
                    input_first,
                    input_first + num_partition_edges,
                    accumulator_first,
                    output_first,
                    cuda::proclaim_return_type<bool>([init] __device__(auto v) { return v != init; }));

    // Single-GPU has one local edge partition, so this chunk is the whole result. (MG with
    // multiple local partitions / cross-rank ownership restructures this tail; see section 5.)
    result_srcs   = std::move(chunk_srcs);
    result_dsts   = std::move(chunk_dsts);
    result_values = std::move(chunk_vals);
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

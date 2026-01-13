/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "detail/optional_dataframe_buffer.hpp"
#include "detail/prim_utils.cuh"
#include "prims/property_op_utils.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include <type_traits>
#include <vector>

namespace cugraph {

namespace detail {

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgeOp>
struct return_e_value_t {
  using e_op_result_t =
    typename detail::edge_op_result_type<typename GraphViewType::vertex_type,
                                         typename GraphViewType::vertex_type,
                                         typename EdgePartitionSrcValueInputWrapper::value_type,
                                         typename EdgePartitionDstValueInputWrapper::value_type,
                                         typename EdgePartitionEdgeValueInputWrapper::value_type,
                                         EdgeOp>::type;

  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu>
    edge_partition{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input{};
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input{};
  EdgeOp e_op{};

  __device__ e_op_result_t
  operator()(cuda::std::tuple<typename GraphViewType::vertex_type,
                              typename GraphViewType::vertex_type,
                              typename GraphViewType::edge_type> edge) const
  {
    using vertex_t = typename GraphViewType::vertex_type;
    using edge_t   = typename GraphViewType::edge_type;

    auto major            = cuda::std::get<0>(edge);
    auto minor            = cuda::std::get<1>(edge);
    auto multi_edge_index = cuda::std::get<2>(edge);

    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    auto major_idx    = edge_partition.major_idx_from_major_nocheck(major);
    assert(major_idx);

    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);

    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    cuda::std::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(*major_idx);
    auto it =
      thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor) + multi_edge_index;
    assert(*it == minor);

    auto src        = GraphViewType::is_storage_transposed ? minor : major;
    auto dst        = GraphViewType::is_storage_transposed ? major : minor;
    auto src_offset = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset = GraphViewType::is_storage_transposed ? major_offset : minor_offset;

    return e_op(src,
                dst,
                edge_partition_src_value_input.get(src_offset),
                edge_partition_dst_value_input.get(dst_offset),
                edge_partition_e_value_input.get(edge_offset + cuda::std::distance(indices, it)));
  }
};

}  // namespace detail

/**
 * @brief Iterate over the edges in the input edge list and gather the edge operator outputs.
 *
 * This function is inspired by thrust::transform() and thrust::gather().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeBucketType Type of the edge bucket class which stores the edge list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for input edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for input edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for input edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam GatheredValueOutputIterator Type of the iterator for gathered output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_list EdgeBucketType class object storing the edge list to update edge property
 * values.
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
 * @param gathered_value_output_first Iterator pointing the beginning of the gathered output buffer.
 * `gathered_value_output_last` (exclusive) is deduced as @p gathered_value_output_first + @p
 * edge_list.size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename GatheredValueOutputIterator>
void transform_gather_e(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        EdgeBucketType const& edge_list,
                        EdgeSrcValueInputWrapper edge_src_value_input,
                        EdgeDstValueInputWrapper edge_dst_value_input,
                        EdgeValueInputWrapper edge_value_input,
                        EdgeOp e_op,
                        GatheredValueOutputIterator gathered_value_output_first,
                        bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(GraphViewType::is_storage_transposed != EdgeBucketType::is_src_major);

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
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_iterator, void*>,
    std::conditional_t<
      std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
      detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
      detail::edge_partition_edge_multi_index_property_device_view_t<edge_t, vertex_t>>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;

  auto major_first =
    GraphViewType::is_storage_transposed ? edge_list.dst_begin() : edge_list.src_begin();
  auto minor_first =
    GraphViewType::is_storage_transposed ? edge_list.src_begin() : edge_list.dst_begin();

  auto pair_first             = thrust::make_zip_iterator(major_first, minor_first);
  auto multi_edge_index_first = edge_list.multi_edge_index_begin();

  CUGRAPH_EXPECTS(graph_view.is_multigraph() == multi_edge_index_first.has_value(),
                  "Invalid input arguments: the edge list should include multi-edge index for a "
                  "multi-graph (and should not for a non-multi-graph).");

  auto edge_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}),
    cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t, edge_t>>(
      [pair_first, multi_edge_index_first] __device__(
        size_t i) -> cuda::std::tuple<vertex_t, vertex_t, edge_t> {
        auto pair = *(pair_first + i);
        if (multi_edge_index_first) {
          return cuda::std::make_tuple(
            cuda::std::get<0>(pair), cuda::std::get<1>(pair), *(*multi_edge_index_first + i));
        } else {
          return cuda::std::make_tuple(cuda::std::get<0>(pair), cuda::std::get<1>(pair), edge_t{0});
        }
      }));

  if (do_expensive_check) {
    if constexpr (EdgeBucketType::is_sorted_unique) {
      CUGRAPH_EXPECTS(
        thrust::is_sorted(handle.get_thrust_policy(), edge_first, edge_first + edge_list.size()),
        "Invalid input arguments: edge_list is not sorted.");
      auto num_uniques = static_cast<size_t>(
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(edge_list.size()),
                         detail::is_first_in_run_t<decltype(edge_first)>{edge_first}));
      CUGRAPH_EXPECTS(num_uniques == edge_list.size(),
                      "Invalid input arguments: edgelist has duplicates.");
    }
  }

  auto output_indices = detail::allocate_optional_dataframe_buffer<
    std::conditional_t<!EdgeBucketType::is_sorted_unique, size_t, void>>(edge_list.size(),
                                                                         handle.get_stream());
  if constexpr (!EdgeBucketType::is_sorted_unique) {
    thrust::sequence(
      handle.get_thrust_policy(), output_indices.begin(), output_indices.end(), size_t{0});
    thrust::sort(handle.get_thrust_policy(),
                 output_indices.begin(),
                 output_indices.end(),
                 cuda::proclaim_return_type<bool>([edge_first] __device__(auto l, auto r) {
                   return *(edge_first + l) < *(edge_first + r);
                 }));
  }

  std::vector<size_t> edge_partition_offsets(graph_view.number_of_local_edge_partitions() + 1, 0);
  if constexpr (GraphViewType::is_multi_gpu) {
    std::vector<vertex_t> h_major_range_lasts(graph_view.number_of_local_edge_partitions());
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      if constexpr (GraphViewType::is_storage_transposed) {
        h_major_range_lasts[i] = graph_view.local_edge_partition_dst_range_last(i);
      } else {
        h_major_range_lasts[i] = graph_view.local_edge_partition_src_range_last(i);
      }
    }
    rmm::device_uvector<vertex_t> d_major_range_lasts(h_major_range_lasts.size(),
                                                      handle.get_stream());
    raft::update_device(d_major_range_lasts.data(),
                        h_major_range_lasts.data(),
                        h_major_range_lasts.size(),
                        handle.get_stream());
    rmm::device_uvector<size_t> d_lower_bounds(d_major_range_lasts.size(), handle.get_stream());
    if constexpr (EdgeBucketType::is_sorted_unique) {
      thrust::lower_bound(handle.get_thrust_policy(),
                          major_first,
                          major_first + edge_list.size(),
                          d_major_range_lasts.begin(),
                          d_major_range_lasts.end(),
                          d_lower_bounds.begin());
    } else {
      auto sorted_major_first =
        thrust::make_permutation_iterator(major_first, output_indices.begin());
      thrust::lower_bound(handle.get_thrust_policy(),
                          sorted_major_first,
                          sorted_major_first + edge_list.size(),
                          d_major_range_lasts.begin(),
                          d_major_range_lasts.end(),
                          d_lower_bounds.begin());
    }
    raft::update_host(edge_partition_offsets.data() + 1,
                      d_lower_bounds.data(),
                      d_lower_bounds.size(),
                      handle.get_stream());
    handle.sync_stream();
  } else {
    edge_partition_offsets.back() = edge_list.size();
  }

  auto edge_mask_view = graph_view.edge_mask_view();

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    if (do_expensive_check) {
      auto edge_partition_e_mask =
        edge_mask_view
          ? cuda::std::make_optional<
              detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
              *edge_mask_view, i)
          : cuda::std::nullopt;

      size_t num_invalids{};
      if constexpr (EdgeBucketType::is_sorted_unique) {
        num_invalids =
          thrust::count_if(handle.get_thrust_policy(),
                           edge_first + edge_partition_offsets[i],
                           edge_first + edge_partition_offsets[i + 1],
                           detail::edge_exists_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>{
                             edge_partition, edge_partition_e_mask});
      } else {
        auto sorted_edge_first =
          thrust::make_permutation_iterator(edge_first, output_indices.begin());
        num_invalids =
          thrust::count_if(handle.get_thrust_policy(),
                           sorted_edge_first + edge_partition_offsets[i],
                           sorted_edge_first + edge_partition_offsets[i + 1],
                           detail::edge_exists_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>{
                             edge_partition, edge_partition_e_mask});
      }
      CUGRAPH_EXPECTS(
        num_invalids == 0,
        "Invalid input arguments: edge_list contains edges that do not exist in the input graph.");
    }

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
    auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, i);

    // no need to check edge_partition_e_mask (edge_list should not have masked-out edges)
    if constexpr (EdgeBucketType::is_sorted_unique) {
      thrust::transform(handle.get_thrust_policy(),
                        edge_first + edge_partition_offsets[i],
                        edge_first + edge_partition_offsets[i + 1],
                        gathered_value_output_first + edge_partition_offsets[i],
                        detail::return_e_value_t<GraphViewType,
                                                 edge_partition_src_input_device_view_t,
                                                 edge_partition_dst_input_device_view_t,
                                                 edge_partition_e_input_device_view_t,
                                                 EdgeOp>{edge_partition,
                                                         edge_partition_src_value_input,
                                                         edge_partition_dst_value_input,
                                                         edge_partition_e_value_input,
                                                         e_op});
    } else {
      auto sorted_edge_first =
        thrust::make_permutation_iterator(edge_first, output_indices.begin());
      auto output_first =
        thrust::make_permutation_iterator(gathered_value_output_first, output_indices.begin());
      thrust::transform(handle.get_thrust_policy(),
                        sorted_edge_first + edge_partition_offsets[i],
                        sorted_edge_first + edge_partition_offsets[i + 1],
                        output_first + edge_partition_offsets[i],
                        detail::return_e_value_t<GraphViewType,
                                                 edge_partition_src_input_device_view_t,
                                                 edge_partition_dst_input_device_view_t,
                                                 edge_partition_e_input_device_view_t,
                                                 EdgeOp>{edge_partition,
                                                         edge_partition_src_value_input,
                                                         edge_partition_dst_value_input,
                                                         edge_partition_e_value_input,
                                                         e_op});
    }
  }
}

}  // namespace cugraph

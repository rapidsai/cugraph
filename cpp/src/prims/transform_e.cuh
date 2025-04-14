/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include <type_traits>
#include <vector>

namespace cugraph {

namespace detail {

int32_t constexpr transform_e_kernel_block_size = 512;

template <bool check_edge_mask,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename EdgePartitionEdgeValueOutputWrapper,
          typename EdgeOp>
__global__ static void transform_e_packed_bool(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask,
  EdgePartitionEdgeValueOutputWrapper edge_partition_e_value_output,
  EdgeOp e_op)
{
  static_assert(EdgePartitionEdgeValueOutputWrapper::is_packed_bool);
  static_assert(raft::warp_size() == packed_bools_per_word());

  using edge_t = typename GraphViewType::edge_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(transform_e_kernel_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<edge_t>(packed_bool_offset(tid));

  auto num_edges = edge_partition.number_of_edges();
  while (idx < static_cast<edge_t>(packed_bool_size(num_edges))) {
    [[maybe_unused]] auto edge_mask =
      packed_bool_full_mask();  // relevant only when check_edge_mask is true
    if constexpr (check_edge_mask) { edge_mask = *(edge_partition_e_mask.value_first() + idx); }

    auto local_edge_idx =
      idx * static_cast<edge_t>(packed_bools_per_word()) + static_cast<edge_t>(lane_id);
    int predicate{0};

    if (local_edge_idx < num_edges) {
      bool compute_predicate = true;
      if constexpr (check_edge_mask) {
        compute_predicate = ((edge_mask & packed_bool_mask(lane_id)) != packed_bool_empty_mask());
      }

      if (compute_predicate) {
        auto major_idx    = edge_partition.major_idx_from_local_edge_idx_nocheck(local_edge_idx);
        auto major        = edge_partition.major_from_major_idx_nocheck(major_idx);
        auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
        auto minor        = *(edge_partition.indices() + local_edge_idx);
        auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);

        auto src        = GraphViewType::is_storage_transposed ? minor : major;
        auto dst        = GraphViewType::is_storage_transposed ? major : minor;
        auto src_offset = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
        auto dst_offset = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
        predicate       = e_op(src,
                         dst,
                         edge_partition_src_value_input.get(src_offset),
                         edge_partition_dst_value_input.get(dst_offset),
                         edge_partition_e_value_input.get(local_edge_idx))
                            ? int{1}
                            : int{0};
      }
    }
    uint32_t new_val = __ballot_sync(raft::warp_full_mask(), predicate);
    if (lane_id == 0) {
      if constexpr (check_edge_mask) {
        auto old_val = *(edge_partition_e_value_output.value_first() + idx);
        *(edge_partition_e_value_output.value_first() + idx) = (old_val & ~edge_mask) | new_val;
      } else {
        *(edge_partition_e_value_output.value_first() + idx) = new_val;
      }
    }

    idx += static_cast<edge_t>(gridDim.x * (blockDim.x / raft::warp_size()));
  }
}

template <bool check_edge_mask,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename EdgeOp,
          typename EdgeValueOutputWrapper>
struct update_e_value_t {
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu>
    edge_partition{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input{};
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input{};
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask{};
  EdgeOp e_op{};
  EdgeValueOutputWrapper edge_partition_e_value_output{};

  __device__ void operator()(thrust::tuple<typename GraphViewType::vertex_type,
                                           typename GraphViewType::vertex_type> edge) const
  {
    using vertex_t = typename GraphViewType::vertex_type;
    using edge_t   = typename GraphViewType::edge_type;

    auto major = thrust::get<0>(edge);
    auto minor = thrust::get<1>(edge);

    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    auto major_idx    = edge_partition.major_idx_from_major_nocheck(major);
    assert(major_idx);

    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);

    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(*major_idx);
    auto lower_it = thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor);
    auto upper_it = thrust::upper_bound(thrust::seq, lower_it, indices + local_degree, minor);

    auto src        = GraphViewType::is_storage_transposed ? minor : major;
    auto dst        = GraphViewType::is_storage_transposed ? major : minor;
    auto src_offset = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset = GraphViewType::is_storage_transposed ? major_offset : minor_offset;

    for (auto it = lower_it; it != upper_it; ++it) {
      assert(*it == minor);
      if constexpr (check_edge_mask) {
        if (edge_partition_e_mask.get(edge_offset + cuda::std::distance(indices, it))) {
          auto e_op_result =
            e_op(src,
                 dst,
                 edge_partition_src_value_input.get(src_offset),
                 edge_partition_dst_value_input.get(dst_offset),
                 edge_partition_e_value_input.get(edge_offset + cuda::std::distance(indices, it)));
          edge_partition_e_value_output.set(edge_offset + cuda::std::distance(indices, it),
                                            e_op_result);
        }
      } else {
        auto e_op_result =
          e_op(src,
               dst,
               edge_partition_src_value_input.get(src_offset),
               edge_partition_dst_value_input.get(dst_offset),
               edge_partition_e_value_input.get(edge_offset + cuda::std::distance(indices, it)));
        edge_partition_e_value_output.set(edge_offset + cuda::std::distance(indices, it),
                                          e_op_result);
      }
    }
  }

  __device__ void operator()(typename GraphViewType::edge_type i) const
  {
    if constexpr (check_edge_mask) {
      if (!edge_partition_e_mask.get(i)) { return; }
    }
    auto major_idx    = edge_partition.major_idx_from_local_edge_idx_nocheck(i);
    auto major        = edge_partition.major_from_major_idx_nocheck(major_idx);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    auto minor        = *(edge_partition.indices() + i);
    auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);

    auto src         = GraphViewType::is_storage_transposed ? minor : major;
    auto dst         = GraphViewType::is_storage_transposed ? major : minor;
    auto src_offset  = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
    auto dst_offset  = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
    auto e_op_result = e_op(src,
                            dst,
                            edge_partition_src_value_input.get(src_offset),
                            edge_partition_dst_value_input.get(dst_offset),
                            edge_partition_e_value_input.get(i));
    edge_partition_e_value_output.set(i, e_op_result);
  }
};

}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and update edge property values.
 *
 * This function is inspired by thrust::transform().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for input edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for input edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for input edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam EdgeValueOutputWrapper Type of the wrapper for output edge property values.
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
 * @param edge_value_output Wrapper used to store edge output property values (for the edges
 * assigned to this process in multi-GPU). Use cugraph::edge_property_t::mutable_view().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeValueOutputWrapper>
void transform_e(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 EdgeSrcValueInputWrapper edge_src_value_input,
                 EdgeDstValueInputWrapper edge_dst_value_input,
                 EdgeValueInputWrapper edge_value_input,
                 EdgeOp e_op,
                 EdgeValueOutputWrapper edge_value_output,
                 bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

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
    std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;
  using edge_partition_e_output_device_view_t = detail::edge_partition_edge_property_device_view_t<
    edge_t,
    typename EdgeValueOutputWrapper::value_iterator,
    typename EdgeValueOutputWrapper::value_type>;

  auto edge_mask_view = graph_view.edge_mask_view();

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

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
    auto edge_partition_e_value_output =
      edge_partition_e_output_device_view_t(edge_value_output, i);

    auto num_edges = edge_partition.number_of_edges();
    if constexpr (edge_partition_e_output_device_view_t::has_packed_bool_element) {
      static_assert(edge_partition_e_output_device_view_t::is_packed_bool,
                    "unimplemented for thrust::tuple types.");
      if (edge_partition.number_of_edges() > edge_t{0}) {
        raft::grid_1d_thread_t update_grid(num_edges,
                                           detail::transform_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        if (edge_partition_e_mask) {
          detail::transform_e_packed_bool<true, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              edge_partition,
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              *edge_partition_e_mask,
              edge_partition_e_value_output,
              e_op);
        } else {
          detail::transform_e_packed_bool<false, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              edge_partition,
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              std::byte{},  // dummy
              edge_partition_e_value_output,
              e_op);
        }
      }
    } else {
      if (edge_partition_e_mask) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(edge_t{0}),
          thrust::make_counting_iterator(num_edges),
          detail::update_e_value_t<true,
                                   GraphViewType,
                                   edge_partition_src_input_device_view_t,
                                   edge_partition_dst_input_device_view_t,
                                   edge_partition_e_input_device_view_t,
                                   std::remove_reference_t<decltype(*edge_partition_e_mask)>,
                                   EdgeOp,
                                   edge_partition_e_output_device_view_t>{
            edge_partition,
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            *edge_partition_e_mask,
            e_op,
            edge_partition_e_value_output});
      } else {
        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(edge_t{0}),
                         thrust::make_counting_iterator(num_edges),
                         detail::update_e_value_t<false,
                                                  GraphViewType,
                                                  edge_partition_src_input_device_view_t,
                                                  edge_partition_dst_input_device_view_t,
                                                  edge_partition_e_input_device_view_t,
                                                  std::byte,  // dummy
                                                  EdgeOp,
                                                  edge_partition_e_output_device_view_t>{
                           edge_partition,
                           edge_partition_src_value_input,
                           edge_partition_dst_value_input,
                           edge_partition_e_value_input,
                           std::byte{},  // dummy
                           e_op,
                           edge_partition_e_value_output});
      }
    }
  }
}

/**
 * @brief Iterate over the edges in the input edge list and update edge property values.
 *
 * This function is inspired by thrust::transform().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeBucketType Type of the edge bucket class which stores the edge list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for input edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for input edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for input edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam EdgeValueOutputWrapper Type of the wrapper for output edge property values.
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
 * @param edge_value_output Wrapper used to store edge output property values (for the edges
 * assigned to this process in multi-GPU). Use cugraph::edge_property_t::mutable_view().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeValueOutputWrapper>
void transform_e(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 EdgeBucketType const& edge_list,
                 EdgeSrcValueInputWrapper edge_src_value_input,
                 EdgeDstValueInputWrapper edge_dst_value_input,
                 EdgeValueInputWrapper edge_value_input,
                 EdgeOp e_op,
                 EdgeValueOutputWrapper edge_value_output,
                 bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(GraphViewType::is_storage_transposed != EdgeBucketType::is_src_major);
  static_assert(EdgeBucketType::is_sorted_unique);
  static_assert(
    std::is_same_v<typename EdgeBucketType::key_type, thrust::tuple<vertex_t, vertex_t>>);

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
    std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;
  using edge_partition_e_output_device_view_t = detail::edge_partition_edge_property_device_view_t<
    edge_t,
    typename EdgeValueOutputWrapper::value_iterator,
    typename EdgeValueOutputWrapper::value_type>;

  auto major_first =
    GraphViewType::is_storage_transposed ? edge_list.dst_begin() : edge_list.src_begin();
  auto minor_first =
    GraphViewType::is_storage_transposed ? edge_list.src_begin() : edge_list.dst_begin();

  auto edge_first = thrust::make_zip_iterator(major_first, minor_first);

  if (do_expensive_check) {
    CUGRAPH_EXPECTS(
      thrust::is_sorted(handle.get_thrust_policy(), edge_first, edge_first + edge_list.size()),
      "Invalid input arguments: edge_list is not sorted.");
  }

  std::vector<size_t> edge_partition_offsets(graph_view.number_of_local_edge_partitions() + 1, 0);
  if constexpr (GraphViewType::is_multi_gpu) {
    std::vector<vertex_t> h_major_range_lasts(graph_view.number_of_local_edge_partitions());
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(i));
      h_major_range_lasts[i] = edge_partition.major_range_last();
    }
    rmm::device_uvector<vertex_t> d_major_range_lasts(h_major_range_lasts.size(),
                                                      handle.get_stream());
    raft::update_device(d_major_range_lasts.data(),
                        h_major_range_lasts.data(),
                        h_major_range_lasts.size(),
                        handle.get_stream());
    rmm::device_uvector<size_t> d_lower_bounds(d_major_range_lasts.size(), handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        major_first,
                        major_first + edge_list.size(),
                        d_major_range_lasts.begin(),
                        d_major_range_lasts.end(),
                        d_lower_bounds.begin());
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
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

    if (do_expensive_check) {
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first + edge_partition_offsets[i],
          edge_first + edge_partition_offsets[i + 1],
          [edge_partition,
           edge_partition_e_mask] __device__(thrust::tuple<vertex_t, vertex_t> edge) {
            auto major     = thrust::get<0>(edge);
            auto minor     = thrust::get<1>(edge);
            auto major_idx = edge_partition.major_idx_from_major_nocheck(major);
            if (!major_idx) { return true; }
            vertex_t const* indices{nullptr};
            edge_t edge_offset{};
            edge_t local_degree{};
            thrust::tie(indices, edge_offset, local_degree) =
              edge_partition.local_edges(*major_idx);
            auto lower_it =
              thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor);
            if (*lower_it != minor) { return true; }
            if (edge_partition_e_mask) {
              auto upper_it =
                thrust::upper_bound(thrust::seq, lower_it, indices + local_degree, minor);
              if (detail::count_set_bits((*edge_partition_e_mask).value_first(),
                                         edge_offset + cuda::std::distance(indices, lower_it),
                                         cuda::std::distance(lower_it, upper_it)) == 0) {
                return true;
              }
            }
            return false;
          }) == 0,
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
    auto edge_partition_e_value_output =
      edge_partition_e_output_device_view_t(edge_value_output, i);

    if (edge_partition_e_mask) {
      thrust::for_each(
        handle.get_thrust_policy(),
        edge_first + edge_partition_offsets[i],
        edge_first + edge_partition_offsets[i + 1],
        detail::update_e_value_t<true,
                                 GraphViewType,
                                 edge_partition_src_input_device_view_t,
                                 edge_partition_dst_input_device_view_t,
                                 edge_partition_e_input_device_view_t,
                                 std::remove_reference_t<decltype(*edge_partition_e_mask)>,
                                 EdgeOp,
                                 edge_partition_e_output_device_view_t>{
          edge_partition,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          *edge_partition_e_mask,
          e_op,
          edge_partition_e_value_output});
    } else {
      thrust::for_each(handle.get_thrust_policy(),
                       edge_first + edge_partition_offsets[i],
                       edge_first + edge_partition_offsets[i + 1],
                       detail::update_e_value_t<false,
                                                GraphViewType,
                                                edge_partition_src_input_device_view_t,
                                                edge_partition_dst_input_device_view_t,
                                                edge_partition_e_input_device_view_t,
                                                std::byte,  // dummy
                                                EdgeOp,
                                                edge_partition_e_output_device_view_t>{
                         edge_partition,
                         edge_partition_src_value_input,
                         edge_partition_dst_value_input,
                         edge_partition_e_value_input,
                         std::byte{},  // dummy
                         e_op,
                         edge_partition_e_value_output});
    }
  }
}

}  // namespace cugraph

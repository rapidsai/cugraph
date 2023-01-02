/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/property_op_utils.cuh>
#include <prims/reduce_op.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/scatter.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>

#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

int32_t constexpr per_v_transform_reduce_e_kernel_block_size = 512;

template <bool update_major,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename T>
__global__ void per_v_transform_reduce_e_hypersparse(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */,
  ReduceOp reduce_op)
{
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(*(edge_partition.major_hypersparse_first()) -
                                                edge_partition.major_range_first());
  auto idx                = static_cast<size_t>(tid);

  auto dcs_nzd_vertex_count = *(edge_partition.dcs_nzd_vertex_count());

  while (idx < static_cast<size_t>(dcs_nzd_vertex_count)) {
    auto major =
      *(edge_partition.major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
    auto major_idx =
      major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_idx));
    auto transform_op = [&edge_partition,
                         &edge_partition_src_value_input,
                         &edge_partition_dst_value_input,
                         &edge_partition_e_value_input,
                         &e_op,
                         major,
                         indices,
                         edge_offset] __device__(auto i) {
      auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
      auto minor        = indices[i];
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      auto src          = GraphViewType::is_storage_transposed ? minor : major;
      auto dst          = GraphViewType::is_storage_transposed ? major : minor;
      auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
      auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
      return e_op(src,
                  dst,
                  edge_partition_src_value_input.get(src_offset),
                  edge_partition_dst_value_input.get(dst_offset),
                  edge_partition_e_value_input.get(edge_offset + i));
    };

    if constexpr (update_major) {
      *(result_value_output + (major - *(edge_partition.major_hypersparse_first()))) =
        thrust::transform_reduce(thrust::seq,
                                 thrust::make_counting_iterator(edge_t{0}),
                                 thrust::make_counting_iterator(local_degree),
                                 transform_op,
                                 init,
                                 reduce_op);
    } else {
      if constexpr (GraphViewType::is_multi_gpu) {
        thrust::for_each(
          thrust::seq,
          thrust::make_counting_iterator(edge_t{0}),
          thrust::make_counting_iterator(local_degree),
          [&edge_partition, indices, &result_value_output, &transform_op] __device__(auto i) {
            auto e_op_result  = transform_op(i);
            auto minor        = indices[i];
            auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
            reduce_op::atomic_reduce<ReduceOp>(result_value_output.get_iter(minor_offset),
                                               e_op_result);
          });
      } else {
        thrust::for_each(
          thrust::seq,
          thrust::make_counting_iterator(edge_t{0}),
          thrust::make_counting_iterator(local_degree),
          [&edge_partition, indices, &result_value_output, &transform_op] __device__(auto i) {
            auto e_op_result  = transform_op(i);
            auto minor        = indices[i];
            auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
            reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
          });
      }
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename T>
__global__ void per_v_transform_reduce_e_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */,
  ReduceOp reduce_op)
{
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  auto idx = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));
    auto transform_op = [&edge_partition,
                         &edge_partition_src_value_input,
                         &edge_partition_dst_value_input,
                         &edge_partition_e_value_input,
                         &e_op,
                         major_offset,
                         indices,
                         edge_offset] __device__(auto i) {
      auto minor        = indices[i];
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      auto src          = GraphViewType::is_storage_transposed
                            ? minor
                            : edge_partition.major_from_major_offset_nocheck(major_offset);
      auto dst          = GraphViewType::is_storage_transposed
                            ? edge_partition.major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto src_offset =
        GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
      auto dst_offset =
        GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
      return e_op(src,
                  dst,
                  edge_partition_src_value_input.get(src_offset),
                  edge_partition_dst_value_input.get(dst_offset),
                  edge_partition_e_value_input.get(edge_offset + i));
    };

    if constexpr (update_major) {
      *(result_value_output + idx) =
        thrust::transform_reduce(thrust::seq,
                                 thrust::make_counting_iterator(edge_t{0}),
                                 thrust::make_counting_iterator(local_degree),
                                 transform_op,
                                 init,
                                 reduce_op);
    } else {
      if constexpr (GraphViewType::is_multi_gpu) {
        thrust::for_each(
          thrust::seq,
          thrust::make_counting_iterator(edge_t{0}),
          thrust::make_counting_iterator(local_degree),
          [&edge_partition, indices, &result_value_output, &transform_op] __device__(auto i) {
            auto e_op_result  = transform_op(i);
            auto minor        = indices[i];
            auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
            reduce_op::atomic_reduce<ReduceOp>(result_value_output.get_iter(minor_offset),
                                               e_op_result);
          });
      } else {
        thrust::for_each(
          thrust::seq,
          thrust::make_counting_iterator(edge_t{0}),
          thrust::make_counting_iterator(local_degree),
          [&edge_partition, indices, &result_value_output, &transform_op] __device__(auto i) {
            auto e_op_result  = transform_op(i);
            auto minor        = indices[i];
            auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
            reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
          });
      }
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename T>
__global__ void per_v_transform_reduce_e_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */,
  ReduceOp reduce_op)
{
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true

  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = T;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(per_v_transform_reduce_e_kernel_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  auto idx = static_cast<size_t>(tid / raft::warp_size());

  using WarpReduce = cub::WarpReduce<e_op_result_t>;
  [[maybe_unused]] __shared__ typename WarpReduce::TempStorage
    temp_storage[per_v_transform_reduce_e_kernel_block_size /
                 raft::warp_size()];  // relevant only if update_major == true

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    [[maybe_unused]] auto reduced_e_op_result =
      lane_id == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      auto minor        = indices[i];
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      auto src          = GraphViewType::is_storage_transposed
                            ? minor
                            : edge_partition.major_from_major_offset_nocheck(major_offset);
      auto dst          = GraphViewType::is_storage_transposed
                            ? edge_partition.major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto src_offset =
        GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
      auto dst_offset =
        GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
      auto e_op_result = e_op(src,
                              dst,
                              edge_partition_src_value_input.get(src_offset),
                              edge_partition_dst_value_input.get(dst_offset),
                              edge_partition_e_value_input.get(edge_offset + i));
      if constexpr (update_major) {
        reduced_e_op_result = reduce_op(reduced_e_op_result, e_op_result);
      } else {
        if constexpr (GraphViewType::is_multi_gpu) {
          reduce_op::atomic_reduce<ReduceOp>(result_value_output.get_iter(minor_offset),
                                             e_op_result);
        } else {
          reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
        }
      }
    }
    if constexpr (update_major) {
      reduced_e_op_result = WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
                              .Reduce(reduced_e_op_result, reduce_op);
      if (lane_id == 0) { *(result_value_output + idx) = reduced_e_op_result; }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <bool update_major,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename T>
__global__ void per_v_transform_reduce_e_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */,
  ReduceOp reduce_op)
{
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true

  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = T;

  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  auto idx = static_cast<size_t>(blockIdx.x);

  using BlockReduce = cub::BlockReduce<e_op_result_t, per_v_transform_reduce_e_kernel_block_size>;
  [[maybe_unused]] __shared__
    typename BlockReduce::TempStorage temp_storage;  // relevant only if update_major == true

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    [[maybe_unused]] auto reduced_e_op_result =
      threadIdx.x == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      auto minor        = indices[i];
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      auto src          = GraphViewType::is_storage_transposed
                            ? minor
                            : edge_partition.major_from_major_offset_nocheck(major_offset);
      auto dst          = GraphViewType::is_storage_transposed
                            ? edge_partition.major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto src_offset =
        GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
      auto dst_offset =
        GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
      auto e_op_result = e_op(src,
                              dst,
                              edge_partition_src_value_input.get(src_offset),
                              edge_partition_dst_value_input.get(dst_offset),
                              edge_partition_e_value_input.get(edge_offset + i));
      if constexpr (update_major) {
        reduced_e_op_result = reduce_op(reduced_e_op_result, e_op_result);
      } else {
        if constexpr (GraphViewType::is_multi_gpu) {
          reduce_op::atomic_reduce<ReduceOp>(result_value_output.get_iter(minor_offset),
                                             e_op_result);
        } else {
          reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
        }
      }
    }
    if constexpr (update_major) {
      reduced_e_op_result = BlockReduce(temp_storage).Reduce(reduced_e_op_result, reduce_op);
      if (threadIdx.x == 0) { *(result_value_output + idx) = reduced_e_op_result; }
    }

    idx += gridDim.x;
  }
}

template <bool incoming,  // iterate over incoming edges (incoming == true) or outgoing edges
                          // (incoming == false)
          typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_e(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              EdgeSrcValueInputWrapper edge_src_value_input,
                              EdgeDstValueInputWrapper edge_dst_value_input,
                              EdgeValueInputWrapper edge_value_input,
                              EdgeOp e_op,
                              T init,
                              ReduceOp reduce_op,
                              VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(ReduceOp::pure_function || reduce_op::has_compatible_raft_comms_op_v<ReduceOp> ||
                reduce_op::has_identity_element_v<ReduceOp>);  // current restriction, to support
                                                               // general reduction, we may need to
                                                               // take a less efficient code path

  constexpr auto update_major = (incoming == GraphViewType::is_storage_transposed);
  [[maybe_unused]] constexpr auto max_segments =
    detail::num_sparse_segments_per_vertex_partition + size_t{1};
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator>>;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  [[maybe_unused]] std::conditional_t<GraphViewType::is_storage_transposed,
                                      edge_src_property_t<GraphViewType, T>,
                                      edge_dst_property_t<GraphViewType, T>>
    minor_tmp_buffer(handle);  // relevant only when (GraphViewType::is_multi_gpu && !update_major
  if constexpr (GraphViewType::is_multi_gpu && !update_major) {
    if constexpr (GraphViewType::is_storage_transposed) {
      minor_tmp_buffer = edge_src_property_t<GraphViewType, T>(handle, graph_view);
    } else {
      minor_tmp_buffer = edge_dst_property_t<GraphViewType, T>(handle, graph_view);
    }
  }

  using edge_partition_minor_output_device_view_t =
    std::conditional_t<update_major,
                       void /* dummy */,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         decltype(minor_tmp_buffer.mutable_view().value_first())>>;

  if constexpr (update_major) {
    size_t partition_idx = 0;
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      partition_idx            = static_cast<size_t>(col_comm_rank);
    }
    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(partition_idx);
    if (segment_offsets) {  // no vertices in the zero degree segment are visited
      thrust::fill(handle.get_thrust_policy(),
                   vertex_value_output_first + *((*segment_offsets).rbegin() + 1),
                   vertex_value_output_first + *((*segment_offsets).rbegin()),
                   init);
    }
  } else {
    auto minor_init = init;
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      minor_init               = (row_comm_rank == 0) ? init : ReduceOp::identity_element;
    }

    if constexpr (GraphViewType::is_multi_gpu) {
      fill_edge_minor_property(handle, graph_view, minor_init, minor_tmp_buffer.mutable_view());
    } else {
      thrust::fill(handle.get_thrust_policy(),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_view.local_vertex_partition_range_size(),
                   minor_init);
    }
  }

  std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    if ((graph_view.local_edge_partition_segment_offsets(0)) &&
        (handle.get_stream_pool_size() >= max_segments)) {
      for (size_t i = 1; i < graph_view.number_of_local_edge_partitions(); ++i) {
        assert(graph_view.local_edge_partition_segment_offsets(i));
      }

      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_size = col_comm.get_size();

      // memory footprint vs parallelism trade-off
      // peak memory requirement per loop is
      // update_major ? V / comm_size * sizeof(T) : 0
      // and limit memory requirement to (E / comm_size) * sizeof(vertex_t)

      size_t num_streams =
        std::min(static_cast<size_t>(col_comm_size) * max_segments,
                 raft::round_down_safe(handle.get_stream_pool_size(), max_segments));
      if constexpr (update_major) {
        size_t value_size{0};
        if constexpr (is_thrust_tuple_of_arithmetic<T>::value) {
          auto elem_sizes = compute_thrust_tuple_element_sizes<T>{}();
          value_size      = std::reduce(elem_sizes.begin(), elem_sizes.end());
        } else {
          value_size = sizeof(T);
        }

        auto avg_vertex_degree = graph_view.number_of_vertices() > 0
                                   ? (static_cast<double>(graph_view.number_of_edges()) /
                                      static_cast<double>(graph_view.number_of_vertices()))
                                   : double{0.0};

        num_streams =
          std::min(static_cast<size_t>(avg_vertex_degree * (static_cast<double>(sizeof(vertex_t)) /
                                                            static_cast<double>(value_size))) *
                     max_segments,
                   num_streams);
      }

      if (num_streams >= max_segments) {
        stream_pool_indices = std::vector<size_t>(num_streams);
        std::iota((*stream_pool_indices).begin(), (*stream_pool_indices).end(), size_t{0});
        handle.sync_stream();
      }
    }
  }

  std::vector<decltype(allocate_dataframe_buffer<T>(0, rmm::cuda_stream_view{}))>
    major_tmp_buffers{};
  if constexpr (GraphViewType::is_multi_gpu && update_major) {
    std::vector<size_t> major_tmp_buffer_sizes(graph_view.number_of_local_edge_partitions(),
                                               size_t{0});
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
      if (segment_offsets) {
        major_tmp_buffer_sizes[i] =
          *((*segment_offsets).rbegin() + 1);  // exclude the zero degree segment
      } else {
        if constexpr (GraphViewType::is_storage_transposed) {
          major_tmp_buffer_sizes[i] = graph_view.local_edge_partition_dst_range_size(i);
        } else {
          major_tmp_buffer_sizes[i] = graph_view.local_edge_partition_src_range_size(i);
        }
      }
    }
    if (stream_pool_indices) {
      auto num_concurrent_loops = (*stream_pool_indices).size() / max_segments;
      major_tmp_buffers.reserve(num_concurrent_loops);
      for (size_t i = 0; i < num_concurrent_loops; ++i) {
        size_t max_size{0};
        for (size_t j = i; j < graph_view.number_of_local_edge_partitions();
             j += num_concurrent_loops) {
          max_size = std::max(major_tmp_buffer_sizes[j], max_size);
        }
        major_tmp_buffers.push_back(allocate_dataframe_buffer<T>(max_size, handle.get_stream()));
      }
    } else {
      major_tmp_buffers.reserve(1);
      major_tmp_buffers.push_back(allocate_dataframe_buffer<T>(
        *std::max_element(major_tmp_buffer_sizes.begin(), major_tmp_buffer_sizes.end()),
        handle.get_stream()));
    }
  } else {  // dummy
    major_tmp_buffers.reserve(1);
    major_tmp_buffers.push_back(allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream()));
  }

  if (stream_pool_indices) { handle.sync_stream(); }

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto major_init = ReduceOp::identity_element;
    if constexpr (update_major) {
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
        auto const col_comm_rank = col_comm.get_rank();
        major_init = (static_cast<int>(i) == col_comm_rank) ? init : ReduceOp::identity_element;
      } else {
        major_init = init;
      }
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

    auto major_buffer_first =
      get_dataframe_buffer_begin(major_tmp_buffers[i % major_tmp_buffers.size()]);

    std::conditional_t<GraphViewType::is_multi_gpu,
                       std::conditional_t<update_major,
                                          decltype(major_buffer_first),
                                          edge_partition_minor_output_device_view_t>,
                       VertexValueOutputIterator>
      output_buffer{};
    if constexpr (GraphViewType::is_multi_gpu) {
      if constexpr (update_major) {
        output_buffer = major_buffer_first;
      } else {
        output_buffer = edge_partition_minor_output_device_view_t(minor_tmp_buffer.mutable_view());
      }
    } else {
      output_buffer = vertex_value_output_first;
    }

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    if (segment_offsets) {
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);

      // FIXME: we may further improve performance by 1) individually tuning block sizes for
      // different segments; and 2) adding one more segment for very high degree vertices and
      // running segmented reduction
      if (edge_partition.dcs_nzd_vertex_count()) {
        auto exec_stream =
          stream_pool_indices
            ? handle.get_stream_from_stream_pool((i * max_segments) % (*stream_pool_indices).size())
            : handle.get_stream();

        if constexpr (update_major) {  // this is necessary as we don't visit every vertex in the
                                       // hypersparse segment
          thrust::fill(rmm::exec_policy(exec_stream),
                       output_buffer + (*segment_offsets)[3],
                       output_buffer + (*segment_offsets)[4],
                       major_init);
        }

        if (*(edge_partition.dcs_nzd_vertex_count()) > 0) {
          raft::grid_1d_thread_t update_grid(*(edge_partition.dcs_nzd_vertex_count()),
                                             detail::per_v_transform_reduce_e_kernel_block_size,
                                             handle.get_device_properties().maxGridSize[0]);
          auto segment_output_buffer = output_buffer;
          if constexpr (update_major) { segment_output_buffer += (*segment_offsets)[3]; }
          detail::per_v_transform_reduce_e_hypersparse<update_major, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
              edge_partition,
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              segment_output_buffer,
              e_op,
              major_init,
              reduce_op);
        }
      }
      if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
        auto exec_stream = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((i * max_segments + 1) %
                                                                  (*stream_pool_indices).size())
                             : handle.get_stream();
        raft::grid_1d_thread_t update_grid((*segment_offsets)[3] - (*segment_offsets)[2],
                                           detail::per_v_transform_reduce_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        auto segment_output_buffer = output_buffer;
        if constexpr (update_major) { segment_output_buffer += (*segment_offsets)[2]; }
        detail::per_v_transform_reduce_e_low_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[2],
            edge_partition.major_range_first() + (*segment_offsets)[3],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            segment_output_buffer,
            e_op,
            major_init,
            reduce_op);
      }
      if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
        auto exec_stream = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((i * max_segments + 2) %
                                                                  (*stream_pool_indices).size())
                             : handle.get_stream();
        raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                         detail::per_v_transform_reduce_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        auto segment_output_buffer = output_buffer;
        if constexpr (update_major) { segment_output_buffer += (*segment_offsets)[1]; }
        detail::per_v_transform_reduce_e_mid_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[1],
            edge_partition.major_range_first() + (*segment_offsets)[2],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            segment_output_buffer,
            e_op,
            major_init,
            reduce_op);
      }
      if ((*segment_offsets)[1] > 0) {
        auto exec_stream = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((i * max_segments + 3) %
                                                                  (*stream_pool_indices).size())
                             : handle.get_stream();
        raft::grid_1d_block_t update_grid((*segment_offsets)[1],
                                          detail::per_v_transform_reduce_e_kernel_block_size,
                                          handle.get_device_properties().maxGridSize[0]);
        detail::per_v_transform_reduce_e_high_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
            edge_partition,
            edge_partition.major_range_first(),
            edge_partition.major_range_first() + (*segment_offsets)[1],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            output_buffer,
            e_op,
            major_init,
            reduce_op);
      }
    } else {
      if (edge_partition.major_range_size() > 0) {
        raft::grid_1d_thread_t update_grid(edge_partition.major_range_size(),
                                           detail::per_v_transform_reduce_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::per_v_transform_reduce_e_low_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first(),
            edge_partition.major_range_last(),
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            output_buffer,
            e_op,
            major_init,
            reduce_op);
      }
    }

    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      auto& comm     = handle.get_comms();
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      if (segment_offsets && stream_pool_indices) {
        if (edge_partition.dcs_nzd_vertex_count()) {
          device_reduce(
            col_comm,
            major_buffer_first + (*segment_offsets)[3],
            vertex_value_output_first + (*segment_offsets)[3],
            (*segment_offsets)[4] - (*segment_offsets)[3],
            ReduceOp::compatible_raft_comms_op,
            static_cast<int>(i),
            handle.get_stream_from_stream_pool((i * max_segments) % (*stream_pool_indices).size()));
        }
        if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
          device_reduce(col_comm,
                        major_buffer_first + (*segment_offsets)[2],
                        vertex_value_output_first + (*segment_offsets)[2],
                        (*segment_offsets)[3] - (*segment_offsets)[2],
                        ReduceOp::compatible_raft_comms_op,
                        static_cast<int>(i),
                        handle.get_stream_from_stream_pool((i * max_segments + 1) %
                                                           (*stream_pool_indices).size()));
        }
        if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
          device_reduce(col_comm,
                        major_buffer_first + (*segment_offsets)[1],
                        vertex_value_output_first + (*segment_offsets)[1],
                        (*segment_offsets)[2] - (*segment_offsets)[1],
                        ReduceOp::compatible_raft_comms_op,
                        static_cast<int>(i),
                        handle.get_stream_from_stream_pool((i * max_segments + 2) %
                                                           (*stream_pool_indices).size()));
        }
        if ((*segment_offsets)[1] > 0) {
          device_reduce(col_comm,
                        major_buffer_first,
                        vertex_value_output_first,
                        (*segment_offsets)[1],
                        ReduceOp::compatible_raft_comms_op,
                        static_cast<int>(i),
                        handle.get_stream_from_stream_pool((i * max_segments + 3) %
                                                           (*stream_pool_indices).size()));
        }
      } else {
        size_t reduction_size = static_cast<size_t>(
          segment_offsets ? *((*segment_offsets).rbegin() + 1) /* exclude the zero degree segment */
                          : edge_partition.major_range_size());
        device_reduce(col_comm,
                      major_buffer_first,
                      vertex_value_output_first,
                      reduction_size,
                      ReduceOp::compatible_raft_comms_op,
                      static_cast<int>(i),
                      handle.get_stream());
      }
    }
  }

  if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }

  if constexpr (GraphViewType::is_multi_gpu && !update_major) {
    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    auto view = minor_tmp_buffer.view();
    if (view.keys()) {
      vertex_t max_vertex_partition_size{0};
      for (int i = 0; i < row_comm_size; ++i) {
        max_vertex_partition_size =
          std::max(max_vertex_partition_size,
                   graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + i));
      }
      auto tx_buffer = allocate_dataframe_buffer<T>(max_vertex_partition_size, handle.get_stream());
      auto tx_first  = get_dataframe_buffer_begin(tx_buffer);
      std::optional<raft::host_span<vertex_t const>> minor_key_offsets{};
      if constexpr (GraphViewType::is_storage_transposed) {
        minor_key_offsets = graph_view.local_sorted_unique_edge_src_vertex_partition_offsets();
      } else {
        minor_key_offsets = graph_view.local_sorted_unique_edge_dst_vertex_partition_offsets();
      }
      for (int i = 0; i < row_comm_size; ++i) {
        thrust::fill(
          handle.get_thrust_policy(),
          tx_first,
          tx_first + graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + i),
          ReduceOp::identity_element);
        thrust::scatter(handle.get_thrust_policy(),
                        view.value_first() + (*minor_key_offsets)[i],
                        view.value_first() + (*minor_key_offsets)[i + 1],
                        thrust::make_transform_iterator(
                          (*(view.keys())).begin() + (*minor_key_offsets)[i],
                          [key_first = graph_view.vertex_partition_range_first(
                             col_comm_rank * row_comm_size + i)] __device__(auto key) {
                            return key - key_first;
                          }),
                        tx_first);
        device_reduce(row_comm,
                      tx_first,
                      vertex_value_output_first,
                      static_cast<size_t>(
                        graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + i)),
                      ReduceOp::compatible_raft_comms_op,
                      i,
                      handle.get_stream());
      }
    } else {
      for (int i = 0; i < row_comm_size; ++i) {
        auto offset = (graph_view.vertex_partition_range_first(col_comm_rank * row_comm_size + i) -
                       graph_view.vertex_partition_range_first(col_comm_rank * row_comm_size));
        device_reduce(row_comm,
                      view.value_first() + offset,
                      vertex_value_output_first,
                      static_cast<size_t>(
                        graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + i)),
                      ReduceOp::compatible_raft_comms_op,
                      i,
                      handle.get_stream());
      }
    }
  }
}

}  // namespace detail

/**
 * @brief Iterate over every vertex's incoming edges to update vertex properties.
 *
 * This function is inspired by thrust::transform_reduce.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
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

  detail::per_v_transform_reduce_e<true>(handle,
                                         graph_view,
                                         edge_src_value_input,
                                         edge_dst_value_input,
                                         edge_value_input,
                                         e_op,
                                         init,
                                         reduce_op,
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
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
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

  detail::per_v_transform_reduce_e<false>(handle,
                                          graph_view,
                                          edge_src_value_input,
                                          edge_dst_value_input,
                                          edge_value_input,
                                          e_op,
                                          init,
                                          reduce_op,
                                          vertex_value_output_first);
}

}  // namespace cugraph

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

#include <prims/property_op_utils.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <type_traits>

namespace cugraph {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_e_kernel_block_size = 128;

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void trasnform_reduce_e_hypersparse(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(*(edge_partition.major_hypersparse_first()) -
                                                edge_partition.major_range_first());
  size_t idx              = static_cast<size_t>(tid);

  auto dcs_nzd_vertex_count = *(edge_partition.dcs_nzd_vertex_count());

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_kernel_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(dcs_nzd_vertex_count)) {
    auto major =
      *(edge_partition.major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
    auto major_idx =
      major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_idx);
    auto sum                                        = thrust::transform_reduce(
      thrust::seq,
      thrust::make_counting_iterator(edge_t{0}),
      thrust::make_counting_iterator(local_degree),
      [&edge_partition,
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
        auto src_offset =
          GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
        auto dst_offset =
          GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
        return e_op(src,
                    dst,
                    edge_partition_src_value_input.get(src_offset),
                    edge_partition_dst_value_input.get(dst_offset),
                    edge_partition_e_value_input.get(edge_offset + i));
      },
      e_op_result_t{},
      edge_property_add);

    e_op_result_sum = edge_property_add(e_op_result_sum, sum);
    idx += gridDim.x * blockDim.x;
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_add_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void trasnform_reduce_e_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid);

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_kernel_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    auto sum                                        = thrust::transform_reduce(
      thrust::seq,
      thrust::make_counting_iterator(edge_t{0}),
      thrust::make_counting_iterator(local_degree),
      [&edge_partition,
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
      },
      e_op_result_t{},
      edge_property_add);

    e_op_result_sum = edge_property_add(e_op_result_sum, sum);
    idx += gridDim.x * blockDim.x;
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_add_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void trasnform_reduce_e_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(transform_reduce_e_kernel_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid / raft::warp_size());

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_kernel_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
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
      e_op_result_sum  = edge_property_add(e_op_result_sum, e_op_result);
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_add_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void trasnform_reduce_e_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(blockIdx.x);

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_kernel_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
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
      e_op_result_sum  = edge_property_add(e_op_result_sum, e_op_result);
    }
    idx += gridDim.x;
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_add_edge_op_result(result_iter, e_op_result_sum); }
}

}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value.
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
 * @param init Initial value to be added to the reduced @p edge_op outputs.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return T Transform-reduced @p edge_op outputs.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
T transform_reduce_e(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     EdgeSrcValueInputWrapper edge_src_value_input,
                     EdgeDstValueInputWrapper edge_dst_value_input,
                     EdgeValueInputWrapper edge_value_input,
                     EdgeOp e_op,
                     T init,
                     bool do_expensive_check = false)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

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

  if (do_expensive_check) {
    // currently, nothing to do
  }

  property_op<T, thrust::plus> edge_property_add{};

  auto result_buffer = allocate_dataframe_buffer<T>(1, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               get_dataframe_buffer_begin(result_buffer),
               get_dataframe_buffer_begin(result_buffer) + 1,
               T{});

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

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

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    if (segment_offsets) {
      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
      // segment for very high degree vertices and running segmented reduction
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
      if ((*segment_offsets)[1] > 0) {
        raft::grid_1d_block_t update_grid((*segment_offsets)[1],
                                          detail::transform_reduce_e_kernel_block_size,
                                          handle.get_device_properties().maxGridSize[0]);
        detail::trasnform_reduce_e_high_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first(),
            edge_partition.major_range_first() + (*segment_offsets)[1],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
        raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                         detail::transform_reduce_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        detail::trasnform_reduce_e_mid_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[1],
            edge_partition.major_range_first() + (*segment_offsets)[2],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
        raft::grid_1d_thread_t update_grid((*segment_offsets)[3] - (*segment_offsets)[2],
                                           detail::transform_reduce_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::trasnform_reduce_e_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[2],
            edge_partition.major_range_first() + (*segment_offsets)[3],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if (edge_partition.dcs_nzd_vertex_count() && (*(edge_partition.dcs_nzd_vertex_count()) > 0)) {
        raft::grid_1d_thread_t update_grid(*(edge_partition.dcs_nzd_vertex_count()),
                                           detail::transform_reduce_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::trasnform_reduce_e_hypersparse<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
    } else {
      if (edge_partition.major_range_size() > 0) {
        raft::grid_1d_thread_t update_grid(edge_partition.major_range_size(),
                                           detail::transform_reduce_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);

        detail::trasnform_reduce_e_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first(),
            edge_partition.major_range_last(),
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
    }
  }

  auto result = thrust::reduce(
    handle.get_thrust_policy(),
    get_dataframe_buffer_begin(result_buffer),
    get_dataframe_buffer_begin(result_buffer) + 1,
    ((GraphViewType::is_multi_gpu) && (handle.get_comms().get_rank() != 0)) ? T{} : init,
    edge_property_add);

  if constexpr (GraphViewType::is_multi_gpu) {
    result = host_scalar_allreduce(
      handle.get_comms(), result, raft::comms::op_t::SUM, handle.get_stream());
  }

  return result;
}

/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
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
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Transform-reduced @p edge_op outputs.
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp>
auto transform_reduce_e(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        EdgeSrcValueInputWrapper edge_src_value_input,
                        EdgeDstValueInputWrapper edge_dst_value_input,
                        EdgeValueInputWrapper edge_value_input,
                        EdgeOp e_op,
                        bool do_expensive_check = false)
{
  using vertex_t    = typename GraphViewType::vertex_type;
  using src_value_t = typename EdgeSrcValueInputWrapper::value_type;
  using dst_value_t = typename EdgeDstValueInputWrapper::value_type;
  using e_value_t   = typename EdgeValueInputWrapper::value_type;
  using T           = typename detail::
    edge_op_result_type<vertex_t, vertex_t, src_value_t, dst_value_t, e_value_t, EdgeOp>::type;
  static_assert(!std::is_same_v<T, void>);

  if (do_expensive_check) {
    // currently, nothing to do
  }

  return transform_reduce_e(
    handle, graph_view, edge_src_value_input, edge_dst_value_input, edge_value_input, e_op, T{});
}

}  // namespace cugraph

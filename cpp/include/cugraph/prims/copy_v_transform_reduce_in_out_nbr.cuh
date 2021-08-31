/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_barrier.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <cub/cub.cuh>

#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

int32_t constexpr copy_v_transform_reduce_nbr_for_all_block_size = 512;

template <bool update_major,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultValueOutputIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_hypersparse(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_hypersparse_first,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_hypersparse_first - matrix_partition.get_major_first());
  auto idx = static_cast<size_t>(tid);

  auto dcs_nzd_vertex_count = *(matrix_partition.get_dcs_nzd_vertex_count());

  property_add<T> edge_property_add{};
  while (idx < static_cast<size_t>(dcs_nzd_vertex_count)) {
    auto major =
      *(matrix_partition.get_major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
    auto major_idx =
      major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_idx));
    auto transform_op = [&matrix_partition,
                         &adj_matrix_row_value_input,
                         &adj_matrix_col_value_input,
                         &e_op,
                         major,
                         indices,
                         weights] __device__(auto i) {
      auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(major);
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed ? minor : major;
      auto col          = GraphViewType::is_adj_matrix_transposed ? major : minor;
      auto row_offset   = GraphViewType::is_adj_matrix_transposed
                            ? minor_offset
                            : static_cast<vertex_t>(major_offset);
      auto col_offset   = GraphViewType::is_adj_matrix_transposed
                            ? static_cast<vertex_t>(major_offset)
                            : minor_offset;
      return evaluate_edge_op<GraphViewType,
                              vertex_t,
                              AdjMatrixRowValueInputWrapper,
                              AdjMatrixColValueInputWrapper,
                              EdgeOp>()
        .compute(row,
                 col,
                 weight,
                 adj_matrix_row_value_input.get(row_offset),
                 adj_matrix_col_value_input.get(col_offset),
                 e_op);
    };

    if (update_major) {
      *(result_value_output_first + (major - major_hypersparse_first)) =
        thrust::transform_reduce(thrust::seq,
                                 thrust::make_counting_iterator(edge_t{0}),
                                 thrust::make_counting_iterator(local_degree),
                                 transform_op,
                                 init,
                                 edge_property_add);
    } else {
      thrust::for_each(
        thrust::seq,
        thrust::make_counting_iterator(edge_t{0}),
        thrust::make_counting_iterator(local_degree),
        [&matrix_partition, indices, &result_value_output_first, &transform_op] __device__(auto i) {
          auto e_op_result  = transform_op(i);
          auto minor        = indices[i];
          auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
          atomic_accumulate_edge_op_result(result_value_output_first + minor_offset, e_op_result);
        });
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultValueOutputIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_low_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(tid);

  property_add<T> edge_property_add{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_offset));
    auto transform_op = [&matrix_partition,
                         &adj_matrix_row_value_input,
                         &adj_matrix_col_value_input,
                         &e_op,
                         major_offset,
                         indices,
                         weights] __device__(auto i) {
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                            ? minor
                            : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col          = GraphViewType::is_adj_matrix_transposed
                            ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto row_offset   = GraphViewType::is_adj_matrix_transposed
                            ? minor_offset
                            : static_cast<vertex_t>(major_offset);
      auto col_offset   = GraphViewType::is_adj_matrix_transposed
                            ? static_cast<vertex_t>(major_offset)
                            : minor_offset;
      return evaluate_edge_op<GraphViewType,
                              vertex_t,
                              AdjMatrixRowValueInputWrapper,
                              AdjMatrixColValueInputWrapper,
                              EdgeOp>()
        .compute(row,
                 col,
                 weight,
                 adj_matrix_row_value_input.get(row_offset),
                 adj_matrix_col_value_input.get(col_offset),
                 e_op);
    };

    if (update_major) {
      *(result_value_output_first + idx) =
        thrust::transform_reduce(thrust::seq,
                                 thrust::make_counting_iterator(edge_t{0}),
                                 thrust::make_counting_iterator(local_degree),
                                 transform_op,
                                 init,
                                 edge_property_add);
    } else {
      thrust::for_each(
        thrust::seq,
        thrust::make_counting_iterator(edge_t{0}),
        thrust::make_counting_iterator(local_degree),
        [&matrix_partition, indices, &result_value_output_first, &transform_op] __device__(auto i) {
          auto e_op_result  = transform_op(i);
          auto minor        = indices[i];
          auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
          atomic_accumulate_edge_op_result(result_value_output_first + minor_offset, e_op_result);
        });
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultValueOutputIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_mid_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = T;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(copy_v_transform_reduce_nbr_for_all_block_size % raft::warp_size() == 0);
  auto const lane_id      = tid % raft::warp_size();
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(tid / raft::warp_size());

  property_add<e_op_result_t> edge_property_add{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    auto e_op_result_sum =
      lane_id == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                            ? minor
                            : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col          = GraphViewType::is_adj_matrix_transposed
                            ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto row_offset   = GraphViewType::is_adj_matrix_transposed
                            ? minor_offset
                            : static_cast<vertex_t>(major_offset);
      auto col_offset   = GraphViewType::is_adj_matrix_transposed
                            ? static_cast<vertex_t>(major_offset)
                            : minor_offset;
      auto e_op_result  = evaluate_edge_op<GraphViewType,
                                          vertex_t,
                                          AdjMatrixRowValueInputWrapper,
                                          AdjMatrixColValueInputWrapper,
                                          EdgeOp>()
                           .compute(row,
                                    col,
                                    weight,
                                    adj_matrix_row_value_input.get(row_offset),
                                    adj_matrix_col_value_input.get(col_offset),
                                    e_op);
      if (update_major) {
        e_op_result_sum = edge_property_add(e_op_result_sum, e_op_result);
      } else {
        atomic_accumulate_edge_op_result(result_value_output_first + minor_offset, e_op_result);
      }
    }
    if (update_major) {
      e_op_result_sum = warp_reduce_edge_op_result<e_op_result_t>().compute(e_op_result_sum);
      if (lane_id == 0) { *(result_value_output_first + idx) = e_op_result_sum; }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <bool update_major,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultValueOutputIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_high_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = T;

  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(blockIdx.x);

  property_add<e_op_result_t> edge_property_add{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    auto e_op_result_sum =
      threadIdx.x == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                            ? minor
                            : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col          = GraphViewType::is_adj_matrix_transposed
                            ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto row_offset   = GraphViewType::is_adj_matrix_transposed
                            ? minor_offset
                            : static_cast<vertex_t>(major_offset);
      auto col_offset   = GraphViewType::is_adj_matrix_transposed
                            ? static_cast<vertex_t>(major_offset)
                            : minor_offset;
      auto e_op_result  = evaluate_edge_op<GraphViewType,
                                          vertex_t,
                                          AdjMatrixRowValueInputWrapper,
                                          AdjMatrixColValueInputWrapper,
                                          EdgeOp>()
                           .compute(row,
                                    col,
                                    weight,
                                    adj_matrix_row_value_input.get(row_offset),
                                    adj_matrix_col_value_input.get(col_offset),
                                    e_op);
      if (update_major) {
        e_op_result_sum = edge_property_add(e_op_result_sum, e_op_result);
      } else {
        atomic_accumulate_edge_op_result(result_value_output_first + minor_offset, e_op_result);
      }
    }
    if (update_major) {
      e_op_result_sum =
        block_reduce_edge_op_result<e_op_result_t, copy_v_transform_reduce_nbr_for_all_block_size>()
          .compute(e_op_result_sum);
      if (threadIdx.x == 0) { *(result_value_output_first + idx) = e_op_result_sum; }
    }

    idx += gridDim.x;
  }
}

template <bool in,  // iterate over incoming edges (in == true) or outgoing edges (in == false)
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_nbr(raft::handle_t const& handle,
                                 GraphViewType const& graph_view,
                                 AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
                                 AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
                                 EdgeOp e_op,
                                 T init,
                                 VertexValueOutputIterator vertex_value_output_first)
{
  constexpr auto update_major = (in == GraphViewType::is_adj_matrix_transposed);
  using vertex_t              = typename GraphViewType::vertex_type;
  using edge_t                = typename GraphViewType::edge_type;
  using weight_t              = typename GraphViewType::weight_type;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  auto minor_tmp_buffer_size =
    (GraphViewType::is_multi_gpu && (in != GraphViewType::is_adj_matrix_transposed))
      ? GraphViewType::is_adj_matrix_transposed
          ? graph_view.get_number_of_local_adj_matrix_partition_rows()
          : graph_view.get_number_of_local_adj_matrix_partition_cols()
      : vertex_t{0};
  auto minor_tmp_buffer = allocate_dataframe_buffer<T>(minor_tmp_buffer_size, handle.get_stream());
  auto minor_buffer_first = get_dataframe_buffer_begin<T>(minor_tmp_buffer);

  if (in != GraphViewType::is_adj_matrix_transposed) {
    auto minor_init = init;
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      minor_init               = (row_comm_rank == 0) ? init : T{};
    }

    auto execution_policy = handle.get_thrust_policy();
    if (GraphViewType::is_multi_gpu) {
      thrust::fill(execution_policy,
                   minor_buffer_first,
                   minor_buffer_first + minor_tmp_buffer_size,
                   minor_init);
    } else {
      thrust::fill(execution_policy,
                   vertex_value_output_first,
                   vertex_value_output_first + graph_view.get_number_of_local_vertices(),
                   minor_init);
    }
  } else {
    assert(minor_tmp_buffer_size == 0);
  }

  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));

    auto major_tmp_buffer_size =
      GraphViewType::is_multi_gpu && update_major ? matrix_partition.get_major_size() : vertex_t{0};
    auto major_tmp_buffer =
      allocate_dataframe_buffer<T>(major_tmp_buffer_size, handle.get_stream());
    auto major_buffer_first = get_dataframe_buffer_begin<T>(major_tmp_buffer);

    auto major_init = T{};
    if (update_major) {
      if (GraphViewType::is_multi_gpu) {
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
        auto const col_comm_rank = col_comm.get_rank();
        major_init               = (col_comm_rank == 0) ? init : T{};
      } else {
        major_init = init;
      }
    }

    auto matrix_partition_row_value_input = adj_matrix_row_value_input;
    auto matrix_partition_col_value_input = adj_matrix_col_value_input;
    if constexpr (GraphViewType::is_adj_matrix_transposed) {
      matrix_partition_col_value_input.add_offset(matrix_partition.get_major_value_start_offset());
    } else {
      matrix_partition_row_value_input.add_offset(matrix_partition.get_major_value_start_offset());
    }

    std::conditional_t<
      GraphViewType::is_multi_gpu,
      std::conditional_t<update_major, decltype(major_buffer_first), decltype(minor_buffer_first)>,
      VertexValueOutputIterator>
      output_buffer_first{};
    if constexpr (GraphViewType::is_multi_gpu) {
      output_buffer_first = update_major ? major_buffer_first : minor_buffer_first;
    } else {
      output_buffer_first = vertex_value_output_first;
    }
    auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);
    if (segment_offsets) {
      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
      // segment for very high degree vertices and running segmented reduction
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
      if ((*segment_offsets)[1] > 0) {
        raft::grid_1d_block_t update_grid((*segment_offsets)[1],
                                          detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_high_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first(),
            matrix_partition.get_major_first() + (*segment_offsets)[1],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            output_buffer_first,
            e_op,
            major_init);
      }
      if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
        raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                         detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_mid_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first() + (*segment_offsets)[1],
            matrix_partition.get_major_first() + (*segment_offsets)[2],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            output_buffer_first + (update_major ? (*segment_offsets)[1] : vertex_t{0}),
            e_op,
            major_init);
      }
      if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
        raft::grid_1d_thread_t update_grid((*segment_offsets)[3] - (*segment_offsets)[2],
                                           detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_low_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first() + (*segment_offsets)[2],
            matrix_partition.get_major_first() + (*segment_offsets)[3],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            output_buffer_first + (update_major ? (*segment_offsets)[2] : vertex_t{0}),
            e_op,
            major_init);
      }
      if (matrix_partition.get_dcs_nzd_vertex_count()) {
        if constexpr (update_major) {  // this is necessary as we don't visit every vertex in the
                                       // hypersparse segment in
                                       // for_all_major_for_all_nbr_hypersparse
          thrust::fill(handle.get_thrust_policy(),
                       output_buffer_first + (*segment_offsets)[3],
                       output_buffer_first + (*segment_offsets)[4],
                       major_init);
        }
        if (*(matrix_partition.get_dcs_nzd_vertex_count()) > 0) {
          raft::grid_1d_thread_t update_grid(*(matrix_partition.get_dcs_nzd_vertex_count()),
                                             detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                             handle.get_device_properties().maxGridSize[0]);
          detail::for_all_major_for_all_nbr_hypersparse<update_major, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              matrix_partition,
              matrix_partition.get_major_first() + (*segment_offsets)[3],
              matrix_partition_row_value_input,
              matrix_partition_col_value_input,
              output_buffer_first + (update_major ? (*segment_offsets)[3] : vertex_t{0}),
              e_op,
              major_init);
        }
      }
    } else {
      if (matrix_partition.get_major_size() > 0) {
        raft::grid_1d_thread_t update_grid(matrix_partition.get_major_size(),
                                           detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_low_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first(),
            matrix_partition.get_major_last(),
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            output_buffer_first,
            e_op,
            major_init);
      }
    }

    if (GraphViewType::is_multi_gpu && update_major) {
      auto& comm     = handle.get_comms();
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
      // two different communicators (beginning of col_comm)
#if 1
      // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
      // and MPI barrier with MPI)
      host_barrier(comm, handle.get_stream_view());
#else
      handle.get_stream_view().synchronize();
      comm.barrier();  // currently, this is ncclAllReduce
#endif

      device_reduce(col_comm,
                    major_buffer_first,
                    vertex_value_output_first,
                    matrix_partition.get_major_size(),
                    raft::comms::op_t::SUM,
                    i,
                    handle.get_stream());

      // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
      // two different communicators (end of col_comm)
#if 1
      // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
      // and MPI barrier with MPI)
      host_barrier(comm, handle.get_stream_view());
#else
      handle.get_stream_view().synchronize();
      comm.barrier();  // currently, this is ncclAllReduce
#endif
    }
  }

  if (GraphViewType::is_multi_gpu && !update_major) {
    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (beginning of row_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif

    for (int i = 0; i < row_comm_size; ++i) {
      auto offset = (graph_view.get_vertex_partition_first(col_comm_rank * row_comm_size + i) -
                     graph_view.get_vertex_partition_first(col_comm_rank * row_comm_size));
      device_reduce(row_comm,
                    minor_buffer_first + offset,
                    vertex_value_output_first,
                    static_cast<size_t>(
                      graph_view.get_vertex_partition_size(col_comm_rank * row_comm_size + i)),
                    raft::comms::op_t::SUM,
                    i,
                    handle.get_stream());
    }

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (end of row_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif
  }
}

}  // namespace detail

/**
 * @brief Iterate over the incoming edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the incoming edges part)
 * and thrust::copy() (update vertex properties part, take transform_reduce output as copy input).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColValueInputWrapper Type of the wrapper for graph adjacency matrix column input
 * properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value for reduction over the incoming edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input Device-copyable wrapper used to access row input properties
 * (for the rows assigned to this process in multi-GPU). Use either
 * cugraph::row_properties_t::device_view() (if @p e_op needs to access row properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access row properties). Use
 * copy_to_adj_matrix_row to fill the wrapper.
 * @param adj_matrix_col_value_input Device-copyable wrapper used to access column input properties
 * (for the columns assigned to this process in multi-GPU). Use either
 * cugraph::col_properties_t::device_view() (if @p e_op needs to access column properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access column properties). Use
 * copy_to_adj_matrix_col to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), properties for the row (i.e. source), and properties for the column  (i.e. destination)
 * and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @p e_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_in_nbr(raft::handle_t const& handle,
                                    GraphViewType const& graph_view,
                                    AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
                                    AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
                                    EdgeOp e_op,
                                    T init,
                                    VertexValueOutputIterator vertex_value_output_first)
{
  detail::copy_v_transform_reduce_nbr<true>(handle,
                                            graph_view,
                                            adj_matrix_row_value_input,
                                            adj_matrix_col_value_input,
                                            e_op,
                                            init,
                                            vertex_value_output_first);
}

/**
 * @brief Iterate over the outgoing edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the outgoing edges
 * part) and thrust::copy() (update vertex properties part, take transform_reduce output as copy
 * input).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColValueInputWrapper Type of the wrapper for graph adjacency matrix column input
 * properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value for reduction over the outgoing edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input Device-copyable wrapper used to access row input properties
 * (for the rows assigned to this process in multi-GPU). Use either
 * cugraph::row_properties_t::device_view() (if @p e_op needs to access row properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access row properties). Use
 * copy_to_adj_matrix_row to fill the wrapper.
 * @param adj_matrix_col_value_input Device-copyable wrapper used to access column input properties
 * (for the columns assigned to this process in multi-GPU). Use either
 * cugraph::col_properties_t::device_view() (if @p e_op needs to access column properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access column properties). Use
 * copy_to_adj_matrix_col to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), properties for the row (i.e. source), and properties for the column  (i.e. destination)
 * and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @p e_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_out_nbr(raft::handle_t const& handle,
                                     GraphViewType const& graph_view,
                                     AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
                                     AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
                                     EdgeOp e_op,
                                     T init,
                                     VertexValueOutputIterator vertex_value_output_first)
{
  detail::copy_v_transform_reduce_nbr<false>(handle,
                                             graph_view,
                                             adj_matrix_row_value_input,
                                             adj_matrix_col_value_input,
                                             e_op,
                                             init,
                                             vertex_value_output_first);
}

}  // namespace cugraph

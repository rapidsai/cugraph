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

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph_view.hpp>
#include <matrix_partition_device.cuh>
#include <patterns/edge_op_utils.cuh>
#include <utilities/dataframe_buffer.cuh>
#include <utilities/error.hpp>
#include <utilities/shuffle_comm.cuh>

#include <raft/handle.hpp>

#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_by_key_e_for_all_block_size = 128;

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename VertexIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_low_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  VertexIterator adj_matrix_row_col_key_first,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  T* values)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_offset));
    if (local_degree > 0) {
      auto transform_op = [&matrix_partition,
                           &adj_matrix_row_value_input_first,
                           &adj_matrix_col_value_input_first,
                           &adj_matrix_row_col_key_first,
                           &e_op,
                           major_offset,
                           indices,
                           weights] __device__(auto i) {
        auto minor        = indices[i];
        auto weight       = weights != nullptr ? weights[i] : weight_t{1.0};
        auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
        auto row          = GraphViewType::is_adj_matrix_transposed
                     ? minor
                     : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
        auto col = GraphViewType::is_adj_matrix_transposed
                     ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                     : minor;
        auto row_offset = GraphViewType::is_adj_matrix_transposed
                            ? minor_offset
                            : static_cast<vertex_t>(major_offset);
        auto col_offset = GraphViewType::is_adj_matrix_transposed
                            ? static_cast<vertex_t>(major_offset)
                            : minor_offset;

        auto key =
          *(adj_matrix_row_col_key_first +
            ((GraphViewType::is_adj_matrix_transposed != adj_matrix_row_key) ? major_offset
                                                                             : minor_offset));
        auto e_op_result = evaluate_edge_op<GraphViewType,
                                            vertex_t,
                                            AdjMatrixRowValueInputIterator,
                                            AdjMatrixColValueInputIterator,
                                            EdgeOp>()
                             .compute(row,
                                      col,
                                      weight,
                                      *(adj_matrix_row_value_input_first + row_offset),
                                      *(adj_matrix_col_value_input_first + col_offset),
                                      e_op);

        return thrust::make_tuple(key, e_op_result);
      };

      auto local_offset = matrix_partition.get_local_offset(major_offset);
      thrust::transform(
        thrust::seq,
        thrust::make_counting_iterator(edge_t{0}),
        thrust::make_counting_iterator(local_degree),
        thrust::make_zip_iterator(thrust::make_tuple(keys + local_offset, values + local_offset)),
        transform_op);
    }

    idx += gridDim.x * blockDim.x;
  }
}

// FIXME: better derive value_t from BufferType
template <typename vertex_t, typename value_t, typename BufferType>
std::tuple<rmm::device_uvector<vertex_t>, BufferType> reduce_to_unique_kv_pairs(
  rmm::device_uvector<vertex_t>&& keys, BufferType&& value_buffer, cudaStream_t stream)
{
  thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
                      keys.begin(),
                      keys.end(),
                      get_dataframe_buffer_begin<value_t>(value_buffer));
  auto num_uniques =
    thrust::count_if(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(keys.size()),
                     [keys = keys.data()] __device__(auto i) {
                       return ((i == 0) || (keys[i] != keys[i - 1])) ? true : false;
                     });

  rmm::device_uvector<vertex_t> unique_keys(num_uniques, stream);
  auto value_for_unique_key_buffer = allocate_dataframe_buffer<value_t>(unique_keys.size(), stream);
  thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                        keys.begin(),
                        keys.end(),
                        get_dataframe_buffer_begin<value_t>(value_buffer),
                        unique_keys.begin(),
                        get_dataframe_buffer_begin<value_t>(value_for_unique_key_buffer));

  return std::make_tuple(std::move(unique_keys), std::move(value_for_unique_key_buffer));
}

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename VertexIterator,
          typename EdgeOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           decltype(allocate_dataframe_buffer<T>(0, cudaStream_t{nullptr}))>
transform_reduce_by_adj_matrix_row_col_key_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  VertexIterator adj_matrix_row_col_key_first,
  EdgeOp e_op,
  T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename std::iterator_traits<VertexIterator>::value_type,
                             typename GraphViewType::vertex_type>::value);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  rmm::device_uvector<vertex_t> keys(0, handle.get_stream());
  auto value_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

    int comm_root_rank = 0;
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      comm_root_rank           = i * row_comm_size + row_comm_rank;
    }

    auto num_edges = thrust::transform_reduce(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      thrust::make_counting_iterator(graph_view.get_vertex_partition_first(comm_root_rank)),
      thrust::make_counting_iterator(graph_view.get_vertex_partition_last(comm_root_rank)),
      [matrix_partition] __device__(auto row) {
        auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
        return matrix_partition.get_local_degree(major_offset);
      },
      edge_t{0},
      thrust::plus<edge_t>());

    rmm::device_uvector<vertex_t> tmp_keys(num_edges, handle.get_stream());
    auto tmp_value_buffer = allocate_dataframe_buffer<T>(tmp_keys.size(), handle.get_stream());

    if (graph_view.get_vertex_partition_size(comm_root_rank) > 0) {
      raft::grid_1d_thread_t update_grid(graph_view.get_vertex_partition_size(comm_root_rank),
                                         detail::transform_reduce_by_key_e_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);

      auto row_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                      ? vertex_t{0}
                                      : matrix_partition.get_major_value_start_offset();
      auto col_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                      ? matrix_partition.get_major_value_start_offset()
                                      : vertex_t{0};

      // FIXME: This is highly inefficient for graphs with high-degree vertices. If we renumber
      // vertices to insure that rows within a partition are sorted by their out-degree in
      // decreasing order, we will apply this kernel only to low out-degree vertices.
      detail::for_all_major_for_all_nbr_low_degree<adj_matrix_row_key>
        <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
          matrix_partition,
          graph_view.get_vertex_partition_first(comm_root_rank),
          graph_view.get_vertex_partition_last(comm_root_rank),
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first + col_value_input_offset,
          adj_matrix_row_col_key_first +
            (adj_matrix_row_key ? row_value_input_offset : col_value_input_offset),
          e_op,
          tmp_keys.data(),
          get_dataframe_buffer_begin<T>(tmp_value_buffer));
    }
    std::tie(tmp_keys, tmp_value_buffer) = reduce_to_unique_kv_pairs<vertex_t, T>(
      std::move(tmp_keys), std::move(tmp_value_buffer), handle.get_stream());

    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      rmm::device_uvector<vertex_t> rx_unique_keys(0, handle.get_stream());
      auto rx_value_for_unique_key_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
      std::tie(rx_unique_keys, rx_value_for_unique_key_buffer, std::ignore) =
        groupby_gpuid_and_shuffle_kv_pairs(
          comm,
          tmp_keys.begin(),
          tmp_keys.end(),
          get_dataframe_buffer_begin<T>(tmp_value_buffer),
          [key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(
            auto val) { return key_func(val); },
          handle.get_stream());

      std::tie(tmp_keys, tmp_value_buffer) = reduce_to_unique_kv_pairs<vertex_t, T>(
        std::move(rx_unique_keys), std::move(rx_value_for_unique_key_buffer), handle.get_stream());
    }

    auto cur_size = keys.size();
    if (cur_size == 0) {
      keys         = std::move(tmp_keys);
      value_buffer = std::move(tmp_value_buffer);
    } else {
      // FIXME: this can lead to frequent costly reallocation; we may be able to avoid this if we
      // can reserve address space to avoid expensive reallocation.
      // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management
      keys.resize(cur_size + tmp_keys.size(), handle.get_stream());
      resize_dataframe_buffer<T>(value_buffer, keys.size(), handle.get_stream());

      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   tmp_keys.begin(),
                   tmp_keys.end(),
                   keys.begin() + cur_size);
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   get_dataframe_buffer_begin<T>(tmp_value_buffer),
                   get_dataframe_buffer_begin<T>(tmp_value_buffer) + tmp_keys.size(),
                   get_dataframe_buffer_begin<T>(value_buffer) + cur_size);
    }
  }

  if (GraphViewType::is_multi_gpu) {
    std::tie(keys, value_buffer) = reduce_to_unique_kv_pairs<vertex_t, T>(
      std::move(keys), std::move(value_buffer), handle.get_stream());
  }

  // FIXME: add init

  return std::make_tuple(std::move(keys), std::move(value_buffer));
}

}  // namespace detail

// FIXME: EdgeOp & VertexOp in update_frontier_v_push_if_out_nbr concatenates push inidicator or
// bucket idx with the value while EdgeOp here does not. This is inconsistent. Better be fixed.
/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs to (key, value) pairs.
 *
 * This function is inspired by thrust::transform_reduce() and thrust::reduce_by_key(). Keys for
 * edges are determined by the graph adjacency matrix rows.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam VertexIterator Type of the iterator for keys in (key, value) pairs (key type should
 * coincide with vertex type).
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the values in (key, value) pairs.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
 * @param adj_matrix_row_key_first Iterator pointing to the adjacency matrix row key for the first
 * (inclusive) column (assigned to this process in multi-GPU). `adj_matrix_row_key_last` (exclusive)
 * is deduced as @p adj_matrix_row_key_first + @p graph_view.get_number_of_local_adj_matrix_rows().
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), *(@p adj_matrix_row_value_input_first + i), and *(@p adj_matrix_col_value_input_first +
 * j) (where i is in [0, graph_view.get_number_of_local_adj_matrix_partition_rows()) and j is in [0,
 * get_number_of_local_adj_matrix_partition_cols())) and returns a transformed value to be reduced.
 * @param init Initial value to be added to the value in each transform-reduced (key, value) pair.
 * @return std::tuple Tuple of rmm::device_uvector<typename GraphView::vertex_type> and
 * rmm::device_uvector<T> (if T is arithmetic scalar) or a tuple of rmm::device_uvector objects (if
 * T is a thrust::tuple type of arithmetic scalar types, one rmm::device_uvector object per scalar
 * type).
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename VertexIterator,
          typename EdgeOp,
          typename T>
auto transform_reduce_by_adj_matrix_row_key_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  VertexIterator adj_matrix_row_key_first,
  EdgeOp e_op,
  T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename std::iterator_traits<VertexIterator>::value_type,
                             typename GraphViewType::vertex_type>::value);

  return detail::transform_reduce_by_adj_matrix_row_col_key_e<true>(
    handle,
    graph_view,
    adj_matrix_row_value_input_first,
    adj_matrix_col_value_input_first,
    adj_matrix_row_key_first,
    e_op,
    init);
}

// FIXME: EdgeOp & VertexOp in update_frontier_v_push_if_out_nbr concatenates push inidicator or
// bucket idx with the value while EdgeOp here does not. This is inconsistent. Better be fixed.
/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs to (key, value) pairs.
 *
 * This function is inspired by thrust::transform_reduce() and thrust::reduce_by_key(). Keys for
 * edges are determined by the graph adjacency matrix columns.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam VertexIterator Type of the iterator for keys in (key, value) pairs (key type should
 * coincide with vertex type).
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the values in (key, value) pairs.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
 * @param adj_matrix_col_key_first Iterator pointing to the adjacency matrix column key for the
 * first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_key_last` (exclusive) is deduced as @p adj_matrix_col_key_first + @p
 * graph_view.get_number_of_local_adj_matrix_cols().
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), *(@p adj_matrix_row_value_input_first + i), and *(@p adj_matrix_col_value_input_first +
 * j) (where i is in [0, graph_view.get_number_of_local_adj_matrix_partition_rows()) and j is in [0,
 * get_number_of_local_adj_matrix_partition_cols())) and returns a transformed value to be reduced.
 * @param init Initial value to be added to the value in each transform-reduced (key, value) pair.
 * @return std::tuple Tuple of rmm::device_uvector<typename GraphView::vertex_type> and
 * rmm::device_uvector<T> (if T is arithmetic scalar) or a tuple of rmm::device_uvector objects (if
 * T is a thrust::tuple type of arithmetic scalar types, one rmm::device_uvector object per scalar
 * type).
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename VertexIterator,
          typename EdgeOp,
          typename T>
auto transform_reduce_by_adj_matrix_col_key_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  VertexIterator adj_matrix_col_key_first,
  EdgeOp e_op,
  T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename std::iterator_traits<VertexIterator>::value_type,
                             typename GraphViewType::vertex_type>::value);

  return detail::transform_reduce_by_adj_matrix_row_col_key_e<false>(
    handle,
    graph_view,
    adj_matrix_row_value_input_first,
    adj_matrix_col_value_input_first,
    adj_matrix_col_key_first,
    e_op,
    init);
}

}  // namespace experimental
}  // namespace cugraph

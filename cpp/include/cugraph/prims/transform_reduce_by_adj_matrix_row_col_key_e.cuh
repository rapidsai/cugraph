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

#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/handle.hpp>

#include <type_traits>

namespace cugraph {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_by_adj_matrix_row_col_key_e_for_all_block_size = 128;

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixRowColKeyInputWrapper,
          typename EdgeOp,
          typename T>
__device__ void update_buffer_element(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu>& matrix_partition,
  typename GraphViewType::vertex_type major,
  typename GraphViewType::vertex_type minor,
  typename GraphViewType::weight_type weight,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  AdjMatrixRowColKeyInputWrapper adj_matrix_row_col_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* key,
  T* value)
{
  using vertex_t = typename GraphViewType::vertex_type;

  auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(major);
  auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
  auto row          = GraphViewType::is_adj_matrix_transposed ? minor : major;
  auto col          = GraphViewType::is_adj_matrix_transposed ? major : minor;
  auto row_offset   = GraphViewType::is_adj_matrix_transposed ? minor_offset : major_offset;
  auto col_offset   = GraphViewType::is_adj_matrix_transposed ? major_offset : minor_offset;

  *key   = adj_matrix_row_col_key_input.get((
    (GraphViewType::is_adj_matrix_transposed != adj_matrix_row_key) ? major_offset : minor_offset));
  *value = evaluate_edge_op<GraphViewType,
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
}

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixRowColKeyInputWrapper,
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
  AdjMatrixRowColKeyInputWrapper adj_matrix_row_col_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  T* values)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_hypersparse_first - matrix_partition.get_major_first());
  auto idx = static_cast<size_t>(tid);

  auto dcs_nzd_vertex_count = *(matrix_partition.get_dcs_nzd_vertex_count());

  while (idx < static_cast<size_t>(dcs_nzd_vertex_count)) {
    auto major =
      *(matrix_partition.get_major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
    auto major_idx =
      major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_idx));
    auto local_offset = matrix_partition.get_local_offset(major_idx);
    for (edge_t i = 0; i < local_degree; ++i) {
      update_buffer_element<adj_matrix_row_key, GraphViewType>(
        matrix_partition,
        major,
        indices[i],
        weights ? (*weights)[i] : weight_t{1.0},
        adj_matrix_row_value_input,
        adj_matrix_col_value_input,
        adj_matrix_row_col_key_input,
        e_op,
        keys + local_offset + i,
        values + local_offset + i);
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixRowColKeyInputWrapper,
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
  AdjMatrixRowColKeyInputWrapper adj_matrix_row_col_key_input,
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
    auto major =
      matrix_partition.get_major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = matrix_partition.get_local_offset(major_offset);
    for (edge_t i = 0; i < local_degree; ++i) {
      update_buffer_element<adj_matrix_row_key, GraphViewType>(
        matrix_partition,
        major,
        indices[i],
        weights ? (*weights)[i] : weight_t{1.0},
        adj_matrix_row_value_input,
        adj_matrix_col_value_input,
        adj_matrix_row_col_key_input,
        e_op,
        keys + local_offset + i,
        values + local_offset + i);
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixRowColKeyInputWrapper,
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
  AdjMatrixRowColKeyInputWrapper adj_matrix_row_col_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  T* values)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(
    transform_reduce_by_adj_matrix_row_col_key_e_for_all_block_size % raft::warp_size() == 0);
  auto const lane_id      = tid % raft::warp_size();
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  size_t idx              = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    auto major =
      matrix_partition.get_major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = matrix_partition.get_local_offset(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      update_buffer_element<adj_matrix_row_key, GraphViewType>(
        matrix_partition,
        major,
        indices[i],
        weights ? (*weights)[i] : weight_t{1.0},
        adj_matrix_row_value_input,
        adj_matrix_col_value_input,
        adj_matrix_row_col_key_input,
        e_op,
        keys + local_offset + i,
        values + local_offset + i);
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixRowColKeyInputWrapper,
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
  AdjMatrixRowColKeyInputWrapper adj_matrix_row_col_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  T* values)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(blockIdx.x);

  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    auto major =
      matrix_partition.get_major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = matrix_partition.get_local_offset(major_offset);
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      update_buffer_element<adj_matrix_row_key, GraphViewType>(
        matrix_partition,
        major,
        indices[i],
        weights ? (*weights)[i] : weight_t{1.0},
        adj_matrix_row_value_input,
        adj_matrix_col_value_input,
        adj_matrix_row_col_key_input,
        e_op,
        keys + local_offset + i,
        values + local_offset + i);
    }

    idx += gridDim.x;
  }
}

// FIXME: better derive value_t from BufferType
template <typename vertex_t, typename value_t, typename BufferType>
std::tuple<rmm::device_uvector<vertex_t>, BufferType> reduce_to_unique_kv_pairs(
  rmm::device_uvector<vertex_t>&& keys, BufferType&& value_buffer, cudaStream_t stream)
{
  thrust::sort_by_key(
    rmm::exec_policy(stream), keys.begin(), keys.end(), get_dataframe_buffer_begin(value_buffer));
  auto num_uniques =
    thrust::count_if(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(keys.size()),
                     [keys = keys.data()] __device__(auto i) {
                       return ((i == 0) || (keys[i] != keys[i - 1])) ? true : false;
                     });

  rmm::device_uvector<vertex_t> unique_keys(num_uniques, stream);
  auto value_for_unique_key_buffer = allocate_dataframe_buffer<value_t>(unique_keys.size(), stream);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        keys.begin(),
                        keys.end(),
                        get_dataframe_buffer_begin(value_buffer),
                        unique_keys.begin(),
                        get_dataframe_buffer_begin(value_for_unique_key_buffer));

  return std::make_tuple(std::move(unique_keys), std::move(value_for_unique_key_buffer));
}

template <bool adj_matrix_row_key,
          typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixRowColKeyInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           decltype(allocate_dataframe_buffer<T>(0, cudaStream_t{nullptr}))>
transform_reduce_by_adj_matrix_row_col_key_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  AdjMatrixRowColKeyInputWrapper adj_matrix_row_col_key_input,
  EdgeOp e_op,
  T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename AdjMatrixRowColKeyInputWrapper::value_type,
                             typename GraphViewType::vertex_type>::value);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  rmm::device_uvector<vertex_t> keys(0, handle.get_stream());
  auto value_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));

    int comm_root_rank = 0;
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      comm_root_rank           = i * row_comm_size + row_comm_rank;
    }

    auto num_edges = matrix_partition.get_number_of_edges();

    rmm::device_uvector<vertex_t> tmp_keys(num_edges, handle.get_stream());
    auto tmp_value_buffer = allocate_dataframe_buffer<T>(tmp_keys.size(), handle.get_stream());

    if (graph_view.get_vertex_partition_size(comm_root_rank) > 0) {
      auto matrix_partition_row_value_input = adj_matrix_row_value_input;
      auto matrix_partition_col_value_input = adj_matrix_col_value_input;
      if constexpr (GraphViewType::is_adj_matrix_transposed) {
        matrix_partition_col_value_input.set_local_adj_matrix_partition_idx(i);
      } else {
        matrix_partition_row_value_input.set_local_adj_matrix_partition_idx(i);
      }
      auto matrix_partition_row_col_key_input = adj_matrix_row_col_key_input;
      if constexpr ((adj_matrix_row_key && !GraphViewType::is_adj_matrix_transposed) ||
                    (!adj_matrix_row_key && GraphViewType::is_adj_matrix_transposed)) {
        matrix_partition_row_col_key_input.set_local_adj_matrix_partition_idx(i);
      }

      auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);
      if (segment_offsets) {
        // FIXME: we may further improve performance by 1) concurrently running kernels on different
        // segments; 2) individually tuning block sizes for different segments; and 3) adding one
        // more segment for very high degree vertices and running segmented reduction
        static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
        if ((*segment_offsets)[1] > 0) {
          raft::grid_1d_block_t update_grid(
            (*segment_offsets)[1],
            detail::transform_reduce_by_adj_matrix_row_col_key_e_for_all_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::for_all_major_for_all_nbr_high_degree<adj_matrix_row_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              matrix_partition,
              matrix_partition.get_major_first(),
              matrix_partition.get_major_first() + (*segment_offsets)[1],
              matrix_partition_row_value_input,
              matrix_partition_col_value_input,
              matrix_partition_row_col_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
        if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
          raft::grid_1d_warp_t update_grid(
            (*segment_offsets)[2] - (*segment_offsets)[1],
            detail::transform_reduce_by_adj_matrix_row_col_key_e_for_all_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::for_all_major_for_all_nbr_mid_degree<adj_matrix_row_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              matrix_partition,
              matrix_partition.get_major_first() + (*segment_offsets)[1],
              matrix_partition.get_major_first() + (*segment_offsets)[2],
              matrix_partition_row_value_input,
              matrix_partition_col_value_input,
              matrix_partition_row_col_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
        if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
          raft::grid_1d_thread_t update_grid(
            (*segment_offsets)[3] - (*segment_offsets)[2],
            detail::transform_reduce_by_adj_matrix_row_col_key_e_for_all_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::for_all_major_for_all_nbr_low_degree<adj_matrix_row_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              matrix_partition,
              matrix_partition.get_major_first() + (*segment_offsets)[2],
              matrix_partition.get_major_first() + (*segment_offsets)[3],
              matrix_partition_row_value_input,
              matrix_partition_col_value_input,
              matrix_partition_row_col_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
        if (matrix_partition.get_dcs_nzd_vertex_count() &&
            (*(matrix_partition.get_dcs_nzd_vertex_count()) > 0)) {
          raft::grid_1d_thread_t update_grid(
            *(matrix_partition.get_dcs_nzd_vertex_count()),
            detail::transform_reduce_by_adj_matrix_row_col_key_e_for_all_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::for_all_major_for_all_nbr_hypersparse<adj_matrix_row_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              matrix_partition,
              matrix_partition.get_major_first() + (*segment_offsets)[3],
              matrix_partition_row_value_input,
              matrix_partition_col_value_input,
              matrix_partition_row_col_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
      } else {
        raft::grid_1d_thread_t update_grid(
          matrix_partition.get_major_size(),
          detail::transform_reduce_by_adj_matrix_row_col_key_e_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_major_for_all_nbr_low_degree<adj_matrix_row_key, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first(),
            matrix_partition.get_major_last(),
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            matrix_partition_row_col_key_input,
            e_op,
            tmp_keys.data(),
            get_dataframe_buffer_begin(tmp_value_buffer));
      }
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
          get_dataframe_buffer_begin(tmp_value_buffer),
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
      resize_dataframe_buffer(value_buffer, keys.size(), handle.get_stream());

      auto execution_policy = handle.get_thrust_policy();
      thrust::copy(execution_policy, tmp_keys.begin(), tmp_keys.end(), keys.begin() + cur_size);
      thrust::copy(execution_policy,
                   get_dataframe_buffer_begin(tmp_value_buffer),
                   get_dataframe_buffer_begin(tmp_value_buffer) + tmp_keys.size(),
                   get_dataframe_buffer_begin(value_buffer) + cur_size);
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
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColValueInputWrapper Type of the wrapper for graph adjacency matrix column input
 * properties.
 * @tparam AdjMatrixRowKeyInputWrapper Type of the wrapper for graph adjacency matrix row keys.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the values in (key, value) pairs.
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
 * @param adj_matrix_row_key_input Device-copyable wrapper used to access row keys(for the rows
 * assigned to this process in multi-GPU). Use either cugraph::row_properties_t::device_view(). Use
 * copy_to_adj_matrix_row to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), properties for the row (i.e. source), and properties for the column  (i.e. destination)
 * and returns a transformed value to be reduced.
 * @param init Initial value to be added to the value in each transform-reduced (key, value) pair.
 * @return std::tuple Tuple of rmm::device_uvector<typename GraphView::vertex_type> and
 * rmm::device_uvector<T> (if T is arithmetic scalar) or a tuple of rmm::device_uvector objects (if
 * T is a thrust::tuple type of arithmetic scalar types, one rmm::device_uvector object per scalar
 * type).
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixRowKeyInputWrapper,
          typename EdgeOp,
          typename T>
auto transform_reduce_by_adj_matrix_row_key_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  AdjMatrixRowKeyInputWrapper adj_matrix_row_key_input,
  EdgeOp e_op,
  T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename AdjMatrixRowKeyInputWrapper::value_type,
                             typename GraphViewType::vertex_type>::value);

  return detail::transform_reduce_by_adj_matrix_row_col_key_e<true>(handle,
                                                                    graph_view,
                                                                    adj_matrix_row_value_input,
                                                                    adj_matrix_col_value_input,
                                                                    adj_matrix_row_key_input,
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
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColValueInputWrapper Type of the wrapper for graph adjacency matrix column input
 * properties.
 * @tparam AdjMatrixColKeyInputWrapper Type of the wrapper for graph adjacency matrix column keys.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the values in (key, value) pairs.
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
 * @param adj_matrix_col_key_input Device-copyable wrapper used to access column keys(for the
 * columns assigned to this process in multi-GPU). Use either
 * cugraph::col_properties_t::device_view(). Use copy_to_adj_matrix_col to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), properties for the row (i.e. source), and properties for the column  (i.e. destination)
 * and returns a transformed value to be reduced.
 * @param init Initial value to be added to the value in each transform-reduced (key, value) pair.
 * @return std::tuple Tuple of rmm::device_uvector<typename GraphView::vertex_type> and
 * rmm::device_uvector<T> (if T is arithmetic scalar) or a tuple of rmm::device_uvector objects (if
 * T is a thrust::tuple type of arithmetic scalar types, one rmm::device_uvector object per scalar
 * type).
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename AdjMatrixColKeyInputWrapper,
          typename EdgeOp,
          typename T>
auto transform_reduce_by_adj_matrix_col_key_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  AdjMatrixColKeyInputWrapper adj_matrix_col_key_input,
  EdgeOp e_op,
  T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename AdjMatrixColKeyInputWrapper::value_type,
                             typename GraphViewType::vertex_type>::value);

  return detail::transform_reduce_by_adj_matrix_row_col_key_e<false>(handle,
                                                                     graph_view,
                                                                     adj_matrix_row_value_input,
                                                                     adj_matrix_col_value_input,
                                                                     adj_matrix_col_key_input,
                                                                     e_op,
                                                                     init);
}

}  // namespace cugraph

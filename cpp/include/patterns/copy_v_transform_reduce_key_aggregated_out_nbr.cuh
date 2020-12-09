/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <experimental/graph_view.hpp>
#include <utilities/error.hpp>

#include <raft/handle.hpp>

#include <cuco/static_map.cuh>

#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr copy_v_transform_reduce_key_aggregated_out_nbr_for_all_block_size = 128;

template <typename GraphViewType, typename KeyIterator>
__global__ void for_all_major_for_all_nbr_low_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  KeyIterator adj_matrix_minor_key_first,
  typename GraphViewType::vertex_type const* major_vertices,
  typename GraphViewType::vertex_type const* minor_keys,
  typename GraphViewType::weight_type const* key_aggregated_edge_weights)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(major_last - major_first)) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    auto major_offset = major_start_offset + idx;
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_offset));
    if (local_degree > 0) {
      auto local_offset    = matrix_partition.get_local_offset(major_offset);
      auto minor_key_first = thrust::make_transform_iterator(indices, [] __device__(auto minor) {
        return *(adj_matrix_minor_key_first +
                 matrix_partition.get_minor_offset_from_minor_nocheck(minor));
      });
      thrust::copy(
        thrust::seq, minor_key_first, minor_key_first + local_degree, minor_keys + local_offset);
      if (weights == nullptr) {
        thrust::sort(
          thrust::seq, minor_keys + local_offset, minor_keys + local_offset + local_degree);
      } else {
        thrust::copy(
          thrust::seq, weights, weights + local_degree, key_aggregated_edge_weights + local_offset);
        thrust::sort_by_key(thrust::seq,
                            minor_keys + local_offset,
                            minor_keys + local_offset + local_degree,
                            key_aggregated_edge_weights + local_offset,
                            key_aggregated_edge_weights + local_offset + local_degree);
      }
      // in-place reduce_by_key
      vertex_t key_idx{0};
      key_aggregated_edge_weights[local_offset + key_idx] =
        weights != nullptr ? weights[0] : weight_t{1.0};
      for (edge_t i = 1; i < local_degree; ++i) {
        if (minor_keys[local_offset + i] == minor_keys[local_offset + key_idx]) {
          key_aggregated_edge_weights[local_offset + key_idx] +=
            weights != nullptr ? weights[i] : weight_t{1.0};
        } else {
          ++key_idx;
          minor_keys[local_offset + key_idx] = minor_keys[local_offset + i];
          key_aggregated_edge_weights[local_offset + key_idx] =
            weights != nullptr ? weights[i] : weight_t{1.0};
        }
      }
      thrust::fill(thrust::seq,
                   major_vertices + local_offset,
                   major_vertices + local_offset + key_idx,
                   matrix_partition.get_major_from_major_offset_nocheck(major_offset));
      thrust::fill(thrust::seq,
                   major_vertices + local_offset + key_idx,
                   major_vertices + local_offset + local_degree,
                   cugraph::experimental::invalid_vertex_id<vertex_t>::value);
    }

    idx += gridDim.x * blockDim.x;
  }
}

}  // namespace detail

/**
 * @brief Iterate over the key-aggregated outgoing edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the outgoing edges
 * part) and thrust::copy() (update vertex properties part, take transform_reduce output as copy
 * input).
 * Unlike copy_v_transform_reduce_out_nbr, this function first aggregates outgoing edges by key to
 * support two level reduction for each vertex.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam KeyIterator Type of the iterator for graph adjacency matrix column key values for
 * aggregation.
 * @tparam ValueIterator Type of the iterator for values in (key, value) pairs.
 * @tparam KeyAggregatedEdgeOp Type of the quinary key-aggregated edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for reduction over the key-aggregated outgoing edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param adj_matrix_col_key_first Iterator pointing to the adjacency matrix column key (for
 * aggregation) for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_key_last` (exclusive) is deduced as @p adj_matrix_col_key_first + @p
 * graph_view.get_number_of_local_adj_matrix_partition_cols().
 * @param map_key_first Iterator pointing to the keys in (key, value) pairs (assigned to this
 * process in multi-GPU, `cugraph::experimental::detail::compute_gpu_id_from_vertex_t` is used to
 * assign keys to processes). (Key, value) pairs may be provided by transform_reduce_by_key_e().
 * @param map_key_last
 * @param map_value_first Iterator pointing to the values in (key, value) pairs (assigned to this
 * process in multi-GPU).
 * @param key_aggregated_e_op Quinary operator takes edge source, key, aggregated edge weight, *(@p
 * adj_matrix_row_value_input_first + i), and value for the key stored in (@p map_key_first, @p
 * map_value_first)
  @p kv_map for the key (where i is in
 * [0, graph_view.get_number_of_local_adj_matrix_partition_rows())) and returns a value to be
 * reduced. weight of 1.0 is assumed if unweighted.
 * @param reduce_op Binary operator takes two input arguments and reduce the two variables to one.
 * @param init Initial value to be added to the reduced @p key_aggregated_e_op return values for
 * each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename KeyIterator,
          typename ValueIterator,
          typename KeyAggregatedEdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_key_aggregated_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  KeyIterator adj_matrix_col_key_first,
  KeyIterator map_key_first,
  KeyIterator map_key_last,
  ValueIterator map_value_first,
  KeyAggregatedEdgeOp key_aggregated_e_op,
  ReduceOp reduce_op,
  T init,
  VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");
  static_assert(std::is_integral<typename std::iterator_traits<KeyIterator>::value_type>::value);

  typename value_t = typname std::iterator_traits<ValueIterator>::value_type;

  double constexpr load_factor = 0.7;

  // 1. build a cuco::static_map object for the k, v pairs.

  auto kv_map_ptr = std::make_unique<cuco::static_map<vertex_t, value_t>>(
    static_cast<size_t>(static_cast<double>(thrust::distance(map_key_first, map_key_last)) /
                        load_factor),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value);
  auto pair_first = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(map_key_first, map_value_first),
    [] __device__(auto val) {
    return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
    });
  kv_map_ptr->insert(pair_first, pair_first + thrust::distance(map_key_first, map_key_last));

  if (GraphViewType::is_multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    rmm::device_uvector<vertex_t> unique_keys(
      graph_view.get_number_of_local_adj_matrix_partition_cols(), handle.get_stream());
    thrust::copy(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      adj_matrix_col_key_first,
      adj_matrix_col_key_first + graph_view.get_number_of_local_adj_matrix_partition_cols(),
      unique_keys.begin());
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 unique_keys.begin(),
                 unique_keys.end());
    auto last = thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                               unique_keys.begin(),
                               unique_keys.end());
    unique_keys.resize(thrust::distance(unique_keys.begin(), last), handle.get_stream());

    auto key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size};
    thrust::sort(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      unique_keys.begin(),
      unique_keys.end(),
      [key_func] __device__(auto lhs, auto rhs) { return key_func(lhs) < key_func(rhs); });

    auto key_first = thrust::make_transform_iterator(
      unique_keys.begin(), [key_func] __device__(auto val) { return key_func(val); });
    rmm::device_uvector<size_t> tx_value_counts(comm_size, handle.get_stream());
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          key_first,
                          key_first + unique_keys.size(),
                          thrust::make_constant_iterator(size_t{1}),
                          thrust::make_discard_iterator(),
                          tx_value_counts.begin());

    rmm::device_uvector<vertex_t> rx_unique_keys(0, handle.get_stream());
    rmm::device_uvector<size_t> rx_value_counts(0, handle.get_stream());

    std::tie(rx_unique_keys, rx_value_counts) =
      cugraph::experimental::detail::shuffle_values(handle, unique_keys.begin(), tx_value_counts);

    rmm::device_uvector<value_t> values_for_unique_keys(rx_unique_keys.size(), handle.get_stream());

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // cuco::static_map currently does not take stream

    kv_map_ptr->find(rx_unique_keys.begin(), rx_unique_keys.end(), values_for_unique_keys.begin());

    rmm::device_uvector<value_t> rx_values_for_unique_keys(0, handle.get_stream());

    std::tie(rx_values_for_unique_keys, std::ignore) =
      cugraph::experimental::detail::shuffle_values(
        handle, values_for_unique_keys.begin(), rx_value_counts);

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // cuco::static_map currently does not take stream

    kv_map_ptr->reset();

    kv_map_ptr = std::make_unique<cuco::static_map<vertex_t, value_t>>(
      static_cast<size_t>(static_cast<double>(unique_keys.size()) / load_factor),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(unique_keys.begin(), rx_values_for_unique_keys.begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });

    kv_map_ptr->insert(pair_first, pair_first + unique_keys.size());
  }

  // 2. aggregate each vertex out-going edges based on keys and transform-reduce.

  auto loop_count = size_t{1};
  if (GraphViewType::is_multi_gpu) {
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    loop_count               = graph_view.is_hypergraph_partitioned()
                   ? graph_view.get_number_of_local_adj_matrix_partitions()
                   : static_cast<size_t>(row_comm_size);
  }

  rmm::device_uvector<vertex_t> major_vertices(0, handle.get_stream());
  auto e_op_result_buffer = allocate_comm_buffer<T>(0, handle.get_stream());
  for (size_t i = 0; i < loop_count; ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(
      graph_view, (GraphViewType::is_multi_gpu && !graph_view.is_hypergraph_partitioned()) ? 0 : i);

    int comm_root_rank = 0;
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      comm_root_rank = graph_view.is_hypergraph_partitioned() ? i * row_comm_size + row_comm_rank
                                                              : col_comm_rank * row_comm_size + i;
    }

    auto num_edges = thrust::transform_reduce(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      graph_view.get_vertex_partition_first(comm_root_rank),
      graph_view.get_vertex_partition_last(comm_root_rank),
      [matrix_partition] __device__(auto row) {
        auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
        return matrix_partition.get_local_degree(row_offset);
      },
      edge_t{0},
      thrust::plus<edge_t>());

    rmm::device_uvector<edge_t> tmp_major_vertices(num_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_minor_keys(tmp_major_vertices.size(), handle.get_stream());
    rmm::device_uvector<weight_t> tmp_key_aggregated_edge_weights(tmp_major_vertices.size(),
                                                                  handle.get_stream());

    if (graph_view.get_vertex_partition_size(comm_root_rank) > 0) {
      raft::grid_1d_thread_t update_grid(
        graph_view.get_vertex_partition_size(comm_root_rank),
        detail::copy_v_transform_reduce_key_aggregated_out_nbr_for_all_block_size,
        handle.get_device_properties().maxGridSize[0]);

      // FIXME: This is highly inefficient for graphs with high-degree vertices. If we renumber
      // vertices to insure that rows within a partition are sorted by their out-degree in
      // decreasing order, we will apply this kernel only to low out-degree vertices.
      detail::for_all_major_for_all_nbr_low_degree<<<update_grid.num_blocks,
                                                     update_grid.block_size,
                                                     0,
                                                     handle.get_stream()>>>(
        matrix_partition,
        graph_view.get_vertex_partition_first(comm_root_rank),
        graph_view.get_vertex_partition_last(comm_root_rank),
        adj_matrix_col_key_first,
        tmp_major_vertices.data(),
        tmp_minor_keys.data(),
        tmp_key_aggregated_edge_weights.data());
    }

    auto triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
      tmp_major_vertices.begin(), tmp_minor_keys.begin(), tmp_key_aggregated_edge_weights.begin()));
    auto last =
      thrust::remove_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        triplet_first,
                        triplet_first + tmp_major_vertices.size(),
                        [] __device__(auto val) {
                          return thrust::get<0>(val) == invalid_vertex_id<vertex_t>::value;
                        });
    tmp_major_vertices.resize(thrust::distance(triplet_first, last), handle.get_stream());
    tmp_minor_keys.resize(tmp_major_vertices.size(), handle.get_stream());
    tmp_key_aggregated_edge_weights.resize(tmp_major_vertices.size(), handle.get_stream());

    if (GraphViewType::is_multi_gpu) {
      auto& sub_comm           = handle.get_subcomm(graph_view.is_hypergraph_partitioned()
                                            ? cugraph::partition_2d::key_naming_t().col_name()
                                            : cugraph::partition_2d::key_naming_t().row_name());
      auto const sub_comm_size = sub_comm.get_size();

      triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_major_vertices.begin(), tmp_minor_keys.begin(), tmp_key_aggregated_edge_weights.begin()));
      auto key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{sub_comm_size};
      thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   triplet_first,
                   triplet_first + tmp_major_vertices.size(),
                   [key_func] __device__(auto lhs, auto rhs) {
                       return key_func(thrust::get<1>(lhs) < key_func(thrust::get<1>(rhs));
                   });
      auto key_first = thrust::make_transform_iterator(
        triplet_first, [key_func] __device__(auto val) { return key_func(thrust::get<1>(val)); });
      rmm::device_uvector<size_t> tx_value_counts(sub_comm.get_size(), handle.get_stream());
      thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                            key_first,
                            key_first + tmp_major_vertices.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            thrust::make_discard_iterator(),
                            tx_value_counts.begin());

      rmm::device_uvector<vertex_t> rx_major_vertices(0, handle.get_stream());
      rmm::device_uvector<vertex_t> rx_minor_keys(0, handle.get_stream());
      rmm::device_uvector<weight_t> rx_key_aggregatd_edge_weights(0, handle.get_stream());

      std::tie(rx_major_vertices, rx_minor_keys, rx_key_aggregated_edge_weights, std::ignore) =
        detail::shuffle_values(sub_comm, triplet_first, tx_value_counts, handle.get_stream());

      tmp_major_vertices              = std::move(rx_major_vertices);
      tmp_minor_keys                  = std::move(rx_minor_keys);
      tmp_key_aggregated_edge_weights = std::move(rx_key_aggregatd_edge_weights);

      CUDA_TRY(
        cudaStreamSynchronize(handle.get_stream()));  // tx_value_counts will become out-of-scope
    }

    auto e_op_result_tmp_buffer =
      allocate_comm_buffer<T>(major_vertices.size(), handle.get_stream());
    auto e_op_result_buffer_first = get_comm_buffer_begin<T>(e_op_result_tmp_buffer);

    triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
      major_vertices.begin(), minor_keys.begin(), key_aggregated_edge_weights.begin()));
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      triplet_first,
                      triplet_first + major_vertices.size(),
                      e_op_result_buffer_first,
                      [] __device__(auto val) {
                        auto major = thrust::get<0>(val);
                        auto key   = thrust::get<1>(val);
                        auto w     = thrust::get<2>(val);
                        return key_aggregated_e_op();
                      });
    minor_keys.resize(0, handle.get_stream());
    key_aggregated_edge_weights.resize(0, handle.get_stream());
    minor_keys.shrink_to_fit(handle.get_stream());
    key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());

    if (GraphViewType::is_multi_gpu) {
      // FIXME: additional optimization is possible if reduce_op is a pure function (and reduce_op
      // can be mapped to ncclRedOp_t).

      device_gatherv();
      e_op_result_tmp_buffer = std::move();
    }
  }

  {
    // FIXME: this runs only on one GPU a subcomm.
    thrust::sort_by_key();
    thrust::reduce_by_key();

    thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 vertex_value_output_first,
                 vertex_value_output_first + graph_view.get_vertex_partition_size(comm_root_rank),
                 init);

    thrust::for_each([] __device__(auto val) {
      auto major                     = ;
      auto val                       = ;
      *(vertex_value_output_first +) = reduce_op(val, *(vertex_value_output_first +));
    });
  }
}

}  // namespace experimental
}  // namespace cugraph

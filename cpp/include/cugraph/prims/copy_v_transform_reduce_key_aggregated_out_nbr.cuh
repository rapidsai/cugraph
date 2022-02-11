/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/detail/decompress_matrix_partition.cuh>
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <cuco/static_map.cuh>
#include <raft/handle.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <type_traits>

namespace cugraph {

namespace detail {

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename AdjMatrixColKeyInputWrapper>
struct minor_to_key_t {
  using vertex_t = typename AdjMatrixColKeyInputWrapper::value_type;
  AdjMatrixColKeyInputWrapper adj_matrix_col_key_input{};
  vertex_t minor_first{};
  __device__ vertex_t operator()(vertex_t minor) const
  {
    return adj_matrix_col_key_input.get(minor - minor_first);
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename edge_t>
struct rebase_offset_t {
  edge_t base_offset{};
  __device__ edge_t operator()(edge_t offset) const { return offset - base_offset; }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t>
struct minor_key_to_col_rank_t {
  compute_gpu_id_from_vertex_t<vertex_t> key_func{};
  int row_comm_size{};
  __device__ int operator()(vertex_t minor_key) const
  {
    return key_func(minor_key) / row_comm_size;
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct triplet_to_col_rank_t {
  compute_gpu_id_from_vertex_t<vertex_t> key_func{};
  int row_comm_size{};
  __device__ int operator()(
    thrust::tuple<vertex_t, vertex_t, weight_t> val /* major, minor key, weight */) const
  {
    return key_func(thrust::get<1>(val)) / row_comm_size;
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t,
          typename weight_t,
          typename AdjMatrixRowValueInputWrapper,
          typename KeyAggregatedEdgeOp,
          typename MatrixPartitionDeviceView,
          typename StaticMapDeviceView>
struct call_key_aggregated_e_op_t {
  AdjMatrixRowValueInputWrapper matrix_partition_row_value_input{};
  KeyAggregatedEdgeOp key_aggregated_e_op{};
  MatrixPartitionDeviceView matrix_partition{};
  StaticMapDeviceView kv_map{};
  __device__ auto operator()(
    thrust::tuple<vertex_t, vertex_t, weight_t> val /* major, minor key, weight */) const
  {
    auto major     = thrust::get<0>(val);
    auto key       = thrust::get<1>(val);
    auto w         = thrust::get<2>(val);
    auto row_value = matrix_partition_row_value_input.get(
      matrix_partition.get_major_offset_from_major_nocheck(major));
    return key_aggregated_e_op(major,
                               key,
                               w,
                               matrix_partition_row_value_input.get(
                                 matrix_partition.get_major_offset_from_major_nocheck(major)),
                               kv_map.find(key)->second.load(cuda::std::memory_order_relaxed));
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t>
struct is_valid_vertex_t {
  __device__ bool operator()(vertex_t v) const { return v != invalid_vertex_id<vertex_t>::value; }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t>
struct invalidate_if_not_first_in_run_t {
  vertex_t const* major_vertices{nullptr};
  __device__ vertex_t operator()(size_t i) const
  {
    return ((i == 0) || (major_vertices[i] != major_vertices[i - 1]))
             ? major_vertices[i]
             : invalid_vertex_id<vertex_t>::value;
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, bool multi_gpu>
struct vertex_local_offset_t {
  vertex_partition_device_view_t<vertex_t, multi_gpu> vertex_partition{};
  __device__ vertex_t operator()(vertex_t v) const
  {
    return vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v);
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename ReduceOp, typename T>
struct reduce_with_init_t {
  ReduceOp reduce_op{};
  T init{};
  __device__ T operator()(T val) const { return reduce_op(val, init); }
};

}  // namespace detail

/**
 * @brief Iterate over every vertex's key-aggregated outgoing edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the outgoing edges
 * part) and thrust::copy() (update vertex properties part, take transform_reduce output as copy
 * input).
 * Unlike copy_v_transform_reduce_out_nbr, this function first aggregates outgoing edges by key to
 * support two level reduction for every vertex.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColKeyInputWrapper Type of the wrapper for graph adjacency matrix column keys.
 * @tparam VertexIterator Type of the iterator for graph adjacency matrix column key values for
 * aggregation (key type should coincide with vertex type).
 * @tparam ValueIterator Type of the iterator for values in (key, value) pairs.
 * @tparam KeyAggregatedEdgeOp Type of the quinary key-aggregated edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for reduction over the key-aggregated outgoing edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input Device-copyable wrapper used to access row input properties
 * (for the rows assigned to this process in multi-GPU). Use either
 * cugraph::row_properties_t::device_view() (if @p e_op needs to access row properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access row properties). Use
 * copy_to_adj_matrix_row to fill the wrapper.
 * @param adj_matrix_col_key_input Device-copyable wrapper used to access column keys (for the
 * columns assigned to this process in multi-GPU). Use either
 * cugraph::col_properties_t::device_view(). Use copy_to_adj_matrix_col to fill the wrapper.
 * @param map_unique_key_first Iterator pointing to the first (inclusive) key in (key, value) pairs
 * (assigned to this process in multi-GPU, `cugraph::detail::compute_gpu_id_from_vertex_t` is used
 * to map keys to processes). (Key, value) pairs may be provided by
 * transform_reduce_by_adj_matrix_row_key_e() or transform_reduce_by_adj_matrix_col_key_e().
 * @param map_unique_key_last Iterator pointing to the last (exclusive) key in (key, value) pairs
 * (assigned to this process in multi-GPU).
 * @param map_value_first Iterator pointing to the first (inclusive) value in (key, value) pairs
 * (assigned to this process in multi-GPU). `map_value_last` (exclusive) is deduced as @p
 * map_value_first + thrust::distance(@p map_unique_key_first, @p map_unique_key_last).
 * @param key_aggregated_e_op Quinary operator takes edge source, key, aggregated edge weight, *(@p
 * adj_matrix_row_value_input_first + i), and value for the key stored in the input (key, value)
 * pairs provided by @p map_unique_key_first, @p map_unique_key_last, and @p map_value_first
 * (aggregated over the entire set of processes in multi-GPU).
 * @param reduce_op Binary operator takes two input arguments and reduce the two variables to one.
 * @param init Initial value to be added to the reduced @p reduce_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColKeyInputWrapper,
          typename VertexIterator,
          typename ValueIterator,
          typename KeyAggregatedEdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_key_aggregated_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColKeyInputWrapper adj_matrix_col_key_input,
  VertexIterator map_unique_key_first,
  VertexIterator map_unique_key_last,
  ValueIterator map_value_first,
  KeyAggregatedEdgeOp key_aggregated_e_op,
  ReduceOp reduce_op,
  T init,
  VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");
  static_assert(std::is_same<typename std::iterator_traits<VertexIterator>::value_type,
                             typename GraphViewType::vertex_type>::value);
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using value_t  = typename std::iterator_traits<ValueIterator>::value_type;

  double constexpr load_factor = 0.7;

  // 1. build a cuco::static_map object for the k, v pairs.

#if 1
handle.sync_stream();
auto time0 = std::chrono::steady_clock::now();
#endif
  auto poly_alloc = rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
  auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(poly_alloc, handle.get_stream());
  auto kv_map_ptr     = std::make_unique<
    cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>>(
    size_t{0},
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value,
    stream_adapter,
    handle.get_stream());
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm               = handle.get_comms();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();

#if 1  // FIXME: compare performance and memory saving in large scale
    auto const comm_size = comm.get_size();
    auto& col_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

    rmm::device_uvector<vertex_t> map_keys(
      adj_matrix_col_key_input.number_of_keys()
        ? *(adj_matrix_col_key_input.number_of_keys())
        : graph_view.get_number_of_local_adj_matrix_partition_cols(),
      handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 adj_matrix_col_key_input.value_data(),
                 adj_matrix_col_key_input.value_data() + map_keys.size(),
                 map_keys.begin());
    thrust::sort(handle.get_thrust_policy(), map_keys.begin(), map_keys.end());
    map_keys.resize(thrust::distance(
                      map_keys.begin(),
                      thrust::unique(handle.get_thrust_policy(), map_keys.begin(), map_keys.end())),
                    handle.get_stream());

    std::tie(map_keys, std::ignore) = groupby_gpuid_and_shuffle_values(
      col_comm,
      map_keys.begin(),
      map_keys.end(),
      detail::minor_key_to_col_rank_t<vertex_t>{
        detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}, row_comm_size},
      handle.get_stream());

    thrust::sort(handle.get_thrust_policy(), map_keys.begin(), map_keys.end());
    map_keys.resize(thrust::distance(
                      map_keys.begin(),
                      thrust::unique(handle.get_thrust_policy(), map_keys.begin(), map_keys.end())),
                    handle.get_stream());

    auto map_value_buffer = cugraph::collect_values_for_keys(
      handle.get_comms(),
      map_unique_key_first,
      map_unique_key_last,
      map_value_first,
      map_keys.begin(),
      map_keys.end(),
      cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size},
      handle.get_stream());
#else
    auto map_counts = host_scalar_allgather(
      row_comm,
      static_cast<size_t>(thrust::distance(map_unique_key_first, map_unique_key_last)),
      handle.get_stream());
    std::vector<size_t> map_displacements(row_comm_size, size_t{0});
    std::partial_sum(map_counts.begin(), map_counts.end() - 1, map_displacements.begin() + 1);
    rmm::device_uvector<vertex_t> map_keys(map_displacements.back() + map_counts.back(),
                                           handle.get_stream());
    auto map_value_buffer =
      allocate_dataframe_buffer<value_t>(map_keys.size(), handle.get_stream());
    for (int i = 0; i < row_comm_size; ++i) {
      device_bcast(row_comm,
                   map_unique_key_first,
                   map_keys.begin() + map_displacements[i],
                   map_counts[i],
                   i,
                   handle.get_stream());
      device_bcast(row_comm,
                   map_value_first,
                   get_dataframe_buffer_begin(map_value_buffer) + map_displacements[i],
                   map_counts[i],
                   i,
                   handle.get_stream());
    }
#endif

    kv_map_ptr.reset();

    kv_map_ptr = std::make_unique<
      cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>>(
      // cuco::static_map requires at least one empty slot
      std::max(
        static_cast<size_t>(static_cast<double>(map_keys.size()) / load_factor),
        static_cast<size_t>(thrust::distance(map_unique_key_first, map_unique_key_last)) + 1),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      stream_adapter,
      handle.get_stream());

    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(map_keys.begin(), get_dataframe_buffer_begin(map_value_buffer)));
    kv_map_ptr->insert(pair_first,
                       pair_first + map_keys.size(),
                       cuco::detail::MurmurHash3_32<vertex_t>{},
                       thrust::equal_to<vertex_t>{},
                       handle.get_stream());
  } else {
    kv_map_ptr.reset();

    kv_map_ptr = std::make_unique<
      cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>>(
      // cuco::static_map requires at least one empty slot
      std::max(
        static_cast<size_t>(
          static_cast<double>(thrust::distance(map_unique_key_first, map_unique_key_last)) /
          load_factor),
        static_cast<size_t>(thrust::distance(map_unique_key_first, map_unique_key_last)) + 1),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      stream_adapter,
      handle.get_stream());

    auto pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(map_unique_key_first, map_value_first));
    kv_map_ptr->insert(pair_first,
                       pair_first + thrust::distance(map_unique_key_first, map_unique_key_last),
                       cuco::detail::MurmurHash3_32<vertex_t>{},
                       thrust::equal_to<vertex_t>{},
                       handle.get_stream());
  }

  // 2. aggregate each vertex out-going edges based on keys and transform-reduce.

#if 1
handle.sync_stream();
auto time1 = std::chrono::steady_clock::now();
#endif
  rmm::device_uvector<vertex_t> major_vertices(0, handle.get_stream());
  auto e_op_result_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
#if 1
handle.sync_stream();
auto loop_time0 = std::chrono::steady_clock::now();
#endif
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));

    rmm::device_uvector<vertex_t> tmp_major_vertices(matrix_partition.get_number_of_edges(),
                                                     handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_minor_keys(tmp_major_vertices.size(), handle.get_stream());
    rmm::device_uvector<weight_t> tmp_key_aggregated_edge_weights(tmp_major_vertices.size(),
                                                                  handle.get_stream());

#if 1
handle.sync_stream();
auto loop_time1 = std::chrono::steady_clock::now();
#endif
    if (matrix_partition.get_number_of_edges() > 0) {
      auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);

      detail::decompress_matrix_partition_to_fill_edgelist_majors(
        handle, matrix_partition, tmp_major_vertices.data(), segment_offsets);

      auto minor_key_first = thrust::make_transform_iterator(
        matrix_partition.get_indices(),
        detail::minor_to_key_t<AdjMatrixColKeyInputWrapper>{adj_matrix_col_key_input,
                                                            matrix_partition.get_minor_first()});

      // to limit memory footprint ((1 << 20) is a tuning parameter)
      auto approx_edges_to_sort_per_iteration =
        static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
      auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_edge_chunks(
        handle,
        matrix_partition.get_offsets(),
        matrix_partition.get_dcs_nzd_vertices()
          ? (*segment_offsets)[detail::num_sparse_segments_per_vertex_partition] +
              *(matrix_partition.get_dcs_nzd_vertex_count())
          : matrix_partition.get_major_size(),
        matrix_partition.get_number_of_edges(),
        approx_edges_to_sort_per_iteration);
      auto num_chunks = h_vertex_offsets.size() - 1;

      size_t max_chunk_size{0};
      for (size_t j = 0; j < num_chunks; ++j) {
        max_chunk_size =
          std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[j + 1] - h_edge_offsets[j]));
      }
      rmm::device_uvector<vertex_t> unreduced_major_vertices(max_chunk_size, handle.get_stream());
      rmm::device_uvector<vertex_t> unreduced_minor_keys(unreduced_major_vertices.size(),
                                                         handle.get_stream());
      rmm::device_uvector<weight_t> unreduced_key_aggregated_edge_weights(
        graph_view.is_weighted() ? unreduced_major_vertices.size() : size_t{0},
        handle.get_stream());
      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

      size_t reduced_size{0};
      for (size_t j = 0; j < num_chunks; ++j) {
        thrust::copy(handle.get_thrust_policy(),
                     minor_key_first + h_edge_offsets[j],
                     minor_key_first + h_edge_offsets[j + 1],
                     tmp_minor_keys.begin() + h_edge_offsets[j]);

        size_t tmp_storage_bytes{0};
        auto offset_first =
          thrust::make_transform_iterator(matrix_partition.get_offsets() + h_vertex_offsets[j],
                                          detail::rebase_offset_t<edge_t>{h_edge_offsets[j]});
        if (graph_view.is_weighted()) {
          cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                              tmp_storage_bytes,
                                              tmp_minor_keys.begin() + h_edge_offsets[j],
                                              unreduced_minor_keys.begin(),
                                              *(matrix_partition.get_weights()) + h_edge_offsets[j],
                                              unreduced_key_aggregated_edge_weights.begin(),
                                              h_edge_offsets[j + 1] - h_edge_offsets[j],
                                              h_vertex_offsets[j + 1] - h_vertex_offsets[j],
                                              offset_first,
                                              offset_first + 1,
                                              handle.get_stream());
        } else {
          cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                             tmp_storage_bytes,
                                             tmp_minor_keys.begin() + h_edge_offsets[j],
                                             unreduced_minor_keys.begin(),
                                             h_edge_offsets[j + 1] - h_edge_offsets[j],
                                             h_vertex_offsets[j + 1] - h_vertex_offsets[j],
                                             offset_first,
                                             offset_first + 1,
                                             handle.get_stream());
        }
        if (tmp_storage_bytes > d_tmp_storage.size()) {
          d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
        }
        if (graph_view.is_weighted()) {
          cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                              tmp_storage_bytes,
                                              tmp_minor_keys.begin() + h_edge_offsets[j],
                                              unreduced_minor_keys.begin(),
                                              *(matrix_partition.get_weights()) + h_edge_offsets[j],
                                              unreduced_key_aggregated_edge_weights.begin(),
                                              h_edge_offsets[j + 1] - h_edge_offsets[j],
                                              h_vertex_offsets[j + 1] - h_vertex_offsets[j],
                                              offset_first,
                                              offset_first + 1,
                                              handle.get_stream());
        } else {
          cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                             tmp_storage_bytes,
                                             tmp_minor_keys.begin() + h_edge_offsets[j],
                                             unreduced_minor_keys.begin(),
                                             h_edge_offsets[j + 1] - h_edge_offsets[j],
                                             h_vertex_offsets[j + 1] - h_vertex_offsets[j],
                                             offset_first,
                                             offset_first + 1,
                                             handle.get_stream());
        }

        thrust::copy(handle.get_thrust_policy(),
                     tmp_major_vertices.begin() + h_edge_offsets[j],
                     tmp_major_vertices.begin() + h_edge_offsets[j + 1],
                     unreduced_major_vertices.begin());
        auto input_key_first = thrust::make_zip_iterator(
          thrust::make_tuple(unreduced_major_vertices.begin(), unreduced_minor_keys.begin()));
        auto output_key_first = thrust::make_zip_iterator(
          thrust::make_tuple(tmp_major_vertices.begin(), tmp_minor_keys.begin()));
        if (graph_view.is_weighted()) {
          reduced_size +=
            thrust::distance(output_key_first + reduced_size,
                             thrust::get<0>(thrust::reduce_by_key(
                               handle.get_thrust_policy(),
                               input_key_first,
                               input_key_first + (h_edge_offsets[j + 1] - h_edge_offsets[j]),
                               unreduced_key_aggregated_edge_weights.begin(),
                               output_key_first + reduced_size,
                               tmp_key_aggregated_edge_weights.begin() + reduced_size)));
        } else {
          reduced_size +=
            thrust::distance(output_key_first + reduced_size,
                             thrust::get<0>(thrust::reduce_by_key(
                               handle.get_thrust_policy(),
                               input_key_first,
                               input_key_first + (h_edge_offsets[j + 1] - h_edge_offsets[j]),
                               thrust::make_constant_iterator(weight_t{1.0}),
                               output_key_first + reduced_size,
                               tmp_key_aggregated_edge_weights.begin() + reduced_size)));
        }
      }
      tmp_major_vertices.resize(reduced_size, handle.get_stream());
      tmp_minor_keys.resize(tmp_major_vertices.size(), handle.get_stream());
      tmp_key_aggregated_edge_weights.resize(tmp_major_vertices.size(), handle.get_stream());
    }
#if 1
handle.sync_stream();
auto loop_time2 = std::chrono::steady_clock::now();
#endif
    tmp_minor_keys.shrink_to_fit(handle.get_stream());
    tmp_key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());
    tmp_major_vertices.shrink_to_fit(handle.get_stream());

#if 1
handle.sync_stream();
auto loop_time3 = std::chrono::steady_clock::now();
#endif
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_size = row_comm.get_size();

      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_size = col_comm.get_size();

      auto triplet_first =
        thrust::make_zip_iterator(thrust::make_tuple(tmp_major_vertices.begin(),
                                                     tmp_minor_keys.begin(),
                                                     tmp_key_aggregated_edge_weights.begin()));
handle.sync_stream(); auto tmp_time0 = std::chrono::steady_clock::now();  // FIXME: delete
      rmm::device_uvector<vertex_t> rx_major_vertices(0, handle.get_stream());
      rmm::device_uvector<vertex_t> rx_minor_keys(0, handle.get_stream());
      rmm::device_uvector<weight_t> rx_key_aggregated_edge_weights(0, handle.get_stream());
      std::forward_as_tuple(
        std::tie(rx_major_vertices, rx_minor_keys, rx_key_aggregated_edge_weights), std::ignore) =
        groupby_gpuid_and_shuffle_values(
          col_comm,
          triplet_first,
          triplet_first + tmp_major_vertices.size(),
          detail::triplet_to_col_rank_t<vertex_t, weight_t>{
            detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}, row_comm_size},
          handle.get_stream());
handle.sync_stream(); auto tmp_time1 = std::chrono::steady_clock::now();  // FIXME: delete

      auto pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(rx_major_vertices.begin(), rx_minor_keys.begin()));
      thrust::sort_by_key(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + rx_major_vertices.size(),
                          rx_key_aggregated_edge_weights.begin());
handle.sync_stream(); auto tmp_time2 = std::chrono::steady_clock::now();  // FIXME: delete
      tmp_major_vertices.resize(rx_major_vertices.size(), handle.get_stream());
      tmp_minor_keys.resize(tmp_major_vertices.size(), handle.get_stream());
      tmp_key_aggregated_edge_weights.resize(tmp_major_vertices.size(), handle.get_stream());
handle.sync_stream(); auto tmp_time3 = std::chrono::steady_clock::now();  // FIXME: delete
      auto pair_it = thrust::reduce_by_key(handle.get_thrust_policy(),
                                           pair_first,
                                           pair_first + rx_major_vertices.size(),
                                           rx_key_aggregated_edge_weights.begin(),
                                           thrust::make_zip_iterator(thrust::make_tuple(
                                             tmp_major_vertices.begin(), tmp_minor_keys.begin())),
                                           tmp_key_aggregated_edge_weights.begin());
handle.sync_stream(); auto tmp_time4 = std::chrono::steady_clock::now();  // FIXME: delete
      tmp_major_vertices.resize(
        thrust::distance(tmp_key_aggregated_edge_weights.begin(), thrust::get<1>(pair_it)),
        handle.get_stream());
      tmp_minor_keys.resize(tmp_major_vertices.size(), handle.get_stream());
      tmp_key_aggregated_edge_weights.resize(tmp_major_vertices.size(), handle.get_stream());
      tmp_major_vertices.shrink_to_fit(handle.get_stream());
      tmp_minor_keys.shrink_to_fit(handle.get_stream());
      tmp_key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());
#if 1  // FIXME: delete
handle.sync_stream(); auto tmp_time5 = std::chrono::steady_clock::now();
std::chrono::duration<double> elapsed_total = tmp_time5 - tmp_time0;
std::chrono::duration<double> elapsed0 = tmp_time1 - tmp_time0;
std::chrono::duration<double> elapsed1 = tmp_time2 - tmp_time1;
std::chrono::duration<double> elapsed2 = tmp_time3 - tmp_time2;
std::chrono::duration<double> elapsed3 = tmp_time4 - tmp_time3;
std::chrono::duration<double> elapsed4 = tmp_time5 - tmp_time4;
std::cout << "\t\ttmp took " << elapsed_total.count() * 1e3 << " breakdown=(" << elapsed0.count() * 1e3 << "," << elapsed1.count() * 1e3 << "," << elapsed2.count() * 1e3 << "," << elapsed3.count() * 1e3 << "," << elapsed4.count() * 1e3 << ") ms." << std::endl;
#endif
    }
#if 1
handle.sync_stream();
auto loop_time4 = std::chrono::steady_clock::now();
#endif

    auto tmp_e_op_result_buffer =
      allocate_dataframe_buffer<T>(tmp_major_vertices.size(), handle.get_stream());

    auto matrix_partition_row_value_input = adj_matrix_row_value_input;
    matrix_partition_row_value_input.set_local_adj_matrix_partition_idx(i);

    auto triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
      tmp_major_vertices.begin(), tmp_minor_keys.begin(), tmp_key_aggregated_edge_weights.begin()));
#if 1
handle.sync_stream();
auto loop_time5 = std::chrono::steady_clock::now();
#endif
    thrust::transform(handle.get_thrust_policy(),
                      triplet_first,
                      triplet_first + tmp_major_vertices.size(),
                      get_dataframe_buffer_begin(tmp_e_op_result_buffer),
                      detail::call_key_aggregated_e_op_t<vertex_t,
                                                         weight_t,
                                                         AdjMatrixRowValueInputWrapper,
                                                         KeyAggregatedEdgeOp,
                                                         decltype(matrix_partition),
                                                         decltype(kv_map_ptr->get_device_view())>{
                        matrix_partition_row_value_input,
                        key_aggregated_e_op,
                        matrix_partition,
                        kv_map_ptr->get_device_view()});
#if 1
handle.sync_stream();
auto loop_time6 = std::chrono::steady_clock::now();
#endif
    tmp_minor_keys.resize(0, handle.get_stream());
    tmp_key_aggregated_edge_weights.resize(0, handle.get_stream());
    tmp_minor_keys.shrink_to_fit(handle.get_stream());
    tmp_key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());

    {
      auto num_uniques =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(tmp_major_vertices.size()),
                         detail::is_first_in_run_t<vertex_t>{tmp_major_vertices.data()});
      rmm::device_uvector<vertex_t> unique_majors(num_uniques, handle.get_stream());
      auto reduced_e_op_result_buffer =
        allocate_dataframe_buffer<T>(unique_majors.size(), handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            tmp_major_vertices.begin(),
                            tmp_major_vertices.end(),
                            get_dataframe_buffer_begin(tmp_e_op_result_buffer),
                            unique_majors.begin(),
                            get_dataframe_buffer_begin(reduced_e_op_result_buffer),
                            thrust::equal_to<vertex_t>{},
                            reduce_op);
      tmp_major_vertices     = std::move(unique_majors);
      tmp_e_op_result_buffer = std::move(reduced_e_op_result_buffer);
    }
#if 1
handle.sync_stream();
auto loop_time7 = std::chrono::steady_clock::now();
#endif

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      // FIXME: additional optimization is possible if reduce_op is a pure function (and reduce_op
      // can be mapped to ncclRedOp_t).

      auto rx_sizes =
        host_scalar_gather(col_comm, tmp_major_vertices.size(), i, handle.get_stream());
      std::vector<size_t> rx_displs{};
      rmm::device_uvector<vertex_t> rx_major_vertices(0, handle.get_stream());
      if (static_cast<size_t>(col_comm_rank) == i) {
        rx_displs.assign(col_comm_size, size_t{0});
        std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);
        rx_major_vertices.resize(rx_displs.back() + rx_sizes.back(), handle.get_stream());
      }
      auto rx_tmp_e_op_result_buffer =
        allocate_dataframe_buffer<T>(rx_major_vertices.size(), handle.get_stream());

      device_gatherv(col_comm,
                     tmp_major_vertices.data(),
                     rx_major_vertices.data(),
                     tmp_major_vertices.size(),
                     rx_sizes,
                     rx_displs,
                     i,
                     handle.get_stream());
      device_gatherv(col_comm,
                     get_dataframe_buffer_begin(tmp_e_op_result_buffer),
                     get_dataframe_buffer_begin(rx_tmp_e_op_result_buffer),
                     tmp_major_vertices.size(),
                     rx_sizes,
                     rx_displs,
                     i,
                     handle.get_stream());

      if (static_cast<size_t>(col_comm_rank) == i) {
        major_vertices     = std::move(rx_major_vertices);
        e_op_result_buffer = std::move(rx_tmp_e_op_result_buffer);
      }
    } else {
      major_vertices     = std::move(tmp_major_vertices);
      e_op_result_buffer = std::move(tmp_e_op_result_buffer);
    }
#if 1
handle.sync_stream();
auto loop_time8 = std::chrono::steady_clock::now();
std::chrono::duration<double> elapsed_total = loop_time8 - loop_time0;
std::chrono::duration<double> elapsed0 = loop_time1 - loop_time0;
std::chrono::duration<double> elapsed1 = loop_time2 - loop_time1;
std::chrono::duration<double> elapsed2 = loop_time3 - loop_time2;
std::chrono::duration<double> elapsed3 = loop_time4 - loop_time3;
std::chrono::duration<double> elapsed4 = loop_time5 - loop_time4;
std::chrono::duration<double> elapsed5 = loop_time6 - loop_time5;
std::chrono::duration<double> elapsed6 = loop_time7 - loop_time6;
std::chrono::duration<double> elapsed7 = loop_time8 - loop_time7;
std::cout << "\tloop i=" << i << " took " << elapsed_total.count() * 1e3 << " breakdown=(" << elapsed0.count() * 1e3 << "," << elapsed1.count() * 1e3 << "," << elapsed2.count() * 1e3 << "," << elapsed3.count() * 1e3 << "," << elapsed4.count() * 1e3 << "," << elapsed5.count() * 1e3 << "," << elapsed6.count() * 1e3 << "," << elapsed7.count() * 1e3 << ") ms." << std::endl;
#endif
  }

#if 1
handle.sync_stream();
auto time2 = std::chrono::steady_clock::now();
#endif
  if constexpr (GraphViewType::is_multi_gpu) {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        major_vertices.begin(),
                        major_vertices.end(),
                        get_dataframe_buffer_begin(e_op_result_buffer));
    auto num_uniques = thrust::count_if(handle.get_thrust_policy(),
                                        thrust::make_counting_iterator(size_t{0}),
                                        thrust::make_counting_iterator(major_vertices.size()),
                                        detail::is_first_in_run_t<vertex_t>{major_vertices.data()});
    rmm::device_uvector<vertex_t> unique_majors(num_uniques, handle.get_stream());
    auto reduced_e_op_result_buffer =
      allocate_dataframe_buffer<T>(unique_majors.size(), handle.get_stream());
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          major_vertices.begin(),
                          major_vertices.end(),
                          get_dataframe_buffer_begin(e_op_result_buffer),
                          unique_majors.begin(),
                          get_dataframe_buffer_begin(reduced_e_op_result_buffer),
                          thrust::equal_to<vertex_t>{},
                          reduce_op);
    major_vertices     = std::move(unique_majors);
    e_op_result_buffer = std::move(reduced_e_op_result_buffer);
  }

  thrust::fill(handle.get_thrust_policy(),
               vertex_value_output_first,
               vertex_value_output_first + graph_view.get_number_of_local_vertices(),
               T{});

  thrust::scatter(handle.get_thrust_policy(),
                  get_dataframe_buffer_begin(e_op_result_buffer),
                  get_dataframe_buffer_end(e_op_result_buffer),
                  thrust::make_transform_iterator(
                    major_vertices.begin(),
                    detail::vertex_local_offset_t<vertex_t, GraphViewType::is_multi_gpu>{
                      graph_view.get_vertex_partition_view()}),
                  vertex_value_output_first);

  thrust::transform(handle.get_thrust_policy(),
                    vertex_value_output_first,
                    vertex_value_output_first + graph_view.get_number_of_local_vertices(),
                    vertex_value_output_first,
                    detail::reduce_with_init_t<ReduceOp, T>{reduce_op, init});
#if 1
handle.sync_stream();
auto time3 = std::chrono::steady_clock::now();
std::chrono::duration<double> elapsed_total = time3 - time0;
std::chrono::duration<double> elapsed0 = time1 - time0;
std::chrono::duration<double> elapsed1 = time2 - time1;
std::chrono::duration<double> elapsed2 = time3 - time2;
std::cout << "copy_v_transform_reduce_key_aggregated_out_nbr took " << elapsed_total.count() * 1e3 << " breakdown=(" << elapsed0.count() * 1e3 << "," << elapsed1.count() * 1e3 << "," << elapsed2.count() * 1e3 << ") ms." << std::endl;
#endif
}

}  // namespace cugraph

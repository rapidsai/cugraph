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

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph_view.hpp>
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
template <typename EdgePartitionDstKeyInputWrapper>
struct minor_to_key_t {
  using vertex_t = typename EdgePartitionDstKeyInputWrapper::value_type;
  EdgePartitionDstKeyInputWrapper edge_partition_dst_key_input{};
  vertex_t minor_range_first{};
  __device__ vertex_t operator()(vertex_t minor) const
  {
    return edge_partition_dst_key_input.get(minor - minor_range_first);
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename edge_t>
struct rebase_offset_t {
  edge_t base_offset{};
  __device__ edge_t operator()(edge_t offset) const { return offset - base_offset; }
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
template <typename vertex_t>
struct pair_to_binary_partition_id_t {
  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> pair) const
  {
    return static_cast<int>(thrust::get<0>(pair) % 2);
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t,
          typename weight_t,
          typename EdgePartitionSrcValueInputWrapper,
          typename KeyAggregatedEdgeOp,
          typename EdgePartitionDeviceView,
          typename StaticMapDeviceView>
struct call_key_aggregated_e_op_t {
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  KeyAggregatedEdgeOp key_aggregated_e_op{};
  EdgePartitionDeviceView edge_partition{};
  StaticMapDeviceView kv_map{};
  __device__ auto operator()(
    thrust::tuple<vertex_t, vertex_t, weight_t> val /* major, minor key, weight */) const
  {
    auto major = thrust::get<0>(val);
    auto key   = thrust::get<1>(val);
    auto w     = thrust::get<2>(val);
    return key_aggregated_e_op(
      major,
      key,
      w,
      edge_partition_src_value_input.get(edge_partition.major_offset_from_major_nocheck(major)),
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
  vertex_t const* majors{nullptr};
  __device__ vertex_t operator()(size_t i) const
  {
    return ((i == 0) || (majors[i] != majors[i - 1])) ? majors[i]
                                                      : invalid_vertex_id<vertex_t>::value;
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, bool multi_gpu>
struct vertex_local_offset_t {
  vertex_partition_device_view_t<vertex_t, multi_gpu> vertex_partition{};
  __device__ vertex_t operator()(vertex_t v) const
  {
    return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
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
 * @tparam EdgePartitionSrcValueInputWrapper Type of the wrapper for edge partition source property
 * values.
 * @tparam EdgePartitionDstKeyInputWrapper Type of the wrapper for edge partition destination key
 * values.
 * @tparam VertexIterator Type of the iterator for keys in (key, value) pairs (key type should
 * coincide with vertex type).
 * @tparam ValueIterator Type of the iterator for values in (key, value) pairs.
 * @tparam KeyAggregatedEdgeOp Type of the quinary key-aggregated edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for reduction over the key-aggregated outgoing edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_partition_src_value_input Device-copyable wrapper used to access source input
 * property values (for the edge sources assigned to this process in multi-GPU). Use either
 * cugraph::edge_partition_src_property_t::device_view() (if @p e_op needs to access source property
 * values) or cugraph::dummy_property_t::device_view() (if @p e_op does not access source property
 * values). Use update_edge_partition_src_property to fill the wrapper.
 * @param edge_partition_dst_key_input Device-copyable wrapper used to access destination input key
 * values (for the edge destinations assigned to this process in multi-GPU). Use
 * cugraph::edge_partition_dst_property_t::device_view(). Use update_edge_partition_dst_property to
 * fill the wrapper.
 * @param map_unique_key_first Iterator pointing to the first (inclusive) key in (key, value) pairs
 * (assigned to this process in multi-GPU, `cugraph::detail::compute_gpu_id_from_vertex_t` is used
 * to map keys to processes). (Key, value) pairs may be provided by
 * transform_reduce_by_src_key_e() or transform_reduce_by_dst_key_e().
 * @param map_unique_key_last Iterator pointing to the last (exclusive) key in (key, value) pairs
 * (assigned to this process in multi-GPU).
 * @param map_value_first Iterator pointing to the first (inclusive) value in (key, value) pairs
 * (assigned to this process in multi-GPU). `map_value_last` (exclusive) is deduced as @p
 * map_value_first + thrust::distance(@p map_unique_key_first, @p map_unique_key_last).
 * @param key_aggregated_e_op Quinary operator takes edge source, key, aggregated edge weight, *(@p
 * edge_partition_src_value_input_first + i), and value for the key stored in the input (key, value)
 * pairs provided by @p map_unique_key_first, @p map_unique_key_last, and @p map_value_first
 * (aggregated over the entire set of processes in multi-GPU).
 * @param reduce_op Binary operator takes two input arguments and reduce the two variables to one.
 * @param init Initial value to be added to the reduced @p reduce_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 */
template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstKeyInputWrapper,
          typename VertexIterator,
          typename ValueIterator,
          typename KeyAggregatedEdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_key_aggregated_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstKeyInputWrapper edge_partition_dst_key_input,
  VertexIterator map_unique_key_first,
  VertexIterator map_unique_key_last,
  ValueIterator map_value_first,
  KeyAggregatedEdgeOp key_aggregated_e_op,
  ReduceOp reduce_op,
  T init,
  VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");
  static_assert(std::is_same<typename std::iterator_traits<VertexIterator>::value_type,
                             typename GraphViewType::vertex_type>::value);
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using value_t  = typename std::iterator_traits<ValueIterator>::value_type;

  double constexpr load_factor = 0.7;

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto element_size     = sizeof(vertex_t) * 2 + sizeof(weight_t);
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach
  [[maybe_unused]] auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  // 1. build a cuco::static_map object for the k, v pairs.

  auto poly_alloc = rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
  auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(poly_alloc, handle.get_stream());
  auto kv_map =
    cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>(
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
  kv_map.insert(pair_first,
                pair_first + thrust::distance(map_unique_key_first, map_unique_key_last),
                cuco::detail::MurmurHash3_32<vertex_t>{},
                thrust::equal_to<vertex_t>{},
                handle.get_stream());

  // 2. aggregate each vertex out-going edges based on keys and transform-reduce.

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  auto e_op_result_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    rmm::device_uvector<vertex_t> tmp_majors(edge_partition.number_of_edges(), handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_minor_keys(tmp_majors.size(), handle.get_stream());
    rmm::device_uvector<weight_t> tmp_key_aggregated_edge_weights(tmp_majors.size(),
                                                                  handle.get_stream());

    if (edge_partition.number_of_edges() > 0) {
      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);

      detail::decompress_edge_partition_to_fill_edgelist_majors(
        handle, edge_partition, tmp_majors.data(), segment_offsets);

      auto minor_key_first = thrust::make_transform_iterator(
        edge_partition.indices(),
        detail::minor_to_key_t<EdgePartitionDstKeyInputWrapper>{
          edge_partition_dst_key_input, edge_partition.minor_range_first()});

      // to limit memory footprint ((1 << 20) is a tuning parameter)
      auto approx_edges_to_sort_per_iteration =
        static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
      auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_edge_chunks(
        handle,
        edge_partition.offsets(),
        edge_partition.dcs_nzd_vertices()
          ? (*segment_offsets)[detail::num_sparse_segments_per_vertex_partition] +
              *(edge_partition.dcs_nzd_vertex_count())
          : edge_partition.major_range_size(),
        edge_partition.number_of_edges(),
        approx_edges_to_sort_per_iteration);
      auto num_chunks = h_vertex_offsets.size() - 1;

      size_t max_chunk_size{0};
      for (size_t j = 0; j < num_chunks; ++j) {
        max_chunk_size =
          std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[j + 1] - h_edge_offsets[j]));
      }
      rmm::device_uvector<vertex_t> unreduced_majors(max_chunk_size, handle.get_stream());
      rmm::device_uvector<vertex_t> unreduced_minor_keys(unreduced_majors.size(),
                                                         handle.get_stream());
      rmm::device_uvector<weight_t> unreduced_key_aggregated_edge_weights(
        graph_view.is_weighted() ? unreduced_majors.size() : size_t{0}, handle.get_stream());
      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

      size_t reduced_size{0};
      for (size_t j = 0; j < num_chunks; ++j) {
        thrust::copy(handle.get_thrust_policy(),
                     minor_key_first + h_edge_offsets[j],
                     minor_key_first + h_edge_offsets[j + 1],
                     tmp_minor_keys.begin() + h_edge_offsets[j]);

        size_t tmp_storage_bytes{0};
        auto offset_first =
          thrust::make_transform_iterator(edge_partition.offsets() + h_vertex_offsets[j],
                                          detail::rebase_offset_t<edge_t>{h_edge_offsets[j]});
        if (graph_view.is_weighted()) {
          cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                              tmp_storage_bytes,
                                              tmp_minor_keys.begin() + h_edge_offsets[j],
                                              unreduced_minor_keys.begin(),
                                              *(edge_partition.weights()) + h_edge_offsets[j],
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
                                              *(edge_partition.weights()) + h_edge_offsets[j],
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
                     tmp_majors.begin() + h_edge_offsets[j],
                     tmp_majors.begin() + h_edge_offsets[j + 1],
                     unreduced_majors.begin());
        auto input_key_first = thrust::make_zip_iterator(
          thrust::make_tuple(unreduced_majors.begin(), unreduced_minor_keys.begin()));
        auto output_key_first =
          thrust::make_zip_iterator(thrust::make_tuple(tmp_majors.begin(), tmp_minor_keys.begin()));
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
      tmp_majors.resize(reduced_size, handle.get_stream());
      tmp_minor_keys.resize(tmp_majors.size(), handle.get_stream());
      tmp_key_aggregated_edge_weights.resize(tmp_majors.size(), handle.get_stream());
    }
    tmp_minor_keys.shrink_to_fit(handle.get_stream());
    tmp_key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());
    tmp_majors.shrink_to_fit(handle.get_stream());

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_size = col_comm.get_size();

      auto triplet_first     = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_majors.begin(), tmp_minor_keys.begin(), tmp_key_aggregated_edge_weights.begin()));
      auto d_tx_value_counts = cugraph::groupby_and_count(
        triplet_first,
        triplet_first + tmp_majors.size(),
        detail::triplet_to_col_rank_t<vertex_t, weight_t>{
          detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}, row_comm_size},
        col_comm_size,
        mem_frugal_threshold,
        handle.get_stream());

      std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
      raft::update_host(h_tx_value_counts.data(),
                        d_tx_value_counts.data(),
                        d_tx_value_counts.size(),
                        handle.get_stream());
      handle.sync_stream();

      rmm::device_uvector<vertex_t> rx_majors(0, handle.get_stream());
      rmm::device_uvector<vertex_t> rx_minor_keys(0, handle.get_stream());
      rmm::device_uvector<weight_t> rx_key_aggregated_edge_weights(0, handle.get_stream());
      auto mem_frugal_flag =
        host_scalar_allreduce(col_comm,
                              tmp_majors.size() > mem_frugal_threshold ? int{1} : int{0},
                              raft::comms::op_t::MAX,
                              handle.get_stream());
      if (mem_frugal_flag) {  // trade-off potential parallelism to lower peak memory
        std::tie(rx_majors, std::ignore) =
          shuffle_values(col_comm, tmp_majors.begin(), h_tx_value_counts, handle.get_stream());
        tmp_majors.resize(0, handle.get_stream());
        tmp_majors.shrink_to_fit(handle.get_stream());

        std::tie(rx_minor_keys, std::ignore) =
          shuffle_values(col_comm, tmp_minor_keys.begin(), h_tx_value_counts, handle.get_stream());
        tmp_minor_keys.resize(0, handle.get_stream());
        tmp_minor_keys.shrink_to_fit(handle.get_stream());

        std::tie(rx_key_aggregated_edge_weights, std::ignore) =
          shuffle_values(col_comm,
                         tmp_key_aggregated_edge_weights.begin(),
                         h_tx_value_counts,
                         handle.get_stream());
        tmp_key_aggregated_edge_weights.resize(0, handle.get_stream());
        tmp_key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());
      } else {
        std::forward_as_tuple(std::tie(rx_majors, rx_minor_keys, rx_key_aggregated_edge_weights),
                              std::ignore) =
          shuffle_values(col_comm, triplet_first, h_tx_value_counts, handle.get_stream());
        tmp_majors.resize(0, handle.get_stream());
        tmp_majors.shrink_to_fit(handle.get_stream());
        tmp_minor_keys.resize(0, handle.get_stream());
        tmp_minor_keys.shrink_to_fit(handle.get_stream());
        tmp_key_aggregated_edge_weights.resize(0, handle.get_stream());
        tmp_key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());
      }

      auto key_pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(rx_majors.begin(), rx_minor_keys.begin()));
      if (rx_majors.size() > mem_frugal_threshold) {  // trade-off parallelism to lower peak memory
        auto second_first =
          detail::mem_frugal_partition(key_pair_first,
                                       key_pair_first + rx_majors.size(),
                                       rx_key_aggregated_edge_weights.begin(),
                                       detail::pair_to_binary_partition_id_t<vertex_t>{},
                                       int{1},
                                       handle.get_stream());

        thrust::sort_by_key(handle.get_thrust_policy(),
                            key_pair_first,
                            std::get<0>(second_first),
                            rx_key_aggregated_edge_weights.begin());

        thrust::sort_by_key(handle.get_thrust_policy(),
                            std::get<0>(second_first),
                            key_pair_first + rx_majors.size(),
                            std::get<1>(second_first));
      } else {
        thrust::sort_by_key(handle.get_thrust_policy(),
                            key_pair_first,
                            key_pair_first + rx_majors.size(),
                            rx_key_aggregated_edge_weights.begin());
      }
      auto num_uniques = thrust::count_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(rx_majors.size()),
        detail::is_first_in_run_pair_t<vertex_t>{rx_majors.data(), rx_minor_keys.data()});
      tmp_majors.resize(num_uniques, handle.get_stream());
      tmp_minor_keys.resize(tmp_majors.size(), handle.get_stream());
      tmp_key_aggregated_edge_weights.resize(tmp_majors.size(), handle.get_stream());
      thrust::reduce_by_key(
        handle.get_thrust_policy(),
        key_pair_first,
        key_pair_first + rx_majors.size(),
        rx_key_aggregated_edge_weights.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(tmp_majors.begin(), tmp_minor_keys.begin())),
        tmp_key_aggregated_edge_weights.begin());
    }

    auto multi_gpu_kv_map_ptr = std::make_unique<
      cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>>(
      size_t{0},
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      stream_adapter,
      handle.get_stream());  // relevant only when GraphViewType::is_multi_gpu is true
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      rmm::device_uvector<vertex_t> unique_minor_keys(tmp_minor_keys.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   tmp_minor_keys.begin(),
                   tmp_minor_keys.end(),
                   unique_minor_keys.begin());
      thrust::sort(handle.get_thrust_policy(), unique_minor_keys.begin(), unique_minor_keys.end());
      unique_minor_keys.resize(thrust::distance(unique_minor_keys.begin(),
                                                thrust::unique(handle.get_thrust_policy(),
                                                               unique_minor_keys.begin(),
                                                               unique_minor_keys.end())),
                               handle.get_stream());
      unique_minor_keys.shrink_to_fit(handle.get_stream());

      auto values_for_unique_keys = allocate_dataframe_buffer<value_t>(0, handle.get_stream());
      std::tie(unique_minor_keys, values_for_unique_keys) =
        collect_values_for_unique_keys<vertex_t,
                                       value_t,
                                       decltype(stream_adapter),
                                       cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>>(
          comm,
          kv_map,
          std::move(unique_minor_keys),
          cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size},
          handle.get_stream());

      multi_gpu_kv_map_ptr.reset();
      multi_gpu_kv_map_ptr = std::make_unique<
        cuco::static_map<vertex_t, value_t, cuda::thread_scope_device, decltype(stream_adapter)>>(
        // cuco::static_map requires at least one empty slot
        std::max(static_cast<size_t>(static_cast<double>(unique_minor_keys.size()) / load_factor),
                 static_cast<size_t>(unique_minor_keys.size()) + 1),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value,
        stream_adapter,
        handle.get_stream());

      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        unique_minor_keys.begin(), get_dataframe_buffer_begin(values_for_unique_keys)));
      multi_gpu_kv_map_ptr->insert(pair_first,
                                   pair_first + unique_minor_keys.size(),
                                   cuco::detail::MurmurHash3_32<vertex_t>{},
                                   thrust::equal_to<vertex_t>{},
                                   handle.get_stream());
    }

    auto tmp_e_op_result_buffer =
      allocate_dataframe_buffer<T>(tmp_majors.size(), handle.get_stream());

    auto edge_partition_src_value_input_copy = edge_partition_src_value_input;
    edge_partition_src_value_input_copy.set_local_edge_partition_idx(i);

    auto triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
      tmp_majors.begin(), tmp_minor_keys.begin(), tmp_key_aggregated_edge_weights.begin()));
    thrust::transform(handle.get_thrust_policy(),
                      triplet_first,
                      triplet_first + tmp_majors.size(),
                      get_dataframe_buffer_begin(tmp_e_op_result_buffer),
                      detail::call_key_aggregated_e_op_t<vertex_t,
                                                         weight_t,
                                                         EdgePartitionSrcValueInputWrapper,
                                                         KeyAggregatedEdgeOp,
                                                         decltype(edge_partition),
                                                         decltype(kv_map.get_device_view())>{
                        edge_partition_src_value_input_copy,
                        key_aggregated_e_op,
                        edge_partition,
                        GraphViewType::is_multi_gpu ? multi_gpu_kv_map_ptr->get_device_view()
                                                    : kv_map.get_device_view()});

    if constexpr (GraphViewType::is_multi_gpu) { multi_gpu_kv_map_ptr.reset(); }
    tmp_minor_keys.resize(0, handle.get_stream());
    tmp_minor_keys.shrink_to_fit(handle.get_stream());
    tmp_key_aggregated_edge_weights.resize(0, handle.get_stream());
    tmp_key_aggregated_edge_weights.shrink_to_fit(handle.get_stream());

    {
      auto num_uniques = thrust::count_if(handle.get_thrust_policy(),
                                          thrust::make_counting_iterator(size_t{0}),
                                          thrust::make_counting_iterator(tmp_majors.size()),
                                          detail::is_first_in_run_t<vertex_t>{tmp_majors.data()});
      rmm::device_uvector<vertex_t> unique_majors(num_uniques, handle.get_stream());
      auto reduced_e_op_result_buffer =
        allocate_dataframe_buffer<T>(unique_majors.size(), handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            tmp_majors.begin(),
                            tmp_majors.end(),
                            get_dataframe_buffer_begin(tmp_e_op_result_buffer),
                            unique_majors.begin(),
                            get_dataframe_buffer_begin(reduced_e_op_result_buffer),
                            thrust::equal_to<vertex_t>{},
                            reduce_op);
      tmp_majors             = std::move(unique_majors);
      tmp_e_op_result_buffer = std::move(reduced_e_op_result_buffer);
    }

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      // FIXME: additional optimization is possible if reduce_op is a pure function (and reduce_op
      // can be mapped to ncclRedOp_t).

      auto rx_sizes = host_scalar_gather(col_comm, tmp_majors.size(), i, handle.get_stream());
      std::vector<size_t> rx_displs{};
      rmm::device_uvector<vertex_t> rx_majors(0, handle.get_stream());
      if (static_cast<size_t>(col_comm_rank) == i) {
        rx_displs.assign(col_comm_size, size_t{0});
        std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);
        rx_majors.resize(rx_displs.back() + rx_sizes.back(), handle.get_stream());
      }
      auto rx_tmp_e_op_result_buffer =
        allocate_dataframe_buffer<T>(rx_majors.size(), handle.get_stream());

      device_gatherv(col_comm,
                     tmp_majors.data(),
                     rx_majors.data(),
                     tmp_majors.size(),
                     rx_sizes,
                     rx_displs,
                     i,
                     handle.get_stream());
      device_gatherv(col_comm,
                     get_dataframe_buffer_begin(tmp_e_op_result_buffer),
                     get_dataframe_buffer_begin(rx_tmp_e_op_result_buffer),
                     tmp_majors.size(),
                     rx_sizes,
                     rx_displs,
                     i,
                     handle.get_stream());

      if (static_cast<size_t>(col_comm_rank) == i) {
        majors             = std::move(rx_majors);
        e_op_result_buffer = std::move(rx_tmp_e_op_result_buffer);
      }
    } else {
      majors             = std::move(tmp_majors);
      e_op_result_buffer = std::move(tmp_e_op_result_buffer);
    }
  }

  if constexpr (GraphViewType::is_multi_gpu) {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        majors.begin(),
                        majors.end(),
                        get_dataframe_buffer_begin(e_op_result_buffer));
    auto num_uniques = thrust::count_if(handle.get_thrust_policy(),
                                        thrust::make_counting_iterator(size_t{0}),
                                        thrust::make_counting_iterator(majors.size()),
                                        detail::is_first_in_run_t<vertex_t>{majors.data()});
    rmm::device_uvector<vertex_t> unique_majors(num_uniques, handle.get_stream());
    auto reduced_e_op_result_buffer =
      allocate_dataframe_buffer<T>(unique_majors.size(), handle.get_stream());
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          majors.begin(),
                          majors.end(),
                          get_dataframe_buffer_begin(e_op_result_buffer),
                          unique_majors.begin(),
                          get_dataframe_buffer_begin(reduced_e_op_result_buffer),
                          thrust::equal_to<vertex_t>{},
                          reduce_op);
    majors             = std::move(unique_majors);
    e_op_result_buffer = std::move(reduced_e_op_result_buffer);
  }

  thrust::fill(handle.get_thrust_policy(),
               vertex_value_output_first,
               vertex_value_output_first + graph_view.local_vertex_partition_range_size(),
               T{});

  thrust::scatter(handle.get_thrust_policy(),
                  get_dataframe_buffer_begin(e_op_result_buffer),
                  get_dataframe_buffer_end(e_op_result_buffer),
                  thrust::make_transform_iterator(
                    majors.begin(),
                    detail::vertex_local_offset_t<vertex_t, GraphViewType::is_multi_gpu>{
                      graph_view.local_vertex_partition_view()}),
                  vertex_value_output_first);

  thrust::transform(handle.get_thrust_policy(),
                    vertex_value_output_first,
                    vertex_value_output_first + graph_view.local_vertex_partition_range_size(),
                    vertex_value_output_first,
                    detail::reduce_with_init_t<ReduceOp, T>{reduce_op, init});
}

}  // namespace cugraph

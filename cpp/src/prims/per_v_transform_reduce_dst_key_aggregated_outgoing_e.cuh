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

#include <detail/graph_utils.cuh>
#include <prims/kv_store.cuh>
#include <utilities/collect_comm.cuh>

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

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
template <typename vertex_t, typename edge_value_t>
struct triplet_to_col_rank_t {
  compute_gpu_id_from_ext_vertex_t<vertex_t> key_func{};
  int row_comm_size{};
  __device__ int operator()(
    thrust::tuple<vertex_t, vertex_t, edge_value_t> val /* major, minor key, edge value */) const
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
          typename edge_value_t,
          typename EdgePartitionSrcValueInputWrapper,
          typename KeyAggregatedEdgeOp,
          typename EdgePartitionDeviceView,
          typename KVStoreDeviceViewType>
struct call_key_aggregated_e_op_t {
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input{};
  KeyAggregatedEdgeOp key_aggregated_e_op{};
  EdgePartitionDeviceView edge_partition{};
  KVStoreDeviceViewType kv_store_device_view{};

  __device__ auto operator()(thrust::tuple<vertex_t, vertex_t, edge_value_t>
                               val /* major, minor key, aggregated edge value */) const
  {
    auto major                 = thrust::get<0>(val);
    auto key                   = thrust::get<1>(val);
    auto aggregated_edge_value = thrust::get<2>(val);
    return key_aggregated_e_op(
      major,
      key,
      edge_partition_src_value_input.get(edge_partition.major_offset_from_major_nocheck(major)),
      kv_store_device_view.find(key),
      aggregated_edge_value);
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
 * @brief Iterate over every vertex's destination key-aggregated outgoing edges to update vertex
 * property values.
 *
 * This function is inspired by thrust::transform_reduce().
 * Unlike per_v_transform_reduce_outgoing_e, this function first aggregates outgoing edges by
 * destination keys to support two level reduction for every vertex.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeDstKeyInputWrapper Type of the wrapper for edge destination key values.
 * @tparam VertexIterator Type of the iterator for keys in (key, value) pairs (key type should
 * coincide with vertex type).
 * @tparam ValueIterator Type of the iterator for values in (key, value) pairs.
 * @tparam KeyAggregatedEdgeOp Type of the quinary key-aggregated edge operator.
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
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param edge_dst_key_input Wrapper used to access destination input key values (for the edge
 * destinations assigned to this process in multi-GPU). Use  cugraph::edge_dst_property_t::view().
 * Use update_edge_dst_property to fill the wrapper.
 * @param map_unique_key_first Iterator pointing to the first (inclusive) key in (key, value) pairs
 * (assigned to this process in multi-GPU, `cugraph::detail::compute_gpu_id_from_ext_vertex_t` is
 * used to map keys to processes). (Key, value) pairs may be provided by
 * transform_reduce_by_src_key_e() or transform_reduce_by_dst_key_e().
 * @param map_unique_key_last Iterator pointing to the last (exclusive) key in (key, value) pairs
 * (assigned to this process in multi-GPU).
 * @param map_value_first Iterator pointing to the first (inclusive) value in (key, value) pairs
 * (assigned to this process in multi-GPU). `map_value_last` (exclusive) is deduced as @p
 * map_value_first + thrust::distance(@p map_unique_key_first, @p map_unique_key_last).
 * @param key_aggregated_e_op Quinary operator takes 1) edge source, 2) key, 3) *(@p
 * edge_partition_src_value_input_first + i), 4) value for the key stored in the input (key, value)
 * pairs provided by @p map_unique_key_first, @p map_unique_key_last, and @p map_value_first
 * (aggregated over the entire set of processes in multi-GPU), and 5) aggregated edge value.
 * @param init Initial value to be reduced with the reduced value for each vertex.
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
          typename EdgeValueInputWrapper,
          typename EdgeDstKeyInputWrapper,
          typename KVStoreViewType,
          typename KeyAggregatedEdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_dst_key_aggregated_outgoing_e(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeValueInputWrapper edge_value_input,
  EdgeDstKeyInputWrapper edge_dst_key_input,
  KVStoreViewType kv_store_view,
  KeyAggregatedEdgeOp key_aggregated_e_op,
  T init,
  ReduceOp reduce_op,
  VertexValueOutputIterator vertex_value_output_first,
  bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");
  static_assert(
    std::is_same_v<typename KVStoreViewType::key_type, typename GraphViewType::vertex_type>);
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using vertex_t        = typename GraphViewType::vertex_type;
  using edge_t          = typename GraphViewType::edge_type;
  using edge_value_t    = typename EdgeValueInputWrapper::value_type;
  using kv_pair_value_t = typename KVStoreViewType::value_type;
  static_assert(
    std::is_arithmetic_v<edge_value_t>,
    "Currently only scalar values are supported, should be extended to support thrust::tuple of "
    "arithmetic types and void (for dummy property values) to be consistent with other "
    "primitives.");  // this will also require a custom edge value aggregation op.

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
  using edge_partition_dst_key_device_view_t =
    std::conditional_t<GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstKeyInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstKeyInputWrapper::value_iterator>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator>>;

  if (do_expensive_check) { /* currently, nothing to do */
  }

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  size_t element_size   = sizeof(vertex_t) * 2;
  if constexpr (!std::is_same_v<edge_value_t, void>) {
    static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<edge_value_t>::value);
    if constexpr (is_thrust_tuple_of_arithmetic<edge_value_t>::value) {
      element_size += sum_thrust_tuple_element_sizes<edge_value_t>();
    } else {
      element_size += sizeof(edge_value_t);
    }
  }
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach
  [[maybe_unused]] auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  // 1. aggregate each vertex out-going edges based on keys and transform-reduce.

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  auto e_op_result_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto edge_partition_src_value_input =
      edge_partition_src_input_device_view_t(edge_src_value_input, i);
    auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, i);

    rmm::device_uvector<vertex_t> tmp_majors(edge_partition.number_of_edges(), handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_minor_keys(tmp_majors.size(), handle.get_stream());
    // FIXME: this doesn't work if edge_value_t is thrust::tuple or void
    rmm::device_uvector<edge_value_t> tmp_key_aggregated_edge_values(tmp_majors.size(),
                                                                     handle.get_stream());

    if (edge_partition.number_of_edges() > 0) {
      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);

      detail::decompress_edge_partition_to_fill_edgelist_majors(
        handle, edge_partition, tmp_majors.data(), segment_offsets);

      auto minor_key_first = thrust::make_transform_iterator(
        edge_partition.indices(),
        detail::minor_to_key_t<edge_partition_dst_key_device_view_t>{
          edge_partition_dst_key_device_view_t(edge_dst_key_input),
          edge_partition.minor_range_first()});

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
      // FIXME: this doesn't work if edge_value_t is thrust::tuple or void
      rmm::device_uvector<edge_value_t> unreduced_key_aggregated_edge_values(
        unreduced_majors.size(), handle.get_stream());
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
        if constexpr (!std::is_same_v<edge_value_t, void>) {
          cub::DeviceSegmentedSort::SortPairs(
            static_cast<void*>(nullptr),
            tmp_storage_bytes,
            tmp_minor_keys.begin() + h_edge_offsets[j],
            unreduced_minor_keys.begin(),
            edge_partition_e_value_input.value_first() + h_edge_offsets[j],
            unreduced_key_aggregated_edge_values.begin(),
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
        if constexpr (!std::is_same_v<edge_value_t, void>) {
          cub::DeviceSegmentedSort::SortPairs(
            d_tmp_storage.data(),
            tmp_storage_bytes,
            tmp_minor_keys.begin() + h_edge_offsets[j],
            unreduced_minor_keys.begin(),
            edge_partition_e_value_input.value_first() + h_edge_offsets[j],
            unreduced_key_aggregated_edge_values.begin(),
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
        if constexpr (!std::is_same_v<edge_value_t, void>) {
          reduced_size +=
            thrust::distance(output_key_first + reduced_size,
                             thrust::get<0>(thrust::reduce_by_key(
                               handle.get_thrust_policy(),
                               input_key_first,
                               input_key_first + (h_edge_offsets[j + 1] - h_edge_offsets[j]),
                               unreduced_key_aggregated_edge_values.begin(),
                               output_key_first + reduced_size,
                               tmp_key_aggregated_edge_values.begin() + reduced_size)));
        } else {
          reduced_size +=
            thrust::distance(output_key_first + reduced_size,
                             thrust::get<0>(thrust::unique(
                               handle.get_thrust_policy(),
                               input_key_first,
                               input_key_first + (h_edge_offsets[j + 1] - h_edge_offsets[j]),
                               output_key_first + reduced_size)));
        }
      }
      tmp_majors.resize(reduced_size, handle.get_stream());
      tmp_minor_keys.resize(tmp_majors.size(), handle.get_stream());
      // FIXME: this doesn't work if edge_value_t is thrust::tuple or void
      tmp_key_aggregated_edge_values.resize(tmp_majors.size(), handle.get_stream());
    }
    tmp_majors.shrink_to_fit(handle.get_stream());
    tmp_minor_keys.shrink_to_fit(handle.get_stream());
    // FIXME: this doesn't work if edge_value_t is thrust::tuple or void
    tmp_key_aggregated_edge_values.shrink_to_fit(handle.get_stream());

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();
      auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_size = col_comm.get_size();

      // FIXME: this doesn't work if edge_value_t is thrust::tuple or void
      auto triplet_first     = thrust::make_zip_iterator(thrust::make_tuple(
        tmp_majors.begin(), tmp_minor_keys.begin(), tmp_key_aggregated_edge_values.begin()));
      auto d_tx_value_counts = cugraph::groupby_and_count(
        triplet_first,
        triplet_first + tmp_majors.size(),
        detail::triplet_to_col_rank_t<vertex_t, edge_value_t>{
          detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size}, row_comm_size},
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
      rmm::device_uvector<edge_value_t> rx_key_aggregated_edge_values(0, handle.get_stream());
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

        std::tie(rx_key_aggregated_edge_values, std::ignore) = shuffle_values(
          col_comm, tmp_key_aggregated_edge_values.begin(), h_tx_value_counts, handle.get_stream());
        tmp_key_aggregated_edge_values.resize(0, handle.get_stream());
        tmp_key_aggregated_edge_values.shrink_to_fit(handle.get_stream());
      } else {
        std::forward_as_tuple(std::tie(rx_majors, rx_minor_keys, rx_key_aggregated_edge_values),
                              std::ignore) =
          shuffle_values(col_comm, triplet_first, h_tx_value_counts, handle.get_stream());
        tmp_majors.resize(0, handle.get_stream());
        tmp_majors.shrink_to_fit(handle.get_stream());
        tmp_minor_keys.resize(0, handle.get_stream());
        tmp_minor_keys.shrink_to_fit(handle.get_stream());
        tmp_key_aggregated_edge_values.resize(0, handle.get_stream());
        tmp_key_aggregated_edge_values.shrink_to_fit(handle.get_stream());
      }

      auto key_pair_first =
        thrust::make_zip_iterator(thrust::make_tuple(rx_majors.begin(), rx_minor_keys.begin()));
      if (rx_majors.size() > mem_frugal_threshold) {  // trade-off parallelism to lower peak memory
        auto second_first =
          detail::mem_frugal_partition(key_pair_first,
                                       key_pair_first + rx_majors.size(),
                                       rx_key_aggregated_edge_values.begin(),
                                       detail::pair_to_binary_partition_id_t<vertex_t>{},
                                       int{1},
                                       handle.get_stream());

        thrust::sort_by_key(handle.get_thrust_policy(),
                            key_pair_first,
                            std::get<0>(second_first),
                            rx_key_aggregated_edge_values.begin());

        thrust::sort_by_key(handle.get_thrust_policy(),
                            std::get<0>(second_first),
                            key_pair_first + rx_majors.size(),
                            std::get<1>(second_first));
      } else {
        thrust::sort_by_key(handle.get_thrust_policy(),
                            key_pair_first,
                            key_pair_first + rx_majors.size(),
                            rx_key_aggregated_edge_values.begin());
      }
      auto num_uniques =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(rx_majors.size()),
                         detail::is_first_in_run_t<decltype(key_pair_first)>{key_pair_first});
      tmp_majors.resize(num_uniques, handle.get_stream());
      tmp_minor_keys.resize(tmp_majors.size(), handle.get_stream());
      tmp_key_aggregated_edge_values.resize(tmp_majors.size(), handle.get_stream());
      thrust::reduce_by_key(
        handle.get_thrust_policy(),
        key_pair_first,
        key_pair_first + rx_majors.size(),
        rx_key_aggregated_edge_values.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(tmp_majors.begin(), tmp_minor_keys.begin())),
        tmp_key_aggregated_edge_values.begin());
    }

    std::unique_ptr<kv_store_t<vertex_t, kv_pair_value_t, KVStoreViewType::binary_search>>
      multi_gpu_kv_map_ptr{nullptr};
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

      auto values_for_unique_keys =
        allocate_dataframe_buffer<kv_pair_value_t>(0, handle.get_stream());
      std::tie(unique_minor_keys, values_for_unique_keys) = collect_values_for_unique_keys(
        comm,
        kv_store_view,
        std::move(unique_minor_keys),
        cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size},
        handle.get_stream());

      if constexpr (KVStoreViewType::binary_search) {
        multi_gpu_kv_map_ptr = std::make_unique<kv_store_t<vertex_t, kv_pair_value_t, true>>(
          std::move(unique_minor_keys),
          std::move(values_for_unique_keys),
          kv_store_view.invalid_value,
          false,
          handle.get_stream());
      } else {
        multi_gpu_kv_map_ptr = std::make_unique<kv_store_t<vertex_t, kv_pair_value_t, false>>(
          unique_minor_keys.begin(),
          unique_minor_keys.begin() + unique_minor_keys.size(),
          get_dataframe_buffer_begin(values_for_unique_keys),
          kv_store_view.cuco_store->get_empty_key_sentinel(),
          kv_store_view.cuco_store->get_empty_value_sentinel(),
          handle.get_stream());
      }
    }

    auto tmp_e_op_result_buffer =
      allocate_dataframe_buffer<T>(tmp_majors.size(), handle.get_stream());

    auto triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
      tmp_majors.begin(), tmp_minor_keys.begin(), tmp_key_aggregated_edge_values.begin()));
    std::conditional_t<KVStoreViewType::binary_search,
                       detail::kv_binary_search_store_device_view_t<KVStoreViewType>,
                       detail::kv_cuco_store_device_view_t<KVStoreViewType>>
      device_view(GraphViewType::is_multi_gpu ? multi_gpu_kv_map_ptr->view() : kv_store_view);
    thrust::transform(
      handle.get_thrust_policy(),
      triplet_first,
      triplet_first + tmp_majors.size(),
      get_dataframe_buffer_begin(tmp_e_op_result_buffer),
      detail::call_key_aggregated_e_op_t<vertex_t,
                                         edge_value_t,
                                         edge_partition_src_input_device_view_t,
                                         KeyAggregatedEdgeOp,
                                         decltype(edge_partition),
                                         decltype(device_view)>{
        edge_partition_src_value_input, key_aggregated_e_op, edge_partition, device_view});

    if constexpr (GraphViewType::is_multi_gpu) { multi_gpu_kv_map_ptr.reset(); }
    tmp_minor_keys.resize(0, handle.get_stream());
    tmp_minor_keys.shrink_to_fit(handle.get_stream());
    tmp_key_aggregated_edge_values.resize(0, handle.get_stream());
    tmp_key_aggregated_edge_values.shrink_to_fit(handle.get_stream());

    {
      auto num_uniques =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(tmp_majors.size()),
                         detail::is_first_in_run_t<vertex_t const*>{tmp_majors.data()});
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
                                        detail::is_first_in_run_t<vertex_t const*>{majors.data()});
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

  // 2. update final results

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

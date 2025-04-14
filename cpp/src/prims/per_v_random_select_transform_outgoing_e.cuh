/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "prims/detail/sample_and_compute_local_nbr_indices.cuh"
#include "prims/property_op_utils.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/random/rng.cuh>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <optional>
#include <tuple>

namespace cugraph {

namespace detail {

template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename key_t>
struct constant_bias_e_op_t {
  __device__ float operator()(key_t,
                              typename GraphViewType::vertex_type,
                              typename EdgeSrcValueInputWrapper::value_type,
                              typename EdgeDstValueInputWrapper::value_type,
                              typename EdgeValueInputWrapper::value_type) const
  {
    return 1.0;
  }
};

template <typename edge_t, typename T>
struct check_invalid_t {
  edge_t invalid_idx{};

  __device__ bool operator()(thrust::tuple<edge_t, T> pair) const
  {
    return thrust::get<0>(pair) == invalid_idx;
  }
};

template <typename GraphViewType,
          typename KeyIterator,
          typename LocalNbrIdxIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
struct transform_local_nbr_indices_t {
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu> edge_partition{};
  cuda::std::optional<size_t const*> local_key_indices{cuda::std::nullopt};
  KeyIterator key_first{};
  LocalNbrIdxIterator local_nbr_idx_first{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input;
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input;
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input;
  EdgeOp e_op{};
  edge_t invalid_idx{};
  cuda::std::optional<T> invalid_value{cuda::std::nullopt};
  size_t K{};

  __device__ T operator()(size_t i) const
  {
    auto key_idx      = local_key_indices ? (*local_key_indices)[i] : (i / K);
    auto key          = *(key_first + key_idx);
    auto major        = thrust_tuple_get_or_identity<key_t, 0>(key);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    edge_t edge_offset{0};
    [[maybe_unused]] edge_t local_degree{0};
    if constexpr (GraphViewType::is_multi_gpu) {
      auto major_hypersparse_first = edge_partition.major_hypersparse_first();
      if (major_hypersparse_first && (major >= *major_hypersparse_first)) {
        auto major_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
        if (major_hypersparse_idx) {
          thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(
            edge_partition.major_offset_from_major_nocheck(*major_hypersparse_first) +
            *major_hypersparse_idx);
        }
      } else {
        thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
      }
    } else {
      thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    }
    auto local_nbr_idx = *(local_nbr_idx_first + i);
    if (local_nbr_idx != invalid_idx) {
      vertex_t minor{};
      minor             = indices[local_nbr_idx];
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);

      std::conditional_t<GraphViewType::is_storage_transposed, vertex_t, key_t>
        key_or_src{};  // key if major
      std::conditional_t<GraphViewType::is_storage_transposed, key_t, vertex_t>
        key_or_dst{};  // key if major
      if constexpr (GraphViewType::is_storage_transposed) {
        key_or_src = minor;
        key_or_dst = key;
      } else {
        key_or_src = key;
        key_or_dst = minor;
      }
      auto src_offset = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
      auto dst_offset = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
      return e_op(key_or_src,
                  key_or_dst,
                  edge_partition_src_value_input.get(src_offset),
                  edge_partition_dst_value_input.get(dst_offset),
                  edge_partition_e_value_input.get(edge_offset + local_nbr_idx));
    } else if (invalid_value) {
      return *invalid_value;
    } else {
      return T{};
    }
  }
};

template <typename edge_t>
struct count_valids_t {
  raft::device_span<edge_t const> sample_local_nbr_indices{};
  size_t K{};
  edge_t invalid_idx{};

  __device__ int32_t operator()(size_t i) const
  {
    auto first = sample_local_nbr_indices.begin() + i * K;
    return static_cast<int32_t>(
      cuda::std::distance(first, thrust::find(thrust::seq, first, first + K, invalid_idx)));
  }
};

struct count_t {
  raft::device_span<int32_t> sample_counts{};

  __device__ size_t operator()(size_t key_idx) const
  {
    cuda::atomic_ref<int32_t, cuda::thread_scope_device> counter(sample_counts[key_idx]);
    return counter.fetch_add(int32_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <bool use_invalid_value>
struct return_value_compute_offset_t {
  raft::device_span<size_t const> sample_key_indices{};
  raft::device_span<int32_t const> sample_intra_partition_displacements{};
  std::conditional_t<use_invalid_value, size_t, raft::device_span<size_t const>>
    K_or_sample_offsets{};

  __device__ size_t operator()(size_t i) const
  {
    auto key_idx = sample_key_indices[i];
    size_t key_start_offset{};
    if constexpr (use_invalid_value) {
      key_start_offset = key_idx * K_or_sample_offsets;
    } else {
      key_start_offset = K_or_sample_offsets[key_idx];
    }
    return key_start_offset + static_cast<size_t>(sample_intra_partition_displacements[i]);
  }
};

template <bool incoming,
          typename GraphViewType,
          typename KeyBucketType,
          typename BiasEdgeSrcValueInputWrapper,
          typename BiasEdgeDstValueInputWrapper,
          typename BiasEdgeValueInputWrapper,
          typename BiasEdgeOp,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeTypeInputWrapper,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
per_v_random_select_transform_e(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                KeyBucketType const& key_list,
                                BiasEdgeSrcValueInputWrapper bias_edge_src_value_input,
                                BiasEdgeDstValueInputWrapper bias_edge_dst_value_input,
                                BiasEdgeValueInputWrapper bias_edge_value_input,
                                BiasEdgeOp bias_e_op,
                                EdgeSrcValueInputWrapper edge_src_value_input,
                                EdgeDstValueInputWrapper edge_dst_value_input,
                                EdgeValueInputWrapper edge_value_input,
                                EdgeOp e_op,
                                EdgeTypeInputWrapper edge_type_input,
                                raft::random::RngState& rng_state,
                                raft::host_span<size_t const> Ks,
                                bool with_replacement,
                                std::optional<T> invalid_value,
                                bool do_expensive_check)
{
  using vertex_t     = typename GraphViewType::vertex_type;
  using edge_t       = typename GraphViewType::edge_type;
  using key_t        = typename KeyBucketType::key_type;
  using key_buffer_t = dataframe_buffer_type_t<key_t>;

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, cuda::std::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, cuda::std::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    edge_partition_endpoint_property_device_view_t<
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

  static_assert(GraphViewType::is_storage_transposed == incoming);
  static_assert(std::is_same_v<
                typename detail::edge_op_result_type<key_t,
                                                     vertex_t,
                                                     typename EdgeSrcValueInputWrapper::value_type,
                                                     typename EdgeDstValueInputWrapper::value_type,
                                                     typename EdgeValueInputWrapper::value_type,
                                                     EdgeOp>::type,
                T>);

  CUGRAPH_EXPECTS(Ks.size() >= 1, "Invalid input argument: Ks.size() should be 1 or larger.");

  if constexpr (std::is_same_v<EdgeTypeInputWrapper, edge_dummy_property_view_t>) {  // homogeneous
    CUGRAPH_EXPECTS(Ks.size() == 1,
                    "Invalid input argument: Ks.size() should be 1 for homogeneous sampling.");
  }

  for (size_t i = 0; i < Ks.size(); ++i) {
    CUGRAPH_EXPECTS(Ks[i] <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    "Invalid input argument: the current implementation expects Ks[] to be no "
                    "larger than std::numeric_limits<int32_t>::max().");
  }

  auto minor_comm_size =
    GraphViewType::is_multi_gpu
      ? handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size()
      : int{1};
  assert(graph_view.number_of_local_edge_partitions() == minor_comm_size);

  if (do_expensive_check) {
    // FIXME: better re-factor this check function?
    auto key_list_vertex_first =
      thrust_tuple_get_or_identity<decltype(key_list.begin()), 0>(key_list.begin());
    auto key_list_vertex_last =
      thrust_tuple_get_or_identity<decltype(key_list.end()), 0>(key_list.end());
    auto num_invalid_keys =
      key_list.size() -
      thrust::count_if(handle.get_thrust_policy(),
                       key_list_vertex_first,
                       key_list_vertex_last,
                       check_in_range_t<vertex_t>{graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last()});
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_keys = host_scalar_allreduce(
        handle.get_comms(), num_invalid_keys, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_keys == size_t{0},
                    "Invalid input argument: key_list includes out-of-range keys.");
  }

  std::vector<size_t> local_key_list_sizes{};
  if (minor_comm_size > 1) {
    auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    local_key_list_sizes = host_scalar_allgather(minor_comm, key_list.size(), handle.get_stream());
  } else {
    local_key_list_sizes = std::vector<size_t>{key_list.size()};
  }
  std::vector<size_t> local_key_list_offsets(local_key_list_sizes.size() + 1);
  local_key_list_offsets[0] = 0;
  std::inclusive_scan(
    local_key_list_sizes.begin(), local_key_list_sizes.end(), local_key_list_offsets.begin() + 1);

  // 1. aggregate key_list

  std::optional<key_buffer_t> aggregate_local_key_list{std::nullopt};
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    aggregate_local_key_list =
      allocate_dataframe_buffer<key_t>(local_key_list_offsets.back(), handle.get_stream());
    device_allgatherv(
      minor_comm,
      key_list.begin(),
      get_dataframe_buffer_begin(*aggregate_local_key_list),
      raft::host_span<size_t const>(local_key_list_sizes.data(), local_key_list_sizes.size()),
      raft::host_span<size_t const>(local_key_list_offsets.data(),
                                    local_key_list_offsets.size() - 1),
      handle.get_stream());
  }

  // 2. randomly select neighbor indices and compute local neighbor indices for every local edge
  // partition

  rmm::device_uvector<edge_t> sample_local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> sample_key_indices{std::nullopt};
  std::vector<size_t> local_key_list_sample_offsets{};
  if constexpr (std::is_same_v<BiasEdgeOp,
                               constant_bias_e_op_t<GraphViewType,
                                                    BiasEdgeSrcValueInputWrapper,
                                                    BiasEdgeDstValueInputWrapper,
                                                    BiasEdgeValueInputWrapper,
                                                    key_t>>) {
    if constexpr (std::is_same_v<EdgeTypeInputWrapper,
                                 edge_dummy_property_view_t>) {  // homogeneous
      std::tie(sample_local_nbr_indices, sample_key_indices, local_key_list_sample_offsets) =
        homogeneous_uniform_sample_and_compute_local_nbr_indices(
          handle,
          graph_view,
          (minor_comm_size > 1) ? get_dataframe_buffer_cbegin(*aggregate_local_key_list)
                                : key_list.begin(),
          raft::host_span<size_t const>(local_key_list_offsets.data(),
                                        local_key_list_offsets.size()),
          rng_state,
          Ks[0],
          with_replacement);
    } else {  // heterogeneous
      std::tie(sample_local_nbr_indices, sample_key_indices, local_key_list_sample_offsets) =
        heterogeneous_uniform_sample_and_compute_local_nbr_indices(
          handle,
          graph_view,
          (minor_comm_size > 1) ? get_dataframe_buffer_cbegin(*aggregate_local_key_list)
                                : key_list.begin(),
          edge_type_input,
          raft::host_span<size_t const>(local_key_list_offsets.data(),
                                        local_key_list_offsets.size()),
          rng_state,
          Ks,
          with_replacement);
    }
  } else {
    if constexpr (std::is_same_v<EdgeTypeInputWrapper,
                                 edge_dummy_property_view_t>) {  // homogeneous
      std::tie(sample_local_nbr_indices, sample_key_indices, local_key_list_sample_offsets) =
        homogeneous_biased_sample_and_compute_local_nbr_indices(
          handle,
          graph_view,
          (minor_comm_size > 1) ? get_dataframe_buffer_cbegin(*aggregate_local_key_list)
                                : key_list.begin(),
          bias_edge_src_value_input,
          bias_edge_dst_value_input,
          bias_edge_value_input,
          bias_e_op,
          raft::host_span<size_t const>(local_key_list_offsets.data(),
                                        local_key_list_offsets.size()),
          rng_state,
          Ks[0],
          with_replacement,
          do_expensive_check);
    } else {  // heterogeneous
      std::tie(sample_local_nbr_indices, sample_key_indices, local_key_list_sample_offsets) =
        heterogeneous_biased_sample_and_compute_local_nbr_indices(
          handle,
          graph_view,
          (minor_comm_size > 1) ? get_dataframe_buffer_cbegin(*aggregate_local_key_list)
                                : key_list.begin(),
          bias_edge_src_value_input,
          bias_edge_dst_value_input,
          bias_edge_value_input,
          bias_e_op,
          edge_type_input,
          raft::host_span<size_t const>(local_key_list_offsets.data(),
                                        local_key_list_offsets.size()),
          rng_state,
          Ks,
          with_replacement,
          do_expensive_check);
    }
  }

  std::vector<size_t> local_key_list_sample_counts(minor_comm_size);
  std::adjacent_difference(local_key_list_sample_offsets.begin() + 1,
                           local_key_list_sample_offsets.end(),
                           local_key_list_sample_counts.begin());

  // 3. transform

  auto K_sum = std::accumulate(Ks.begin(), Ks.end(), size_t{0});

  auto sample_e_op_results =
    allocate_dataframe_buffer<T>(local_key_list_sample_offsets.back(), handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto edge_partition_key_list_first =
      ((minor_comm_size > 1) ? get_dataframe_buffer_cbegin(*aggregate_local_key_list)
                             : key_list.begin()) +
      local_key_list_offsets[i];
    auto edge_partition_sample_local_nbr_index_first =
      sample_local_nbr_indices.begin() + local_key_list_sample_offsets[i];

    auto edge_partition_sample_e_op_result_first =
      get_dataframe_buffer_begin(sample_e_op_results) + local_key_list_sample_offsets[i];

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

    if (sample_key_indices) {
      auto edge_partition_sample_key_index_first =
        (*sample_key_indices).begin() + local_key_list_sample_offsets[i];
      thrust::transform(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(local_key_list_sample_counts[i]),
        edge_partition_sample_e_op_result_first,
        transform_local_nbr_indices_t<GraphViewType,
                                      decltype(edge_partition_key_list_first),
                                      decltype(edge_partition_sample_local_nbr_index_first),
                                      edge_partition_src_input_device_view_t,
                                      edge_partition_dst_input_device_view_t,
                                      edge_partition_e_input_device_view_t,
                                      EdgeOp,
                                      T>{
          edge_partition,
          cuda::std::make_optional(edge_partition_sample_key_index_first),
          edge_partition_key_list_first,
          edge_partition_sample_local_nbr_index_first,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          e_op,
          cugraph::invalid_edge_id_v<edge_t>,
          to_thrust_optional(invalid_value),
          K_sum});
    } else {
      thrust::transform(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(key_list.size() * K_sum),
        edge_partition_sample_e_op_result_first,
        transform_local_nbr_indices_t<GraphViewType,
                                      decltype(edge_partition_key_list_first),
                                      decltype(edge_partition_sample_local_nbr_index_first),
                                      edge_partition_src_input_device_view_t,
                                      edge_partition_dst_input_device_view_t,
                                      edge_partition_e_input_device_view_t,
                                      EdgeOp,
                                      T>{edge_partition,
                                         cuda::std::nullopt,
                                         edge_partition_key_list_first,
                                         edge_partition_sample_local_nbr_index_first,
                                         edge_partition_src_value_input,
                                         edge_partition_dst_value_input,
                                         edge_partition_e_value_input,
                                         e_op,
                                         cugraph::invalid_edge_id_v<edge_t>,
                                         to_thrust_optional(invalid_value),
                                         K_sum});
    }
  }
  aggregate_local_key_list = std::nullopt;

  // 4. shuffle randomly selected & transformed results and update sample_offsets

  auto sample_offsets = invalid_value ? std::nullopt
                                      : std::make_optional<rmm::device_uvector<size_t>>(
                                          key_list.size() + 1, handle.get_stream());
  assert(K_sum <= std::numeric_limits<int32_t>::max());
  if (minor_comm_size > 1) {
    sample_local_nbr_indices.resize(0, handle.get_stream());
    sample_local_nbr_indices.shrink_to_fit(handle.get_stream());

    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    std::tie(sample_e_op_results, std::ignore) =
      shuffle_values(minor_comm,
                     get_dataframe_buffer_begin(sample_e_op_results),
                     raft::host_span<size_t const>(local_key_list_sample_counts.data(),
                                                   local_key_list_sample_counts.size()),
                     handle.get_stream());
    std::tie(sample_key_indices, std::ignore) =
      shuffle_values(minor_comm,
                     (*sample_key_indices).begin(),
                     raft::host_span<size_t const>(local_key_list_sample_counts.data(),
                                                   local_key_list_sample_counts.size()),
                     handle.get_stream());

    rmm::device_uvector<int32_t> sample_counts(key_list.size(), handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), sample_counts.begin(), sample_counts.end(), int32_t{0});
    auto sample_intra_partition_displacements =
      rmm::device_uvector<int32_t>((*sample_key_indices).size(), handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(),
      (*sample_key_indices).begin(),
      (*sample_key_indices).end(),
      sample_intra_partition_displacements.begin(),
      count_t{raft::device_span<int32_t>(sample_counts.data(), sample_counts.size())});
    auto tmp_sample_e_op_results = allocate_dataframe_buffer<T>(0, handle.get_stream());
    if (invalid_value) {
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(
        tmp_sample_e_op_results, key_list.size() * K_sum, handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(tmp_sample_e_op_results),
                   get_dataframe_buffer_end(tmp_sample_e_op_results),
                   *invalid_value);
      thrust::scatter(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(sample_e_op_results),
        get_dataframe_buffer_end(sample_e_op_results),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          return_value_compute_offset_t<true>{
            raft::device_span<size_t const>((*sample_key_indices).data(),
                                            (*sample_key_indices).size()),
            raft::device_span<int32_t const>(sample_intra_partition_displacements.data(),
                                             sample_intra_partition_displacements.size()),
            K_sum}),
        get_dataframe_buffer_begin(tmp_sample_e_op_results));
    } else {
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      auto typecasted_sample_count_first =
        thrust::make_transform_iterator(sample_counts.begin(), typecast_t<int32_t, size_t>{});
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             typecasted_sample_count_first,
                             typecasted_sample_count_first + sample_counts.size(),
                             (*sample_offsets).begin() + 1);
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(tmp_sample_e_op_results,
                              (*sample_offsets).back_element(handle.get_stream()),
                              handle.get_stream());
      thrust::scatter(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(sample_e_op_results),
        get_dataframe_buffer_end(sample_e_op_results),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          return_value_compute_offset_t<false>{
            raft::device_span<size_t const>((*sample_key_indices).data(),
                                            (*sample_key_indices).size()),
            raft::device_span<int32_t const>(sample_intra_partition_displacements.data(),
                                             sample_intra_partition_displacements.size()),
            raft::device_span<size_t const>((*sample_offsets).data(), (*sample_offsets).size())}),
        get_dataframe_buffer_begin(tmp_sample_e_op_results));
    }
    sample_e_op_results = std::move(tmp_sample_e_op_results);
  } else {
    if (!invalid_value) {
      rmm::device_uvector<int32_t> sample_counts(key_list.size(), handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        sample_counts.begin(),
        sample_counts.end(),
        count_valids_t<edge_t>{raft::device_span<edge_t const>(sample_local_nbr_indices.data(),
                                                               sample_local_nbr_indices.size()),
                               K_sum,
                               cugraph::invalid_edge_id_v<edge_t>});
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      auto typecasted_sample_count_first =
        thrust::make_transform_iterator(sample_counts.begin(), typecast_t<int32_t, size_t>{});
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             typecasted_sample_count_first,
                             typecasted_sample_count_first + sample_counts.size(),
                             (*sample_offsets).begin() + 1);
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        sample_local_nbr_indices.begin(), get_dataframe_buffer_begin(sample_e_op_results)));
      auto pair_last =
        thrust::remove_if(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + sample_local_nbr_indices.size(),
                          check_invalid_t<edge_t, T>{cugraph::invalid_edge_id_v<edge_t>});
      sample_local_nbr_indices.resize(0, handle.get_stream());
      sample_local_nbr_indices.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(
        sample_e_op_results, cuda::std::distance(pair_first, pair_last), handle.get_stream());
      shrink_to_fit_dataframe_buffer(sample_e_op_results, handle.get_stream());
    }
  }

  return std::make_tuple(std::move(sample_offsets), std::move(sample_e_op_results));
}

}  // namespace detail

/**
 * @brief Randomly select and transform the input (tagged-)vertices' outgoing edges.
 *
 * This function assumes that every outgoing edge of a given vertex has the same odd to be selected
 * (uniform neighbor sampling).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to sample outgoing
 * edges.
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
 * @param e_op Quinary operator takes (tagged-)edge source, edge destination, property values for
 * the source, destination, and edge and returns a value to be collected in the output. This
 * function is called only for the selected edges.
 * @param K Number of outgoing edges to select per (tagged-)vertex.
 * @param with_replacement A flag to specify whether a single outgoing edge can be selected multiple
 * times (if @p with_replacement = true) or can be selected only once (if @p with_replacement =
 * false).
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the zero out-degree vertices (if @p with_replacement = true) or the vertices
 * with their out-degrees smaller than @p K (if @p with_replacement = false). If @p
 * invalid_value.has_value() is false, fewer than @p K values can be returned for the vertices with
 * fewer than @p K selected edges. See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p key_list.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p key_list.size() * @p K elements). If
 * @p invalid_value.has_value() is true, @p K values are returned for each key in @p key_list. Among
 * the K_sum values, valid values proceed the invalid values; ordering of the valid values can be
 * arbitrary.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         KeyBucketType const& key_list,
                                         EdgeSrcValueInputWrapper edge_src_value_input,
                                         EdgeDstValueInputWrapper edge_dst_value_input,
                                         EdgeValueInputWrapper edge_value_input,
                                         EdgeOp e_op,
                                         raft::random::RngState& rng_state,
                                         size_t K,
                                         bool with_replacement,
                                         std::optional<T> invalid_value,
                                         bool do_expensive_check = false)
{
  return detail::per_v_random_select_transform_e<false>(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_dummy_property_t{}.view(),
    detail::constant_bias_e_op_t<GraphViewType,
                                 detail::edge_endpoint_dummy_property_view_t,
                                 detail::edge_endpoint_dummy_property_view_t,
                                 edge_dummy_property_view_t,
                                 typename KeyBucketType::key_type>{},
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    edge_dummy_property_view_t{},
    rng_state,
    raft::host_span<size_t const>(&K, size_t{1}),
    with_replacement,
    invalid_value,
    do_expensive_check);
}

/**
 * @brief Randomly select (per-type) and transform the input (tagged-)vertices' outgoing edges.
 *
 * This function assumes that every outgoing edge of a given vertex has the same odd to be selected
 * (uniform neighbor sampling).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam EdgeTypeInputWrapper Type of the wrapper for edge type values.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to sample outgoing
 * edges.
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
 * @param e_op Quinary operator takes (tagged-)edge source, edge destination, property values for
 * the source, destination, and edge and returns a value to be collected in the output. This
 * function is called only for the selected edges.
 * @param edge_type_input Wrapper used to access edge type value (for the edges assigned to this
 * process in multi-GPU). This parameter is used in per-type (heterogeneous) sampling. Use
 * cugraph::edge_property_t::view().
 * @param Ks Number of outgoing edges to select per (tagged-)vertex for each edge type (size = #
 * edge types).
 * @param with_replacement A flag to specify whether a single outgoing edge can be selected multiple
 * times (if @p with_replacement = true) or can be selected only once (if @p with_replacement =
 * false).
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the zero out-degree vertices (if @p with_replacement = true) or the vertices
 * with their out-degrees smaller than @p K (if @p with_replacement = false). If @p
 * invalid_value.has_value() is false, fewer than @p K values can be returned for the vertices with
 * fewer than @p K selected edges. See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p key_list.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p key_list.size() * @p K elements). If
 * @p invalid_value.has_value() is true, K_sum = std::reduce(@p Ks.begin(), @p Ks.end()) values are
 * returned for each key in @p key_list. Among the K_sum values, valid values proceed the invalid
 * values; ordering of the valid values can be arbitrary.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeTypeInputWrapper,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         KeyBucketType const& key_list,
                                         EdgeSrcValueInputWrapper edge_src_value_input,
                                         EdgeDstValueInputWrapper edge_dst_value_input,
                                         EdgeValueInputWrapper edge_value_input,
                                         EdgeOp e_op,
                                         EdgeTypeInputWrapper edge_type_input,
                                         raft::random::RngState& rng_state,
                                         raft::host_span<size_t const> Ks,
                                         bool with_replacement,
                                         std::optional<T> invalid_value,
                                         bool do_expensive_check = false)
{
  return detail::per_v_random_select_transform_e<false>(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_dummy_property_t{}.view(),
    detail::constant_bias_e_op_t<GraphViewType,
                                 detail::edge_endpoint_dummy_property_view_t,
                                 detail::edge_endpoint_dummy_property_view_t,
                                 edge_dummy_property_view_t,
                                 typename KeyBucketType::key_type>{},
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    edge_type_input,
    rng_state,
    Ks,
    with_replacement,
    invalid_value,
    do_expensive_check);
}

/**
 * @brief Randomly select and transform the input (tagged-)vertices' outgoing edges with biases.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam BiasEdgeSrcValueInputWrapper Type of the wrapper for edge source property values (for
 * BiasEdgeOp).
 * @tparam BiasEdgeDstValueInputWrapper Type of the wrapper for edge destination property values
 * (for BiasEdgeOp).
 * @tparam BiasEdgeValueInputWrapper Type of the wrapper for edge property values  (for BiasEdgeOp).
 * @tparam BiasEdgeOp Type of the quinary edge operator to set-up selection bias
 * values.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to sample outgoing
 * edges.
 * @param bias_edge_src_value_input Wrapper used to access source input property values (for the
 * edge sources assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p bias_e_op. Use either cugraph::edge_src_property_t::view() (if @p
 * e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p
 * e_op does not access source property values). Use update_edge_src_property to fill the wrapper.
 * @param bias_edge_dst_value_input Wrapper used to access destination input property values (for
 * the edge destinations assigned to this process in multi-GPU). This parameter is used to pass an
 * edge source property value to @p bias_e_op. Use either cugraph::edge_dst_property_t::view() (if
 * @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param bias_edge_value_input Wrapper used to access edge input property values (for the edges
 * assigned to this process in multi-GPU). This parameter is used to pass an edge source property
 * value to @p bias_e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access
 * edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge
 * property values).
 * @param bias_e_op Quinary operator takes (tagged-)edge source, edge destination, property values
 * for the source, destination, and edge and returns a floating point bias value to be used in
 * biased random selection. The return value should be non-negative. The bias value of 0 indicates
 * that the corresponding edge cannot be selected. Assuming that the return value type is bias_t,
 * the sum of the bias values for any seed vertex should not exceed
 * std::numeric_limits<bias_t>::max().
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). This parameter is used to pass an edge source
 * property value to @p e_op. Use either cugraph::edge_src_property_t::view() (if @p e_op needs to
 * access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p e_op does not
 * access source property values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p e_op. Use either cugraph::edge_dst_property_t::view() (if @p e_op
 * needs to access destination property values) or cugraph::edge_dst_dummy_property_t::view() (if @p
 * e_op does not access destination property values). Use update_edge_dst_property to fill the
 * wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). This parameter is used to pass an edge source property value to @p
 * e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access edge property
 * values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge property
 * values).
 * @param e_op Quinary operator takes (tagged-)edge source, edge destination, property values for
 * the source, destination, and edge and returns a value to be collected in the output. This
 * function is called only for the selected edges.
 * @param K Number of outgoing edges to select per (tagged-)vertex.
 * @param with_replacement A flag to specify whether a single outgoing edge can be selected multiple
 * times (if @p with_replacement = true) or can be selected only once (if @p with_replacement =
 * false).
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the zero out-degree vertices (if @p with_replacement = true) or the vertices
 * with their out-degrees smaller than @p K (if @p with_replacement = false). If @p
 * invalid_value.has_value() is false, fewer than @p K values can be returned for the vertices with
 * fewer than @p K selected edges. See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p key_list.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p key_list.size() * @p K elements). If
 * @p invalid_value.has_value() is true, @p K values are returned for each key in @p key_list. Among
 * the K_sum values, valid values proceed the invalid values; ordering of the valid values can be
 * arbitrary.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename BiasEdgeSrcValueInputWrapper,
          typename BiasEdgeDstValueInputWrapper,
          typename BiasEdgeValueInputWrapper,
          typename BiasEdgeOp,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         KeyBucketType const& key_list,
                                         BiasEdgeSrcValueInputWrapper bias_edge_src_value_input,
                                         BiasEdgeDstValueInputWrapper bias_edge_dst_value_input,
                                         BiasEdgeValueInputWrapper bias_edge_value_input,
                                         BiasEdgeOp bias_e_op,
                                         EdgeSrcValueInputWrapper edge_src_value_input,
                                         EdgeDstValueInputWrapper edge_dst_value_input,
                                         EdgeValueInputWrapper edge_value_input,
                                         EdgeOp e_op,
                                         raft::random::RngState& rng_state,
                                         size_t K,
                                         bool with_replacement,
                                         std::optional<T> invalid_value,
                                         bool do_expensive_check = false)
{
  return detail::per_v_random_select_transform_e<false>(
    handle,
    graph_view,
    key_list,
    bias_edge_src_value_input,
    bias_edge_dst_value_input,
    bias_edge_value_input,
    bias_e_op,
    edge_src_value_input,
    edge_dst_value_input,
    edge_value_input,
    e_op,
    edge_dummy_property_view_t{},
    rng_state,
    raft::host_span<size_t const>(&K, size_t{1}),
    with_replacement,
    invalid_value,
    do_expensive_check);
}

/**
 * @brief Randomly select (per edge type) and transform the input (tagged-)vertices' outgoing edges
 * with biases.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam BiasEdgeSrcValueInputWrapper Type of the wrapper for edge source property values (for
 * BiasEdgeOp).
 * @tparam BiasEdgeDstValueInputWrapper Type of the wrapper for edge destination property values
 * (for BiasEdgeOp).
 * @tparam BiasEdgeValueInputWrapper Type of the wrapper for edge property values  (for BiasEdgeOp).
 * @tparam BiasEdgeOp Type of the quinary edge operator to set-up selection bias
 * values.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam EdgeTypeInputWrapper Type of the wrapper for edge type values.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to sample outgoing
 * edges.
 * @param bias_edge_src_value_input Wrapper used to access source input property values (for the
 * edge sources assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p bias_e_op. Use either cugraph::edge_src_property_t::view() (if @p
 * e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p
 * e_op does not access source property values). Use update_edge_src_property to fill the wrapper.
 * @param bias_edge_dst_value_input Wrapper used to access destination input property values (for
 * the edge destinations assigned to this process in multi-GPU). This parameter is used to pass an
 * edge source property value to @p bias_e_op. Use either cugraph::edge_dst_property_t::view() (if
 * @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param bias_edge_value_input Wrapper used to access edge input property values (for the edges
 * assigned to this process in multi-GPU). This parameter is used to pass an edge source property
 * value to @p bias_e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access
 * edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge
 * property values).
 * @param bias_e_op Quinary operator takes (tagged-)edge source, edge destination, property values
 * for the source, destination, and edge and returns a floating point bias value to be used in
 * biased random selection. The return value should be non-negative. The bias value of 0 indicates
 * that the corresponding edge cannot be selected. Assuming that the return value type is bias_t,
 * the sum of the bias values for any seed vertex should not exceed
 * std::numeric_limits<bias_t>::max().
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). This parameter is used to pass an edge source
 * property value to @p e_op. Use either cugraph::edge_src_property_t::view() (if @p e_op needs to
 * access source property values) or cugraph::edge_src_dummy_property_t::view() (if @p e_op does not
 * access source property values). Use update_edge_src_property to fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). This parameter is used to pass an edge
 * source property value to @p e_op. Use either cugraph::edge_dst_property_t::view() (if @p e_op
 * needs to access destination property values) or cugraph::edge_dst_dummy_property_t::view() (if @p
 * e_op does not access destination property values). Use update_edge_dst_property to fill the
 * wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). This parameter is used to pass an edge source property value to @p
 * e_op. Use either cugraph::edge_property_t::view() (if @p e_op needs to access edge property
 * values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not access edge property
 * values).
 * @param e_op Quinary operator takes (tagged-)edge source, edge destination, property values for
 * the source, destination, and edge and returns a value to be collected in the output. This
 * function is called only for the selected edges.
 * @param edge_type_input Wrapper used to access edge type value (for the edges assigned to this
 * process in multi-GPU). This parameter is used in per-type (heterogeneous) sampling. Use
 * cugraph::edge_property_t::view().
 * @param Ks Number of outgoing edges to select per (tagged-)vertex for each edge type (size = #
 * edge types).
 * @param with_replacement A flag to specify whether a single outgoing edge can be selected multiple
 * times (if @p with_replacement = true) or can be selected only once (if @p with_replacement =
 * false).
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the zero out-degree vertices (if @p with_replacement = true) or the vertices
 * with their out-degrees smaller than @p K (if @p with_replacement = false). If @p
 * invalid_value.has_value() is false, fewer than @p K values can be returned for the vertices with
 * fewer than @p K selected edges. See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p key_list.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p key_list.size() * @p K elements). If
 * @p invalid_value.has_value() is true, K_sum = std::reduce(@p Ks.begin(), @p Ks.end()) values are
 * returned for each key in @p key_list. Among the K_sum values, valid values proceed the invalid
 * values; ordering of the valid values can be arbitrary.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename BiasEdgeSrcValueInputWrapper,
          typename BiasEdgeDstValueInputWrapper,
          typename BiasEdgeValueInputWrapper,
          typename BiasEdgeOp,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename EdgeTypeInputWrapper,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         KeyBucketType const& key_list,
                                         BiasEdgeSrcValueInputWrapper bias_edge_src_value_input,
                                         BiasEdgeDstValueInputWrapper bias_edge_dst_value_input,
                                         BiasEdgeValueInputWrapper bias_edge_value_input,
                                         BiasEdgeOp bias_e_op,
                                         EdgeSrcValueInputWrapper edge_src_value_input,
                                         EdgeDstValueInputWrapper edge_dst_value_input,
                                         EdgeValueInputWrapper edge_value_input,
                                         EdgeOp e_op,
                                         EdgeTypeInputWrapper edge_type_input,
                                         raft::random::RngState& rng_state,
                                         raft::host_span<size_t const> Ks,
                                         bool with_replacement,
                                         std::optional<T> invalid_value,
                                         bool do_expensive_check = false)
{
  return detail::per_v_random_select_transform_e<false>(handle,
                                                        graph_view,
                                                        key_list,
                                                        bias_edge_src_value_input,
                                                        bias_edge_dst_value_input,
                                                        bias_edge_value_input,
                                                        bias_e_op,
                                                        edge_src_value_input,
                                                        edge_dst_value_input,
                                                        edge_value_input,
                                                        e_op,
                                                        edge_type_input,
                                                        rng_state,
                                                        Ks,
                                                        with_replacement,
                                                        invalid_value,
                                                        do_expensive_check);
}

}  // namespace cugraph

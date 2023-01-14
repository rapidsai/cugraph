/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/random/rng.cuh>
#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

#include <cuda/atomic>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>

#include <optional>
#include <tuple>

namespace cugraph {

namespace detail {

// convert a (neighbor index, key index) pair  to a (col_comm_rank, neighbor index, key index)
// triplet, col_comm_rank is set to -1 if an neighbor index is invalid
template <typename edge_t>
struct convert_pair_to_triplet_t {
  raft::device_span<edge_t const> gathered_local_degrees{};
  size_t stride{};
  size_t K{};
  int32_t col_comm_size{};
  edge_t invalid_idx{};

  __device__ thrust::tuple<int32_t, edge_t, size_t> operator()(
    thrust::tuple<edge_t, size_t> index_pair) const
  {
    auto nbr_idx       = thrust::get<0>(index_pair);
    auto key_idx       = thrust::get<1>(index_pair);
    auto local_nbr_idx = nbr_idx;
    int32_t col_comm_rank{-1};
    if (nbr_idx != invalid_idx) {
      col_comm_rank = col_comm_size - 1;
      for (int rank = 0; rank < col_comm_size - 1; ++rank) {
        auto local_degree = gathered_local_degrees[stride * rank + key_idx];
        if (local_nbr_idx < local_degree) {
          col_comm_rank = rank;
          break;
        } else {
          local_nbr_idx -= local_degree;
        }
      }
    }
    return thrust::make_tuple(col_comm_rank, local_nbr_idx, key_idx);
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

template <typename edge_t>
struct invalid_col_comm_rank_t {
  int32_t invalid_col_comm_rank{};
  __device__ bool operator()(thrust::tuple<edge_t, int32_t, size_t> triplet) const
  {
    return thrust::get<1>(triplet) == invalid_col_comm_rank;
  }
};

template <typename GraphViewType,
          typename KeyIterator,
          typename LocalNbrIdxIterator,
          typename OutputValueIterator,
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
  thrust::optional<size_t const*> local_key_indices{thrust::nullopt};
  KeyIterator key_first{};
  LocalNbrIdxIterator local_nbr_idx_first{};
  OutputValueIterator output_value_first{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input;
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input;
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input;
  EdgeOp e_op{};
  edge_t invalid_idx{};
  thrust::optional<T> invalid_value{thrust::nullopt};
  size_t K{};

  __device__ void operator()(size_t i) const
  {
    auto key_idx = local_key_indices ? (*local_key_indices)[i] : (i / K);
    auto key     = *(key_first + key_idx);
    vertex_t major{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      major = key;
    } else {
      major = thrust::get<0>(key);
    }
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
      auto minor        = indices[local_nbr_idx];
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
      *(output_value_first + i) =
        e_op(key_or_src,
             key_or_dst,
             edge_partition_src_value_input.get(src_offset),
             edge_partition_dst_value_input.get(dst_offset),
             edge_partition_e_value_input.get(edge_offset + local_nbr_idx));
    } else if (invalid_value) {
      *(output_value_first + i) = *invalid_value;
    }
  }
};

template <typename edge_t>
struct count_valids_t {
  raft::device_span<edge_t const> sample_local_nbr_indices{};
  size_t K{};
  edge_t invalid_idx{};

  __device__ size_t operator()(size_t i) const
  {
    auto first = sample_local_nbr_indices.begin() + i * K;
    return static_cast<size_t>(
      thrust::distance(first, thrust::find(thrust::seq, first, first + K, invalid_idx)));
  }
};

struct count_t {
  raft::device_span<size_t> sample_counts{};

  __device__ void operator()(size_t key_idx) const
  {
    cuda::std::atomic_ref<size_t> counter(sample_counts[key_idx]);
    counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <typename T, typename Iterator>
struct copy_with_counter_t {
  raft::device_span<size_t> sample_counts{};
  thrust::optional<raft::device_span<size_t const>> sample_offsets{};
  Iterator output_first{};
  size_t K{};

  __device__ void operator()(thrust::tuple<T, size_t> pair) const
  {
    auto e_op_result = thrust::get<0>(pair);
    auto key_idx     = thrust::get<1>(pair);
    cuda::std::atomic_ref<size_t> counter(*(sample_counts.begin() + key_idx));
    auto sample_idx    = counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
    auto output_offset = (sample_offsets ? (*sample_offsets)[key_idx] : key_idx * K) + sample_idx;
    *(output_first + output_offset) = e_op_result;
  }
};

template <bool incoming,
          typename GraphViewType,
          typename VertexFrontierBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>,
           decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>
per_v_random_select_transform_e(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexFrontierBucketType const& frontier,
                                EdgeSrcValueInputWrapper edge_src_value_input,
                                EdgeDstValueInputWrapper edge_dst_value_input,
                                EdgeValueInputWrapper edge_value_input,
                                EdgeOp e_op,
                                raft::random::RngState& rng_state,
                                size_t K,
                                bool with_replacement,
                                std::optional<T> invalid_value,
                                bool do_expensive_check)
{
#ifndef NO_CUGRAPH_OPS
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename VertexFrontierBucketType::key_type;
  using key_buffer_t =
    decltype(allocate_dataframe_buffer<key_t>(size_t{0}, rmm::cuda_stream_view{}));

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>,
                       edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>,
                       edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator>>;

  static_assert(GraphViewType::is_storage_transposed == incoming);
  static_assert(std::is_same_v<
                typename detail::edge_op_result_type<key_t,
                                                     vertex_t,
                                                     typename EdgeSrcValueInputWrapper::value_type,
                                                     typename EdgeDstValueInputWrapper::value_type,
                                                     typename EdgeValueInputWrapper::value_type,
                                                     EdgeOp>::type,
                T>);

  CUGRAPH_EXPECTS(K >= size_t{1},
                  "Invalid input argument: invalid K, K should be a positive integer.");

  auto col_comm_size =
    GraphViewType::is_multi_gpu
      ? handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name()).get_size()
      : int{1};

  if (do_expensive_check) {
    // FIXME: better re-factor this check function?
    vertex_t const* frontier_vertex_first{nullptr};
    vertex_t const* frontier_vertex_last{nullptr};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      frontier_vertex_first = frontier.begin();
      frontier_vertex_last  = frontier.end();
    } else {
      frontier_vertex_first = thrust::get<0>(frontier.begin().get_iterator_tuple());
      frontier_vertex_last  = thrust::get<0>(frontier.end().get_iterator_tuple());
    }
    auto num_invalid_keys =
      frontier.size() -
      thrust::count_if(handle.get_thrust_policy(),
                       frontier_vertex_first,
                       frontier_vertex_last,
                       check_in_range_t<vertex_t>{graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last()});
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_keys = host_scalar_allreduce(
        handle.get_comms(), num_invalid_keys, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_keys == size_t{0},
                    "Invalid input argument: frontier includes out-of-range keys.");
  }

  auto frontier_key_first = frontier.begin();
  auto frontier_key_last  = frontier.end();

  std::vector<size_t> local_frontier_sizes{};
  if (col_comm_size > 1) {
    auto& col_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    local_frontier_sizes = host_scalar_allgather(
      col_comm,
      static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
      handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(
      static_cast<vertex_t>(thrust::distance(frontier_key_first, frontier_key_last)))};
  }
  std::vector<size_t> local_frontier_displacements(local_frontier_sizes.size());
  std::exclusive_scan(local_frontier_sizes.begin(),
                      local_frontier_sizes.end(),
                      local_frontier_displacements.begin(),
                      size_t{0});

  // 1. aggregate frontier

  auto aggregate_local_frontier_keys =
    (col_comm_size > 1)
      ? std::make_optional<key_buffer_t>(
          local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream())
      : std::nullopt;
  if (col_comm_size > 1) {
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    device_allgatherv(col_comm,
                      frontier_key_first,
                      get_dataframe_buffer_begin(*aggregate_local_frontier_keys),
                      local_frontier_sizes,
                      local_frontier_displacements,
                      handle.get_stream());
  }

  // 2. compute degrees

  auto aggregate_local_frontier_local_degrees =
    (col_comm_size > 1)
      ? std::make_optional<rmm::device_uvector<edge_t>>(
          local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream())
      : std::nullopt;
  rmm::device_uvector<edge_t> frontier_degrees(frontier.size(), handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    vertex_t const* edge_partition_frontier_major_first{nullptr};

    auto edge_partition_frontier_key_first =
      ((col_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier_keys)
                           : frontier_key_first) +
      local_frontier_displacements[i];
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      edge_partition_frontier_major_first = edge_partition_frontier_key_first;
    } else {
      edge_partition_frontier_major_first = thrust::get<0>(edge_partition_frontier_key_first);
    }

    auto edge_partition_frontier_local_degrees = edge_partition.compute_local_degrees(
      raft::device_span<vertex_t const>(edge_partition_frontier_major_first,
                                        local_frontier_sizes[i]),
      handle.get_stream());

    if (col_comm_size > 1) {
      // FIXME: this copy is unnecessary if edge_partition.compute_local_degrees() takes a pointer
      // to the output array
      thrust::copy(
        handle.get_thrust_policy(),
        edge_partition_frontier_local_degrees.begin(),
        edge_partition_frontier_local_degrees.end(),
        (*aggregate_local_frontier_local_degrees).begin() + local_frontier_displacements[i]);
    } else {
      frontier_degrees = std::move(edge_partition_frontier_local_degrees);
    }
  }

  auto frontier_gathered_local_degrees =
    (col_comm_size > 1)
      ? std::make_optional<rmm::device_uvector<edge_t>>(size_t{0}, handle.get_stream())
      : std::nullopt;
  if (col_comm_size > 1) {
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

    std::tie(frontier_gathered_local_degrees, std::ignore) =
      shuffle_values(col_comm,
                     (*aggregate_local_frontier_local_degrees).begin(),
                     local_frontier_sizes,
                     handle.get_stream());
    thrust::tabulate(handle.get_thrust_policy(),
                     frontier_degrees.begin(),
                     frontier_degrees.end(),
                     strided_sum_t<edge_t>{(*frontier_gathered_local_degrees).data(),
                                           frontier.size(),
                                           static_cast<size_t>(col_comm_size)});
    aggregate_local_frontier_local_degrees = std::nullopt;
  }

  // 3. randomly select neighbor indices

  rmm::device_uvector<edge_t> sample_nbr_indices(frontier.size() * K, handle.get_stream());
  // FIXME: get_sampling_index is inefficient when degree >> K & with_replacement = false
  if (frontier_degrees.size() > 0) {
    cugraph::ops::gnn::graph::get_sampling_index(sample_nbr_indices.data(),
                                                 rng_state,
                                                 frontier_degrees.data(),
                                                 static_cast<edge_t>(frontier_degrees.size()),
                                                 static_cast<int32_t>(K),
                                                 with_replacement,
                                                 handle.get_stream());
  }
  frontier_degrees.resize(0, handle.get_stream());
  frontier_degrees.shrink_to_fit(handle.get_stream());

  // 4. shuffle randomly selected indices

  auto sample_local_nbr_indices = std::move(
    sample_nbr_indices);  // neighbor index within an edge partition (note that each vertex's
                          // neighbors are distributed in col_comm_size partitions)
  std::optional<rmm::device_uvector<size_t>> sample_key_indices{
    std::nullopt};  // relevant only when (col_comm_size > 1)
  auto local_frontier_sample_counts        = std::vector<size_t>{};
  auto local_frontier_sample_displacements = std::vector<size_t>{};
  if (col_comm_size > 1) {
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

    sample_key_indices =
      rmm::device_uvector<size_t>(sample_local_nbr_indices.size(), handle.get_stream());
    auto col_comm_ranks =
      rmm::device_uvector<int32_t>(sample_local_nbr_indices.size(), handle.get_stream());
    auto input_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(),
                         thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                                         divider_t<size_t>{K})));
    thrust::transform(
      handle.get_thrust_policy(),
      input_pair_first,
      input_pair_first + sample_local_nbr_indices.size(),
      thrust::make_zip_iterator(thrust::make_tuple(
        col_comm_ranks.begin(), sample_local_nbr_indices.begin(), (*sample_key_indices).begin())),
      convert_pair_to_triplet_t<edge_t>{
        raft::device_span<edge_t const>((*frontier_gathered_local_degrees).data(),
                                        (*frontier_gathered_local_degrees).size()),
        frontier.size(),
        K,
        col_comm_size,
        cugraph::ops::gnn::graph::INVALID_ID<edge_t>});

    frontier_gathered_local_degrees = std::nullopt;

    auto triplet_first = thrust::make_zip_iterator(thrust::make_tuple(
      sample_local_nbr_indices.begin(), col_comm_ranks.begin(), (*sample_key_indices).begin()));
    sample_local_nbr_indices.resize(
      thrust::distance(triplet_first,
                       thrust::remove_if(handle.get_thrust_policy(),
                                         triplet_first,
                                         triplet_first + sample_local_nbr_indices.size(),
                                         invalid_col_comm_rank_t<edge_t>{int32_t{-1}})),
      handle.get_stream());
    col_comm_ranks.resize(sample_local_nbr_indices.size(), handle.get_stream());
    (*sample_key_indices).resize(sample_local_nbr_indices.size(), handle.get_stream());

    auto d_tx_counts =
      groupby_and_count(col_comm_ranks.begin(),
                        col_comm_ranks.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                          sample_local_nbr_indices.begin(), (*sample_key_indices).begin())),
                        thrust::identity<int32_t>{},
                        col_comm_size,
                        std::numeric_limits<size_t>::max(),
                        handle.get_stream());

    std::vector<size_t> h_tx_counts(d_tx_counts.size());
    raft::update_host(
      h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
    handle.sync_stream();

    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(), (*sample_key_indices).begin()));
    auto [rx_value_buffer, rx_counts] =
      shuffle_values(col_comm, pair_first, h_tx_counts, handle.get_stream());

    sample_local_nbr_indices            = std::move(std::get<0>(rx_value_buffer));
    sample_key_indices                  = std::move(std::get<1>(rx_value_buffer));
    local_frontier_sample_displacements = std::vector<size_t>(rx_counts.size());
    std::exclusive_scan(
      rx_counts.begin(), rx_counts.end(), local_frontier_sample_displacements.begin(), size_t{0});
    local_frontier_sample_counts = std::move(rx_counts);
  } else {
    local_frontier_sample_counts.push_back(frontier.size() * K);
    local_frontier_sample_displacements.push_back(size_t{0});
  }

  // 5. transform

  auto sample_e_op_results = allocate_dataframe_buffer<T>(
    local_frontier_sample_displacements.back() + local_frontier_sample_counts.back(),
    handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto edge_partition_frontier_key_first =
      ((col_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier_keys)
                           : frontier_key_first) +
      local_frontier_displacements[i];
    auto edge_partition_sample_local_nbr_index_first =
      sample_local_nbr_indices.begin() + local_frontier_sample_displacements[i];

    auto edge_partition_sample_e_op_result_first =
      get_dataframe_buffer_begin(sample_e_op_results) + local_frontier_sample_displacements[i];

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

    if (col_comm_size > 1) {
      auto edge_partition_sample_key_index_first =
        (*sample_key_indices).begin() + local_frontier_sample_displacements[i];
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(local_frontier_sample_counts[i]),
        transform_local_nbr_indices_t<GraphViewType,
                                      decltype(edge_partition_frontier_key_first),
                                      decltype(edge_partition_sample_local_nbr_index_first),
                                      decltype(edge_partition_sample_e_op_result_first),
                                      edge_partition_src_input_device_view_t,
                                      edge_partition_dst_input_device_view_t,
                                      edge_partition_e_input_device_view_t,
                                      EdgeOp,
                                      T>{
          edge_partition,
          thrust::make_optional(edge_partition_sample_key_index_first),
          edge_partition_frontier_key_first,
          edge_partition_sample_local_nbr_index_first,
          edge_partition_sample_e_op_result_first,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          e_op,
          cugraph::ops::gnn::graph::INVALID_ID<edge_t>,
          to_thrust_optional(invalid_value),
          K});
    } else {
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(frontier.size() * K),
        transform_local_nbr_indices_t<GraphViewType,
                                      decltype(edge_partition_frontier_key_first),
                                      decltype(edge_partition_sample_local_nbr_index_first),
                                      decltype(edge_partition_sample_e_op_result_first),
                                      edge_partition_src_input_device_view_t,
                                      edge_partition_dst_input_device_view_t,
                                      edge_partition_e_input_device_view_t,
                                      EdgeOp,
                                      T>{edge_partition,
                                         thrust::nullopt,
                                         edge_partition_frontier_key_first,
                                         edge_partition_sample_local_nbr_index_first,
                                         edge_partition_sample_e_op_result_first,
                                         edge_partition_src_value_input,
                                         edge_partition_dst_value_input,
                                         edge_partition_e_value_input,
                                         e_op,
                                         cugraph::ops::gnn::graph::INVALID_ID<edge_t>,
                                         to_thrust_optional(invalid_value),
                                         K});
    }
  }

  // 6. shuffle randomly selected & transformed results and update sample_offsets

  auto sample_offsets = invalid_value ? std::nullopt
                                      : std::make_optional<rmm::device_uvector<size_t>>(
                                          frontier.size() + 1, handle.get_stream());
  if (col_comm_size > 1) {
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

    std::tie(sample_e_op_results, std::ignore) =
      shuffle_values(col_comm,
                     get_dataframe_buffer_begin(sample_e_op_results),
                     local_frontier_sample_counts,
                     handle.get_stream());
    std::tie(sample_key_indices, std::ignore) = shuffle_values(
      col_comm, (*sample_key_indices).begin(), local_frontier_sample_counts, handle.get_stream());

    rmm::device_uvector<size_t> sample_counts(frontier.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), sample_counts.begin(), sample_counts.end(), size_t{0});
    auto input_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      get_dataframe_buffer_begin(sample_e_op_results), (*sample_key_indices).begin()));
    if (invalid_value) {
      auto tmp_sample_e_op_results =
        allocate_dataframe_buffer<T>(frontier.size() * K, handle.get_stream());
      auto tmp_sample_e_op_result_first = get_dataframe_buffer_begin(tmp_sample_e_op_results);
      thrust::fill(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(tmp_sample_e_op_results),
                   get_dataframe_buffer_end(tmp_sample_e_op_results),
                   *invalid_value);
      thrust::for_each(handle.get_thrust_policy(),
                       input_pair_first,
                       input_pair_first + size_dataframe_buffer(sample_e_op_results),
                       copy_with_counter_t<T, decltype(tmp_sample_e_op_result_first)>{
                         raft::device_span<size_t>(sample_counts.data(), sample_counts.size()),
                         thrust::nullopt,
                         tmp_sample_e_op_result_first,
                         K});
      sample_e_op_results = std::move(tmp_sample_e_op_results);
    } else {
      thrust::for_each(
        handle.get_thrust_policy(),
        (*sample_key_indices).begin(),
        (*sample_key_indices).end(),
        count_t{raft::device_span<size_t>(sample_counts.data(), sample_counts.size())});
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             sample_counts.begin(),
                             sample_counts.end(),
                             (*sample_offsets).begin() + 1);
      auto tmp_sample_e_op_results = allocate_dataframe_buffer<T>(
        (*sample_offsets).back_element(handle.get_stream()), handle.get_stream());
      auto tmp_sample_e_op_result_first = get_dataframe_buffer_begin(tmp_sample_e_op_results);
      thrust::fill(
        handle.get_thrust_policy(), sample_counts.begin(), sample_counts.end(), size_t{0});
      thrust::for_each(
        handle.get_thrust_policy(),
        input_pair_first,
        input_pair_first + size_dataframe_buffer(sample_e_op_results),
        copy_with_counter_t<T, decltype(tmp_sample_e_op_result_first)>{
          raft::device_span<size_t>(sample_counts.data(), sample_counts.size()),
          raft::device_span<size_t const>((*sample_offsets).data(), (*sample_offsets).size()),
          tmp_sample_e_op_result_first,
          K});
      sample_e_op_results = std::move(tmp_sample_e_op_results);
    }
  } else {
    if (!invalid_value) {
      rmm::device_uvector<size_t> sample_counts(frontier.size(), handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        sample_counts.begin(),
        sample_counts.end(),
        count_valids_t<edge_t>{raft::device_span<edge_t const>(sample_local_nbr_indices.data(),
                                                               sample_local_nbr_indices.size()),
                               K,
                               cugraph::ops::gnn::graph::INVALID_ID<edge_t>});
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             sample_counts.begin(),
                             sample_counts.end(),
                             (*sample_offsets).begin() + 1);
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        sample_local_nbr_indices.begin(), get_dataframe_buffer_begin(sample_e_op_results)));
      resize_dataframe_buffer(
        sample_e_op_results,
        thrust::distance(pair_first,
                         thrust::remove_if(handle.get_thrust_policy(),
                                           pair_first,
                                           pair_first + sample_local_nbr_indices.size(),
                                           check_invalid_t<edge_t, T>{
                                             cugraph::ops::gnn::graph::INVALID_ID<edge_t>})),
        handle.get_stream());
      shrink_to_fit_dataframe_buffer(sample_e_op_results, handle.get_stream());
    }
  }

  return std::make_tuple(std::move(sample_offsets), std::move(sample_e_op_results));
#else
  CUGRAPH_FAIL("unimplemented.");
#endif
}

}  // namespace detail

/**
 * @brief Randomly select and transform the input (tagged-)vertices' outgoing edges with biases.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexFrontierBucketType Type of the vertex frontier bucket class which abstracts the
 * current (tagged-)vertex frontier.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeBiasOp Type of the quaternary (or quinary) edge operator to set-up selection bias
 * values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontierBucketType class object to store the (tagged-)vertex list to sample
 * outgoing edges.
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
 * @param e_bias_op Quinary operator takes edge source, edge destination, property values for the
 * source, destination, and edge and returns a floating point bias value to be used in biased random
 * selection.
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be collected in the output. This function is called
 * only for the selected edges.
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
 * valid and has the size of @p frontier.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p frontier.size() * @p K elements).
 */
template <typename GraphViewType,
          typename VertexFrontierBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeBiasOp,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>,
           decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         VertexFrontierBucketType const& frontier,
                                         EdgeSrcValueInputWrapper edge_src_value_input,
                                         EdgeDstValueInputWrapper edge_dst_value_input,
                                         EdgeValueInputWrapper egde_value_input,
                                         EdgeBiasOp e_bias_op,
                                         EdgeOp e_op,
                                         raft::random::RngState& rng_state,
                                         size_t K,
                                         bool with_replacement,
                                         std::optional<T> invalid_value,
                                         bool do_expensive_check = false)
{
  CUGRAPH_FAIL("unimplemented.");

  return std::make_tuple(std::nullopt,
                         allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}));
}

/**
 * @brief Randomly select and transform the input (tagged-)vertices' outgoing edges.
 *
 * This function assumes that every outgoing edge of a given vertex has the same odd to be selected
 * (uniform neighbor sampling).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexFrontierBucketType Type of the vertex frontier bucket class which abstracts the
 * current (tagged-)vertex frontier.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontierBucketType class object to store the (tagged-)vertex list to sample
 * outgoing edges.
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
 * destination, and edge and returns a value to be collected in the output. This function is called
 * only for the selected edges.
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
 * valid and has the size of @p frontier.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p frontier.size() * @p K elements).
 */
template <typename GraphViewType,
          typename VertexFrontierBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>,
           decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         VertexFrontierBucketType const& frontier,
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
  return detail::per_v_random_select_transform_e<false>(handle,
                                                        graph_view,
                                                        frontier,
                                                        edge_src_value_input,
                                                        edge_dst_value_input,
                                                        edge_value_input,
                                                        e_op,
                                                        rng_state,
                                                        K,
                                                        with_replacement,
                                                        invalid_value,
                                                        do_expensive_check);
}

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/graph.hpp>
#include <cugraph/prims/detail/sample_and_compute_local_nbr_indices.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <cassert>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace detail {
namespace per_v_random_select_transform_refactor {

/**
 * Skeleton for splitting per_v_random_select_transform_e into smaller units.
 *
 * This file is intentionally not wired into existing call paths. The goal is to
 * provide a side-by-side scaffold for refactoring without modifying behavior in
 * the current implementation.
 */

enum class sampling_bias_mode_t { k_uniform, k_biased };
enum class sampling_type_mode_t { k_homogeneous, k_heterogeneous };

// -----------------------------------------------------------------------------
// Phase A: Validation + key aggregation
//
// Intended template dependence:
// - GraphViewType, KeyBucketType
// Not needed here:
// - BiasEdgeOp, EdgeOp, output property wrappers
//
// Returns tuple:
// - [0] optional aggregate_local_key_list (set if minor_comm_size > 1)
// - [1] local_key_list_offsets
// -----------------------------------------------------------------------------
template <typename GraphViewType, typename KeyBucketType>
std::tuple<std::optional<dataframe_buffer_type_t<typename KeyBucketType::key_type>>,
           std::vector<size_t>>
prepare_keys(raft::handle_t const& handle,
             GraphViewType const& graph_view,
             KeyBucketType const& key_list,
             raft::host_span<size_t const> Ks,
             bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using key_t    = typename KeyBucketType::key_type;

  CUGRAPH_EXPECTS(Ks.size() >= 1, "Invalid input argument: Ks.size() should be 1 or larger.");

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

  std::optional<dataframe_buffer_type_t<key_t>> aggregate_local_key_list{std::nullopt};
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

  return std::make_tuple(std::move(aggregate_local_key_list), std::move(local_key_list_offsets));
}

// -----------------------------------------------------------------------------
// Phase B: Sampling local neighbor indices
//
// Intended template dependence:
// - GraphViewType, key iterator/value type, bias/type mode-related wrappers
// Not needed here:
// - EdgeOp (transform op)
//
// Returns tuple:
// - [0] sample_local_nbr_indices
// - [1] optional sample_key_indices
// - [2] local_key_list_sample_offsets
// - [3] K_sum
// -----------------------------------------------------------------------------
template <typename GraphViewType, typename KeyIterator, typename EdgeTypeInputWrapper>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>,
           size_t>
sample_uniform_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_key_first,
  EdgeTypeInputWrapper edge_type_input,
  std::tuple<std::optional<
               dataframe_buffer_type_t<typename thrust::iterator_traits<KeyIterator>::value_type>>,
             std::vector<size_t>> const& prepared_keys,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks,
  bool with_replacement,
  sampling_type_mode_t type_mode);

template <typename GraphViewType,
          typename KeyIterator,
          typename BiasEdgeSrcValueInputWrapper,
          typename BiasEdgeDstValueInputWrapper,
          typename BiasEdgeValueInputWrapper,
          typename BiasEdgeOp,
          typename EdgeTypeInputWrapper>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>,
           size_t>
sample_biased_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_key_first,
  BiasEdgeSrcValueInputWrapper bias_edge_src_value_input,
  BiasEdgeDstValueInputWrapper bias_edge_dst_value_input,
  BiasEdgeValueInputWrapper bias_edge_value_input,
  BiasEdgeOp bias_e_op,
  EdgeTypeInputWrapper edge_type_input,
  std::tuple<std::optional<
               dataframe_buffer_type_t<typename thrust::iterator_traits<KeyIterator>::value_type>>,
             std::vector<size_t>> const& prepared_keys,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks,
  bool with_replacement,
  bool do_expensive_check,
  sampling_type_mode_t type_mode);

// -----------------------------------------------------------------------------
// Phase C: Transform sampled indices into result values
//
// Intended template dependence:
// - EdgeOp + property wrappers + T (this is where most e_op-specific codegen lives)
//
// Returns tuple:
// - [0] sample_e_op_results
// -----------------------------------------------------------------------------
template <bool incoming,
          typename GraphViewType,
          typename KeyIterator,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<dataframe_buffer_type_t<T>> apply_transform(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_key_first,
  std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
             std::optional<rmm::device_uvector<size_t>>,
             std::vector<size_t>,
             size_t> const& sampled_indices,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  EdgeOp e_op,
  std::optional<T> invalid_value);

// -----------------------------------------------------------------------------
// Phase D: Shuffle/finalize offsets — implemented in finalize_random_select_output.cuh.
//
// Intended template dependence:
// - edge_t, T, comm topology
// Not needed here:
// - EdgeOp, BiasEdgeOp, property wrapper types
//
// Returns tuple:
// - [0] optional sample_offsets
// - [1] sample_e_op_results
// -----------------------------------------------------------------------------
template <typename edge_t, typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>> finalize_output(
  raft::handle_t const& handle,
  std::tuple<rmm::device_uvector<edge_t>,
             std::optional<rmm::device_uvector<size_t>>,
             std::vector<size_t>,
             size_t>&& sampled_indices,
  std::tuple<dataframe_buffer_type_t<T>>&& transformed_values,
  size_t key_list_size,
  std::optional<T> invalid_value);

// -----------------------------------------------------------------------------
// Optional orchestrator skeleton
//
// This intentionally has no definition yet; it documents the target API shape
// for a future split implementation. The existing implementation remains the
// source of truth during migration.
// -----------------------------------------------------------------------------
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
per_v_random_select_transform_e_refactor_entry(
  raft::handle_t const& handle,
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
  using edge_t = typename GraphViewType::edge_type;
  using key_t  = typename KeyBucketType::key_type;

  auto prepared_keys = prepare_keys(handle, graph_view, key_list, Ks, do_expensive_check);

  constexpr auto type_mode = std::is_same_v<EdgeTypeInputWrapper, edge_dummy_property_view_t>
                               ? sampling_type_mode_t::k_homogeneous
                               : sampling_type_mode_t::k_heterogeneous;

  auto sampled_indices = [&]() {
    auto maybe_aggregate_local_key_list = std::get<0>(prepared_keys);

    if constexpr (std::is_same_v<BiasEdgeOp,
                                 constant_bias_e_op_t<GraphViewType,
                                                      BiasEdgeSrcValueInputWrapper,
                                                      BiasEdgeDstValueInputWrapper,
                                                      BiasEdgeValueInputWrapper,
                                                      key_t>>) {
      if (maybe_aggregate_local_key_list) {
        return sample_uniform_indices(handle,
                                      graph_view,
                                      get_dataframe_buffer_cbegin(*maybe_aggregate_local_key_list),
                                      edge_type_input,
                                      prepared_keys,
                                      rng_state,
                                      Ks,
                                      with_replacement,
                                      type_mode);
      } else {
        return sample_uniform_indices(handle,
                                      graph_view,
                                      key_list.begin(),
                                      edge_type_input,
                                      prepared_keys,
                                      rng_state,
                                      Ks,
                                      with_replacement,
                                      type_mode);
      }
    } else {
      if (maybe_aggregate_local_key_list) {
        return sample_biased_indices(handle,
                                     graph_view,
                                     get_dataframe_buffer_cbegin(*maybe_aggregate_local_key_list),
                                     bias_edge_src_value_input,
                                     bias_edge_dst_value_input,
                                     bias_edge_value_input,
                                     bias_e_op,
                                     edge_type_input,
                                     prepared_keys,
                                     rng_state,
                                     Ks,
                                     with_replacement,
                                     do_expensive_check,
                                     type_mode);
      } else {
        return sample_biased_indices(handle,
                                     graph_view,
                                     key_list.begin(),
                                     bias_edge_src_value_input,
                                     bias_edge_dst_value_input,
                                     bias_edge_value_input,
                                     bias_e_op,
                                     edge_type_input,
                                     prepared_keys,
                                     rng_state,
                                     Ks,
                                     with_replacement,
                                     do_expensive_check,
                                     type_mode);
      }
    }
  }();

  auto transformed_values = [&]() {
    auto maybe_aggregate_local_key_list = std::get<0>(prepared_keys);
    if (maybe_aggregate_local_key_list) {
      return apply_transform<incoming>(handle,
                                       graph_view,
                                       get_dataframe_buffer_cbegin(*maybe_aggregate_local_key_list),
                                       sampled_indices,
                                       edge_src_value_input,
                                       edge_dst_value_input,
                                       edge_value_input,
                                       e_op,
                                       invalid_value);
    } else {
      return apply_transform<incoming>(handle,
                                       graph_view,
                                       key_list.begin(),
                                       sampled_indices,
                                       edge_src_value_input,
                                       edge_dst_value_input,
                                       edge_value_input,
                                       e_op,
                                       invalid_value);
    }
  }();

  return finalize_output<edge_t, T>(handle,
                                    std::move(sampled_indices),
                                    std::move(transformed_values),
                                    key_list.size(),
                                    invalid_value);
}

}  // namespace per_v_random_select_transform_refactor
}  // namespace detail
}  // namespace cugraph

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

#include "cugraph/utilities/shuffle_comm.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"
#include "utilities/tuple_with_optionals_dispatching.hpp"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <optional>

namespace cugraph {
namespace detail {

template <typename label_t>
struct shuffle_to_output_comm_rank_t {
  raft::device_span<int32_t const> output_rank_;

  template <typename key_t>
  __device__ int32_t operator()(key_t key) const
  {
    return output_rank_[key];
  }
};

template <typename vertex_t,
          typename weight_t,
          typename edge_t,
          typename edge_type_t,
          typename edge_time_t,
          typename label_t>
void sort_sampled_tuples(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& majors,
                         rmm::device_uvector<vertex_t>& minors,
                         std::optional<rmm::device_uvector<weight_t>>& weights,
                         std::optional<rmm::device_uvector<edge_t>>& edge_ids,
                         std::optional<rmm::device_uvector<edge_type_t>>& edge_types,
                         std::optional<rmm::device_uvector<edge_time_t>>& edge_start_times,
                         std::optional<rmm::device_uvector<edge_time_t>>& edge_end_times,
                         std::optional<rmm::device_uvector<int32_t>>& hops,
                         rmm::device_uvector<label_t>& labels)
{
  rmm::device_uvector<size_t> indices(majors.size(), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
  rmm::device_uvector<label_t> tmp_labels(indices.size(), handle.get_stream());

  if (hops) {
    rmm::device_uvector<int32_t> tmp_hops(indices.size(), handle.get_stream());

    thrust::sort(
      handle.get_thrust_policy(),
      indices.begin(),
      indices.end(),
      [labels = raft::device_span<label_t const>(labels.data(), labels.size()),
       hops   = raft::device_span<int32_t const>(hops->data(), hops->size())] __device__(size_t l,
                                                                                       size_t r) {
        return thrust::make_tuple(labels[l], hops[l]) < thrust::make_tuple(labels[r], hops[r]);
      });

    thrust::gather(handle.get_thrust_policy(),
                   indices.begin(),
                   indices.end(),
                   thrust::make_zip_iterator(labels.begin(), hops->begin()),
                   thrust::make_zip_iterator(tmp_labels.begin(), tmp_hops.begin()));
    hops = std::move(tmp_hops);
  } else {
    thrust::sort(
      handle.get_thrust_policy(),
      indices.begin(),
      indices.end(),
      [labels = raft::device_span<label_t const>(labels.data(), labels.size())] __device__(
        size_t l, size_t r) { return labels[l] < labels[r]; });
    thrust::gather(handle.get_thrust_policy(),
                   indices.begin(),
                   indices.end(),
                   labels.begin(),
                   tmp_labels.begin());
  }
  labels = std::move(tmp_labels);

  rmm::device_uvector<vertex_t> tmp_majors(indices.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> tmp_minors(indices.size(), handle.get_stream());
  thrust::gather(handle.get_thrust_policy(),
                 indices.begin(),
                 indices.end(),
                 thrust::make_zip_iterator(majors.begin(), minors.begin()),
                 thrust::make_zip_iterator(tmp_majors.begin(), tmp_minors.begin()));
  majors = std::move(tmp_majors);
  minors = std::move(tmp_minors);

  if (weights) {
    rmm::device_uvector<weight_t> tmp_weights(indices.size(), handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   indices.begin(),
                   indices.end(),
                   weights->begin(),
                   tmp_weights.begin());
    weights = std::move(tmp_weights);
  }

  if (edge_ids) {
    rmm::device_uvector<edge_t> tmp_edge_ids(indices.size(), handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   indices.begin(),
                   indices.end(),
                   edge_ids->begin(),
                   tmp_edge_ids.begin());
    edge_ids = std::move(tmp_edge_ids);
  }

  if (edge_types) {
    rmm::device_uvector<edge_type_t> tmp_edge_types(indices.size(), handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   indices.begin(),
                   indices.end(),
                   edge_types->begin(),
                   tmp_edge_types.begin());
    edge_types = std::move(tmp_edge_types);
  }

  if (edge_start_times) {
    rmm::device_uvector<edge_time_t> tmp_edge_start_times(indices.size(), handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   indices.begin(),
                   indices.end(),
                   edge_start_times->begin(),
                   tmp_edge_start_times.begin());
    edge_start_times = std::move(tmp_edge_start_times);
  }

  if (edge_end_times) {
    rmm::device_uvector<edge_time_t> tmp_edge_end_times(indices.size(), handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   indices.begin(),
                   indices.end(),
                   edge_end_times->begin(),
                   tmp_edge_end_times.begin());
    edge_end_times = std::move(tmp_edge_end_times);
  }
}

template <typename KeyIterator, typename KeyToGroupIdOp>
struct groupby_and_count_functor_t {
  raft::handle_t const& handle;
  KeyIterator first;
  KeyIterator last;
  KeyToGroupIdOp key_to_group_id_op;
  int comm_size;
  size_t mem_frugal_threshold;

  template <bool... Flags, typename TupleType>
  auto operator()(TupleType tup)
  {
    if constexpr (std::is_same_v<TupleType, std::tuple<>>) {
      return cugraph::groupby_and_count(
        first, last, key_to_group_id_op, comm_size, mem_frugal_threshold, handle.get_stream());
    } else {
      return cugraph::groupby_and_count(first,
                                        last,
                                        get_dataframe_buffer_begin(tup),
                                        key_to_group_id_op,
                                        comm_size,
                                        mem_frugal_threshold,
                                        handle.get_stream());
    }
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<label_t>>&& labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank)
{
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

  if (labels) {
    sort_sampled_tuples(handle,
                        majors,
                        minors,
                        weights,
                        edge_ids,
                        edge_types,
                        edge_start_times,
                        edge_end_times,
                        hops,
                        *labels);

    if (label_to_output_comm_rank) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      auto total_global_mem = handle.get_device_properties().totalGlobalMem;
      auto element_size     = sizeof(vertex_t) * 2 + (weights ? sizeof(weight_t) : size_t{0}) +
                          (edge_ids ? sizeof(edge_t) : size_t{0}) +
                          (edge_types ? sizeof(edge_type_t) : size_t{0}) +
                          (hops ? sizeof(int32_t) : size_t{0}) + sizeof(label_t);

      auto constexpr mem_frugal_ratio =
        0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
      // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
      // group-by by default, and thrust::sort requires temporary buffer comparable to the input
      // data size)
      auto mem_frugal_threshold = static_cast<size_t>(
        static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

      groupby_and_count_functor_t<label_t*, shuffle_to_output_comm_rank_t<label_t>>
        groupby_and_count_functor{
          handle,
          labels->begin(),
          labels->end(),
          shuffle_to_output_comm_rank_t<label_t>{*label_to_output_comm_rank},
          comm_size,
          mem_frugal_threshold};

      auto d_tx_value_counts = tuple_with_optionals_dispatch(
        groupby_and_count_functor,
        std::make_optional<raft::device_span<vertex_t>>(majors.data(), majors.size()),
        std::make_optional<raft::device_span<vertex_t>>(minors.data(), minors.size()),
        weights ? std::make_optional<raft::device_span<weight_t>>(weights->data(), weights->size())
                : std::nullopt,
        edge_ids ? std::make_optional<raft::device_span<edge_t>>(edge_ids->data(), edge_ids->size())
                 : std::nullopt,
        edge_types ? std::make_optional<raft::device_span<edge_type_t>>(edge_types->data(),
                                                                        edge_types->size())
                   : std::nullopt,
        edge_start_times ? std::make_optional<raft::device_span<edge_time_t>>(
                             edge_start_times->data(), edge_start_times->size())
                         : std::nullopt,
        edge_end_times ? std::make_optional<raft::device_span<edge_time_t>>(edge_end_times->data(),
                                                                            edge_end_times->size())
                       : std::nullopt,
        hops ? std::make_optional<raft::device_span<int32_t>>(hops->data(), hops->size())
             : std::nullopt);

      std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
      raft::update_host(h_tx_value_counts.data(),
                        d_tx_value_counts.data(),
                        d_tx_value_counts.size(),
                        handle.get_stream());
      handle.sync_stream();

      raft::host_span<size_t const> h_tx_value_counts_span{h_tx_value_counts.data(),
                                                           h_tx_value_counts.size()};

      std::forward_as_tuple(std::tie(majors, minors, labels), std::ignore) =
        shuffle_values(comm,
                       thrust::make_zip_iterator(majors.begin(), minors.begin(), labels->begin()),
                       h_tx_value_counts_span,
                       handle.get_stream());

      if (weights)
        std::tie(weights, std::ignore) =
          shuffle_values(comm, weights->begin(), h_tx_value_counts_span, handle.get_stream());

      if (edge_ids)
        std::tie(edge_ids, std::ignore) =
          shuffle_values(comm, edge_ids->begin(), h_tx_value_counts_span, handle.get_stream());

      if (edge_types)
        std::tie(edge_types, std::ignore) =
          shuffle_values(comm, edge_types->begin(), h_tx_value_counts_span, handle.get_stream());

      if (edge_start_times)
        std::tie(edge_start_times, std::ignore) = shuffle_values(
          comm, edge_start_times->begin(), h_tx_value_counts_span, handle.get_stream());

      if (edge_end_times)
        std::tie(edge_end_times, std::ignore) = shuffle_values(
          comm, edge_end_times->begin(), h_tx_value_counts_span, handle.get_stream());

      if (hops)
        std::tie(hops, std::ignore) =
          shuffle_values(comm, hops->begin(), h_tx_value_counts_span, handle.get_stream());

      sort_sampled_tuples(handle,
                          majors,
                          minors,
                          weights,
                          edge_ids,
                          edge_types,
                          edge_start_times,
                          edge_end_times,
                          hops,
                          *labels);
    }

    size_t num_unique_labels =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<size_t>(0),
                       thrust::make_counting_iterator<size_t>(labels->size()),
                       is_first_in_run_t<label_t const*>{labels->data()});

    rmm::device_uvector<label_t> unique_labels(num_unique_labels, handle.get_stream());
    offsets = rmm::device_uvector<size_t>(num_unique_labels + 1, handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          labels->begin(),
                          labels->end(),
                          thrust::make_constant_iterator(size_t{1}),
                          unique_labels.begin(),
                          offsets->begin());

    thrust::exclusive_scan(
      handle.get_thrust_policy(), offsets->begin(), offsets->end(), offsets->begin());
    labels = std::move(unique_labels);
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(edge_start_times),
                         std::move(edge_end_times),
                         std::move(hops),
                         std::move(labels),
                         std::move(offsets));
}

}  // namespace detail
}  // namespace cugraph

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

#include <prims/update_edge_src_dst_property.cuh>  // ??
#include <prims/vertex_frontier.cuh>
#include <structure/detail/structure_utils.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename label_t>
struct shuffle_to_output_comm_rank_t {
  raft::device_span<label_t const> output_label_;
  raft::device_span<int32_t const> output_rank_;

  template <typename key_t>
  __device__ int32_t operator()(key_t key) const
  {
    auto pos = thrust::lower_bound(thrust::seq, output_label_.begin(), output_label_.end(), key);
    return output_rank_[thrust::distance(output_label_.begin(), pos)];
  }
};

template <typename vertex_t,
          typename weight_t,
          typename edge_t,
          typename edge_type_t,
          typename label_t>
void sort_sampled_tuples(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& majors,
                         rmm::device_uvector<vertex_t>& minors,
                         std::optional<rmm::device_uvector<weight_t>>& weights,
                         std::optional<rmm::device_uvector<edge_t>>& edge_ids,
                         std::optional<rmm::device_uvector<edge_type_t>>& edge_types,
                         std::optional<rmm::device_uvector<int32_t>>& hops,
                         std::optional<rmm::device_uvector<label_t>>& labels)
{
  if (weights) {
    if (edge_ids) {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              thrust::make_zip_iterator(labels->begin(), hops->begin()),
                              thrust::make_zip_iterator(labels->end(), hops->end()),
                              thrust::make_zip_iterator(majors.begin(),
                                                        minors.begin(),
                                                        weights->begin(),
                                                        edge_ids->begin(),
                                                        edge_types->begin()));
        } else {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(majors.begin(),
                                                        minors.begin(),
                                                        weights->begin(),
                                                        edge_ids->begin(),
                                                        edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(labels->begin(), hops->begin()),
            thrust::make_zip_iterator(labels->end(), hops->end()),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), weights->begin(), edge_ids->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), weights->begin(), edge_ids->begin()));
        }
      }
    } else {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(labels->begin(), hops->begin()),
            thrust::make_zip_iterator(labels->end(), hops->end()),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), weights->begin(), edge_types->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), weights->begin(), edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(labels->begin(), hops->begin()),
            thrust::make_zip_iterator(labels->end(), hops->end()),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()));
        }
      }
    }
  } else {
    if (edge_ids) {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(labels->begin(), hops->begin()),
            thrust::make_zip_iterator(labels->end(), hops->end()),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), edge_ids->begin(), edge_types->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), edge_ids->begin(), edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(labels->begin(), hops->begin()),
            thrust::make_zip_iterator(labels->end(), hops->end()),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()));
        }
      }
    } else {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            thrust::make_zip_iterator(labels->begin(), hops->begin()),
            thrust::make_zip_iterator(labels->end(), hops->end()),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              thrust::make_zip_iterator(labels->begin(), hops->begin()),
                              thrust::make_zip_iterator(labels->end(), hops->end()),
                              thrust::make_zip_iterator(majors.begin(), minors.begin()));
        } else {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(majors.begin(), minors.begin()));
        }
      }
    }
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
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
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<label_t>>&& labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank)
{
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

  if (labels) {
    sort_sampled_tuples(handle, majors, minors, weights, edge_ids, edge_types, hops, labels);

    if (label_to_output_comm_rank) {
      CUGRAPH_EXPECTS(labels, "labels must be specified in order to shuffle sampling results");

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

      if (weights) {
        if (edge_ids) {
          if (edge_types) {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(),
                                          minors.begin(),
                                          weights->begin(),
                                          edge_ids->begin(),
                                          edge_types->begin(),
                                          hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());

              handle.sync_stream();

              std::forward_as_tuple(
                std::tie(majors, minors, weights, edge_ids, edge_types, hops, labels),
                std::ignore) = shuffle_values(comm,
                                              thrust::make_zip_iterator(majors.begin(),
                                                                        minors.begin(),
                                                                        weights->begin(),
                                                                        edge_ids->begin(),
                                                                        edge_types->begin(),
                                                                        hops->begin(),
                                                                        labels->begin()),
                                              h_tx_value_counts,
                                              handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(),
                                          minors.begin(),
                                          weights->begin(),
                                          edge_ids->begin(),
                                          edge_types->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids, edge_types, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_ids->begin(),
                                                         edge_types->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          } else {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(),
                                          minors.begin(),
                                          weights->begin(),
                                          edge_ids->begin(),
                                          hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_ids->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(
                  majors.begin(), minors.begin(), weights->begin(), edge_ids->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_ids->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          }
        } else {
          if (edge_types) {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(),
                                          minors.begin(),
                                          weights->begin(),
                                          edge_types->begin(),
                                          hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, weights, edge_types, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_types->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(
                  majors.begin(), minors.begin(), weights->begin(), edge_types->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, weights, edge_types, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_types->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          } else {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(
                  majors.begin(), minors.begin(), weights->begin(), hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, weights, hops, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, weights, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(
                                 majors.begin(), minors.begin(), weights->begin(), labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          }
        }
      } else {
        if (edge_ids) {
          if (edge_types) {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(),
                                          minors.begin(),
                                          edge_ids->begin(),
                                          edge_types->begin(),
                                          hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, edge_ids, edge_types, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_ids->begin(),
                                                         edge_types->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(
                  majors.begin(), minors.begin(), edge_ids->begin(), edge_types->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, edge_ids, edge_types, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_ids->begin(),
                                                         edge_types->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          } else {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(
                  majors.begin(), minors.begin(), edge_ids->begin(), hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, edge_ids, hops, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_ids->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, edge_ids, labels), std::ignore) =
                shuffle_values(
                  comm,
                  thrust::make_zip_iterator(
                    majors.begin(), minors.begin(), edge_ids->begin(), labels->begin()),
                  h_tx_value_counts,
                  handle.get_stream());
            }
          }
        } else {
          if (edge_types) {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(
                  majors.begin(), minors.begin(), edge_types->begin(), hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, edge_types, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_types->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, edge_types, labels), std::ignore) =
                shuffle_values(
                  comm,
                  thrust::make_zip_iterator(
                    majors.begin(), minors.begin(), edge_types->begin(), labels->begin()),
                  h_tx_value_counts,
                  handle.get_stream());
            }
          } else {
            if (hops) {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(), minors.begin(), hops->begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, hops, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(
                                 majors.begin(), minors.begin(), hops->begin(), labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              auto d_tx_value_counts = cugraph::groupby_and_count(
                labels->begin(),
                labels->end(),
                thrust::make_zip_iterator(majors.begin(), minors.begin()),
                shuffle_to_output_comm_rank_t<label_t>{std::get<0>(*label_to_output_comm_rank),
                                                       std::get<1>(*label_to_output_comm_rank)},
                comm_size,
                mem_frugal_threshold,
                handle.get_stream());

              std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
              raft::update_host(h_tx_value_counts.data(),
                                d_tx_value_counts.data(),
                                d_tx_value_counts.size(),
                                handle.get_stream());
              handle.sync_stream();

              std::forward_as_tuple(std::tie(majors, minors, labels), std::ignore) = shuffle_values(
                comm,
                thrust::make_zip_iterator(majors.begin(), minors.begin(), labels->begin()),
                h_tx_value_counts,
                handle.get_stream());
            }
          }
        }
      }

      sort_sampled_tuples(handle, majors, minors, weights, edge_ids, edge_types, hops, labels);
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
                         std::move(hops),
                         std::move(labels),
                         std::move(offsets));
}

}  // namespace detail
}  // namespace cugraph

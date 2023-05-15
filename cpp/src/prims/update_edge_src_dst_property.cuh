/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <detail/graph_partition_utils.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <utility>
#include <variant>

namespace cugraph {

namespace detail {

template <typename Iterator, typename vertex_t>
__device__ void packed_bool_atomic_set(Iterator value_first, vertex_t offset, bool val)
{
  auto packed_output_offset = packed_bool_offset(offset);
  auto packed_output_mask   = packed_bool_mask(offset);
  if (val) {
    atomicOr(value_first + packed_output_offset, packed_output_mask);
  } else {
    atomicAnd(value_first + packed_output_offset, ~packed_output_mask);
  }
}

template <typename BoolInputIterator, typename PackedBoolOutputIterator>
void pack_bools(raft::handle_t const& handle,
                BoolInputIterator input_first,
                BoolInputIterator input_last,
                PackedBoolOutputIterator output_first)
{
  auto num_bools   = static_cast<size_t>(thrust::distance(input_first, input_last));
  auto packed_size = cugraph::packed_bool_size(num_bools);
  thrust::tabulate(handle.get_thrust_policy(),
                   output_first,
                   output_first + packed_size,
                   pack_bool_t<BoolInputIterator>{input_first, num_bools});
}

template <typename BoolInputIterator, typename PackedBoolOutputIterator>
void pack_unaligned_bools(raft::handle_t const& handle,
                          BoolInputIterator input_first,
                          BoolInputIterator input_last,
                          PackedBoolOutputIterator output_first,
                          size_t intraword_start_offset)
{
  auto num_bools            = static_cast<size_t>(thrust::distance(input_first, input_last));
  auto num_first_word_bools = std::min(num_bools, packed_bools_per_word() - intraword_start_offset);
  auto num_aligned_bools    = (num_bools - num_first_word_bools) -
                           (num_bools - num_first_word_bools) % packed_bools_per_word();
  auto num_last_word_bools = num_bools - (num_first_word_bools + num_aligned_bools);

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(num_first_word_bools + num_last_word_bools),
    [intraword_start_offset,
     num_first_word_bools,
     num_last_word_bools,
     first_word_input_bool_first = input_first,
     last_word_input_bool_first  = input_first + (num_first_word_bools + num_aligned_bools),
     first_word_output           = output_first,
     last_word_output            = output_first + ((num_first_word_bools > 0 ? 1 : 0) +
                                        packed_bool_size(num_aligned_bools))] __device__(size_t i) {
      if (i < num_first_word_bools) {
        auto val = *(first_word_input_bool_first + i);
        packed_bool_atomic_set(first_word_output, intraword_start_offset + i, val);
      } else {
        auto val = *(last_word_input_bool_first + (i - num_first_word_bools));
        packed_bool_atomic_set(last_word_output, i - num_first_word_bools, val);
      }
    });

  pack_bools(handle,
             input_first + num_first_word_bools,
             input_first + num_first_word_bools + num_aligned_bools,
             output_first + (num_first_word_bools > 0 ? 1 : 0));
}

template <typename GraphViewType,
          typename VertexPropertyInputIterator,
          typename EdgeMajorPropertyOutputWrapper>
void update_edge_major_property(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexPropertyInputIterator vertex_property_input_first,
                                EdgeMajorPropertyOutputWrapper edge_major_property_output)
{
  constexpr bool packed_bool =
    std::is_same_v<typename EdgeMajorPropertyOutputWrapper::value_type, bool>;

  auto edge_partition_value_firsts = edge_major_property_output.value_firsts();
  if constexpr (GraphViewType::is_multi_gpu) {
    using vertex_t = typename GraphViewType::vertex_type;

    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    auto edge_partition_keys = edge_major_property_output.keys();
    if (edge_partition_keys) {
      vertex_t max_rx_size{0};
      for (int i = 0; i < minor_comm_size; ++i) {
        auto major_range_vertex_partition_id =
          compute_local_edge_partition_major_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        max_rx_size = std::max(
          max_rx_size, graph_view.vertex_partition_range_size(major_range_vertex_partition_id));
      }
      auto rx_value_buffer = allocate_dataframe_buffer<
        std::conditional_t<packed_bool,
                           uint32_t,
                           typename EdgeMajorPropertyOutputWrapper::value_type>>(
        packed_bool ? packed_bool_size(max_rx_size) : max_rx_size, handle.get_stream());
      auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffer);
      for (int i = 0; i < minor_comm_size; ++i) {
        auto major_range_vertex_partition_id =
          compute_local_edge_partition_major_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        if constexpr (packed_bool) {
          if (i == minor_comm_rank) {
            pack_bools(handle,
                       vertex_property_input_first,
                       vertex_property_input_first +
                         graph_view.vertex_partition_range_size(major_range_vertex_partition_id),
                       rx_value_first);
          }
          device_bcast(minor_comm,
                       rx_value_first,
                       rx_value_first,
                       packed_bool_size(
                         graph_view.vertex_partition_range_size(major_range_vertex_partition_id)),
                       i,
                       handle.get_stream());
          auto bool_first = thrust::make_transform_iterator(
            (*edge_partition_keys)[i].begin(),
            [rx_value_first,
             v_first = graph_view.vertex_partition_range_first(
               major_range_vertex_partition_id)] __device__(auto v) {
              auto v_offset = v - v_first;
              return static_cast<bool>(*(rx_value_first + packed_bool_offset(v_offset)) &
                                       packed_bool_mask(v_offset));
            });
          pack_bools(handle,
                     bool_first,
                     bool_first + (*edge_partition_keys)[i].size(),
                     edge_partition_value_firsts[i]);
        } else {
          device_bcast(minor_comm,
                       vertex_property_input_first,
                       rx_value_first,
                       graph_view.vertex_partition_range_size(major_range_vertex_partition_id),
                       i,
                       handle.get_stream());

          auto v_offset_first = thrust::make_transform_iterator(
            (*edge_partition_keys)[i].begin(),
            [v_first = graph_view.vertex_partition_range_first(
               major_range_vertex_partition_id)] __device__(auto v) { return v - v_first; });
          thrust::gather(handle.get_thrust_policy(),
                         v_offset_first,
                         v_offset_first + (*edge_partition_keys)[i].size(),
                         rx_value_first,
                         edge_partition_value_firsts[i]);
        }
      }
    } else {
      for (int i = 0; i < minor_comm_size; ++i) {
        auto major_range_vertex_partition_id =
          compute_local_edge_partition_major_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        if constexpr (packed_bool) {
          if (i == minor_comm_rank) {
            pack_bools(handle,
                       vertex_property_input_first,
                       vertex_property_input_first +
                         graph_view.vertex_partition_range_size(major_range_vertex_partition_id),
                       edge_partition_value_firsts[i]);
          }
          device_bcast(minor_comm,
                       edge_partition_value_firsts[i],
                       edge_partition_value_firsts[i],
                       packed_bool_size(
                         graph_view.vertex_partition_range_size(major_range_vertex_partition_id)),
                       i,
                       handle.get_stream());
        } else {
          device_bcast(minor_comm,
                       vertex_property_input_first,
                       edge_partition_value_firsts[i],
                       graph_view.vertex_partition_range_size(major_range_vertex_partition_id),
                       i,
                       handle.get_stream());
        }
      }
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_dst_range_size()
             : graph_view.local_edge_partition_src_range_size());
    assert(edge_partition_value_firsts.size() == size_t{1});
    if constexpr (packed_bool) {
      pack_bools(handle,
                 vertex_property_input_first,
                 vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                 edge_partition_value_firsts[0]);
    } else {
      thrust::copy(handle.get_thrust_policy(),
                   vertex_property_input_first,
                   vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                   edge_partition_value_firsts[0]);
    }
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexPropertyInputIterator,
          typename EdgeMajorPropertyOutputWrapper>
void update_edge_major_property(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexIterator vertex_first,
                                VertexIterator vertex_last,
                                VertexPropertyInputIterator vertex_property_input_first,
                                EdgeMajorPropertyOutputWrapper edge_major_property_output)
{
  constexpr bool packed_bool =
    std::is_same_v<typename EdgeMajorPropertyOutputWrapper::value_type, bool>;

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_firsts = edge_major_property_output.value_firsts();
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    auto rx_counts =
      host_scalar_allgather(minor_comm,
                            static_cast<size_t>(thrust::distance(vertex_first, vertex_last)),
                            handle.get_stream());
    auto max_rx_size =
      std::reduce(rx_counts.begin(), rx_counts.end(), size_t{0}, [](auto lhs, auto rhs) {
        return std::max(lhs, rhs);
      });
    rmm::device_uvector<vertex_t> rx_vertices(max_rx_size, handle.get_stream());
    auto rx_tmp_buffer = allocate_dataframe_buffer<
      std::
        conditional_t<packed_bool, uint32_t, typename EdgeMajorPropertyOutputWrapper::value_type>>(
      packed_bool ? packed_bool_size(max_rx_size) : max_rx_size, handle.get_stream());
    auto rx_value_first = get_dataframe_buffer_begin(rx_tmp_buffer);

    auto edge_partition_keys = edge_major_property_output.keys();
    for (int i = 0; i < minor_comm_size; ++i) {
      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(i));

      if (i == minor_comm_rank) {
        auto vertex_partition =
          vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
            graph_view.local_vertex_partition_view());
        if constexpr (packed_bool) {
          auto bool_first = thrust::make_transform_iterator(
            vertex_first, [vertex_property_input_first, vertex_partition] __device__(auto v) {
              auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
              return static_cast<bool>(
                *(vertex_property_input_first + packed_bool_offset(v_offset)) &
                packed_bool_mask(v_offset));
            });
          pack_bools(handle,
                     bool_first,
                     bool_first + thrust::distance(vertex_first, vertex_last),
                     rx_value_first);
        } else {
          auto map_first =
            thrust::make_transform_iterator(vertex_first, [vertex_partition] __device__(auto v) {
              return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            });
          // FIXME: this gather (and temporary buffer) is unnecessary if NCCL directly takes a
          // permutation iterator (and directly gathers to the internal buffer)
          thrust::gather(handle.get_thrust_policy(),
                         map_first,
                         map_first + thrust::distance(vertex_first, vertex_last),
                         vertex_property_input_first,
                         rx_value_first);
        }
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(
        minor_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());
      device_bcast(minor_comm,
                   rx_value_first,
                   rx_value_first,
                   packed_bool ? packed_bool_size(rx_counts[i]) : rx_counts[i],
                   i,
                   handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(rx_counts[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_value_first,
           edge_partition_key_first   = ((*edge_partition_keys)[i]).begin(),
           edge_partition_key_last    = ((*edge_partition_keys)[i]).end(),
           edge_partition_value_first = edge_partition_value_firsts[i]] __device__(size_t i) {
            auto major = *(rx_vertex_first + i);
            auto it    = thrust::lower_bound(
              thrust::seq, edge_partition_key_first, edge_partition_key_last, major);
            if ((it != edge_partition_key_last) && (*it == major)) {
              auto edge_partition_offset = thrust::distance(edge_partition_key_first, it);
              if constexpr (packed_bool) {
                auto rx_value = static_cast<bool>(*(rx_value_first + packed_bool_offset(i)) &
                                                  packed_bool_mask(i));
                packe_bool_atomic_set(edge_partition_value_first, edge_partition_offset, rx_value);
              } else {
                auto rx_value                                         = *(rx_value_first + i);
                *(edge_partition_value_first + edge_partition_offset) = rx_value;
              }
            }
          });
      } else {
        if constexpr (packed_bool) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(vertex_t{0}),
            thrust::make_counting_iterator(static_cast<vertex_t>(rx_counts[i])),
            [edge_partition,
             rx_vertex_first = rx_vertices.begin(),
             rx_value_first,
             output_value_first = edge_partition_value_firsts[i]] __device__(auto i) {
              auto rx_vertex = *(rx_vertex_first + i);
              auto rx_value =
                static_cast<bool>(*(rx_value_first + packed_bool_offset(i)) & packed_bool_mask(i));
              auto major_offset = edge_partition.major_offset_from_major_nocheck(rx_vertex);
              packed_bool_atomic_set(output_value_first, major_offset, rx_value);
            });
        } else {
          auto map_first = thrust::make_transform_iterator(
            rx_vertices.begin(), [edge_partition] __device__(auto v) {
              return edge_partition.major_offset_from_major_nocheck(v);
            });
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(handle.get_thrust_policy(),
                          rx_value_first,
                          rx_value_first + rx_counts[i],
                          map_first,
                          edge_partition_value_firsts[i]);
        }
      }
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_dst_range_size()
             : graph_view.local_edge_partition_src_range_size());
    assert(edge_partition_value_firsts.size() == size_t{1});
    if constexpr (packed_bool) {
      thrust::for_each(handle.get_thrust_policy(),
                       vertex_first,
                       vertex_last,
                       [vertex_property_input_first,
                        output_value_first = edge_partition_value_firsts[0]] __device__(auto v) {
                         bool val = static_cast<bool>(*(vertex_property_input_first + v));
                         packed_bool_atomic_set(output_value_first, v, val);
                       });
    } else {
      auto val_first = thrust::make_permutation_iterator(vertex_property_input_first, vertex_first);
      thrust::scatter(handle.get_thrust_policy(),
                      val_first,
                      val_first + thrust::distance(vertex_first, vertex_last),
                      vertex_first,
                      edge_partition_value_firsts[0]);
    }
  }
}

template <typename GraphViewType,
          typename VertexPropertyInputIterator,
          typename EdgeMinorPropertyOutputWrapper>
void update_edge_minor_property(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexPropertyInputIterator vertex_property_input_first,
                                EdgeMinorPropertyOutputWrapper edge_minor_property_output)
{
  constexpr bool packed_bool =
    std::is_same_v<typename EdgeMinorPropertyOutputWrapper::value_type, bool>;

  auto edge_partition_value_first = edge_minor_property_output.value_first();
  if constexpr (GraphViewType::is_multi_gpu) {
    using vertex_t = typename GraphViewType::vertex_type;
    using bcast_buffer_type =
      decltype(allocate_dataframe_buffer<
               std::conditional_t<packed_bool,
                                  uint32_t,
                                  typename EdgeMinorPropertyOutputWrapper::value_type>>(
        size_t{0}, handle.get_stream()));

    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    // memory footprint vs parallelism trade-off
    // memory requirement per loop is
    // (V/comm_size) * sizeof(value_t)
    // and limit memory requirement to (E / comm_size) * sizeof(vertex_t)
    auto bcast_size = static_cast<size_t>(graph_view.number_of_vertices()) / comm_size;
    if constexpr (packed_bool) {
      bcast_size /= 8;  // bits to bytes
    } else {
      bcast_size *= sizeof(typename EdgeMinorPropertyOutputWrapper::value_type);
    }
    auto num_concurrent_bcasts =
      (static_cast<size_t>(graph_view.number_of_edges() / comm_size) * sizeof(vertex_t)) /
      std::max(bcast_size, size_t{1});
    num_concurrent_bcasts = std::max(num_concurrent_bcasts, size_t{1});
    num_concurrent_bcasts = std::min(num_concurrent_bcasts, static_cast<size_t>(major_comm_size));
    auto num_rounds = (static_cast<size_t>(major_comm_size) + num_concurrent_bcasts - size_t{1}) /
                      num_concurrent_bcasts;

    auto edge_partition_keys = edge_minor_property_output.keys();

    std::optional<std::vector<bcast_buffer_type>> rx_value_buffers{std::nullopt};
    if (packed_bool || edge_partition_keys) {
      rx_value_buffers = std::vector<bcast_buffer_type>{};
      (*rx_value_buffers).reserve(num_concurrent_bcasts);
      for (size_t i = 0; i < num_concurrent_bcasts; ++i) {
        size_t max_size{0};
        for (size_t round = 0; round < num_rounds; ++round) {
          auto j = num_rounds * i + round;
          if (j < static_cast<size_t>(major_comm_size)) {
            auto minor_range_vertex_partition_id =
              compute_local_edge_partition_minor_range_vertex_partition_id_t{
                major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(j);
            max_size = std::max(max_size,
                                static_cast<size_t>(graph_view.vertex_partition_range_size(
                                  minor_range_vertex_partition_id)));
          }
        }
        (*rx_value_buffers)
          .push_back(allocate_dataframe_buffer<
                     std::conditional_t<packed_bool,
                                        uint32_t,
                                        typename EdgeMinorPropertyOutputWrapper::value_type>>(
            packed_bool ? packed_bool_size(max_size) : max_size, handle.get_stream()));
      }
    }

    std::variant<raft::host_span<vertex_t const>, std::vector<size_t>>
      key_offsets_or_rx_displacements{};
    if (edge_partition_keys) {
      if constexpr (GraphViewType::is_storage_transposed) {
        key_offsets_or_rx_displacements =
          *(graph_view.local_sorted_unique_edge_src_vertex_partition_offsets());
      } else {
        key_offsets_or_rx_displacements =
          *(graph_view.local_sorted_unique_edge_dst_vertex_partition_offsets());
      }
    } else {
      std::vector<size_t> rx_counts(major_comm_size, size_t{0});
      for (int i = 0; i < major_comm_size; ++i) {
        auto minor_range_vertex_partition_id =
          compute_local_edge_partition_minor_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        rx_counts[i] = graph_view.vertex_partition_range_size(minor_range_vertex_partition_id);
      }
      std::vector<size_t> rx_displacements(major_comm_size, size_t{0});
      std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
      key_offsets_or_rx_displacements = std::move(rx_displacements);
    }

    for (size_t round = 0; round < num_rounds; ++round) {
      if constexpr (packed_bool) {
        for (size_t i = 0; i < num_concurrent_bcasts; ++i) {
          auto j = static_cast<int>(num_rounds * i + round);
          if (j == major_comm_rank) {
            auto minor_range_vertex_partition_id =
              compute_local_edge_partition_minor_range_vertex_partition_id_t{
                major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(j);
            auto rx_value_first = get_dataframe_buffer_begin((*rx_value_buffers)[i]);
            pack_bools(handle,
                       vertex_property_input_first,
                       vertex_property_input_first +
                         graph_view.vertex_partition_range_size(minor_range_vertex_partition_id),
                       rx_value_first);
          }
        }
      }

      device_group_start(major_comm);
      for (size_t i = 0; i < num_concurrent_bcasts; ++i) {
        auto j = num_rounds * i + round;
        if (j < static_cast<size_t>(major_comm_size)) {
          auto minor_range_vertex_partition_id =
            compute_local_edge_partition_minor_range_vertex_partition_id_t{
              major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(j);
          auto rx_value_first =
            rx_value_buffers ? get_dataframe_buffer_begin((*rx_value_buffers)[i])
                             : edge_partition_value_first +
                                 std::get<std::vector<size_t>>(key_offsets_or_rx_displacements)[j];
          if constexpr (packed_bool) {
            device_bcast(major_comm,
                         rx_value_first,
                         rx_value_first,
                         packed_bool_size(
                           graph_view.vertex_partition_range_size(minor_range_vertex_partition_id)),
                         j,
                         handle.get_stream());
          } else {
            device_bcast(major_comm,
                         vertex_property_input_first,
                         rx_value_first,
                         graph_view.vertex_partition_range_size(minor_range_vertex_partition_id),
                         j,
                         handle.get_stream());
          }
        }
      }
      device_group_end(major_comm);

      if (rx_value_buffers) {
        for (size_t i = 0; i < num_concurrent_bcasts; ++i) {
          auto j = num_rounds * i + round;
          if (j < static_cast<size_t>(major_comm_size)) {
            auto minor_range_vertex_partition_id =
              compute_local_edge_partition_minor_range_vertex_partition_id_t{
                major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(j);
            auto rx_value_first = get_dataframe_buffer_begin((*rx_value_buffers)[i]);
            if constexpr (packed_bool) {
              if (edge_partition_keys) {
                auto key_offsets =
                  std::get<raft::host_span<vertex_t const>>(key_offsets_or_rx_displacements);

                auto bool_first = thrust::make_transform_iterator(
                  (*edge_partition_keys).begin() + key_offsets[j],
                  [rx_value_first,
                   v_first = graph_view.vertex_partition_range_first(
                     minor_range_vertex_partition_id)] __device__(auto v) {
                    auto v_offset = v - v_first;
                    return static_cast<bool>(*(rx_value_first + packed_bool_offset(v_offset)) &
                                             packed_bool_mask(v_offset));
                  });
                pack_unaligned_bools(
                  handle,
                  bool_first,
                  bool_first + (key_offsets[j + 1] - key_offsets[j]),
                  edge_partition_value_first + packed_bool_offset(key_offsets[j]),
                  key_offsets[j] % packed_bools_per_word());
              } else {
                auto rx_displacements =
                  std::get<std::vector<size_t>>(key_offsets_or_rx_displacements);
                auto bool_first = thrust::make_transform_iterator(
                  thrust::make_counting_iterator(vertex_t{0}),
                  [rx_value_first] __device__(vertex_t v_offset) {
                    return static_cast<bool>(*(rx_value_first + packed_bool_offset(v_offset)) &
                                             packed_bool_mask(v_offset));
                  });
                pack_unaligned_bools(
                  handle,
                  bool_first,
                  bool_first +
                    graph_view.vertex_partition_range_size(minor_range_vertex_partition_id),
                  edge_partition_value_first + packed_bool_offset(rx_displacements[j]),
                  rx_displacements[j] % packed_bools_per_word());
              }
            } else {
              assert(edge_partition_keys);
              auto key_offsets =
                std::get<raft::host_span<vertex_t const>>(key_offsets_or_rx_displacements);

              auto v_offset_first = thrust::make_transform_iterator(
                (*edge_partition_keys).begin() + key_offsets[j],
                [v_first = graph_view.vertex_partition_range_first(
                   minor_range_vertex_partition_id)] __device__(auto v) { return v - v_first; });
              thrust::gather(handle.get_thrust_policy(),
                             v_offset_first,
                             v_offset_first + (key_offsets[j + 1] - key_offsets[j]),
                             rx_value_first,
                             edge_partition_value_first + key_offsets[j]);
            }
          }
        }
      }
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_src_range_size()
             : graph_view.local_edge_partition_dst_range_size());
    if constexpr (packed_bool) {
      pack_bools(handle,
                 vertex_property_input_first,
                 vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                 edge_partition_value_first);
    } else {
      thrust::copy(handle.get_thrust_policy(),
                   vertex_property_input_first,
                   vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                   edge_partition_value_first);
    }
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexPropertyInputIterator,
          typename EdgeMinorPropertyOutputWrapper>
void update_edge_minor_property(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexIterator vertex_first,
                                VertexIterator vertex_last,
                                VertexPropertyInputIterator vertex_property_input_first,
                                EdgeMinorPropertyOutputWrapper edge_minor_property_output)
{
  constexpr bool packed_bool =
    std::is_same_v<typename EdgeMinorPropertyOutputWrapper::value_type, bool>;

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_first = edge_minor_property_output.value_first();
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();

    auto rx_counts =
      host_scalar_allgather(major_comm,
                            static_cast<size_t>(thrust::distance(vertex_first, vertex_last)),
                            handle.get_stream());
    auto max_rx_size =
      std::reduce(rx_counts.begin(), rx_counts.end(), size_t{0}, [](auto lhs, auto rhs) {
        return std::max(lhs, rhs);
      });
    rmm::device_uvector<vertex_t> rx_vertices(max_rx_size, handle.get_stream());
    auto rx_tmp_buffer = allocate_dataframe_buffer<
      std::
        conditional_t<packed_bool, uint32_t, typename EdgeMinorPropertyOutputWrapper::value_type>>(
      packed_bool ? packed_bool_size(max_rx_size) : max_rx_size, handle.get_stream());
    auto rx_value_first = get_dataframe_buffer_begin(rx_tmp_buffer);

    std::optional<raft::host_span<vertex_t const>> key_offsets{};
    if constexpr (GraphViewType::is_storage_transposed) {
      key_offsets = graph_view.local_sorted_unique_edge_src_vertex_partition_offsets();
    } else {
      key_offsets = graph_view.local_sorted_unique_edge_dst_vertex_partition_offsets();
    }

    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(size_t{0}));
    auto edge_partition_keys = edge_minor_property_output.keys();
    for (int i = 0; i < major_comm_size; ++i) {
      if (i == major_comm_rank) {
        auto vertex_partition =
          vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
            graph_view.local_vertex_partition_view());
        if constexpr (packed_bool) {
          auto bool_first = thrust::make_transform_iterator(
            vertex_first, [vertex_property_input_first, vertex_partition] __device__(auto v) {
              auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
              return static_cast<bool>(
                *(vertex_property_input_first + packed_bool_offset(v_offset)) &
                packed_bool_mask(v_offset));
            });
          pack_bools(handle,
                     bool_first,
                     bool_first + thrust::distance(vertex_first, vertex_last),
                     rx_value_first);
        } else {
          auto map_first =
            thrust::make_transform_iterator(vertex_first, [vertex_partition] __device__(auto v) {
              return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            });
          // FIXME: this gather (and temporary buffer) is unnecessary if NCCL directly takes a
          // permutation iterator (and directly gathers to the internal buffer)
          thrust::gather(handle.get_thrust_policy(),
                         map_first,
                         map_first + thrust::distance(vertex_first, vertex_last),
                         vertex_property_input_first,
                         rx_value_first);
        }
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(
        major_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());
      device_bcast(major_comm,
                   rx_value_first,
                   rx_value_first,
                   packed_bool ? packed_bool_size(rx_counts[i]) : rx_counts[i],
                   i,
                   handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(rx_counts[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_value_first,
           subrange_key_first         = (*edge_partition_keys).begin() + (*key_offsets)[i],
           subrange_key_last          = (*edge_partition_keys).begin() + (*key_offsets)[i + 1],
           edge_partition_value_first = edge_partition_value_first,
           subrange_start_offset      = (*key_offsets)[i]] __device__(auto i) {
            auto minor = *(rx_vertex_first + i);
            auto it =
              thrust::lower_bound(thrust::seq, subrange_key_first, subrange_key_last, minor);
            if ((it != subrange_key_last) && (*it == minor)) {
              auto subrange_offset = thrust::distance(subrange_key_first, it);
              if constexpr (packed_bool) {
                auto rx_value = static_cast<bool>(*(rx_value_first + packed_bool_offset(i)) &
                                                  packed_bool_mask(i));
                packed_bool_atomic_set(
                  edge_partition_value_first, subrange_start_offset + subrange_offset, rx_value);
              } else {
                auto rx_value = *(rx_value_first + i);
                *(edge_partition_value_first + subrange_start_offset + subrange_offset) = rx_value;
              }
            }
          });
      } else {
        if constexpr (packed_bool) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(vertex_t{0}),
            thrust::make_counting_iterator(static_cast<vertex_t>(rx_counts[i])),
            [edge_partition,
             rx_vertex_first = rx_vertices.begin(),
             rx_value_first,
             output_value_first = edge_partition_value_first] __device__(auto i) {
              auto rx_vertex = *(rx_vertex_first + i);
              auto rx_value =
                static_cast<bool>(*(rx_value_first + packed_bool_offset(i)) & packed_bool_mask(i));
              auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(rx_vertex);
              packed_bool_atomic_set(output_value_first, minor_offset, rx_value);
            });
        } else {
          auto map_first = thrust::make_transform_iterator(
            rx_vertices.begin(), [edge_partition] __device__(auto v) {
              return edge_partition.minor_offset_from_minor_nocheck(v);
            });
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(handle.get_thrust_policy(),
                          rx_value_first,
                          rx_value_first + rx_counts[i],
                          map_first,
                          edge_partition_value_first);
        }
      }
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() ==
           graph_view.local_edge_partition_src_range_size());
    if constexpr (packed_bool) {
      thrust::for_each(handle.get_thrust_policy(),
                       vertex_first,
                       vertex_last,
                       [vertex_property_input_first,
                        output_value_first = edge_partition_value_first] __device__(auto v) {
                         bool val = static_cast<bool>(*(vertex_property_input_first + v));
                         packed_bool_atomic_set(output_value_first, v, val);
                       });
    } else {
      auto val_first = thrust::make_permutation_iterator(vertex_property_input_first, vertex_first);
      thrust::scatter(handle.get_thrust_policy(),
                      val_first,
                      val_first + thrust::distance(vertex_first, vertex_last),
                      vertex_first,
                      edge_partition_value_first);
    }
  }
}

}  // namespace detail

/**
 * @brief Update graph edge source property values from the input vertex property values.
 *
 * This version updates graph edge source property values for the entire edge source ranges
 * (assigned to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexPropertyInputIterator Type of the iterator for vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_src_property_output edge_src_property_t class object to store source
 * property values (for the edge sources assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexPropertyInputIterator>
void update_edge_src_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_src_property_t<GraphViewType,
                      typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_src_property_output,
  bool do_expensive_check = false)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_minor_property(
      handle, graph_view, vertex_property_input_first, edge_src_property_output.mutable_view());
  } else {
    detail::update_edge_major_property(
      handle, graph_view, vertex_property_input_first, edge_src_property_output.mutable_view());
  }
}

/**
 * @brief Update graph edge source property values from the input vertex property values.
 *
 * This version updates only a subset of graph edge source property values. [@p vertex_first, @p
 * vertex_last) specifies the vertices with new property values to be updated.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexPropertyInputIterator Type of the iterator for vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex with a new value to be
 * updated. v in [vertex_first, vertex_last) should be distinct (and should belong to the vertex
 * partition assigned to this process in multi-GPU), otherwise undefined behavior.
 * @param vertex_last Iterator pointing to the last (exclusive) vertex with a new value.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_src_property_output edge_src_property_t class object to store source
 * property values (for the edge sources assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexIterator, typename VertexPropertyInputIterator>
void update_edge_src_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_src_property_t<GraphViewType,
                      typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_src_property_output,
  bool do_expensive_check = false)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  if (do_expensive_check) {
    auto num_invalids = thrust::count_if(
      handle.get_thrust_policy(),
      vertex_first,
      vertex_last,
      [local_vertex_partition_range_first = graph_view.local_vertex_partition_range_first(),
       local_vertex_partition_range_last =
         graph_view.local_vertex_partition_range_last()] __device__(auto v) {
        return (v < local_vertex_partition_range_first) || (v >= local_vertex_partition_range_last);
      });
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm = handle.get_comms();
      num_invalids =
        host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(
      num_invalids == 0,
      "Invalid input argument: invalid or non-local vertices in [vertex_first, vertex_last).");
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_minor_property(handle,
                                       graph_view,
                                       vertex_first,
                                       vertex_last,
                                       vertex_property_input_first,
                                       edge_src_property_output.mutable_view());
  } else {
    detail::update_edge_major_property(handle,
                                       graph_view,
                                       vertex_first,
                                       vertex_last,
                                       vertex_property_input_first,
                                       edge_src_property_output.mutable_view());
  }
}

/**
 * @brief Update graph edge destination property values from the input vertex property values.
 *
 * This version updates graph edge destination property values for the entire edge destination
 * ranges (assigned to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexPropertyInputIterator Type of the iterator for vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_dst_property_output edge_dst_property_t class object to store destination
 * property values (for the edge destinations assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexPropertyInputIterator>
void update_edge_dst_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_dst_property_t<GraphViewType,
                      typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_dst_property_output,
  bool do_expensive_check = false)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_major_property(
      handle, graph_view, vertex_property_input_first, edge_dst_property_output.mutable_view());
  } else {
    detail::update_edge_minor_property(
      handle, graph_view, vertex_property_input_first, edge_dst_property_output.mutable_view());
  }
}

/**
 * @brief Update graph edge destination property values from the input vertex property values.
 *
 * This version updates only a subset of graph edge destination property values. [@p vertex_first,
 * @p vertex_last) specifies the vertices with new property values to be updated.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexPropertyInputIterator Type of the iterator for vertex property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex with a new value to be
 * updated. v in [vertex_first, vertex_last) should be distinct (and should belong to the vertex
 * partition assigned to this process in multi-GPU), otherwise undefined behavior.
 * @param vertex_last Iterator pointing to the last (exclusive) vertex with a new value.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_dst_property_output edge_dst_property_t class object to store destination
 * property values (for the edge destinations assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexIterator, typename VertexPropertyInputIterator>
void update_edge_dst_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_dst_property_t<GraphViewType,
                      typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_dst_property_output,
  bool do_expensive_check = false)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  if (do_expensive_check) {
    auto num_invalids = thrust::count_if(
      handle.get_thrust_policy(),
      vertex_first,
      vertex_last,
      [local_vertex_partition_range_first = graph_view.local_vertex_partition_range_first(),
       local_vertex_partition_range_last =
         graph_view.local_vertex_partition_range_last()] __device__(auto v) {
        return (v < local_vertex_partition_range_first) || (v >= local_vertex_partition_range_last);
      });
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm = handle.get_comms();
      num_invalids =
        host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(
      num_invalids == 0,
      "Invalid input argument: invalid or non-local vertices in [vertex_first, vertex_last).");
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_major_property(handle,
                                       graph_view,
                                       vertex_first,
                                       vertex_last,
                                       vertex_property_input_first,
                                       edge_dst_property_output.mutable_view());
  } else {
    detail::update_edge_minor_property(handle,
                                       graph_view,
                                       vertex_first,
                                       vertex_last,
                                       vertex_property_input_first,
                                       edge_dst_property_output.mutable_view());
  }
}

}  // namespace cugraph

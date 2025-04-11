/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/atomic_ops.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
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

template <typename BoolInputIterator, typename PackedBoolOutputIterator>
void pack_bools(raft::handle_t const& handle,
                BoolInputIterator input_first,
                BoolInputIterator input_last,
                PackedBoolOutputIterator output_first)
{
  auto num_bools   = static_cast<size_t>(cuda::std::distance(input_first, input_last));
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
  auto num_bools            = static_cast<size_t>(cuda::std::distance(input_first, input_last));
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
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMajorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMajorPropertyOutputWrapper::value_type>();
  static_assert(!contains_packed_bool_element ||
                  std::is_arithmetic_v<typename EdgeMajorPropertyOutputWrapper::value_type>,
                "unimplemented for thrust::tuple types with a packed bool element.");

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
        std::conditional_t<contains_packed_bool_element,
                           uint32_t,
                           typename EdgeMajorPropertyOutputWrapper::value_type>>(
        contains_packed_bool_element ? packed_bool_size(max_rx_size) : max_rx_size,
        handle.get_stream());
      auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffer);
      for (int i = 0; i < minor_comm_size; ++i) {
        auto major_range_vertex_partition_id =
          compute_local_edge_partition_major_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        if constexpr (contains_packed_bool_element) {
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
            cuda::proclaim_return_type<bool>(
              [rx_value_first,
               v_first = graph_view.vertex_partition_range_first(
                 major_range_vertex_partition_id)] __device__(auto v) {
                auto v_offset = v - v_first;
                return static_cast<bool>(*(rx_value_first + packed_bool_offset(v_offset)) &
                                         packed_bool_mask(v_offset));
              }));
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
            cuda::proclaim_return_type<vertex_t>(
              [v_first = graph_view.vertex_partition_range_first(
                 major_range_vertex_partition_id)] __device__(auto v) { return v - v_first; }));
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
        if constexpr (contains_packed_bool_element) {
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
    if constexpr (contains_packed_bool_element) {
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
                                VertexIterator sorted_unique_vertex_first,
                                VertexIterator sorted_unique_vertex_last,
                                VertexPropertyInputIterator vertex_property_input_first,
                                EdgeMajorPropertyOutputWrapper edge_major_property_output)
{
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMajorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMajorPropertyOutputWrapper::value_type>();
  static_assert(!contains_packed_bool_element ||
                  std::is_arithmetic_v<typename EdgeMajorPropertyOutputWrapper::value_type>,
                "unimplemented for thrust::tuple types with a packed bool element.");

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_firsts = edge_major_property_output.value_firsts();
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    auto local_v_list_sizes =
      host_scalar_allgather(minor_comm,
                            static_cast<size_t>(cuda::std::distance(sorted_unique_vertex_first,
                                                                    sorted_unique_vertex_last)),
                            handle.get_stream());
    auto max_rx_size = std::reduce(
      local_v_list_sizes.begin(), local_v_list_sizes.end(), size_t{0}, [](auto lhs, auto rhs) {
        return std::max(lhs, rhs);
      });
    rmm::device_uvector<vertex_t> rx_vertices(max_rx_size, handle.get_stream());
    auto rx_tmp_buffer = allocate_dataframe_buffer<
      std::conditional_t<contains_packed_bool_element,
                         uint32_t,
                         typename EdgeMajorPropertyOutputWrapper::value_type>>(
      contains_packed_bool_element ? packed_bool_size(max_rx_size) : max_rx_size,
      handle.get_stream());
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
        if constexpr (contains_packed_bool_element) {
          auto bool_first = thrust::make_transform_iterator(
            sorted_unique_vertex_first,
            cuda::proclaim_return_type<bool>([vertex_property_input_first,
                                              vertex_partition] __device__(auto v) {
              auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
              return static_cast<bool>(
                *(vertex_property_input_first + packed_bool_offset(v_offset)) &
                packed_bool_mask(v_offset));
            }));
          pack_bools(
            handle,
            bool_first,
            bool_first + cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last),
            rx_value_first);
        } else {
          auto map_first = thrust::make_transform_iterator(
            sorted_unique_vertex_first,
            cuda::proclaim_return_type<vertex_t>([vertex_partition] __device__(auto v) {
              return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            }));
          // FIXME: this gather (and temporary buffer) is unnecessary if NCCL directly takes a
          // permutation iterator (and directly gathers to the internal buffer)
          thrust::gather(
            handle.get_thrust_policy(),
            map_first,
            map_first + cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last),
            vertex_property_input_first,
            rx_value_first);
        }
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(minor_comm,
                   sorted_unique_vertex_first,
                   rx_vertices.begin(),
                   local_v_list_sizes[i],
                   i,
                   handle.get_stream());
      device_bcast(minor_comm,
                   rx_value_first,
                   rx_value_first,
                   contains_packed_bool_element ? packed_bool_size(local_v_list_sizes[i])
                                                : local_v_list_sizes[i],
                   i,
                   handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(local_v_list_sizes[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_value_first,
           edge_partition_key_first   = ((*edge_partition_keys)[i]).begin(),
           edge_partition_key_last    = ((*edge_partition_keys)[i]).end(),
           edge_partition_value_first = edge_partition_value_firsts[i]] __device__(size_t i) {
            auto major = *(rx_vertex_first + i);
            auto it    = thrust::lower_bound(
              thrust::seq, edge_partition_key_first, edge_partition_key_last, major);
            if ((it != edge_partition_key_last) && (*it == major)) {
              auto edge_partition_offset = cuda::std::distance(edge_partition_key_first, it);
              if constexpr (contains_packed_bool_element) {
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
        if constexpr (contains_packed_bool_element) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(vertex_t{0}),
            thrust::make_counting_iterator(static_cast<vertex_t>(local_v_list_sizes[i])),
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
            rx_vertices.begin(),
            cuda::proclaim_return_type<vertex_t>([edge_partition] __device__(auto v) {
              return edge_partition.major_offset_from_major_nocheck(v);
            }));
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(handle.get_thrust_policy(),
                          rx_value_first,
                          rx_value_first + local_v_list_sizes[i],
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
    if constexpr (contains_packed_bool_element) {
      thrust::for_each(handle.get_thrust_policy(),
                       sorted_unique_vertex_first,
                       sorted_unique_vertex_last,
                       [vertex_property_input_first,
                        output_value_first = edge_partition_value_firsts[0]] __device__(auto v) {
                         bool val = static_cast<bool>(*(vertex_property_input_first + v));
                         packed_bool_atomic_set(output_value_first, v, val);
                       });
    } else {
      auto val_first =
        thrust::make_permutation_iterator(vertex_property_input_first, sorted_unique_vertex_first);
      thrust::scatter(
        handle.get_thrust_policy(),
        val_first,
        val_first + cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last),
        sorted_unique_vertex_first,
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
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMinorPropertyOutputWrapper::value_type>();
  static_assert(!contains_packed_bool_element ||
                  std::is_arithmetic_v<typename EdgeMinorPropertyOutputWrapper::value_type>,
                "unimplemented for thrust::tuple types with a packed bool element.");

  auto edge_partition_value_first = edge_minor_property_output.value_first();
  if constexpr (GraphViewType::is_multi_gpu) {
    using vertex_t          = typename GraphViewType::vertex_type;
    using bcast_buffer_type = dataframe_buffer_type_t<
      std::conditional_t<contains_packed_bool_element,
                         uint32_t,
                         typename EdgeMinorPropertyOutputWrapper::value_type>>;

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
    if constexpr (contains_packed_bool_element) {
      bcast_size /= 8;  // bits to bytes
    } else {
      bcast_size *= sizeof(typename EdgeMinorPropertyOutputWrapper::value_type);
    }
    auto num_concurrent_bcasts =
      (static_cast<size_t>(graph_view.compute_number_of_edges(handle) / comm_size) *
       sizeof(vertex_t)) /
      std::max(bcast_size, size_t{1});
    num_concurrent_bcasts =
      std::min(std::max(num_concurrent_bcasts, size_t{1}), static_cast<size_t>(major_comm_size));
    auto num_rounds = (static_cast<size_t>(major_comm_size) + num_concurrent_bcasts - size_t{1}) /
                      num_concurrent_bcasts;

    auto edge_partition_keys = edge_minor_property_output.keys();

    std::optional<std::vector<bcast_buffer_type>> rx_value_buffers{std::nullopt};
    if (contains_packed_bool_element || edge_partition_keys) {
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
                     std::conditional_t<contains_packed_bool_element,
                                        uint32_t,
                                        typename EdgeMinorPropertyOutputWrapper::value_type>>(
            contains_packed_bool_element ? packed_bool_size(max_size) : max_size,
            handle.get_stream()));
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
      std::vector<size_t> local_v_list_sizes(major_comm_size, size_t{0});
      for (int i = 0; i < major_comm_size; ++i) {
        auto minor_range_vertex_partition_id =
          compute_local_edge_partition_minor_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        local_v_list_sizes[i] =
          graph_view.vertex_partition_range_size(minor_range_vertex_partition_id);
      }
      std::vector<size_t> rx_displacements(major_comm_size, size_t{0});
      std::exclusive_scan(
        local_v_list_sizes.begin(), local_v_list_sizes.end(), rx_displacements.begin(), size_t{0});
      key_offsets_or_rx_displacements = std::move(rx_displacements);
    }

    for (size_t round = 0; round < num_rounds; ++round) {
      if constexpr (contains_packed_bool_element) {
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
          if constexpr (contains_packed_bool_element) {
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
            if constexpr (contains_packed_bool_element) {
              if (edge_partition_keys) {
                auto key_offsets =
                  std::get<raft::host_span<vertex_t const>>(key_offsets_or_rx_displacements);

                auto bool_first = thrust::make_transform_iterator(
                  (*edge_partition_keys).begin() + key_offsets[j],
                  cuda::proclaim_return_type<bool>(
                    [rx_value_first,
                     v_first = graph_view.vertex_partition_range_first(
                       minor_range_vertex_partition_id)] __device__(auto v) {
                      auto v_offset = v - v_first;
                      return static_cast<bool>(*(rx_value_first + packed_bool_offset(v_offset)) &
                                               packed_bool_mask(v_offset));
                    }));
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
                  cuda::proclaim_return_type<bool>([rx_value_first] __device__(vertex_t v_offset) {
                    return static_cast<bool>(*(rx_value_first + packed_bool_offset(v_offset)) &
                                             packed_bool_mask(v_offset));
                  }));
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
                cuda::proclaim_return_type<vertex_t>(
                  [v_first = graph_view.vertex_partition_range_first(
                     minor_range_vertex_partition_id)] __device__(auto v) { return v - v_first; }));
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
    if constexpr (contains_packed_bool_element) {
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
                                VertexIterator sorted_unique_vertex_first,
                                VertexIterator sorted_unique_vertex_last,
                                VertexPropertyInputIterator vertex_property_input_first,
                                EdgeMinorPropertyOutputWrapper edge_minor_property_output)
{
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMinorPropertyOutputWrapper::value_type>();
  static_assert(!contains_packed_bool_element ||
                  std::is_arithmetic_v<typename EdgeMinorPropertyOutputWrapper::value_type>,
                "unimplemented for thrust::tuple types with a packed bool element.");

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_first = edge_minor_property_output.value_first();
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();

    auto v_list_size = static_cast<size_t>(
      cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last));
    std::array<vertex_t, 2> v_list_range = {vertex_t{0}, vertex_t{0}};
    if (v_list_size > 0) {
      rmm::device_uvector<vertex_t> tmps(2, handle.get_stream());
      thrust::tabulate(handle.get_thrust_policy(),
                       tmps.begin(),
                       tmps.end(),
                       [sorted_unique_vertex_first, v_list_size] __device__(size_t i) {
                         return (i == 0) ? *sorted_unique_vertex_first
                                         : (*(sorted_unique_vertex_first + (v_list_size - 1)) + 1);
                       });
      raft::update_host(v_list_range.data(), tmps.data(), 2, handle.get_stream());
      handle.sync_stream();
    }

    auto local_v_list_sizes = host_scalar_allgather(major_comm, v_list_size, handle.get_stream());
    auto local_v_list_range_firsts =
      host_scalar_allgather(major_comm, v_list_range[0], handle.get_stream());
    auto local_v_list_range_lasts =
      host_scalar_allgather(major_comm, v_list_range[1], handle.get_stream());

    std::optional<rmm::device_uvector<uint32_t>> v_list_bitmap{std::nullopt};
    if (major_comm_size > 1) {
      double avg_fill_ratio{0.0};
      for (int i = 0; i < major_comm_size; ++i) {
        auto num_keys   = static_cast<double>(local_v_list_sizes[i]);
        auto range_size = local_v_list_range_lasts[i] - local_v_list_range_firsts[i];
        avg_fill_ratio +=
          (range_size > 0) ? (num_keys / static_cast<double>(range_size)) : double{0.0};
      }
      avg_fill_ratio /= static_cast<double>(major_comm_size);

      constexpr double threshold_ratio =
        0.0 /* tuning parameter */ / static_cast<double>(sizeof(vertex_t) * 8);
      if (avg_fill_ratio > threshold_ratio) {
        v_list_bitmap = compute_vertex_list_bitmap_info(sorted_unique_vertex_first,
                                                        sorted_unique_vertex_last,
                                                        local_v_list_range_firsts[major_comm_rank],
                                                        local_v_list_range_lasts[major_comm_rank],
                                                        handle.get_stream());
      }
    }

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
      rmm::device_uvector<vertex_t> rx_vertices(local_v_list_sizes[i], handle.get_stream());
      auto rx_tmp_buffer = allocate_dataframe_buffer<
        std::conditional_t<contains_packed_bool_element,
                           uint32_t,
                           typename EdgeMinorPropertyOutputWrapper::value_type>>(
        contains_packed_bool_element ? packed_bool_size(local_v_list_sizes[i])
                                     : local_v_list_sizes[i],
        handle.get_stream());
      auto rx_value_first = get_dataframe_buffer_begin(rx_tmp_buffer);

      if (i == major_comm_rank) {
        auto vertex_partition =
          vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
            graph_view.local_vertex_partition_view());
        if constexpr (contains_packed_bool_element) {
          auto bool_first = thrust::make_transform_iterator(
            sorted_unique_vertex_first,
            cuda::proclaim_return_type<bool>([vertex_property_input_first,
                                              vertex_partition] __device__(auto v) {
              auto v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
              return static_cast<bool>(
                *(vertex_property_input_first + packed_bool_offset(v_offset)) &
                packed_bool_mask(v_offset));
            }));
          pack_bools(
            handle,
            bool_first,
            bool_first + cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last),
            rx_value_first);
        } else {
          auto map_first = thrust::make_transform_iterator(
            sorted_unique_vertex_first,
            cuda::proclaim_return_type<vertex_t>([vertex_partition] __device__(auto v) {
              return vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
            }));
          // FIXME: this gather (and temporary buffer) is unnecessary if NCCL directly takes a
          // permutation iterator (and directly gathers to the internal buffer)
          thrust::gather(
            handle.get_thrust_policy(),
            map_first,
            map_first + cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last),
            vertex_property_input_first,
            rx_value_first);
        }
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      std::variant<raft::device_span<uint32_t const>, decltype(sorted_unique_vertex_first)>
        v_list{};
      if (v_list_bitmap) {
        v_list =
          (i == major_comm_rank)
            ? raft::device_span<uint32_t const>((*v_list_bitmap).data(), (*v_list_bitmap).size())
            : raft::device_span<uint32_t const>(static_cast<uint32_t const*>(nullptr), size_t{0});
      } else {
        v_list = sorted_unique_vertex_first;
      }
      device_bcast_vertex_list(major_comm,
                               v_list,
                               rx_vertices.begin(),
                               local_v_list_range_firsts[i],
                               local_v_list_range_lasts[i],
                               local_v_list_sizes[i],
                               i,
                               handle.get_stream());
      device_bcast(major_comm,
                   rx_value_first,
                   rx_value_first,
                   contains_packed_bool_element ? packed_bool_size(local_v_list_sizes[i])
                                                : local_v_list_sizes[i],
                   i,
                   handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(local_v_list_sizes[i]),
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
              auto subrange_offset = cuda::std::distance(subrange_key_first, it);
              if constexpr (contains_packed_bool_element) {
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
        if constexpr (contains_packed_bool_element) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(vertex_t{0}),
            thrust::make_counting_iterator(static_cast<vertex_t>(local_v_list_sizes[i])),
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
            rx_vertices.begin(),
            cuda::proclaim_return_type<vertex_t>([edge_partition] __device__(auto v) {
              return edge_partition.minor_offset_from_minor_nocheck(v);
            }));
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(handle.get_thrust_policy(),
                          rx_value_first,
                          rx_value_first + local_v_list_sizes[i],
                          map_first,
                          edge_partition_value_first);
        }
      }
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() ==
           graph_view.local_edge_partition_src_range_size());
    if constexpr (contains_packed_bool_element) {
      thrust::for_each(handle.get_thrust_policy(),
                       sorted_unique_vertex_first,
                       sorted_unique_vertex_last,
                       [vertex_property_input_first,
                        output_value_first = edge_partition_value_first] __device__(auto v) {
                         bool val = static_cast<bool>(*(vertex_property_input_first + v));
                         packed_bool_atomic_set(output_value_first, v, val);
                       });
    } else {
      auto val_first =
        thrust::make_permutation_iterator(vertex_property_input_first, sorted_unique_vertex_first);
      thrust::scatter(
        handle.get_thrust_policy(),
        val_first,
        val_first + cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last),
        sorted_unique_vertex_first,
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
 * @tparam EdgeSrcValueOutputWrapper Type of the wrapper for output edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_src_property_output edge_src_property_view_t class object to store source
 * property values (for the edge sources assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexPropertyInputIterator,
          typename EdgeSrcValueOutputWrapper>
void update_edge_src_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexPropertyInputIterator vertex_property_input_first,
                              EdgeSrcValueOutputWrapper edge_src_property_output,
                              bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_minor_property(
      handle, graph_view, vertex_property_input_first, edge_src_property_output);
  } else {
    detail::update_edge_major_property(
      handle, graph_view, vertex_property_input_first, edge_src_property_output);
  }
}

/**
 * @brief Update graph edge source property values from the input vertex property values.
 *
 * This version updates only a subset of graph edge source property values. [@p
 * sorted_unique_vertex_first, @p sorted_unique_vertex_last) specifies the vertices with new
 * property values to be updated.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexPropertyInputIterator Type of the iterator for vertex property values.
 * @tparam EdgeSrcValueOutputWrapper Type of the wrapper for output edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param sorted_unique_vertex_first Iterator pointing to the first (inclusive) vertex with a new
 * value to be updated. v in [sorted_unique_vertex_first, sorted_unique_vertex_last) should be
 * sorted & distinct (and should belong to the vertex partition assigned to this process in
 * multi-GPU), otherwise undefined behavior.
 * @param sorted_unique_vertex_last Iterator pointing to the last (exclusive) vertex with a new
 * value.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_src_property_output edge_src_property_view_t class object to store source
 * property values (for the edge sources assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename VertexPropertyInputIterator,
          typename EdgeSrcValueOutputWrapper>
void update_edge_src_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexIterator sorted_unique_vertex_first,
                              VertexIterator sorted_unique_vertex_last,
                              VertexPropertyInputIterator vertex_property_input_first,
                              EdgeSrcValueOutputWrapper edge_src_property_output,
                              bool do_expensive_check = false)
{
  if (do_expensive_check) {
    auto num_invalids = thrust::count_if(
      handle.get_thrust_policy(),
      sorted_unique_vertex_first,
      sorted_unique_vertex_last,
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
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input argument: invalid or non-local vertices in "
                    "[sorted_unique_vertex_first, sorted_unique_vertex_last).");
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_minor_property(handle,
                                       graph_view,
                                       sorted_unique_vertex_first,
                                       sorted_unique_vertex_last,
                                       vertex_property_input_first,
                                       edge_src_property_output);
  } else {
    detail::update_edge_major_property(handle,
                                       graph_view,
                                       sorted_unique_vertex_first,
                                       sorted_unique_vertex_last,
                                       vertex_property_input_first,
                                       edge_src_property_output);
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
 * @tparam EdgeDstValueOutputWrapper Type of the wrapper for output edge destination property
 * values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_dst_property_output edge_dst_property_view_t class object to store
 * destination property values (for the edge destinations assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexPropertyInputIterator,
          typename EdgeDstValueOutputWrapper>
void update_edge_dst_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexPropertyInputIterator vertex_property_input_first,
                              EdgeDstValueOutputWrapper edge_dst_property_output,
                              bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_major_property(
      handle, graph_view, vertex_property_input_first, edge_dst_property_output);
  } else {
    detail::update_edge_minor_property(
      handle, graph_view, vertex_property_input_first, edge_dst_property_output);
  }
}

/**
 * @brief Update graph edge destination property values from the input vertex property values.
 *
 * This version updates only a subset of graph edge destination property values. [@p
 * sorted_unique_vertex_first, @p sorted_unique_vertex_last) specifies the vertices with new
 * property values to be updated.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexPropertyInputIterator Type of the iterator for vertex property values.
 * @tparam EdgeDstValueOutputWrapper Type of the wrapper for output edge destination property
 * values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param sorted_unique_vertex_first Iterator pointing to the first (inclusive) vertex with a new
 * value to be updated. v in [sorted_unique_vertex_first, sorted_unique_vertex_last) should be
 * sorted & distinct (and should belong to the vertex partition assigned to this process in
 * multi-GPU), otherwise undefined behavior.
 * @param sorted_unique_vertex_last Iterator pointing to the last (exclusive) vertex with a new
 * value.
 * @param vertex_property_input_first Iterator pointing to the vertex property value for the first
 * (inclusive) vertex (of the vertex partition assigned to this process in multi-GPU).
 * `vertex_property_input_last` (exclusive) is deduced as @p vertex_property_input_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param edge_partition_dst_property_output edge_dst_property_view_t class object to store
 * destination property values (for the edge destinations assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename VertexPropertyInputIterator,
          typename EdgeDstValueOutputWrapper>
void update_edge_dst_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexIterator sorted_unique_vertex_first,
                              VertexIterator sorted_unique_vertex_last,
                              VertexPropertyInputIterator vertex_property_input_first,
                              EdgeDstValueOutputWrapper edge_dst_property_output,
                              bool do_expensive_check = false)
{
  if (do_expensive_check) {
    auto num_invalids = thrust::count_if(
      handle.get_thrust_policy(),
      sorted_unique_vertex_first,
      sorted_unique_vertex_last,
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
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input argument: invalid or non-local vertices in "
                    "[sorted_unique_vertex_first, sorted_unique_vertex_last).");
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_major_property(handle,
                                       graph_view,
                                       sorted_unique_vertex_first,
                                       sorted_unique_vertex_last,
                                       vertex_property_input_first,
                                       edge_dst_property_output);
  } else {
    detail::update_edge_minor_property(handle,
                                       graph_view,
                                       sorted_unique_vertex_first,
                                       sorted_unique_vertex_last,
                                       vertex_property_input_first,
                                       edge_dst_property_output);
  }
}

}  // namespace cugraph

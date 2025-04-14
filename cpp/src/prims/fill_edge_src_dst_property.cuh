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

#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/atomic_ops.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>

#include <cstddef>
#include <utility>

namespace cugraph {

namespace detail {

template <typename Iterator, typename T>
__device__ std::enable_if_t<std::is_same_v<T, bool>, void> fill_thrust_tuple_element(Iterator iter,
                                                                                     size_t offset,
                                                                                     T value)
{
  packed_bool_atomic_set(iter, offset, value);
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<!std::is_same_v<T, bool>, void> fill_thrust_tuple_element(Iterator iter,
                                                                                      size_t offset,
                                                                                      T value)
{
  *(iter + offset) = value;
}

template <typename Iterator, typename T, std::size_t... Is>
__device__ void fill_thrust_tuple(Iterator iter, size_t offset, T value, std::index_sequence<Is...>)
{
  ((fill_thrust_tuple_element(
     thrust::get<Is>(iter.get_iterator_tuple()), offset, thrust::get<Is>(value))),
   ...);
}

template <typename Iterator, typename T>
__device__ void fill_scalar_or_thrust_tuple(Iterator iter, size_t offset, T value)
{
  if constexpr (std::is_arithmetic_v<T>) {
    if constexpr (cugraph::is_packed_bool<Iterator, T>) {
      packed_bool_atomic_set(iter, offset, value);
    } else {
      *(iter + offset) = value;
    }
  } else {
    if constexpr (cugraph::has_packed_bool_element<Iterator, T>) {
      fill_thrust_tuple(
        iter, offset, value, std::make_index_sequence<thrust::tuple_size<T>::value>());
    } else {
      *(iter + offset) = value;
    }
  }
}

template <typename GraphViewType, typename EdgeMajorPropertyOutputWrapper, typename T>
void fill_edge_major_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              EdgeMajorPropertyOutputWrapper edge_major_property_output,
                              T input)
{
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMajorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMajorPropertyOutputWrapper::value_type>();
  static_assert(std::is_same_v<T, typename EdgeMajorPropertyOutputWrapper::value_type>);
  static_assert(!contains_packed_bool_element ||
                  std::is_arithmetic_v<typename EdgeMajorPropertyOutputWrapper::value_type>,
                "unimplemented for thrust::tuple types with a packed bool element.");

  auto keys         = edge_major_property_output.keys();
  auto value_firsts = edge_major_property_output.value_firsts();
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    size_t num_buffer_elements{0};
    if (keys) {
      num_buffer_elements = (*keys)[i].size();
    } else {
      if constexpr (GraphViewType::is_storage_transposed) {
        num_buffer_elements =
          static_cast<size_t>(graph_view.local_edge_partition_dst_range_size(i));
      } else {
        num_buffer_elements =
          static_cast<size_t>(graph_view.local_edge_partition_src_range_size(i));
      }
    }
    if constexpr (contains_packed_bool_element) {
      auto packed_input = input ? packed_bool_full_mask() : packed_bool_empty_mask();
      thrust::fill_n(handle.get_thrust_policy(),
                     value_firsts[i],
                     packed_bool_size(num_buffer_elements),
                     packed_input);
    } else {
      thrust::fill_n(handle.get_thrust_policy(), value_firsts[i], num_buffer_elements, input);
    }
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename EdgeMajorPropertyOutputWrapper,
          typename T>
void fill_edge_major_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexIterator sorted_unique_vertex_first,
                              VertexIterator sorted_unique_vertex_last,
                              EdgeMajorPropertyOutputWrapper edge_major_property_output,
                              T input)
{
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMajorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMajorPropertyOutputWrapper::value_type>();
  static_assert(std::is_same_v<T, typename EdgeMajorPropertyOutputWrapper::value_type>);
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

    auto edge_partition_keys = edge_major_property_output.keys();
    for (int i = 0; i < minor_comm_size; ++i) {
      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(i));

      device_bcast(minor_comm,
                   sorted_unique_vertex_first,
                   rx_vertices.begin(),
                   local_v_list_sizes[i],
                   i,
                   handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(local_v_list_sizes[i]),
          [rx_vertex_first = rx_vertices.begin(),
           input,
           edge_partition_key_first   = ((*edge_partition_keys)[i]).begin(),
           edge_partition_key_last    = ((*edge_partition_keys)[i]).end(),
           edge_partition_value_first = edge_partition_value_firsts[i]] __device__(size_t i) {
            auto major = *(rx_vertex_first + i);
            auto it    = thrust::lower_bound(
              thrust::seq, edge_partition_key_first, edge_partition_key_last, major);
            if ((it != edge_partition_key_last) && (*it == major)) {
              auto edge_partition_offset = cuda::std::distance(edge_partition_key_first, it);
              if constexpr (contains_packed_bool_element) {
                packed_bool_atomic_set(edge_partition_value_first, edge_partition_offset, input);
              } else {
                *(edge_partition_value_first + edge_partition_offset) = input;
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
             input,
             output_value_first = edge_partition_value_firsts[i]] __device__(auto i) {
              auto rx_vertex    = *(rx_vertex_first + i);
              auto major_offset = edge_partition.major_offset_from_major_nocheck(rx_vertex);
              packed_bool_atomic_set(output_value_first, major_offset, input);
            });
        } else {
          auto map_first = thrust::make_transform_iterator(
            rx_vertices.begin(),
            cuda::proclaim_return_type<vertex_t>([edge_partition] __device__(auto v) {
              return edge_partition.major_offset_from_major_nocheck(v);
            }));
          auto val_first = thrust::make_constant_iterator(input);
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(handle.get_thrust_policy(),
                          val_first,
                          val_first + local_v_list_sizes[i],
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
                       [input, output_value_first = edge_partition_value_firsts[0]] __device__(
                         auto v) { packed_bool_atomic_set(output_value_first, v, input); });
    } else {
      auto val_first = thrust::make_constant_iterator(input);
      thrust::scatter(
        handle.get_thrust_policy(),
        val_first,
        val_first + cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last),
        sorted_unique_vertex_first,
        edge_partition_value_firsts[0]);
    }
  }
}

template <typename GraphViewType, typename EdgeMinorPropertyOutputWrapper, typename T>
void fill_edge_minor_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              EdgeMinorPropertyOutputWrapper edge_minor_property_output,
                              T input)
{
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMinorPropertyOutputWrapper::value_type>();
  static_assert(std::is_same_v<T, typename EdgeMinorPropertyOutputWrapper::value_type>);

  auto keys = edge_minor_property_output.keys();
  size_t num_buffer_elements{0};
  if (keys) {
    num_buffer_elements = (*keys).size();
  } else {
    if constexpr (GraphViewType::is_storage_transposed) {
      num_buffer_elements = static_cast<size_t>(graph_view.local_edge_partition_src_range_size());
    } else {
      num_buffer_elements = static_cast<size_t>(graph_view.local_edge_partition_dst_range_size());
    }
  }
  auto value_first = edge_minor_property_output.value_first();
  if constexpr (contains_packed_bool_element) {
    static_assert(std::is_arithmetic_v<T>, "unimplemented for thrust::tuple types.");
    auto packed_input = input ? packed_bool_full_mask() : packed_bool_empty_mask();
    thrust::fill_n(
      handle.get_thrust_policy(), value_first, packed_bool_size(num_buffer_elements), packed_input);
  } else {
    thrust::fill_n(handle.get_thrust_policy(), value_first, num_buffer_elements, input);
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename EdgeMinorPropertyOutputWrapper,
          typename T>
void fill_edge_minor_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexIterator sorted_unique_vertex_first,
                              VertexIterator sorted_unique_vertex_last,
                              EdgeMinorPropertyOutputWrapper edge_minor_property_output,
                              T input)
{
  constexpr bool contains_packed_bool_element =
    cugraph::has_packed_bool_element<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                                     typename EdgeMinorPropertyOutputWrapper::value_type>();
  static_assert(std::is_same_v<T, typename EdgeMinorPropertyOutputWrapper::value_type>);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_first = edge_minor_property_output.value_first();
  vertex_t minor_range_first{};
  if constexpr (GraphViewType::is_storage_transposed) {
    minor_range_first = graph_view.local_edge_partition_src_range_first();
  } else {
    minor_range_first = graph_view.local_edge_partition_dst_range_first();
  }

  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    constexpr size_t packed_bool_word_bcast_alignment =
      128 /
      sizeof(
        uint32_t);  // 128B cache line alignment (unaligned ncclBroadcast operations are slower)

    std::vector<size_t> max_tmp_buffer_sizes{};
    std::vector<vertex_t> local_v_list_sizes{};
    std::vector<vertex_t> local_v_list_range_firsts{};
    std::vector<vertex_t> local_v_list_range_lasts{};
    {
      auto v_list_size = static_cast<vertex_t>(
        cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last));
      rmm::device_uvector<size_t> d_aggregate_tmps(major_comm_size * size_t{4},
                                                   handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        d_aggregate_tmps.begin() + major_comm_rank * size_t{4},
        d_aggregate_tmps.begin() + (major_comm_rank + 1) * size_t{4},
        [max_tmp_buffer_size = static_cast<size_t>(
           static_cast<double>(handle.get_device_properties().totalGlobalMem) * 0.05),
         sorted_unique_vertex_first,
         v_list_size,
         vertex_partition_range_first =
           graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
          if (i == 0) {
            return max_tmp_buffer_size;
          } else if (i == 1) {
            return static_cast<size_t>(v_list_size);
          } else if (i == 2) {
            vertex_t first{};
            if (v_list_size > 0) {
              first = *sorted_unique_vertex_first;
            } else {
              first = vertex_partition_range_first;
            }
            assert(static_cast<vertex_t>(static_cast<size_t>(first)) == first);
            return static_cast<size_t>(first);
          } else {
            vertex_t last{};
            if (v_list_size > 0) {
              last = *(sorted_unique_vertex_first + (v_list_size - 1)) + 1;
            } else {
              last = vertex_partition_range_first;
            }
            assert(static_cast<vertex_t>(static_cast<size_t>(last)) == last);
            return static_cast<size_t>(last);
          }
        });

      if (major_comm_size > 1) {  // allgather max_tmp_buffer_size, v_list_size, v_list_range_first
                                  // (inclusive), v_list_range_last (exclusive)
        device_allgather(major_comm,
                         d_aggregate_tmps.data() + major_comm_rank * size_t{4},
                         d_aggregate_tmps.data(),
                         size_t{4},
                         handle.get_stream());
      }

      std::vector<size_t> h_aggregate_tmps(d_aggregate_tmps.size());
      raft::update_host(h_aggregate_tmps.data(),
                        d_aggregate_tmps.data(),
                        d_aggregate_tmps.size(),
                        handle.get_stream());
      handle.sync_stream();
      max_tmp_buffer_sizes      = std::vector<size_t>(major_comm_size);
      local_v_list_sizes        = std::vector<vertex_t>(major_comm_size);
      local_v_list_range_firsts = std::vector<vertex_t>(major_comm_size);
      local_v_list_range_lasts  = std::vector<vertex_t>(major_comm_size);
      for (int i = 0; i < major_comm_size; ++i) {
        max_tmp_buffer_sizes[i]      = h_aggregate_tmps[i * size_t{4}];
        local_v_list_sizes[i]        = static_cast<vertex_t>(h_aggregate_tmps[i * size_t{4} + 1]);
        local_v_list_range_firsts[i] = static_cast<vertex_t>(h_aggregate_tmps[i * size_t{4} + 2]);
        local_v_list_range_lasts[i]  = static_cast<vertex_t>(h_aggregate_tmps[i * size_t{4} + 3]);
      }
    }

    auto edge_partition_keys = edge_minor_property_output.keys();

    std::optional<rmm::device_uvector<uint32_t>> v_list_bitmap{std::nullopt};
    std::optional<rmm::device_uvector<uint32_t>> compressed_v_list{std::nullopt};
    if (major_comm_size > 1) {
      bool v_compressible{false};
      if constexpr (sizeof(vertex_t) > sizeof(uint32_t)) {
        vertex_t local_v_list_max_range_size{0};
        for (int i = 0; i < major_comm_size; ++i) {
          auto range_size             = local_v_list_range_lasts[i] - local_v_list_range_firsts[i];
          local_v_list_max_range_size = std::max(range_size, local_v_list_max_range_size);
        }
        if (local_v_list_max_range_size <=
            std::numeric_limits<uint32_t>::max()) {  // broadcast 32bit offset values instead of 64
                                                     // bit vertex IDs
          v_compressible = true;
        }
      }

      double avg_fill_ratio{0.0};
      for (int i = 0; i < major_comm_size; ++i) {
        auto num_keys   = static_cast<double>(local_v_list_sizes[i]);
        auto range_size = local_v_list_range_lasts[i] - local_v_list_range_firsts[i];
        avg_fill_ratio += (range_size > 0)
                            ? (static_cast<double>(num_keys) / static_cast<double>(range_size))
                            : double{0.0};
      }
      avg_fill_ratio /= static_cast<double>(major_comm_size);
      double threshold_ratio =
        1.0 / static_cast<double>((v_compressible ? sizeof(uint32_t) : sizeof(vertex_t)) * 8);
      auto avg_v_list_size = std::reduce(local_v_list_sizes.begin(), local_v_list_sizes.end()) /
                             static_cast<vertex_t>(major_comm_size);

      if ((avg_fill_ratio > threshold_ratio) &&
          (static_cast<size_t>(avg_v_list_size) > packed_bool_word_bcast_alignment)) {
        if (is_packed_bool<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                           typename EdgeMinorPropertyOutputWrapper::value_type>() &&
            !edge_partition_keys) {  // directly update edge_minor_property_output (with special
                                     // care for unaligned boundaries)
          rmm::device_uvector<uint32_t> boundary_words(
            packed_bool_word_bcast_alignment,
            handle.get_stream());  // for unaligned boundaries
          auto leading_boundary_words =
            (packed_bool_word_bcast_alignment -
             packed_bool_offset(local_v_list_range_firsts[major_comm_rank] - minor_range_first) %
               packed_bool_word_bcast_alignment) %
            packed_bool_word_bcast_alignment;
          if ((leading_boundary_words == 0) &&
              (packed_bool_offset(local_v_list_range_firsts[major_comm_rank] - minor_range_first) ==
               packed_bool_offset(graph_view.local_vertex_partition_range_first() -
                                  minor_range_first)) &&
              (((local_v_list_range_firsts[major_comm_rank] - minor_range_first) %
                packed_bools_per_word()) !=
               0)) {  // there are unaligned bits (fewer than packed_bools_per_word()) in the vertex
                      // partition boundary
            leading_boundary_words = packed_bool_word_bcast_alignment;
          }
          thrust::fill(handle.get_thrust_policy(),
                       boundary_words.begin(),
                       boundary_words.begin() + leading_boundary_words,
                       packed_bool_empty_mask());
          if (local_v_list_range_firsts[major_comm_rank] <
              local_v_list_range_lasts[major_comm_rank]) {
            auto word_offset_first =
              packed_bool_offset(local_v_list_range_firsts[major_comm_rank] - minor_range_first);
            auto word_offset_last =
              packed_bool_offset((local_v_list_range_lasts[major_comm_rank] - 1) -
                                 minor_range_first) +
              1;
            thrust::for_each(
              handle.get_thrust_policy(),
              thrust::make_counting_iterator(word_offset_first),
              thrust::make_counting_iterator(word_offset_last),
              [sorted_unique_vertex_first,
               sorted_unique_vertex_last,
               input,
               minor_range_first,
               leading_boundary_words,
               word_offset_first,
               vertex_partition_range_last = graph_view.local_vertex_partition_range_last(),
               output_value_first          = edge_partition_value_first,
               boundary_words              = raft::device_span<uint32_t>(
                 boundary_words.data(), boundary_words.size())] __device__(auto i) {
                auto& word = ((i - word_offset_first) < leading_boundary_words)
                               ? boundary_words[i - word_offset_first]
                               : *(output_value_first + i);
                auto word_v_first =
                  minor_range_first + static_cast<vertex_t>(i * packed_bools_per_word());
                auto word_v_last =
                  ((vertex_partition_range_last - word_v_first) <= packed_bools_per_word())
                    ? vertex_partition_range_last
                    : (word_v_first + static_cast<vertex_t>(packed_bools_per_word()));
                auto it = thrust::lower_bound(
                  thrust::seq, sorted_unique_vertex_first, sorted_unique_vertex_last, word_v_first);
                while ((it != sorted_unique_vertex_last) && (*it < word_v_last)) {
                  auto v_offset = *it - minor_range_first;
                  if (input) {
                    word |= packed_bool_mask(v_offset);
                  } else {
                    word &= ~packed_bool_mask(v_offset);
                  }
                  ++it;
                }
              });
          }
          rmm::device_uvector<uint32_t> aggregate_boundary_words(
            major_comm_size * packed_bool_word_bcast_alignment, handle.get_stream());
          device_allgather(major_comm,
                           boundary_words.data(),
                           aggregate_boundary_words.data(),
                           packed_bool_word_bcast_alignment,
                           handle.get_stream());
          v_list_bitmap = std::move(aggregate_boundary_words);
        } else {
          v_list_bitmap =
            compute_vertex_list_bitmap_info(sorted_unique_vertex_first,
                                            sorted_unique_vertex_last,
                                            local_v_list_range_firsts[major_comm_rank],
                                            local_v_list_range_lasts[major_comm_rank],
                                            handle.get_stream());
        }
      } else if (v_compressible) {
        rmm::device_uvector<uint32_t> tmps(local_v_list_sizes[major_comm_rank],
                                           handle.get_stream());
        thrust::transform(handle.get_thrust_policy(),
                          sorted_unique_vertex_first,
                          sorted_unique_vertex_last,
                          tmps.begin(),
                          cuda::proclaim_return_type<uint32_t>(
                            [range_first = local_v_list_range_firsts[major_comm_rank]] __device__(
                              auto v) { return static_cast<uint32_t>(v - range_first); }));
        compressed_v_list = std::move(tmps);
      }
    }

    std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
    size_t num_concurrent_bcasts{1};
    {
      size_t tmp_buffer_size_per_loop{};
      for (int i = 0; i < major_comm_size; ++i) {
        if (is_packed_bool<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                           typename EdgeMinorPropertyOutputWrapper::value_type>() &&
            !edge_partition_keys && v_list_bitmap) {
          tmp_buffer_size_per_loop += 0;
        } else if (v_list_bitmap) {
          tmp_buffer_size_per_loop +=
            packed_bool_size(local_v_list_range_lasts[i] - local_v_list_range_firsts[i]) *
              sizeof(uint32_t) +
            static_cast<size_t>(local_v_list_sizes[i]) * sizeof(vertex_t);
        } else {
          tmp_buffer_size_per_loop += static_cast<size_t>(local_v_list_sizes[i]) * sizeof(vertex_t);
        }
      }
      tmp_buffer_size_per_loop /= major_comm_size;
      size_t max_streams =
        static_cast<size_t>(major_comm_size);  // to allow setting num_concurrent_bcasts above
                                               // hnadle.get_stream_pool_size()
      stream_pool_indices = init_stream_pool_indices(
        std::reduce(max_tmp_buffer_sizes.begin(), max_tmp_buffer_sizes.end()) /
          static_cast<size_t>(major_comm_size),
        tmp_buffer_size_per_loop,
        major_comm_size,
        1,
        max_streams);
      num_concurrent_bcasts = (*stream_pool_indices).size();
      if ((*stream_pool_indices).size() > handle.get_stream_pool_size()) {
        (*stream_pool_indices).resize(handle.get_stream_pool_size());
      }
      if ((*stream_pool_indices).size() <= 1) { stream_pool_indices = std::nullopt; }
    }

    std::optional<raft::host_span<vertex_t const>> key_offsets{};
    if constexpr (GraphViewType::is_storage_transposed) {
      key_offsets = graph_view.local_sorted_unique_edge_src_vertex_partition_offsets();
    } else {
      key_offsets = graph_view.local_sorted_unique_edge_dst_vertex_partition_offsets();
    }

    for (size_t i = 0; i < static_cast<size_t>(major_comm_size); i += num_concurrent_bcasts) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto sub0       = std::chrono::steady_clock::now();
      auto loop_count = std::min(num_concurrent_bcasts, static_cast<size_t>(major_comm_size) - i);

      if (is_packed_bool<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                         typename EdgeMinorPropertyOutputWrapper::value_type>() &&
          !edge_partition_keys && v_list_bitmap) {
        std::vector<size_t> leading_boundary_word_counts(loop_count);
        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;
          auto leading_boundary_words =
            (packed_bool_word_bcast_alignment -
             packed_bool_offset(local_v_list_range_firsts[partition_idx] - minor_range_first) %
               packed_bool_word_bcast_alignment) %
            packed_bool_word_bcast_alignment;
          auto vertex_partition_id =
            partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
              major_comm_size, minor_comm_size, partition_idx, minor_comm_rank);
          if ((leading_boundary_words == 0) &&
              (packed_bool_offset(local_v_list_range_firsts[partition_idx] - minor_range_first) ==
               packed_bool_offset(graph_view.vertex_partition_range_first(vertex_partition_id) -
                                  minor_range_first)) &&
              (((local_v_list_range_firsts[partition_idx] - minor_range_first) %
                packed_bools_per_word()) != 0)) {
            leading_boundary_words = packed_bool_word_bcast_alignment;
          }
          leading_boundary_word_counts[j] = leading_boundary_words;
        }
        device_group_start(major_comm);
        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;
          size_t bcast_size{0};
          vertex_t packed_bool_offset_first{0};
          if (local_v_list_range_firsts[partition_idx] < local_v_list_range_lasts[partition_idx]) {
            auto leading_boundary_words = leading_boundary_word_counts[j];
            packed_bool_offset_first =
              packed_bool_offset(local_v_list_range_firsts[partition_idx] - minor_range_first) +
              static_cast<vertex_t>(leading_boundary_words);
            auto packed_bool_offset_last =
              packed_bool_offset(local_v_list_range_lasts[partition_idx] - 1 - minor_range_first);
            if (packed_bool_offset_first <= packed_bool_offset_last) {
              bcast_size = (packed_bool_offset_last - packed_bool_offset_first) + 1;
            }
          }

          device_bcast(major_comm,
                       edge_partition_value_first + packed_bool_offset_first,
                       edge_partition_value_first + packed_bool_offset_first,
                       bcast_size,
                       static_cast<int>(partition_idx),
                       handle.get_stream());
        }
        device_group_end(major_comm);

        rmm::device_uvector<size_t> d_leading_boundary_word_counts(
          leading_boundary_word_counts.size(), handle.get_stream());
        raft::update_device(d_leading_boundary_word_counts.data(),
                            leading_boundary_word_counts.data(),
                            leading_boundary_word_counts.size(),
                            handle.get_stream());

        rmm::device_uvector<vertex_t> d_local_v_list_range_firsts(loop_count, handle.get_stream());
        raft::update_device(d_local_v_list_range_firsts.data(),
                            local_v_list_range_firsts.data() + i,
                            loop_count,
                            handle.get_stream());

        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(loop_count * packed_bool_word_bcast_alignment),
          [input,
           minor_range_first,
           leading_boundary_word_counts = raft::device_span<size_t const>(
             d_leading_boundary_word_counts.data(), d_leading_boundary_word_counts.size()),
           local_v_list_range_firsts = raft::device_span<vertex_t const>(
             d_local_v_list_range_firsts.data(), d_local_v_list_range_firsts.size()),
           aggregate_boundary_words = raft::device_span<uint32_t const>(
             (*v_list_bitmap).data() + i * packed_bool_word_bcast_alignment,
             loop_count * packed_bool_word_bcast_alignment),
           output_value_first = edge_partition_value_first] __device__(size_t i) {
            auto j                      = i / packed_bool_word_bcast_alignment;
            auto leading_boundary_words = leading_boundary_word_counts[j];
            if ((i % packed_bool_word_bcast_alignment) < leading_boundary_words) {
              auto boundary_word = aggregate_boundary_words[i];
              if (boundary_word != packed_bool_empty_mask()) {
                auto word_offset =
                  packed_bool_offset(local_v_list_range_firsts[j] - minor_range_first) +
                  (i % packed_bool_word_bcast_alignment);
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
                  *(output_value_first + word_offset));
                if (input) {
                  word.fetch_or(aggregate_boundary_words[i], cuda::std::memory_order_relaxed);
                } else {
                  word.fetch_and(~aggregate_boundary_words[i], cuda::std::memory_order_relaxed);
                }
              }
            }
          });
      } else {
        std::vector<std::variant<rmm::device_uvector<vertex_t>, rmm::device_uvector<uint32_t>>>
          edge_partition_v_buffers{};
        edge_partition_v_buffers.reserve(loop_count);
        rmm::device_uvector<size_t> dummy_counters(loop_count, handle.get_stream());
        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;

          std::variant<rmm::device_uvector<vertex_t>, rmm::device_uvector<uint32_t>> v_buffer =
            rmm::device_uvector<vertex_t>(0, handle.get_stream());
          if (v_list_bitmap) {
            v_buffer = rmm::device_uvector<uint32_t>(
              packed_bool_size(local_v_list_range_lasts[partition_idx] -
                               local_v_list_range_firsts[partition_idx]),
              handle.get_stream());
          } else if (compressed_v_list) {
            v_buffer =
              rmm::device_uvector<uint32_t>(local_v_list_sizes[partition_idx], handle.get_stream());
          } else {
            std::get<0>(v_buffer).resize(local_v_list_sizes[partition_idx], handle.get_stream());
          }
          edge_partition_v_buffers.push_back(std::move(v_buffer));
        }

        device_group_start(major_comm);
        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;

          auto& v_buffer = edge_partition_v_buffers[j];
          if (v_list_bitmap) {
            device_bcast(major_comm,
                         (*v_list_bitmap).data(),
                         std::get<1>(v_buffer).data(),
                         std::get<1>(v_buffer).size(),
                         static_cast<int>(partition_idx),
                         handle.get_stream());
          } else if (compressed_v_list) {
            device_bcast(major_comm,
                         (*compressed_v_list).data(),
                         std::get<1>(v_buffer).data(),
                         std::get<1>(v_buffer).size(),
                         static_cast<int>(partition_idx),
                         handle.get_stream());
          } else {
            device_bcast(major_comm,
                         (static_cast<int>(partition_idx) == major_comm_rank)
                           ? sorted_unique_vertex_first
                           : static_cast<vertex_t const*>(nullptr),
                         std::get<0>(v_buffer).data(),
                         std::get<0>(v_buffer).size(),
                         static_cast<int>(partition_idx),
                         handle.get_stream());
          }
        }
        device_group_end(major_comm);
        bool kernel_fusion =
          !edge_partition_keys && !v_list_bitmap && (loop_count > 1) &&
          (static_cast<size_t>(std::reduce(local_v_list_sizes.begin() + i,
                                           local_v_list_sizes.begin() + (i + loop_count))) <
           size_t{256 * 1024} /* tuning parameter (binary search vs kernel launch overhead) */ *
             loop_count);  // FIXME: kernle fusion can be useful even when
                           // edge_partition_keys.has_value() is true

        if (!kernel_fusion) {
          if (stream_pool_indices) { handle.sync_stream(); }
        }

        if (!kernel_fusion) {
          size_t stream_pool_size{0};
          if (stream_pool_indices) { stream_pool_size = (*stream_pool_indices).size(); }
          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto loop_stream =
              stream_pool_indices
                ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j % stream_pool_size])
                : handle.get_stream();

            if (v_list_bitmap) {
              auto const& rx_bitmap = std::get<1>(edge_partition_v_buffers[j]);
              rmm::device_uvector<vertex_t> rx_vertices(local_v_list_sizes[partition_idx],
                                                        loop_stream);
              retrieve_vertex_list_from_bitmap(
                raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                rx_vertices.begin(),
                raft::device_span<size_t>(dummy_counters.data() + j, size_t{1}),
                local_v_list_range_firsts[partition_idx],
                local_v_list_range_lasts[partition_idx],
                loop_stream);
              edge_partition_v_buffers[j] = std::move(rx_vertices);
            }

            if (edge_partition_keys) {
              thrust::for_each(
                rmm::exec_policy_nosync(loop_stream),
                thrust::make_counting_iterator(vertex_t{0}),
                thrust::make_counting_iterator(local_v_list_sizes[partition_idx]),
                [rx_vertex_first            = compressed_v_list
                                                ? static_cast<vertex_t const*>(nullptr)
                                                : std::get<0>(edge_partition_v_buffers[j]).data(),
                 rx_compressed_vertex_first = compressed_v_list
                                                ? std::get<1>(edge_partition_v_buffers[j]).data()
                                                : static_cast<uint32_t const*>(nullptr),
                 range_first                = local_v_list_range_firsts[partition_idx],
                 input,
                 subrange_key_first =
                   (*edge_partition_keys).begin() + (*key_offsets)[partition_idx],
                 subrange_key_last =
                   (*edge_partition_keys).begin() + (*key_offsets)[partition_idx + 1],
                 edge_partition_value_first = edge_partition_value_first,
                 subrange_start_offset      = (*key_offsets)[partition_idx]] __device__(auto i) {
                  vertex_t minor{};
                  if (rx_vertex_first != nullptr) {
                    minor = *(rx_vertex_first + i);
                  } else {
                    minor = range_first + *(rx_compressed_vertex_first + i);
                  }
                  auto it =
                    thrust::lower_bound(thrust::seq, subrange_key_first, subrange_key_last, minor);
                  if ((it != subrange_key_last) && (*it == minor)) {
                    auto subrange_offset = cuda::std::distance(subrange_key_first, it);
                    if constexpr (contains_packed_bool_element) {
                      fill_scalar_or_thrust_tuple(
                        edge_partition_value_first, subrange_start_offset + subrange_offset, input);
                    } else {
                      *(edge_partition_value_first + subrange_start_offset + subrange_offset) =
                        input;
                    }
                  }
                });
            } else {
              if constexpr (contains_packed_bool_element) {
                thrust::for_each(
                  rmm::exec_policy_nosync(loop_stream),
                  thrust::make_counting_iterator(vertex_t{0}),
                  thrust::make_counting_iterator(local_v_list_sizes[partition_idx]),
                  [minor_range_first,
                   rx_vertex_first            = compressed_v_list
                                                  ? static_cast<vertex_t const*>(nullptr)
                                                  : std::get<0>(edge_partition_v_buffers[j]).data(),
                   rx_compressed_vertex_first = compressed_v_list
                                                  ? std::get<1>(edge_partition_v_buffers[j]).data()
                                                  : static_cast<uint32_t const*>(nullptr),
                   range_first                = local_v_list_range_firsts[partition_idx],
                   input,
                   output_value_first = edge_partition_value_first] __device__(auto i) {
                    vertex_t minor{};
                    if (rx_vertex_first != nullptr) {
                      minor = *(rx_vertex_first + i);
                    } else {
                      minor = range_first + *(rx_compressed_vertex_first + i);
                    }
                    auto minor_offset = minor - minor_range_first;
                    fill_scalar_or_thrust_tuple(output_value_first, minor_offset, input);
                  });
              } else {
                if (compressed_v_list) {
                  auto map_first = thrust::make_transform_iterator(
                    std::get<1>(edge_partition_v_buffers[j]).begin(),
                    cuda::proclaim_return_type<vertex_t>(
                      [minor_range_first,
                       range_first =
                         local_v_list_range_firsts[partition_idx]] __device__(auto v_offset) {
                        return static_cast<vertex_t>(v_offset + (range_first - minor_range_first));
                      }));
                  auto val_first = thrust::make_constant_iterator(input);
                  thrust::scatter(rmm::exec_policy_nosync(loop_stream),
                                  val_first,
                                  val_first + local_v_list_sizes[partition_idx],
                                  map_first,
                                  edge_partition_value_first);
                } else {
                  auto map_first = thrust::make_transform_iterator(
                    std::get<0>(edge_partition_v_buffers[j]).begin(),
                    cuda::proclaim_return_type<vertex_t>(
                      [minor_range_first] __device__(auto v) { return v - minor_range_first; }));
                  auto val_first = thrust::make_constant_iterator(input);
                  thrust::scatter(rmm::exec_policy_nosync(loop_stream),
                                  val_first,
                                  val_first + local_v_list_sizes[partition_idx],
                                  map_first,
                                  edge_partition_value_first);
                }
              }
            }
          }
          if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
        } else {  // kernel fusion
          std::vector<vertex_t> h_vertex_vars(loop_count /* range_first values */ +
                                              (loop_count + 1) /* loop offsets */);
          std::copy(local_v_list_range_firsts.begin() + i,
                    local_v_list_range_firsts.begin() + (i + loop_count),
                    h_vertex_vars.begin());
          h_vertex_vars[loop_count] = 0;
          std::inclusive_scan(local_v_list_sizes.begin() + i,
                              local_v_list_sizes.begin() + (i + loop_count),
                              h_vertex_vars.begin() + (loop_count + 1));
          std::vector<void const*> h_ptrs(loop_count);
          if (compressed_v_list) {
            for (size_t j = 0; j < loop_count; ++j) {
              h_ptrs[j] = static_cast<void const*>(std::get<1>(edge_partition_v_buffers[j]).data());
            }
          } else {
            for (size_t j = 0; j < loop_count; ++j) {
              h_ptrs[j] = static_cast<void const*>(std::get<0>(edge_partition_v_buffers[j]).data());
            }
          }
          rmm::device_uvector<vertex_t> d_vertex_vars(h_vertex_vars.size(), handle.get_stream());
          rmm::device_uvector<void const*> d_ptrs(h_ptrs.size(), handle.get_stream());
          raft::update_device(
            d_vertex_vars.data(), h_vertex_vars.data(), h_vertex_vars.size(), handle.get_stream());
          raft::update_device(d_ptrs.data(), h_ptrs.data(), h_ptrs.size(), handle.get_stream());

          raft::device_span<vertex_t const> range_firsts(d_vertex_vars.data(), loop_count);
          raft::device_span<vertex_t const> loop_offsets(d_vertex_vars.data() + loop_count,
                                                         loop_count + 1);
          if constexpr (contains_packed_bool_element) {
            thrust::for_each(
              handle.get_thrust_policy(),
              thrust::make_counting_iterator(vertex_t{0}),
              thrust::make_counting_iterator(h_vertex_vars.back()),
              [range_firsts,
               loop_offsets,
               minor_range_first,
               input,
               rx_firsts = raft::device_span<void const* const>(d_ptrs.data(), d_ptrs.size()),
               output_value_first = edge_partition_value_first,
               compressed         = compressed_v_list.has_value()] __device__(auto i) {
                auto loop_idx = cuda::std::distance(
                  loop_offsets.begin() + 1,
                  thrust::upper_bound(
                    thrust::seq, loop_offsets.begin() + 1, loop_offsets.end(), i));
                auto rx_first = rx_firsts[loop_idx];
                vertex_t minor{};
                if (compressed) {
                  minor = range_firsts[loop_idx] +
                          *(static_cast<uint32_t const*>(rx_first) + (i - loop_offsets[loop_idx]));
                } else {
                  minor = *(static_cast<vertex_t const*>(rx_first) + (i - loop_offsets[loop_idx]));
                }
                auto minor_offset = minor - minor_range_first;
                fill_scalar_or_thrust_tuple(output_value_first, minor_offset, input);
              });
          } else {
            auto val_first = thrust::make_constant_iterator(input);
            if (compressed_v_list) {
              auto map_first = thrust::make_transform_iterator(
                thrust::make_counting_iterator(vertex_t{0}),
                cuda::proclaim_return_type<vertex_t>(
                  [range_firsts,
                   loop_offsets,
                   rx_firsts = raft::device_span<void const* const>(d_ptrs.data(), d_ptrs.size()),
                   minor_range_first] __device__(auto i) {
                    auto loop_idx = cuda::std::distance(
                      loop_offsets.begin() + 1,
                      thrust::upper_bound(
                        thrust::seq, loop_offsets.begin() + 1, loop_offsets.end(), i));
                    auto minor =
                      range_firsts[loop_idx] + *(static_cast<uint32_t const*>(rx_firsts[loop_idx]) +
                                                 (i - loop_offsets[loop_idx]));
                    return minor - minor_range_first;
                  }));
              thrust::scatter(handle.get_thrust_policy(),
                              val_first,
                              val_first + h_vertex_vars.back(),
                              map_first,
                              edge_partition_value_first);
            } else {
              auto map_first = thrust::make_transform_iterator(
                thrust::make_counting_iterator(vertex_t{0}),
                cuda::proclaim_return_type<vertex_t>(
                  [loop_offsets,
                   rx_firsts = raft::device_span<void const* const>(d_ptrs.data(), d_ptrs.size()),
                   minor_range_first] __device__(auto i) {
                    auto loop_idx = cuda::std::distance(
                      loop_offsets.begin() + 1,
                      thrust::upper_bound(
                        thrust::seq, loop_offsets.begin() + 1, loop_offsets.end(), i));
                    auto minor = *(static_cast<vertex_t const*>(rx_firsts[loop_idx]) +
                                   (i - loop_offsets[loop_idx]));
                    return minor - minor_range_first;
                  }));
              thrust::scatter(handle.get_thrust_policy(),
                              val_first,
                              val_first + h_vertex_vars.back(),
                              map_first,
                              edge_partition_value_first);
            }
          }
        }
      }
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() ==
           (GraphViewType::is_storage_transposed
              ? graph_view.local_edge_partition_src_range_size()
              : graph_view.local_edge_partition_dst_range_size()));
    if constexpr (contains_packed_bool_element) {
      thrust::for_each(handle.get_thrust_policy(),
                       sorted_unique_vertex_first,
                       sorted_unique_vertex_last,
                       [input, output_value_first = edge_partition_value_first] __device__(auto v) {
                         fill_scalar_or_thrust_tuple(output_value_first, v, input);
                       });
    } else {
      auto val_first = thrust::make_constant_iterator(input);
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
 * @brief Fill graph edge source property values to the input value.
 *
 * This version fills graph edge source property values for the entire edge source ranges (assigned
 * to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueOutputWrapper Type of the wrapper for output edge source property values.
 * @tparam T Type of the edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_property_output edge_src_property_view_t class object to store source property
 * values (for the edge source assigned to this process in multi-GPU).
 * @param input Edge source property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename EdgeSrcValueOutputWrapper, typename T>
void fill_edge_src_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            EdgeSrcValueOutputWrapper edge_src_property_output,
                            T input,
                            bool do_expensive_check = false)
{
  static_assert(std::is_same_v<T, typename EdgeSrcValueOutputWrapper::value_type>);
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::fill_edge_minor_property(handle, graph_view, edge_src_property_output, input);
  } else {
    detail::fill_edge_major_property(handle, graph_view, edge_src_property_output, input);
  }
}

/**
 * @brief Fill graph edge source property values to the input value.
 *
 * This version fills only a subset of graph edge source property values. [@p
 * sorted_unique_vertex_first, @p sorted_unique_vertex_last) specifies the vertices to be filled.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator Type of the iterator for vertex identifiers.
 * @tparam EdgeSrcValueOutputWrapper Type of the wrapper for output edge source property values.
 * @tparam T Type of the edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param sorted_unique_vertex_first Iterator pointing to the first (inclusive) vertex with a value
 * to be filled. v in [vertex_first, sorted_unique_vertex_last) should be sorted & distinct (and
 * should belong to the vertex partition assigned to this process in multi-GPU), otherwise undefined
 * behavior.
 * @param sorted_unique_vertex_last Iterator pointing to the last (exclusive) vertex with a value to
 * be filled.
 * @param edge_src_property_output edge_src_property_view_t class object to store source property
 * values (for the edge source assigned to this process in multi-GPU).
 * @param input Edge source property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename EdgeSrcValueOutputWrapper,
          typename T>
void fill_edge_src_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexIterator sorted_unique_vertex_first,
                            VertexIterator sorted_unique_vertex_last,
                            EdgeSrcValueOutputWrapper edge_src_property_output,
                            T input,
                            bool do_expensive_check = false)
{
  static_assert(std::is_same_v<T, typename EdgeSrcValueOutputWrapper::value_type>);
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
    detail::fill_edge_minor_property(handle,
                                     graph_view,
                                     sorted_unique_vertex_first,
                                     sorted_unique_vertex_last,
                                     edge_src_property_output,
                                     input);
  } else {
    detail::fill_edge_major_property(handle,
                                     graph_view,
                                     sorted_unique_vertex_first,
                                     sorted_unique_vertex_last,
                                     edge_src_property_output,
                                     input);
  }
}

/**
 * @brief Fill graph edge destination property values to the input value.
 *
 * This version fills graph edge destination property values for the entire edge destination ranges
 * (assigned to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeDstValueOutputWrapper Type of the wrapper for output edge destination property
 * values.
 * @tparam T Type of the edge destination property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_dst_property_output edge_dst_property_view_t class object to store destination
 * property values (for the edge destinations assigned to this process in multi-GPU).
 * @param input Edge destination property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename EdgeDstValueOutputWrapper, typename T>
void fill_edge_dst_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            EdgeDstValueOutputWrapper edge_dst_property_output,
                            T input,
                            bool do_expensive_check = false)
{
  static_assert(std::is_same_v<T, typename EdgeDstValueOutputWrapper::value_type>);
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::fill_edge_major_property(handle, graph_view, edge_dst_property_output, input);
  } else {
    detail::fill_edge_minor_property(handle, graph_view, edge_dst_property_output, input);
  }
}

/**
 * @brief Fill graph edge destination property values to the input value.
 *
 * This version fills only a subset of graph edge destination property values. [@p
 * sorted_unique_vertex_first, @p sorted_unique_vertex_last) specifies the vertices to be filled.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator Type of the iterator for vertex identifiers.
 * @tparam EdgeDstValueOutputWrapper Type of the wrapper for output edge destination property
 * values.
 * @tparam T Type of the edge destination property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param sorted_unique_vertex_first Iterator pointing to the first (inclusive) vertex with a value
 * to be filled. v in [sorted_unique_vertex_first, sorted_unique_vertex_last) should be sorted &
 * distinct (and should belong to the vertex partition assigned to this process in multi-GPU),
 * otherwise undefined behavior.
 * @param sorted_unique_vertex_last Iterator pointing to the last (exclusive) vertex with a value to
 * be filled.
 * @param edge_dst_property_output edge_dst_property_view_t class object to store destination
 * property values (for the edge destinations assigned to this process in multi-GPU).
 * @param input Edge destination property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename EdgeDstValueOutputWrapper,
          typename T>
void fill_edge_dst_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexIterator sorted_unique_vertex_first,
                            VertexIterator sorted_unique_vertex_last,
                            EdgeDstValueOutputWrapper edge_dst_property_output,
                            T input,
                            bool do_expensive_check = false)
{
  static_assert(std::is_same_v<T, typename EdgeDstValueOutputWrapper::value_type>);
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
    detail::fill_edge_major_property(handle,
                                     graph_view,
                                     sorted_unique_vertex_first,
                                     sorted_unique_vertex_last,
                                     edge_dst_property_output,
                                     input);
  } else {
    detail::fill_edge_minor_property(handle,
                                     graph_view,
                                     sorted_unique_vertex_first,
                                     sorted_unique_vertex_last,
                                     edge_dst_property_output,
                                     input);
  }
}

}  // namespace cugraph

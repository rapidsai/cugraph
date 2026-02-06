/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/host_staging_buffer_manager.hpp>
#include <cugraph/utilities/atomic_ops.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

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
     cuda::std::get<Is>(iter.get_iterator_tuple()), offset, cuda::std::get<Is>(value))),
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
        iter, offset, value, std::make_index_sequence<cuda::std::tuple_size<T>::value>());
    } else {
      *(iter + offset) = value;
    }
  }
}

template <typename vertex_t>
struct within_valid_range_t {
  raft::device_span<vertex_t const> local_v_list_sizes{};
  vertex_t max_padded_local_v_list_size{};
  __device__ bool operator()(vertex_t i) const
  {
    auto loop_idx = i / max_padded_local_v_list_size;
    auto offset   = i % max_padded_local_v_list_size;
    return offset < local_v_list_sizes[loop_idx];
  }
};

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
                "unimplemented for cuda::std::tuple types with a packed bool element.");

  auto keys         = edge_major_property_output.major_keys();
  auto value_firsts = edge_major_property_output.major_value_firsts();
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
                "unimplemented for cuda::std::tuple types with a packed bool element.");

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_firsts = edge_major_property_output.major_value_firsts();
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

#if 1  // FIXME: we should add host_allgather to raft
    auto local_v_list_sizes =
      host_scalar_allgather(comm,
                            static_cast<size_t>(cuda::std::distance(sorted_unique_vertex_first,
                                                                    sorted_unique_vertex_last)),
                            handle.get_stream());
#else
    std::vector<size_t> local_v_list_sizes(minor_comm_size, 0);
    local_v_list_sizes[minor_comm_rank] = static_cast<size_t>(
      cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last));
    minor_comm.host_allgather(local_v_list_sizes.data(), local_v_list_sizes.data(), size_t{1});
#endif
    auto max_rx_size = std::reduce(
      local_v_list_sizes.begin(), local_v_list_sizes.end(), size_t{0}, [](auto lhs, auto rhs) {
        return std::max(lhs, rhs);
      });
    rmm::device_uvector<vertex_t> rx_vertices(max_rx_size, handle.get_stream());

    auto edge_partition_keys = edge_major_property_output.major_keys();
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

  auto keys = edge_minor_property_output.minor_keys();
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
  auto value_first = edge_minor_property_output.minor_value_first();
  if constexpr (contains_packed_bool_element) {
    static_assert(std::is_arithmetic_v<T>, "unimplemented for cuda::std::tuple types.");
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

  auto edge_partition_value_first = edge_minor_property_output.minor_value_first();
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
      cache_line_size /
      sizeof(uint32_t);  // cache line alignment,  unaligned ncclBroadcast operations are slower

    // we should consider reducing the life-time of this variable
    // oncermm::rm::pool_memory_resource<rmm::mr::pinned_memory_resource> is updated to honor stream
    // semantics (github.com/rapidsai/rmm/issues/2053)
    rmm::device_uvector<int64_t> h_staging_buffer(0, handle.get_stream());
    {
      size_t staging_buffer_size{};  // should be large enough to cover all update_host &
                                     // update_device calls in this primitive
      if (GraphViewType::is_multi_gpu) {
        auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        staging_buffer_size =
          static_cast<size_t>(major_comm_size) *
          (size_t{3} + (raft::round_up_safe(packed_bool_word_bcast_alignment, size_t{2}) /
                        2 /* two packed bool words per 64 bit */));
      } else {
        staging_buffer_size =
          size_t{3} + (raft::round_up_safe(packed_bool_word_bcast_alignment, size_t{2}) /
                       2 /* two packed bool words per 64 bit */);
      }
      h_staging_buffer = host_staging_buffer_manager::allocate_staging_buffer<int64_t>(
        staging_buffer_size, handle.get_stream());
    }

    auto edge_partition_keys = edge_minor_property_output.minor_keys();

    std::vector<vertex_t> local_v_list_sizes{};
    std::vector<vertex_t> local_v_list_range_firsts{};
    std::vector<vertex_t> local_v_list_range_lasts{};
    bool direct_bcast{false};  // directly broadcast to the array pointed by edge_minor_value_first
                               // (with special care for unaligned boundaries)
    vertex_t aggregate_local_v_list_size{0};
    std::optional<rmm::device_uvector<uint32_t>> v_list_bitmap{std::nullopt};
    std::optional<rmm::device_uvector<uint32_t>> compressed_v_list{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> padded_v_list{std::nullopt};
    {
      auto v_list_size = static_cast<vertex_t>(
        cuda::std::distance(sorted_unique_vertex_first, sorted_unique_vertex_last));
      auto range_first = graph_view.local_vertex_partition_range_first();
      auto range_last  = range_first;
      if (v_list_size > 0) {
        auto h_staging_buffer_ptr = reinterpret_cast<vertex_t*>(h_staging_buffer.data());
        assert(h_staging_buffer.size() >= size_t{2});
        if constexpr (std::is_pointer_v<std::decay_t<VertexIterator>>) {
          raft::update_host(
            h_staging_buffer_ptr, sorted_unique_vertex_first, size_t{1}, handle.get_stream());
          raft::update_host(h_staging_buffer_ptr + 1,
                            sorted_unique_vertex_first + (v_list_size - 1),
                            size_t{1},
                            handle.get_stream());
        } else {
          rmm::device_uvector<vertex_t> tmps(2, handle.get_stream());
          thrust::tabulate(handle.get_thrust_policy(),
                           tmps.begin(),
                           tmps.end(),
                           cuda::proclaim_return_type<vertex_t>(
                             [sorted_unique_vertex_first, v_list_size] __device__(size_t i) {
                               if (i == 0) {
                                 return *sorted_unique_vertex_first;
                               } else {
                                 assert(i == 1);
                                 return *(sorted_unique_vertex_first + (v_list_size - 1));
                               }
                             }));
          raft::update_host(h_staging_buffer_ptr, tmps.data(), size_t{2}, handle.get_stream());
        }
        handle.sync_stream();
        range_first = h_staging_buffer_ptr[0];
        range_last  = h_staging_buffer_ptr[1] + 1;
      }

      auto leading_boundary_words =
        (packed_bool_word_bcast_alignment -
         packed_bool_offset(range_first - minor_range_first) % packed_bool_word_bcast_alignment) %
        packed_bool_word_bcast_alignment;
      if ((leading_boundary_words == 0) &&
          (packed_bool_offset(range_first - minor_range_first) ==
           packed_bool_offset(graph_view.local_vertex_partition_range_first() -
                              minor_range_first)) &&
          (((range_first - minor_range_first) % packed_bools_per_word()) !=
           0)) {  // there are unaligned bits (fewer than packed_bools_per_word()) in the vertex
                  // partition boundary
        leading_boundary_words = packed_bool_word_bcast_alignment;
      }
      auto word_offset_first = packed_bool_offset(range_first - minor_range_first);

      constexpr size_t num_words_per_vertex =
        (sizeof(vertex_t) > sizeof(uint32_t) ? size_t{2} : size_t{1});
      auto num_leading_words =
        size_t{3} /* local_v_list_size, local_v_list_range_first, local_v_list_range_last */ *
        num_words_per_vertex;
      rmm::device_uvector<uint32_t> d_tmps(packed_bool_word_bcast_alignment, handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        d_tmps.begin(),
        d_tmps.end(),
        [sorted_unique_vertex_first,
         sorted_unique_vertex_last,
         vertex_partition_range_last = graph_view.local_vertex_partition_range_last(),
         minor_range_first,
         word_offset_first,
         leading_boundary_words,
         input] __device__(size_t i) {
          uint32_t word{0};
          if (i < leading_boundary_words) {
            auto word_v_first = minor_range_first + static_cast<vertex_t>((word_offset_first + i) *
                                                                          packed_bools_per_word());
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
          }
          return word;
        });

      auto h_aggregate_tmps = reinterpret_cast<uint32_t*>(h_staging_buffer.data());
      assert(h_staging_buffer.size() >=
             major_comm_size *
               (raft::round_up_safe(num_leading_words + packed_bool_word_bcast_alignment, 2) / 2));
      auto h_this_rank_aggregate_tmps =
        h_aggregate_tmps +
        (major_comm_rank * (num_leading_words + packed_bool_word_bcast_alignment));
      if constexpr (num_words_per_vertex == 1) {
        h_this_rank_aggregate_tmps[0] = v_list_size;
        h_this_rank_aggregate_tmps[1] = range_first;
        h_this_rank_aggregate_tmps[2] = range_last;
      } else {
        h_this_rank_aggregate_tmps[0] = static_cast<uint32_t>(v_list_size & 0xfffffffful);
        h_this_rank_aggregate_tmps[1] = static_cast<uint32_t>(v_list_size >> 32);
        h_this_rank_aggregate_tmps[2] = static_cast<uint32_t>(range_first & 0xfffffffful);
        h_this_rank_aggregate_tmps[3] = static_cast<uint32_t>(range_first >> 32);
        h_this_rank_aggregate_tmps[4] = static_cast<uint32_t>(range_last & 0xfffffffful);
        h_this_rank_aggregate_tmps[5] = static_cast<uint32_t>(range_last >> 32);
      }
      raft::update_host(h_this_rank_aggregate_tmps + num_leading_words,
                        d_tmps.data(),
                        d_tmps.size(),
                        handle.get_stream());
      handle.sync_stream();

      if (major_comm_size >
          1) {  // allgather v_list_size, v_list_range_first (inclusive), v_list_range_last
                // (exclusive), and packed bool data in the partition boundary (relevant only if
                // direct_bcast is set to true, but we pre-calculate to reduce the number of
                // device_allgather operations)
#if 1           // FIXME: we should add host_allgather to raft
        rmm::device_uvector<uint32_t> d_aggregate_tmps(
          (num_leading_words + packed_bool_word_bcast_alignment) * major_comm_size,
          handle.get_stream());
        raft::update_device(
          d_aggregate_tmps.data() +
            (num_leading_words + packed_bool_word_bcast_alignment) * major_comm_rank,
          h_aggregate_tmps +
            (num_leading_words + packed_bool_word_bcast_alignment) * major_comm_rank,
          num_leading_words + packed_bool_word_bcast_alignment,
          handle.get_stream());
        device_allgather(major_comm,
                         d_aggregate_tmps.data() +
                           (num_leading_words + packed_bool_word_bcast_alignment) * major_comm_rank,
                         d_aggregate_tmps.data(),
                         num_leading_words + packed_bool_word_bcast_alignment,
                         handle.get_stream());
        raft::update_host(h_aggregate_tmps,
                          d_aggregate_tmps.data(),
                          (num_leading_words + packed_bool_word_bcast_alignment) * major_comm_size,
                          handle.get_stream());
        handle.sync_stream();
#else
        major_comm.host_allgather(
          h_aggregate_tmps, h_aggregate_tmps, num_leading_words + packed_bool_word_bcast_alignment);
#endif
      }

      local_v_list_sizes        = std::vector<vertex_t>(major_comm_size);
      local_v_list_range_firsts = std::vector<vertex_t>(major_comm_size);
      local_v_list_range_lasts  = std::vector<vertex_t>(major_comm_size);
      for (int i = 0; i < major_comm_size; ++i) {
        if constexpr (num_words_per_vertex == 1) {
          local_v_list_sizes[i] =
            h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 0];
          local_v_list_range_firsts[i] =
            h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 1];
          local_v_list_range_lasts[i] =
            h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 2];
        } else {
          local_v_list_sizes[i] =
            static_cast<vertex_t>(
              h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 0]) |
            (static_cast<vertex_t>(
               h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 1])
             << 32);
          local_v_list_range_firsts[i] =
            static_cast<vertex_t>(
              h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 2]) |
            (static_cast<vertex_t>(
               h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 3])
             << 32);
          local_v_list_range_lasts[i] =
            static_cast<vertex_t>(
              h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 4]) |
            (static_cast<vertex_t>(
               h_aggregate_tmps[i * (num_leading_words + packed_bool_word_bcast_alignment) + 5])
             << 32);
        }
      }

      aggregate_local_v_list_size =
        std::reduce(local_v_list_sizes.begin(), local_v_list_sizes.end());
      vertex_t max_local_v_list_size =
        std::reduce(local_v_list_sizes.begin() + 1,
                    local_v_list_sizes.end(),
                    local_v_list_sizes[0],
                    [](vertex_t lhs, vertex_t rhs) { return std::max(lhs, rhs); });
      vertex_t max_local_v_list_range_size{0};
      for (int i = 0; i < major_comm_size; ++i) {
        auto range_size             = local_v_list_range_lasts[i] - local_v_list_range_firsts[i];
        max_local_v_list_range_size = std::max(range_size, max_local_v_list_range_size);
      }
      bool v_compressible{false};
      if constexpr (sizeof(vertex_t) > sizeof(uint32_t)) {
        if (max_local_v_list_range_size <=
            std::numeric_limits<uint32_t>::max()) {  // broadcast 32bit offset values instead of
                                                     // 64 bit vertex IDs
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
      auto avg_v_list_size = static_cast<vertex_t>(
        static_cast<double>(aggregate_local_v_list_size) / static_cast<double>(major_comm_size));
      if ((avg_fill_ratio > threshold_ratio) &&
          (static_cast<size_t>(avg_v_list_size) > packed_bool_word_bcast_alignment)) {
        direct_bcast = is_packed_bool<typename EdgeMinorPropertyOutputWrapper::value_iterator,
                                      typename EdgeMinorPropertyOutputWrapper::value_type>() &&
                       !edge_partition_keys;
        if (direct_bcast) {
          if (local_v_list_range_firsts[major_comm_rank] <
              local_v_list_range_lasts[major_comm_rank]) {
            auto word_offset_last =
              packed_bool_offset((local_v_list_range_lasts[major_comm_rank] - vertex_t{1}) -
                                 minor_range_first) +
              vertex_t{1};
            if (word_offset_first + static_cast<vertex_t>(leading_boundary_words) <
                word_offset_last) {
              thrust::for_each(
                handle.get_thrust_policy(),
                thrust::make_counting_iterator(word_offset_first) + leading_boundary_words,
                thrust::make_counting_iterator(word_offset_last),
                [sorted_unique_vertex_first,
                 sorted_unique_vertex_last,
                 input,
                 minor_range_first,
                 vertex_partition_range_last = graph_view.local_vertex_partition_range_last(),
                 output_value_first          = edge_partition_value_first] __device__(auto i) {
                  auto& word = *(output_value_first + i);
                  auto word_v_first =
                    minor_range_first + static_cast<vertex_t>(i * packed_bools_per_word());
                  auto word_v_last =
                    ((vertex_partition_range_last - word_v_first) <= packed_bools_per_word())
                      ? vertex_partition_range_last
                      : (word_v_first + static_cast<vertex_t>(packed_bools_per_word()));
                  auto it = thrust::lower_bound(thrust::seq,
                                                sorted_unique_vertex_first,
                                                sorted_unique_vertex_last,
                                                word_v_first);
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
          }

          rmm::device_uvector<uint32_t> d_aggregate_tmps(
            major_comm_size * (num_leading_words + packed_bool_word_bcast_alignment),
            handle.get_stream());
          raft::update_device(
            d_aggregate_tmps.data(),
            h_aggregate_tmps,
            major_comm_size * (num_leading_words + packed_bool_word_bcast_alignment),
            handle.get_stream());
          rmm::device_uvector<uint32_t> aggregate_boundary_words(
            major_comm_size * packed_bool_word_bcast_alignment, handle.get_stream());
          auto map_first = thrust::make_transform_iterator(
            thrust::make_counting_iterator(size_t{0}),
            cuda::proclaim_return_type<size_t>([num_leading_words] __device__(auto i) {
              return (i / packed_bool_word_bcast_alignment) *
                       (num_leading_words + packed_bool_word_bcast_alignment) +
                     (num_leading_words + (i % packed_bool_word_bcast_alignment));
            }));
          thrust::gather(handle.get_thrust_policy(),
                         map_first,
                         map_first + aggregate_boundary_words.size(),
                         d_aggregate_tmps.begin(),
                         aggregate_boundary_words.begin());
          v_list_bitmap = std::move(aggregate_boundary_words);
        } else {
          v_list_bitmap = compute_vertex_list_bitmap_info(
            sorted_unique_vertex_first,
            sorted_unique_vertex_last,
            local_v_list_range_firsts[major_comm_rank],
            local_v_list_range_firsts[major_comm_rank] +
              raft::round_up_safe(max_local_v_list_range_size,
                                  static_cast<vertex_t>((cache_line_size / sizeof(uint32_t)) *
                                                        packed_bools_per_word())),
            handle.get_stream());  // use the maximum padded size instead of
                                   // (local_v_list_range_lasts[major_comm_rank] -
                                   // local_v_list_range_firsts[major_comm_rank]) to ensure that
                                   // the buffer size is identical in every major_comm_rank (to
                                   // use devcie_allgather) and cache line aligned
          assert(retinerpret_cast<uintptr_t>(v_list_bitmap.data()) % cache_line_size == 0);
        }
      } else {
        if (aggregate_local_v_list_size >=
            vertex_t{4 * 1024} /* tuning parameter */ * static_cast<vertex_t>(major_comm_size)) {
          if (v_compressible) {
            rmm::device_uvector<uint32_t> tmps(
              raft::round_up_safe(max_local_v_list_size,
                                  static_cast<vertex_t>(cache_line_size / sizeof(uint32_t))),
              handle.get_stream());  // use the maximum padded size instead of
                                     // local_v_list_sizes[major_comm_rank] to ensure that the
                                     // buffer size is identical in every major_comm_rank (to use
                                     // device_allgather) and cache line aligned
            thrust::transform(
              handle.get_thrust_policy(),
              sorted_unique_vertex_first,
              sorted_unique_vertex_last,
              tmps.begin(),
              cuda::proclaim_return_type<uint32_t>(
                [range_first = local_v_list_range_firsts[major_comm_rank]] __device__(auto v) {
                  return static_cast<uint32_t>(v - range_first);
                }));  // last tmps.size() - local_v_list_sizes[major_comm_rank] elements
                      // have garbage values (this is OK as we won't use them)
            compressed_v_list = std::move(tmps);
            assert(retinerpret_cast<uintptr_t>(compressed_v_list->data()) % cache_line_size == 0);
          }
        }
        if (!compressed_v_list) {
          rmm::device_uvector<vertex_t> tmps(
            raft::round_up_safe(max_local_v_list_size,
                                static_cast<vertex_t>(cache_line_size / sizeof(vertex_t))),
            handle.get_stream());  // use the maximum padded size instead of
                                   // local_v_list_sizes[major_comm_rank] to ensure that the
                                   // buffer size is identical in every major_comm_rank (to use
                                   // device_allgather) and cache line aligned
          thrust::copy(
            handle.get_thrust_policy(),
            sorted_unique_vertex_first,
            sorted_unique_vertex_last,
            tmps.begin());  // last tmps.size() - local_v_list_sizes[major_comm_rank]
                            // elements have garbage values (this is OK as we won't use them)
          padded_v_list = std::move(tmps);
          assert(retinerpret_cast<uintptr_t>(padded_v_list->data()) % cache_line_size == 0);
        }
      }
    }

    if (aggregate_local_v_list_size == 0) {
      return;
    }

    std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
    if ((major_comm_size > 1) && (handle.get_stream_pool_size() > 1)) {
      stream_pool_indices = std::vector<size_t>(
        std::min(handle.get_stream_pool_size(), static_cast<size_t>(major_comm_size)));
      std::iota(stream_pool_indices->begin(), stream_pool_indices->end(), size_t{0});
    }

    std::optional<raft::host_span<vertex_t const>> key_offsets{};
    if constexpr (GraphViewType::is_storage_transposed) {
      key_offsets = graph_view.local_sorted_unique_edge_src_vertex_partition_offsets();
    } else {
      key_offsets = graph_view.local_sorted_unique_edge_dst_vertex_partition_offsets();
    }

    {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto sub0 = std::chrono::steady_clock::now();

      if (direct_bcast) {
        std::vector<size_t> leading_boundary_word_counts(major_comm_size);
        for (int i = 0; i < major_comm_size; ++i) {
          auto leading_boundary_words =
            (packed_bool_word_bcast_alignment -
             packed_bool_offset(local_v_list_range_firsts[i] - minor_range_first) %
               packed_bool_word_bcast_alignment) %
            packed_bool_word_bcast_alignment;
          auto vertex_partition_id =
            partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
              major_comm_size, minor_comm_size, i, minor_comm_rank);
          if ((leading_boundary_words == 0) &&
              (packed_bool_offset(local_v_list_range_firsts[i] - minor_range_first) ==
               packed_bool_offset(graph_view.vertex_partition_range_first(vertex_partition_id) -
                                  minor_range_first)) &&
              (((local_v_list_range_firsts[i] - minor_range_first) % packed_bools_per_word()) !=
               0)) {
            leading_boundary_words = packed_bool_word_bcast_alignment;
          }
          leading_boundary_word_counts[i] = leading_boundary_words;
        }
        device_group_start(major_comm);
        for (int i = 0; i < major_comm_size; ++i) {
          size_t bcast_size{0};
          vertex_t packed_bool_offset_first{0};
          if (local_v_list_range_firsts[i] < local_v_list_range_lasts[i]) {
            auto leading_boundary_words = leading_boundary_word_counts[i];
            packed_bool_offset_first =
              packed_bool_offset(local_v_list_range_firsts[i] - minor_range_first) +
              static_cast<vertex_t>(leading_boundary_words);
            auto packed_bool_offset_last =
              packed_bool_offset(local_v_list_range_lasts[i] - 1 - minor_range_first);
            if (packed_bool_offset_first <= packed_bool_offset_last) {
              bcast_size = (packed_bool_offset_last - packed_bool_offset_first) + 1;
            }
          }

          device_bcast(major_comm,
                       edge_partition_value_first + packed_bool_offset_first,
                       edge_partition_value_first + packed_bool_offset_first,
                       bcast_size,
                       static_cast<int>(i),
                       handle.get_stream());
        }
        device_group_end(major_comm);

        rmm::device_uvector<int64_t> d_tmp_vars(
          leading_boundary_word_counts.size() + major_comm_size, handle.get_stream());
        static_assert((sizeof(int64_t) >= sizeof(size_t)) && (sizeof(int64_t) >= sizeof(vertex_t)));
        {
          std::byte* h_staging_buffer_ptr = reinterpret_cast<std::byte*>(h_staging_buffer.data());
          assert(h_staging_buffer.size() >= d_tmp_vars.size());
          std::copy(leading_boundary_word_counts.begin(),
                    leading_boundary_word_counts.end(),
                    reinterpret_cast<size_t*>(h_staging_buffer_ptr));
          std::copy(local_v_list_range_firsts.begin(),
                    local_v_list_range_firsts.begin() + major_comm_size,
                    reinterpret_cast<vertex_t*>(
                      h_staging_buffer_ptr + leading_boundary_word_counts.size() * sizeof(size_t)));
          raft::update_device(d_tmp_vars.data(),
                              reinterpret_cast<int64_t const*>(h_staging_buffer_ptr),
                              d_tmp_vars.size(),
                              handle.get_stream());
        }
        raft::device_span<size_t const> d_leading_boundary_word_counts(
          reinterpret_cast<size_t const*>(d_tmp_vars.data()), leading_boundary_word_counts.size());
        raft::device_span<vertex_t const> d_local_v_list_range_firsts(
          reinterpret_cast<vertex_t const*>(reinterpret_cast<std::byte const*>(d_tmp_vars.data()) +
                                            leading_boundary_word_counts.size() * sizeof(size_t)),
          major_comm_size);

        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(major_comm_size * packed_bool_word_bcast_alignment),
          [input,
           minor_range_first,
           leading_boundary_word_counts = d_leading_boundary_word_counts,
           local_v_list_range_firsts    = d_local_v_list_range_firsts,
           aggregate_boundary_words     = raft::device_span<uint32_t const>(
             v_list_bitmap->data(), major_comm_size * packed_bool_word_bcast_alignment),
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
        std::variant<rmm::device_uvector<vertex_t>, rmm::device_uvector<uint32_t>>
          allgather_buffer = rmm::device_uvector<vertex_t>(0, handle.get_stream());
        rmm::device_uvector<size_t> dummy_counters(major_comm_size, handle.get_stream());
        std::optional<vertex_t> v_list_bitmap_size{std::nullopt};
        std::optional<vertex_t> compressed_v_list_size{std::nullopt};
        std::optional<vertex_t> padded_v_list_size{std::nullopt};
        if (major_comm_size > 1) {
          if (v_list_bitmap) {
            allgather_buffer = rmm::device_uvector<uint32_t>(
              v_list_bitmap->size() * major_comm_size, handle.get_stream());
          } else if (compressed_v_list) {
            allgather_buffer = rmm::device_uvector<uint32_t>(
              compressed_v_list->size() * major_comm_size, handle.get_stream());
          } else {
            assert(padded_v_list);
            allgather_buffer = rmm::device_uvector<vertex_t>(
              padded_v_list->size() * major_comm_size, handle.get_stream());
          }
          if (v_list_bitmap) {
            device_allgather(major_comm,
                             v_list_bitmap->data(),
                             std::get<1>(allgather_buffer).data(),
                             v_list_bitmap->size(),
                             handle.get_stream());
            v_list_bitmap_size = v_list_bitmap->size();
            v_list_bitmap      = std::nullopt;
          } else if (compressed_v_list) {
            device_allgather(major_comm,
                             compressed_v_list->data(),
                             std::get<1>(allgather_buffer).data(),
                             compressed_v_list->size(),
                             handle.get_stream());
            compressed_v_list_size = compressed_v_list->size();
            compressed_v_list      = std::nullopt;
          } else {
            assert(padded_v_list);
            device_allgather(major_comm,
                             padded_v_list->data(),
                             std::get<0>(allgather_buffer).data(),
                             padded_v_list->size(),
                             handle.get_stream());
            padded_v_list_size = padded_v_list->size();
            padded_v_list      = std::nullopt;
          }
        } else {
          allgather_buffer = rmm::device_uvector<vertex_t>(
            raft::round_up_safe(local_v_list_sizes[0],
                                static_cast<vertex_t>(cache_line_size / sizeof(vertex_t))),
            handle.get_stream());
          thrust::copy(handle.get_thrust_policy(),
                       sorted_unique_vertex_first,
                       sorted_unique_vertex_last,
                       std::get<0>(allgather_buffer).begin());
          padded_v_list_size = std::get<0>(allgather_buffer).size();
        }

        bool kernel_fusion =
          !edge_partition_keys && !v_list_bitmap_size;  // FIXME: kernle fusion can be useful even
                                                        // when edge_partition_keys.has_value() or
                                                        // v_list_bitmap.has_value() are true

        if (!kernel_fusion) {
          if (stream_pool_indices) { handle.sync_stream(); }
        }

        rmm::device_uvector<vertex_t> d_vertex_vars(0, handle.get_stream());
        raft::device_span<vertex_t const> d_local_v_list_sizes{};
        raft::device_span<vertex_t const> d_local_v_list_range_firsts{};
        {
          vertex_t* h_staging_buffer_ptr = reinterpret_cast<vertex_t*>(h_staging_buffer.data());
          std::copy(local_v_list_sizes.begin(), local_v_list_sizes.end(), h_staging_buffer_ptr);
          std::copy(local_v_list_range_firsts.begin(),
                    local_v_list_range_firsts.end(),
                    h_staging_buffer_ptr + major_comm_size);
          d_vertex_vars.resize(major_comm_size * 2, handle.get_stream());
          raft::update_device(
            d_vertex_vars.data(), h_staging_buffer_ptr, major_comm_size * 2, handle.get_stream());

          d_local_v_list_sizes =
            raft::device_span<vertex_t const>(d_vertex_vars.data(), major_comm_size);
          d_local_v_list_range_firsts = raft::device_span<vertex_t const>(
            d_vertex_vars.data() + major_comm_size, major_comm_size);
        }

        if (!kernel_fusion) {
          size_t stream_pool_size = stream_pool_indices ? stream_pool_indices->size() : size_t{0};
          for (int i = 0; i < major_comm_size; ++i) {
            auto loop_stream =
              stream_pool_indices
                ? handle.get_stream_from_stream_pool((*stream_pool_indices)[i % stream_pool_size])
                : handle.get_stream();

            std::optional<rmm::device_uvector<vertex_t>> rx_vertices{std::nullopt};
            if (v_list_bitmap_size) {
              auto const& rx_bitmap = std::get<1>(allgather_buffer);
              rx_vertices = rmm::device_uvector<vertex_t>(local_v_list_sizes[i], loop_stream);
              retrieve_vertex_list_from_bitmap(
                raft::device_span<uint32_t const>(rx_bitmap.data() + (*v_list_bitmap_size * i),
                                                  *v_list_bitmap_size),
                rx_vertices->begin(),
                raft::device_span<size_t>(dummy_counters.data() + i, size_t{1}),
                local_v_list_range_firsts[i],
                local_v_list_range_lasts[i],
                loop_stream);
            }

            auto rx_vertex_first = compressed_v_list_size
                                     ? static_cast<vertex_t const*>(nullptr)
                                     : (v_list_bitmap_size ? rx_vertices->data()
                                                           : std::get<0>(allgather_buffer).data() +
                                                               *padded_v_list_size * i);
            auto rx_compressed_vertex_first =
              compressed_v_list_size
                ? std::get<1>(allgather_buffer).data() + (*compressed_v_list_size * i)
                : static_cast<uint32_t const*>(nullptr);
            if (edge_partition_keys) {
              thrust::for_each(
                rmm::exec_policy_nosync(loop_stream),
                thrust::make_counting_iterator(vertex_t{0}),
                thrust::make_counting_iterator(local_v_list_sizes[i]),
                [rx_vertex_first,
                 rx_compressed_vertex_first,
                 range_first = local_v_list_range_firsts[i],
                 input,
                 subrange_key_first = (*edge_partition_keys).begin() + (*key_offsets)[i],
                 subrange_key_last  = (*edge_partition_keys).begin() + (*key_offsets)[i + 1],
                 edge_partition_value_first = edge_partition_value_first,
                 subrange_start_offset      = (*key_offsets)[i]] __device__(auto i) {
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
                  thrust::make_counting_iterator(local_v_list_sizes[i]),
                  [minor_range_first,
                   rx_vertex_first,
                   rx_compressed_vertex_first,
                   range_first = local_v_list_range_firsts[i],
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
                auto stencil_first = thrust::make_counting_iterator(vertex_t{0});
                if (compressed_v_list_size) {
                  auto map_first = thrust::make_transform_iterator(
                    rx_compressed_vertex_first,
                    cuda::proclaim_return_type<vertex_t>(
                      [minor_range_first,
                       range_first = local_v_list_range_firsts[i]] __device__(auto v_offset) {
                        return static_cast<vertex_t>(v_offset + (range_first - minor_range_first));
                      }));
                  auto val_first = thrust::make_constant_iterator(input);
                  thrust::scatter(rmm::exec_policy_nosync(loop_stream),
                                  val_first,
                                  val_first + local_v_list_sizes[i],
                                  map_first,
                                  edge_partition_value_first);
                } else {
                  auto map_first = thrust::make_transform_iterator(
                    rx_vertex_first,
                    cuda::proclaim_return_type<vertex_t>(
                      [minor_range_first] __device__(auto v) { return v - minor_range_first; }));
                  auto val_first = thrust::make_constant_iterator(input);
                  thrust::scatter(rmm::exec_policy_nosync(loop_stream),
                                  val_first,
                                  val_first + local_v_list_sizes[i],
                                  map_first,
                                  edge_partition_value_first);
                }
              }
            }
          }
          if (stream_pool_indices) { RAFT_CUDA_TRY(cudaDeviceSynchronize()); }
        } else {  // kernel fusion
          auto max_padded_local_v_list_size =
            compressed_v_list_size ? *compressed_v_list_size : *padded_v_list_size;
          if constexpr (contains_packed_bool_element) {
            thrust::for_each(
              handle.get_thrust_policy(),
              thrust::make_counting_iterator(vertex_t{0}),
              thrust::make_counting_iterator(max_padded_local_v_list_size *
                                             static_cast<vertex_t>(major_comm_size)),
              [local_v_list_sizes        = d_local_v_list_sizes,
               local_v_list_range_firsts = d_local_v_list_range_firsts,
               rx_first                  = allgather_buffer.index() == 0
                                             ? static_cast<void const*>(std::get<0>(allgather_buffer).data())
                                             : static_cast<void const*>(std::get<1>(allgather_buffer).data()),
               output_value_first        = edge_partition_value_first,
               compressed                = compressed_v_list_size.has_value(),
               minor_range_first,
               input,
               max_padded_local_v_list_size] __device__(auto i) {
                auto loop_idx = i / max_padded_local_v_list_size;
                auto offset   = i % max_padded_local_v_list_size;
                if (offset < local_v_list_sizes[loop_idx]) {
                  vertex_t minor{};
                  if (compressed) {
                    minor = local_v_list_range_firsts[loop_idx] +
                            *(static_cast<uint32_t const*>(rx_first) + i);
                  } else {
                    minor = *(static_cast<vertex_t const*>(rx_first) + i);
                  }
                  auto minor_offset = minor - minor_range_first;
                  fill_scalar_or_thrust_tuple(output_value_first, minor_offset, input);
                }
              });
          } else {
            auto val_first     = thrust::make_constant_iterator(input);
            auto stencil_first = thrust::make_counting_iterator(vertex_t{0});
            if (compressed_v_list_size) {
              auto map_first = thrust::make_transform_iterator(
                thrust::make_counting_iterator(vertex_t{0}),
                cuda::proclaim_return_type<vertex_t>(
                  [local_v_list_sizes        = d_local_v_list_sizes,
                   local_v_list_range_firsts = d_local_v_list_range_firsts,
                   rx_first                  = std::get<1>(allgather_buffer).begin(),
                   minor_range_first,
                   max_padded_local_v_list_size] __device__(auto i) {
                    auto loop_idx = i / max_padded_local_v_list_size;
                    auto offset   = i % max_padded_local_v_list_size;
                    if (offset < local_v_list_sizes[loop_idx]) {
                      auto minor = local_v_list_range_firsts[loop_idx] + *(rx_first + i);
                      return minor - minor_range_first;
                    } else {
                      return vertex_t{0};  // dummy
                    }
                  }));
              thrust::scatter_if(
                handle.get_thrust_policy(),
                val_first,
                val_first + (max_padded_local_v_list_size * major_comm_size),
                map_first,
                stencil_first,
                edge_partition_value_first,
                within_valid_range_t<vertex_t>{d_local_v_list_sizes, max_padded_local_v_list_size});
            } else {
              auto map_first = thrust::make_transform_iterator(
                thrust::make_counting_iterator(vertex_t{0}),
                cuda::proclaim_return_type<vertex_t>(
                  [local_v_list_sizes = d_local_v_list_sizes,
                   rx_first           = std::get<0>(allgather_buffer).begin(),
                   minor_range_first,
                   max_padded_local_v_list_size] __device__(auto i) {
                    auto loop_idx = i / max_padded_local_v_list_size;
                    auto offset   = i % max_padded_local_v_list_size;
                    if (offset < local_v_list_sizes[loop_idx]) {
                      auto minor = *(rx_first + i);
                      return minor - minor_range_first;
                    } else {
                      return vertex_t{0};  // dummy
                    }
                  }));
              thrust::scatter_if(
                handle.get_thrust_policy(),
                val_first,
                val_first + (max_padded_local_v_list_size * major_comm_size),
                map_first,
                stencil_first,
                edge_partition_value_first,
                within_valid_range_t<vertex_t>{d_local_v_list_sizes, max_padded_local_v_list_size});
            }
          }
        }
      }
    }
    handle.sync_stream();  // to ensure that the above update_device operations complete before
                           // h_staging_buffer goes out-of-scope, currently, rmm::device_uvector
                           // with mr=rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>
                           // does not support stream semantics
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
#if 1  // FIXME: we should add host_allreduce to raft
      num_invalids =
        host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
      comm.host_allreduce(std::addressof(num_invalids),
                          std::addressof(num_invalids),
                          size_t{1},
                          raft::comms::op_t::SUM);
#endif
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
#if 1  // FIXME: we should add host_allreduce to raft
      num_invalids =
        host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
      comm.host_allreduce(std::addressof(num_invalids),
                          std::addressof(num_invalids),
                          size_t{1},
                          raft::comms::op_t::SUM);
#endif
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

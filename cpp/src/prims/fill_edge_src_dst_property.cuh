/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/atomic_ops.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>

#include <cstddef>

namespace cugraph {

namespace detail {

template <typename GraphViewType, typename T, typename EdgeMajorPropertyOutputWrapper>
void fill_edge_major_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              T input,
                              EdgeMajorPropertyOutputWrapper edge_major_property_output)
{
  static_assert(std::is_same_v<T, typename EdgeMajorPropertyOutputWrapper::value_type>);

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
    if constexpr (cugraph::has_packed_bool_element<
                    std::remove_reference_t<decltype(value_firsts[i])>,
                    T>()) {
      static_assert(std::is_arithmetic_v<T>, "unimplemented for thrust::tuple types.");
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
          typename T,
          typename EdgeMajorPropertyOutputWrapper>
void fill_edge_major_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexIterator vertex_first,
                              VertexIterator vertex_last,
                              T input,
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

    auto edge_partition_keys = edge_major_property_output.keys();
    for (int i = 0; i < minor_comm_size; ++i) {
      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(i));

      device_bcast(
        minor_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(rx_counts[i]),
          [rx_vertex_first = rx_vertices.begin(),
           input,
           edge_partition_key_first   = ((*edge_partition_keys)[i]).begin(),
           edge_partition_key_last    = ((*edge_partition_keys)[i]).end(),
           edge_partition_value_first = edge_partition_value_firsts[i]] __device__(size_t i) {
            auto major = *(rx_vertex_first + i);
            auto it    = thrust::lower_bound(
              thrust::seq, edge_partition_key_first, edge_partition_key_last, major);
            if ((it != edge_partition_key_last) && (*it == major)) {
              auto edge_partition_offset = thrust::distance(edge_partition_key_first, it);
              if constexpr (packed_bool) {
                packe_bool_atomic_set(edge_partition_value_first, edge_partition_offset, input);
              } else {
                *(edge_partition_value_first + edge_partition_offset) = input;
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
                          val_first + rx_counts[i],
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
                       [input, output_value_first = edge_partition_value_firsts[0]] __device__(
                         auto v) { packed_bool_atomic_set(output_value_first, v, input); });
    } else {
      auto val_first = thrust::make_constant_iterator(input);
      thrust::scatter(handle.get_thrust_policy(),
                      val_first,
                      val_first + thrust::distance(vertex_first, vertex_last),
                      vertex_first,
                      edge_partition_value_firsts[0]);
    }
  }
}

template <typename GraphViewType, typename T, typename EdgeMinorPropertyOutputWrapper>
void fill_edge_minor_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              T input,
                              EdgeMinorPropertyOutputWrapper edge_minor_property_output)
{
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
  if constexpr (cugraph::has_packed_bool_element<decltype(value_first), T>()) {
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
          typename T,
          typename EdgeMinorPropertyOutputWrapper>
void fill_edge_minor_property(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              VertexIterator vertex_first,
                              VertexIterator vertex_last,
                              T input,
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
      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(
        major_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(rx_counts[i]),
          [rx_vertex_first = rx_vertices.begin(),
           input,
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
                packed_bool_atomic_set(
                  edge_partition_value_first, subrange_start_offset + subrange_offset, input);
              } else {
                *(edge_partition_value_first + subrange_start_offset + subrange_offset) = input;
              }
            }
          });
      } else {
        if constexpr (packed_bool) {
          thrust::for_each(handle.get_thrust_policy(),
                           thrust::make_counting_iterator(vertex_t{0}),
                           thrust::make_counting_iterator(static_cast<vertex_t>(rx_counts[i])),
                           [edge_partition,
                            rx_vertex_first = rx_vertices.begin(),
                            input,
                            output_value_first = edge_partition_value_first] __device__(auto i) {
                             auto rx_vertex = *(rx_vertex_first + i);
                             auto minor_offset =
                               edge_partition.minor_offset_from_minor_nocheck(rx_vertex);
                             packed_bool_atomic_set(output_value_first, minor_offset, input);
                           });
        } else {
          auto map_first = thrust::make_transform_iterator(
            rx_vertices.begin(),
            cuda::proclaim_return_type<vertex_t>([edge_partition] __device__(auto v) {
              return edge_partition.minor_offset_from_minor_nocheck(v);
            }));
          auto val_first = thrust::make_constant_iterator(input);
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(handle.get_thrust_policy(),
                          val_first,
                          val_first + rx_counts[i],
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
                       [input, output_value_first = edge_partition_value_first] __device__(auto v) {
                         packed_bool_atomic_set(output_value_first, v, input);
                       });
    } else {
      auto val_first = thrust::make_constant_iterator(input);
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
 * @brief Fill graph edge source property values to the input value.
 *
 * This version fills graph edge source property values for the entire edge source ranges (assigned
 * to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge source property values will be set to @p input.
 * @param edge_src_property_output edge_src_property_t class object to store source property values
 * (for the edge source assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename T>
void fill_edge_src_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            T input,
                            edge_src_property_t<GraphViewType, T>& edge_src_property_output,
                            bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::fill_edge_minor_property(
      handle, graph_view, input, edge_src_property_output.mutable_view());
  } else {
    detail::fill_edge_major_property(
      handle, graph_view, input, edge_src_property_output.mutable_view());
  }
}

/**
 * @brief Fill graph edge source property values to the input value.
 *
 * This version fills only a subset of graph edge source property values. [@p vertex_first,
 * @p vertex_last) specifies the vertices to be filled.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator Type of the iterator for vertex identifiers.
 * @tparam T Type of the edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex with a value to be filled.
 * v in [vertex_first, vertex_last) should be distinct (and should belong to the vertex partition
 * assigned to this process in multi-GPU), otherwise undefined behavior.
 * @param vertex_last Iterator pointing to the last (exclusive) vertex with a value to be filled.
 * @param input Edge source property values will be set to @p input.
 * @param edge_src_property_output edge_src_property_t class object to store source property values
 * (for the edge source assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexIterator, typename T>
void fill_edge_src_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            T input,
                            edge_src_property_t<GraphViewType, T>& edge_src_property_output,
                            bool do_expensive_check = false)
{
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
    detail::fill_edge_minor_property(handle,
                                     graph_view,
                                     vertex_first,
                                     vertex_last,
                                     input,
                                     edge_src_property_output.mutable_view());
  } else {
    detail::fill_edge_major_property(handle,
                                     graph_view,
                                     vertex_first,
                                     vertex_last,
                                     input,
                                     edge_src_property_output.mutable_view());
  }
}

/**
 * @brief Fill graph edge destination property values to the input value.
 *
 * This version fills graph edge destination property values for the entire edge destination ranges
 * (assigned to this process in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge destination property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge destination property values will be set to @p input.
 * @param edge_dst_property_output edge_dst_property_t class object to store destination property
 * values (for the edge destinations assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename T>
void fill_edge_dst_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            T input,
                            edge_dst_property_t<GraphViewType, T>& edge_dst_property_output,
                            bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  if constexpr (GraphViewType::is_storage_transposed) {
    detail::fill_edge_major_property(
      handle, graph_view, input, edge_dst_property_output.mutable_view());
  } else {
    detail::fill_edge_minor_property(
      handle, graph_view, input, edge_dst_property_output.mutable_view());
  }
}

/**
 * @brief Fill graph edge destination property values to the input value.
 *
 * This version fills only a subset of graph edge destination property values. [@p vertex_first,
 * @p vertex_last) specifies the vertices to be filled.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator Type of the iterator for vertex identifiers.
 * @tparam T Type of the edge destination property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex with a value to be filled.
 * v in [vertex_first, vertex_last) should be distinct (and should belong to the vertex partition
 * assigned to this process in multi-GPU), otherwise undefined behavior.
 * @param vertex_last Iterator pointing to the last (exclusive) vertex with a value to be filled.
 * @param input Edge destination property values will be set to @p input.
 * @param edge_dst_property_output edge_dst_property_t class object to store destination property
 * values (for the edge destinations assigned to this process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename VertexIterator, typename T>
void fill_edge_dst_property(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            T input,
                            edge_dst_property_t<GraphViewType, T>& edge_dst_property_output,
                            bool do_expensive_check = false)
{
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
    detail::fill_edge_major_property(handle,
                                     graph_view,
                                     vertex_first,
                                     vertex_last,
                                     input,
                                     edge_dst_property_output.mutable_view());
  } else {
    detail::fill_edge_minor_property(handle,
                                     graph_view,
                                     vertex_first,
                                     vertex_last,
                                     input,
                                     edge_dst_property_output.mutable_view());
  }
}

}  // namespace cugraph

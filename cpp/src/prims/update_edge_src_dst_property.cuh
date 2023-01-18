/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

namespace cugraph {

namespace detail {

template <typename GraphViewType,
          typename VertexPropertyInputIterator,
          typename EdgeMajorPropertyOutputWrapper>
void update_edge_major_property(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexPropertyInputIterator vertex_property_input_first,
                                EdgeMajorPropertyOutputWrapper edge_major_property_output)
{
  auto edge_partition_value_firsts = edge_major_property_output.value_firsts();
  if constexpr (GraphViewType::is_multi_gpu) {
    using vertex_t = typename GraphViewType::vertex_type;

    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    auto edge_partition_keys = edge_major_property_output.keys();
    if (edge_partition_keys) {
      vertex_t max_rx_size{0};
      for (int i = 0; i < col_comm_size; ++i) {
        max_rx_size = std::max(
          max_rx_size, graph_view.vertex_partition_range_size(i * row_comm_size + row_comm_rank));
      }
      auto rx_value_buffer = allocate_dataframe_buffer<
        typename std::iterator_traits<VertexPropertyInputIterator>::value_type>(
        max_rx_size, handle.get_stream());
      auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffer);
      for (int i = 0; i < col_comm_size; ++i) {
        device_bcast(col_comm,
                     vertex_property_input_first,
                     rx_value_first,
                     graph_view.vertex_partition_range_size(i * row_comm_size + row_comm_rank),
                     i,
                     handle.get_stream());

        auto v_offset_first = thrust::make_transform_iterator(
          (*edge_partition_keys)[i].begin(),
          [v_first = graph_view.vertex_partition_range_first(
             i * row_comm_size + row_comm_rank)] __device__(auto v) { return v - v_first; });
        thrust::gather(handle.get_thrust_policy(),
                       v_offset_first,
                       v_offset_first + (*edge_partition_keys)[i].size(),
                       rx_value_first,
                       edge_partition_value_firsts[i]);
      }
    } else {
      for (int i = 0; i < col_comm_size; ++i) {
        device_bcast(col_comm,
                     vertex_property_input_first,
                     edge_partition_value_firsts[i],
                     graph_view.vertex_partition_range_size(i * row_comm_size + row_comm_rank),
                     i,
                     handle.get_stream());
      }
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_dst_range_size()
             : graph_view.local_edge_partition_src_range_size());
    assert(edge_partition_value_firsts.size() == size_t{1});
    thrust::copy(handle.get_thrust_policy(),
                 vertex_property_input_first,
                 vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                 edge_partition_value_firsts[0]);
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
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_firsts = edge_major_property_output.value_firsts();
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    auto rx_counts =
      host_scalar_allgather(col_comm,
                            static_cast<size_t>(thrust::distance(vertex_first, vertex_last)),
                            handle.get_stream());
    auto max_rx_size =
      std::reduce(rx_counts.begin(), rx_counts.end(), size_t{0}, [](auto lhs, auto rhs) {
        return std::max(lhs, rhs);
      });
    rmm::device_uvector<vertex_t> rx_vertices(max_rx_size, handle.get_stream());
    auto rx_tmp_buffer = allocate_dataframe_buffer<
      typename std::iterator_traits<VertexPropertyInputIterator>::value_type>(max_rx_size,
                                                                              handle.get_stream());
    auto rx_value_first = get_dataframe_buffer_begin(rx_tmp_buffer);

    auto edge_partition_keys = edge_major_property_output.keys();
    for (int i = 0; i < col_comm_size; ++i) {
      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(i));

      if (i == col_comm_rank) {
        auto vertex_partition =
          vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
            graph_view.local_vertex_partition_view());
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

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(
        col_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());
      device_bcast(col_comm, rx_value_first, rx_value_first, rx_counts[i], i, handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator(static_cast<vertex_t>((*edge_partition_keys)[i].size())),
          [rx_vertex_first = rx_vertices.begin(),
           rx_vertex_last  = rx_vertices.end(),
           rx_value_first,
           output_key_first   = ((*edge_partition_keys)[i]).begin(),
           output_value_first = edge_partition_value_firsts[i]] __device__(auto i) {
            auto major = *(output_key_first + i);
            auto it    = thrust::lower_bound(thrust::seq, rx_vertex_first, rx_vertex_last, major);
            if ((it != rx_vertex_last) && (*it == major)) {
              auto rx_value             = *(rx_value_first + thrust::distance(rx_vertex_first, it));
              *(output_value_first + i) = rx_value;
            }
          });
      } else {
        auto map_first =
          thrust::make_transform_iterator(rx_vertices.begin(), [edge_partition] __device__(auto v) {
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
  } else {
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_dst_range_size()
             : graph_view.local_edge_partition_src_range_size());
    assert(edge_partition_value_firsts.size() == size_t{1});
    auto val_first = thrust::make_permutation_iterator(vertex_property_input_first, vertex_first);
    thrust::scatter(handle.get_thrust_policy(),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    edge_partition_value_firsts[0]);
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
  auto edge_partition_value_first = edge_minor_property_output.value_first();
  if constexpr (GraphViewType::is_multi_gpu) {
    using vertex_t = typename GraphViewType::vertex_type;
    using value_t  = typename thrust::iterator_traits<VertexPropertyInputIterator>::value_type;

    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto const comm_size     = comm.get_size();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    auto edge_partition_keys = edge_minor_property_output.keys();
    if (edge_partition_keys) {
      raft::host_span<vertex_t const> key_offsets{};
      if constexpr (GraphViewType::is_storage_transposed) {
        key_offsets = *(graph_view.local_sorted_unique_edge_src_vertex_partition_offsets());
      } else {
        key_offsets = *(graph_view.local_sorted_unique_edge_dst_vertex_partition_offsets());
      }

      // memory footprint vs parallelism trade-off
      // memory requirement per loop is
      // (V/comm_size) * sizeof(value_t)
      // and limit memory requirement to (E / comm_size) * sizeof(vertex_t)
      auto num_concurrent_bcasts =
        (static_cast<size_t>(graph_view.number_of_edges() / comm_size) * sizeof(vertex_t)) /
        std::max(static_cast<size_t>(graph_view.number_of_vertices() / comm_size) * sizeof(value_t),
                 size_t{1});
      num_concurrent_bcasts = std::max(num_concurrent_bcasts, size_t{1});
      num_concurrent_bcasts = std::min(num_concurrent_bcasts, static_cast<size_t>(row_comm_size));
      auto num_rounds = (static_cast<size_t>(row_comm_size) + num_concurrent_bcasts - size_t{1}) /
                        num_concurrent_bcasts;

      std::vector<decltype(allocate_dataframe_buffer<value_t>(size_t{0}, handle.get_stream()))>
        rx_value_buffers{};
      rx_value_buffers.reserve(num_concurrent_bcasts);
      for (size_t i = 0; i < num_concurrent_bcasts; ++i) {
        size_t max_size{0};
        for (size_t round = 0; round < num_rounds; ++round) {
          auto j = num_rounds * i + round;
          if (j < static_cast<size_t>(row_comm_size)) {
            max_size = std::max(max_size,
                                static_cast<size_t>(graph_view.vertex_partition_range_size(
                                  col_comm_rank * row_comm_size + j)));
          }
        }
        rx_value_buffers.push_back(
          allocate_dataframe_buffer<value_t>(max_size, handle.get_stream()));
      }

      for (size_t round = 0; round < num_rounds; ++round) {
        device_group_start(row_comm);
        for (size_t i = 0; i < num_concurrent_bcasts; ++i) {
          auto j = num_rounds * i + round;
          if (j < static_cast<size_t>(row_comm_size)) {
            auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffers[i]);
            device_bcast(row_comm,
                         vertex_property_input_first,
                         rx_value_first,
                         graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + j),
                         j,
                         handle.get_stream());
          }
        }
        device_group_end(row_comm);

        for (size_t i = 0; i < num_concurrent_bcasts; ++i) {
          auto j = num_rounds * i + round;
          if (j < static_cast<size_t>(row_comm_size)) {
            auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffers[i]);
            auto v_offset_first = thrust::make_transform_iterator(
              (*edge_partition_keys).begin() + key_offsets[j],
              [v_first = graph_view.vertex_partition_range_first(
                 col_comm_rank * row_comm_size + j)] __device__(auto v) { return v - v_first; });
            thrust::gather(handle.get_thrust_policy(),
                           v_offset_first,
                           v_offset_first + (key_offsets[j + 1] - key_offsets[j]),
                           rx_value_first,
                           edge_partition_value_first + key_offsets[j]);
          }
        }
      }
    } else {
      std::vector<size_t> rx_counts(row_comm_size, size_t{0});
      std::vector<size_t> displacements(row_comm_size, size_t{0});
      for (int i = 0; i < row_comm_size; ++i) {
        rx_counts[i] = graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + i);
        displacements[i] = (i == 0) ? 0 : displacements[i - 1] + rx_counts[i - 1];
      }
      device_allgatherv(row_comm,
                        vertex_property_input_first,
                        edge_partition_value_first,
                        rx_counts,
                        displacements,
                        handle.get_stream());
    }
  } else {
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_src_range_size()
             : graph_view.local_edge_partition_dst_range_size());
    thrust::copy(handle.get_thrust_policy(),
                 vertex_property_input_first,
                 vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                 edge_partition_value_first);
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
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto edge_partition_value_first = edge_minor_property_output.value_first();
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    auto rx_counts =
      host_scalar_allgather(row_comm,
                            static_cast<size_t>(thrust::distance(vertex_first, vertex_last)),
                            handle.get_stream());
    auto max_rx_size =
      std::reduce(rx_counts.begin(), rx_counts.end(), size_t{0}, [](auto lhs, auto rhs) {
        return std::max(lhs, rhs);
      });
    rmm::device_uvector<vertex_t> rx_vertices(max_rx_size, handle.get_stream());
    auto rx_tmp_buffer = allocate_dataframe_buffer<
      typename std::iterator_traits<VertexPropertyInputIterator>::value_type>(max_rx_size,
                                                                              handle.get_stream());
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
    for (int i = 0; i < row_comm_size; ++i) {
      if (i == row_comm_rank) {
        auto vertex_partition =
          vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
            graph_view.local_vertex_partition_view());
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

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(
        row_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());
      device_bcast(row_comm, rx_value_first, rx_value_first, rx_counts[i], i, handle.get_stream());

      if (edge_partition_keys) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator((*key_offsets)[i + 1] - (*key_offsets)[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_vertex_last  = rx_vertices.end(),
           rx_value_first,
           output_key_first   = (*edge_partition_keys).begin() + (*key_offsets)[i],
           output_value_first = edge_partition_value_first + (*key_offsets)[i]] __device__(auto i) {
            auto minor = *(output_key_first + i);
            auto it    = thrust::lower_bound(thrust::seq, rx_vertex_first, rx_vertex_last, minor);
            if ((it != rx_vertex_last) && (*it == minor)) {
              auto rx_value             = *(rx_value_first + thrust::distance(rx_vertex_first, it));
              *(output_value_first + i) = rx_value;
            }
          });
      } else {
        auto map_first =
          thrust::make_transform_iterator(rx_vertices.begin(), [edge_partition] __device__(auto v) {
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
  } else {
    assert(graph_view.local_vertex_partition_range_size() ==
           graph_view.local_edge_partition_src_range_size());
    auto val_first = thrust::make_permutation_iterator(vertex_property_input_first, vertex_first);
    thrust::scatter(handle.get_thrust_policy(),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    edge_partition_value_first);
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

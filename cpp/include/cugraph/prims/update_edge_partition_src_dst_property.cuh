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
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/edge_partition_src_dst_property.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

template <typename GraphViewType,
          typename VertexPropertyInputIterator,
          typename EdgePartitionMajorPropertyOutputWrapper>
void update_edge_partition_major_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPropertyInputIterator vertex_property_input_first,
  EdgePartitionMajorPropertyOutputWrapper& edge_partition_major_property_output)
{
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

    if (edge_partition_major_property_output.key_first()) {
      auto key_offsets = GraphViewType::is_storage_transposed
                           ? *(graph_view.local_sorted_unique_edge_dst_offsets())
                           : *(graph_view.local_sorted_unique_edge_src_offsets());

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
          *(edge_partition_major_property_output.key_first()) + key_offsets[i],
          [v_first = graph_view.vertex_partition_range_first(
             i * row_comm_size + row_comm_rank)] __device__(auto v) { return v - v_first; });
        thrust::gather(handle.get_thrust_policy(),
                       v_offset_first,
                       v_offset_first + (key_offsets[i + 1] - key_offsets[i]),
                       rx_value_first,
                       edge_partition_major_property_output.value_data() + key_offsets[i]);
      }
    } else {
      std::vector<size_t> rx_counts(col_comm_size, size_t{0});
      std::vector<size_t> displacements(col_comm_size, size_t{0});
      for (int i = 0; i < col_comm_size; ++i) {
        rx_counts[i] = graph_view.vertex_partition_range_size(i * row_comm_size + row_comm_rank);
        displacements[i] = (i == 0) ? 0 : displacements[i - 1] + rx_counts[i - 1];
      }
      device_allgatherv(col_comm,
                        vertex_property_input_first,
                        edge_partition_major_property_output.value_data(),
                        rx_counts,
                        displacements,
                        handle.get_stream());
    }
  } else {
    assert(!(edge_partition_major_property_output.key_first()));
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_dst_range_size()
             : graph_view.local_edge_partition_src_range_size());
    thrust::copy(handle.get_thrust_policy(),
                 vertex_property_input_first,
                 vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                 edge_partition_major_property_output.value_data());
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexPropertyInputIterator,
          typename EdgePartitionMajorPropertyOutputWrapper>
void update_edge_partition_major_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexPropertyInputIterator vertex_property_input_first,
  EdgePartitionMajorPropertyOutputWrapper& edge_partition_major_property_output)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

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

    auto key_offsets = GraphViewType::is_storage_transposed
                         ? graph_view.local_sorted_unique_edge_dst_offsets()
                         : graph_view.local_sorted_unique_edge_src_offsets();

    for (int i = 0; i < col_comm_size; ++i) {
      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
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

      if (edge_partition_major_property_output.key_first()) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator((*key_offsets)[i + 1] - (*key_offsets)[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_vertex_last  = rx_vertices.end(),
           rx_value_first,
           output_key_first =
             *(edge_partition_major_property_output.key_first()) + (*key_offsets)[i],
           output_value_first = edge_partition_major_property_output.value_data() +
                                (*key_offsets)[i]] __device__(auto i) {
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
                        edge_partition_major_property_output.value_data() +
                          edge_partition.major_value_start_offset());
      }
    }
  } else {
    assert(!(edge_partition_major_property_output.key_first()));
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_dst_range_size()
             : graph_view.local_edge_partition_src_range_size());
    auto val_first = thrust::make_permutation_iterator(vertex_property_input_first, vertex_first);
    thrust::scatter(handle.get_thrust_policy(),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    edge_partition_major_property_output.value_data());
  }
}

template <typename GraphViewType,
          typename VertexPropertyInputIterator,
          typename EdgePartitionMinorPropertyOutputWrapper>
void update_edge_partition_minor_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPropertyInputIterator vertex_property_input_first,
  EdgePartitionMinorPropertyOutputWrapper& edge_partition_minor_property_output)
{
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

    if (edge_partition_minor_property_output.key_first()) {
      auto key_offsets = GraphViewType::is_storage_transposed
                           ? *(graph_view.local_sorted_unique_edge_src_offsets())
                           : *(graph_view.local_sorted_unique_edge_dst_offsets());

      vertex_t max_rx_size{0};
      for (int i = 0; i < row_comm_size; ++i) {
        max_rx_size = std::max(
          max_rx_size, graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + i));
      }
      auto rx_value_buffer = allocate_dataframe_buffer<
        typename std::iterator_traits<VertexPropertyInputIterator>::value_type>(
        max_rx_size, handle.get_stream());
      auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffer);
      for (int i = 0; i < row_comm_size; ++i) {
        device_bcast(row_comm,
                     vertex_property_input_first,
                     rx_value_first,
                     graph_view.vertex_partition_range_size(col_comm_rank * row_comm_size + i),
                     i,
                     handle.get_stream());

        auto v_offset_first = thrust::make_transform_iterator(
          *(edge_partition_minor_property_output.key_first()) + key_offsets[i],
          [v_first = graph_view.vertex_partition_range_first(
             col_comm_rank * row_comm_size + i)] __device__(auto v) { return v - v_first; });
        thrust::gather(handle.get_thrust_policy(),
                       v_offset_first,
                       v_offset_first + (key_offsets[i + 1] - key_offsets[i]),
                       rx_value_first,
                       edge_partition_minor_property_output.value_data() + key_offsets[i]);
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
                        edge_partition_minor_property_output.value_data(),
                        rx_counts,
                        displacements,
                        handle.get_stream());
    }
  } else {
    assert(!(edge_partition_minor_property_output.key_first()));
    assert(graph_view.local_vertex_partition_range_size() == GraphViewType::is_storage_transposed
             ? graph_view.local_edge_partition_src_range_size()
             : graph_view.local_edge_partition_dst_range_size());
    thrust::copy(handle.get_thrust_policy(),
                 vertex_property_input_first,
                 vertex_property_input_first + graph_view.local_vertex_partition_range_size(),
                 edge_partition_minor_property_output.value_data());
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexPropertyInputIterator,
          typename EdgePartitionMinorPropertyOutputWrapper>
void update_edge_partition_minor_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexPropertyInputIterator vertex_property_input_first,
  EdgePartitionMinorPropertyOutputWrapper& edge_partition_minor_property_output)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

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

    auto key_offsets = GraphViewType::is_storage_transposed
                         ? graph_view.local_sorted_unique_edge_src_offsets()
                         : graph_view.local_sorted_unique_edge_dst_offsets();

    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(size_t{0}));
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

      if (edge_partition_minor_property_output.key_first()) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator((*key_offsets)[i + 1] - (*key_offsets)[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_vertex_last  = rx_vertices.end(),
           rx_value_first,
           output_key_first =
             *(edge_partition_minor_property_output.key_first()) + (*key_offsets)[i],
           output_value_first = edge_partition_minor_property_output.value_data() +
                                (*key_offsets)[i]] __device__(auto i) {
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
                        edge_partition_minor_property_output.value_data());
      }
    }
  } else {
    assert(!(edge_partition_minor_property_output.key_first()));
    assert(graph_view.local_vertex_partition_range_size() ==
           graph_view.local_edge_partition_src_range_size());
    auto val_first = thrust::make_permutation_iterator(vertex_property_input_first, vertex_first);
    thrust::scatter(handle.get_thrust_policy(),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    edge_partition_minor_property_output.value_data());
  }
}

}  // namespace detail

/**
 * @brief Update graph edge partition source property values from the input vertex property values.
 *
 * This version updates graph edge partition property values for the entire edge partition source
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
 * @param edge_partition_src_property_output Device-copyable wrapper used to store source property
 * values (for the edge sources assigned to this process in multi-GPU). Use
 * cugraph::edge_partition_src_property_t::device_view().
 */
template <typename GraphViewType, typename VertexPropertyInputIterator>
void update_edge_partition_src_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_partition_src_property_t<
    GraphViewType,
    typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_partition_src_property_output)
{
  if constexpr (GraphViewType::is_storage_transposed) {
    update_edge_partition_minor_property(
      handle, graph_view, vertex_property_input_first, edge_partition_src_property_output);
  } else {
    update_edge_partition_major_property(
      handle, graph_view, vertex_property_input_first, edge_partition_src_property_output);
  }
}

/**
 * @brief Update graph edge partition source property values from the input vertex property values.
 *
 * This version updates only a subset of graph edge partition source property values. [@p
 * vertex_first, @p vertex_last) specifies the vertices with new property values to be updated.
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
 * @param edge_partition_src_property_output Device-copyable wrapper used to store source property
 * values (for the edge sources assigned to this process in multi-GPU). Use
 * cugraph::edge_partition_src_property_t::device_view().
 */
template <typename GraphViewType, typename VertexIterator, typename VertexPropertyInputIterator>
void update_edge_partition_src_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_partition_src_property_t<
    GraphViewType,
    typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_partition_src_property_output)
{
  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_partition_minor_property(handle,
                                                 graph_view,
                                                 vertex_first,
                                                 vertex_last,
                                                 vertex_property_input_first,
                                                 edge_partition_src_property_output);
  } else {
    detail::update_edge_partition_major_property(handle,
                                                 graph_view,
                                                 vertex_first,
                                                 vertex_last,
                                                 vertex_property_input_first,
                                                 edge_partition_src_property_output);
  }
}

/**
 * @brief Update graph edge partition destination property values from the input vertex property
 * values.
 *
 * This version updates graph edge partition property values for the entire edge partition
 * destination ranges (assigned to this process in multi-GPU).
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
 * @param edge_partition_dst_property_output Device-copyable wrapper used to store destination
 * property values (for the edge destinations assigned to this process in multi-GPU). Use
 * cugraph::edge_partition_dst_property_t::device_view().
 */
template <typename GraphViewType, typename VertexPropertyInputIterator>
void update_edge_partition_dst_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_partition_dst_property_t<
    GraphViewType,
    typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_partition_dst_property_output)
{
  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_partition_major_property(
      handle, graph_view, vertex_property_input_first, edge_partition_dst_property_output);
  } else {
    detail::update_edge_partition_minor_property(
      handle, graph_view, vertex_property_input_first, edge_partition_dst_property_output);
  }
}

/**
 * @brief Update graph edge partition destination property values from the input vertex property
 * values.
 *
 * This version updates only a subset of graph edge partition destination property values. [@p
 * vertex_first, @p vertex_last) specifies the vertices with new property values to be updated.
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
 * @param edge_partition_dst_property_output Device-copyable wrapper used to store destination
 * property values (for the edge destinations assigned to this process in multi-GPU). Use
 * cugraph::edge_partition_dst_property_t::device_view().
 */
template <typename GraphViewType, typename VertexIterator, typename VertexPropertyInputIterator>
void update_edge_partition_dst_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexPropertyInputIterator vertex_property_input_first,
  edge_partition_dst_property_t<
    GraphViewType,
    typename std::iterator_traits<VertexPropertyInputIterator>::value_type>&
    edge_partition_dst_property_output)
{
  if constexpr (GraphViewType::is_storage_transposed) {
    detail::update_edge_partition_major_property(handle,
                                                 graph_view,
                                                 vertex_first,
                                                 vertex_last,
                                                 vertex_property_input_first,
                                                 edge_partition_dst_property_output);
  } else {
    detail::update_edge_partition_minor_property(handle,
                                                 graph_view,
                                                 vertex_first,
                                                 vertex_last,
                                                 vertex_property_input_first,
                                                 edge_partition_dst_property_output);
  }
}

}  // namespace cugraph

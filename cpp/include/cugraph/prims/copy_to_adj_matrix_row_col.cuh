/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cugraph/graph_view.hpp>
#include <cugraph/matrix_partition_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/row_col_properties.cuh>
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
          typename VertexValueInputIterator,
          typename MatrixMajorValueOutputWrapper>
void copy_to_matrix_major(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMajorValueOutputWrapper& matrix_major_value_output)
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

    if (matrix_major_value_output.key_first()) {
      auto key_offsets = GraphViewType::is_adj_matrix_transposed
                           ? *(graph_view.get_local_sorted_unique_edge_col_offsets())
                           : *(graph_view.get_local_sorted_unique_edge_row_offsets());

      vertex_t max_rx_size{0};
      for (int i = 0; i < col_comm_size; ++i) {
        max_rx_size = std::max(
          max_rx_size, graph_view.get_vertex_partition_size(i * row_comm_size + row_comm_rank));
      }
      auto rx_value_buffer = allocate_dataframe_buffer<
        typename std::iterator_traits<VertexValueInputIterator>::value_type>(max_rx_size,
                                                                             handle.get_stream());
      auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffer);
      for (int i = 0; i < col_comm_size; ++i) {
        device_bcast(col_comm,
                     vertex_value_input_first,
                     rx_value_first,
                     graph_view.get_vertex_partition_size(i * row_comm_size + row_comm_rank),
                     i,
                     handle.get_stream());

        auto v_offset_first = thrust::make_transform_iterator(
          *(matrix_major_value_output.key_first()) + key_offsets[i],
          [v_first = graph_view.get_vertex_partition_first(
             i * row_comm_size + row_comm_rank)] __device__(auto v) { return v - v_first; });
        thrust::gather(handle.get_thrust_policy(),
                       v_offset_first,
                       v_offset_first + (key_offsets[i + 1] - key_offsets[i]),
                       rx_value_first,
                       matrix_major_value_output.value_data() + key_offsets[i]);
      }
    } else {
      std::vector<size_t> rx_counts(col_comm_size, size_t{0});
      std::vector<size_t> displacements(col_comm_size, size_t{0});
      for (int i = 0; i < col_comm_size; ++i) {
        rx_counts[i]     = graph_view.get_vertex_partition_size(i * row_comm_size + row_comm_rank);
        displacements[i] = (i == 0) ? 0 : displacements[i - 1] + rx_counts[i - 1];
      }
      device_allgatherv(col_comm,
                        vertex_value_input_first,
                        matrix_major_value_output.value_data(),
                        rx_counts,
                        displacements,
                        handle.get_stream());
    }
  } else {
    assert(!(matrix_major_value_output.key_first()));
    assert(graph_view.get_number_of_local_vertices() == GraphViewType::is_adj_matrix_transposed
             ? graph_view.get_number_of_local_adj_matrix_partition_cols()
             : graph_view.get_number_of_local_adj_matrix_partition_rows());
    thrust::copy(handle.get_thrust_policy(),
                 vertex_value_input_first,
                 vertex_value_input_first + graph_view.get_number_of_local_vertices(),
                 matrix_major_value_output.value_data());
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename MatrixMajorValueOutputWrapper>
void copy_to_matrix_major(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexIterator vertex_first,
                          VertexIterator vertex_last,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMajorValueOutputWrapper& matrix_major_value_output)
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
      typename std::iterator_traits<VertexValueInputIterator>::value_type>(max_rx_size,
                                                                           handle.get_stream());
    auto rx_value_first = get_dataframe_buffer_begin(rx_tmp_buffer);

    auto key_offsets = GraphViewType::is_adj_matrix_transposed
                         ? graph_view.get_local_sorted_unique_edge_col_offsets()
                         : graph_view.get_local_sorted_unique_edge_row_offsets();

    for (int i = 0; i < col_comm_size; ++i) {
      auto matrix_partition =
        matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
          graph_view.get_matrix_partition_view(i));

      if (col_comm_rank == i) {
        auto vertex_partition =
          vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
            graph_view.get_vertex_partition_view());
        auto map_first =
          thrust::make_transform_iterator(vertex_first, [vertex_partition] __device__(auto v) {
            return vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v);
          });
        // FIXME: this gather (and temporary buffer) is unnecessary if NCCL directly takes a
        // permutation iterator (and directly gathers to the internal buffer)
        thrust::gather(handle.get_thrust_policy(),
                       map_first,
                       map_first + thrust::distance(vertex_first, vertex_last),
                       vertex_value_input_first,
                       rx_value_first);
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(
        col_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());
      device_bcast(col_comm, rx_value_first, rx_value_first, rx_counts[i], i, handle.get_stream());

      if (matrix_major_value_output.key_first()) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator((*key_offsets)[i + 1] - (*key_offsets)[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_vertex_last  = rx_vertices.end(),
           rx_value_first,
           output_key_first = *(matrix_major_value_output.key_first()) + (*key_offsets)[i],
           output_value_first =
             matrix_major_value_output.value_data() + (*key_offsets)[i]] __device__(auto i) {
            auto major = *(output_key_first + i);
            auto it    = thrust::lower_bound(thrust::seq, rx_vertex_first, rx_vertex_last, major);
            if ((it != rx_vertex_last) && (*it == major)) {
              auto rx_value             = *(rx_value_first + thrust::distance(rx_vertex_first, it));
              *(output_value_first + i) = rx_value;
            }
          });
      } else {
        auto map_first = thrust::make_transform_iterator(
          rx_vertices.begin(), [matrix_partition] __device__(auto v) {
            return matrix_partition.get_major_offset_from_major_nocheck(v);
          });
        // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
        // directly scatters from the internal buffer)
        thrust::scatter(
          handle.get_thrust_policy(),
          rx_value_first,
          rx_value_first + rx_counts[i],
          map_first,
          matrix_major_value_output.value_data() + matrix_partition.get_major_value_start_offset());
      }
    }
  } else {
    assert(!(matrix_major_value_output.key_first()));
    assert(graph_view.get_number_of_local_vertices() == GraphViewType::is_adj_matrix_transposed
             ? graph_view.get_number_of_local_adj_matrix_partition_cols()
             : graph_view.get_number_of_local_adj_matrix_partition_rows());
    auto val_first = thrust::make_permutation_iterator(vertex_value_input_first, vertex_first);
    thrust::scatter(handle.get_thrust_policy(),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    matrix_major_value_output.value_data());
  }
}

template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename MatrixMinorValueOutputWrapper>
void copy_to_matrix_minor(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMinorValueOutputWrapper& matrix_minor_value_output)
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

    if (matrix_minor_value_output.key_first()) {
      auto key_offsets = GraphViewType::is_adj_matrix_transposed
                           ? *(graph_view.get_local_sorted_unique_edge_row_offsets())
                           : *(graph_view.get_local_sorted_unique_edge_col_offsets());

      vertex_t max_rx_size{0};
      for (int i = 0; i < row_comm_size; ++i) {
        max_rx_size = std::max(
          max_rx_size, graph_view.get_vertex_partition_size(col_comm_rank * row_comm_size + i));
      }
      auto rx_value_buffer = allocate_dataframe_buffer<
        typename std::iterator_traits<VertexValueInputIterator>::value_type>(max_rx_size,
                                                                             handle.get_stream());
      auto rx_value_first = get_dataframe_buffer_begin(rx_value_buffer);
      for (int i = 0; i < row_comm_size; ++i) {
        device_bcast(row_comm,
                     vertex_value_input_first,
                     rx_value_first,
                     graph_view.get_vertex_partition_size(col_comm_rank * row_comm_size + i),
                     i,
                     handle.get_stream());

        auto v_offset_first = thrust::make_transform_iterator(
          *(matrix_minor_value_output.key_first()) + key_offsets[i],
          [v_first = graph_view.get_vertex_partition_first(
             col_comm_rank * row_comm_size + i)] __device__(auto v) { return v - v_first; });
        thrust::gather(handle.get_thrust_policy(),
                       v_offset_first,
                       v_offset_first + (key_offsets[i + 1] - key_offsets[i]),
                       rx_value_first,
                       matrix_minor_value_output.value_data() + key_offsets[i]);
      }
    } else {
      std::vector<size_t> rx_counts(row_comm_size, size_t{0});
      std::vector<size_t> displacements(row_comm_size, size_t{0});
      for (int i = 0; i < row_comm_size; ++i) {
        rx_counts[i]     = graph_view.get_vertex_partition_size(col_comm_rank * row_comm_size + i);
        displacements[i] = (i == 0) ? 0 : displacements[i - 1] + rx_counts[i - 1];
      }
      device_allgatherv(row_comm,
                        vertex_value_input_first,
                        matrix_minor_value_output.value_data(),
                        rx_counts,
                        displacements,
                        handle.get_stream());
    }
  } else {
    assert(!(matrix_minor_value_output.key_first()));
    assert(graph_view.get_number_of_local_vertices() == GraphViewType::is_adj_matrix_transposed
             ? graph_view.get_number_of_local_adj_matrix_partition_rows()
             : graph_view.get_number_of_local_adj_matrix_partition_cols());
    thrust::copy(handle.get_thrust_policy(),
                 vertex_value_input_first,
                 vertex_value_input_first + graph_view.get_number_of_local_vertices(),
                 matrix_minor_value_output.value_data());
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename MatrixMinorValueOutputWrapper>
void copy_to_matrix_minor(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexIterator vertex_first,
                          VertexIterator vertex_last,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMinorValueOutputWrapper& matrix_minor_value_output)
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
      typename std::iterator_traits<VertexValueInputIterator>::value_type>(max_rx_size,
                                                                           handle.get_stream());
    auto rx_value_first = get_dataframe_buffer_begin(rx_tmp_buffer);

    auto key_offsets = GraphViewType::is_adj_matrix_transposed
                         ? graph_view.get_local_sorted_unique_edge_row_offsets()
                         : graph_view.get_local_sorted_unique_edge_col_offsets();

    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(size_t{0}));
    for (int i = 0; i < row_comm_size; ++i) {
      if (row_comm_rank == i) {
        auto vertex_partition =
          vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
            graph_view.get_vertex_partition_view());
        auto map_first =
          thrust::make_transform_iterator(vertex_first, [vertex_partition] __device__(auto v) {
            return vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v);
          });
        // FIXME: this gather (and temporary buffer) is unnecessary if NCCL directly takes a
        // permutation iterator (and directly gathers to the internal buffer)
        thrust::gather(handle.get_thrust_policy(),
                       map_first,
                       map_first + thrust::distance(vertex_first, vertex_last),
                       vertex_value_input_first,
                       rx_value_first);
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      device_bcast(
        row_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());
      device_bcast(row_comm, rx_value_first, rx_value_first, rx_counts[i], i, handle.get_stream());

      if (matrix_minor_value_output.key_first()) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(vertex_t{0}),
          thrust::make_counting_iterator((*key_offsets)[i + 1] - (*key_offsets)[i]),
          [rx_vertex_first = rx_vertices.begin(),
           rx_vertex_last  = rx_vertices.end(),
           rx_value_first,
           output_key_first = *(matrix_minor_value_output.key_first()) + (*key_offsets)[i],
           output_value_first =
             matrix_minor_value_output.value_data() + (*key_offsets)[i]] __device__(auto i) {
            auto minor = *(output_key_first + i);
            auto it    = thrust::lower_bound(thrust::seq, rx_vertex_first, rx_vertex_last, minor);
            if ((it != rx_vertex_last) && (*it == minor)) {
              auto rx_value             = *(rx_value_first + thrust::distance(rx_vertex_first, it));
              *(output_value_first + i) = rx_value;
            }
          });
      } else {
        auto map_first = thrust::make_transform_iterator(
          rx_vertices.begin(), [matrix_partition] __device__(auto v) {
            return matrix_partition.get_minor_offset_from_minor_nocheck(v);
          });
        // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
        // directly scatters from the internal buffer)
        thrust::scatter(handle.get_thrust_policy(),
                        rx_value_first,
                        rx_value_first + rx_counts[i],
                        map_first,
                        matrix_minor_value_output.value_data());
      }
    }
  } else {
    assert(!(matrix_minor_value_output.key_first()));
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_local_adj_matrix_partition_rows());
    auto val_first = thrust::make_permutation_iterator(vertex_value_input_first, vertex_first);
    thrust::scatter(handle.get_thrust_policy(),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    matrix_minor_value_output.value_data());
  }
}

}  // namespace detail

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix row property
 * variables.
 *
 * This version fills the entire set of graph adjacency matrix row property values.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_row_value_output Wrapper used to access data storage to copy row properties
 * (for the rows assigned to this process in multi-GPU).
 */
template <typename GraphViewType, typename VertexValueInputIterator>
void copy_to_adj_matrix_row(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexValueInputIterator vertex_value_input_first,
  row_properties_t<GraphViewType,
                   typename std::iterator_traits<VertexValueInputIterator>::value_type>&
    adj_matrix_row_value_output)
{
  if constexpr (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_minor(handle, graph_view, vertex_value_input_first, adj_matrix_row_value_output);
  } else {
    copy_to_matrix_major(handle, graph_view, vertex_value_input_first, adj_matrix_row_value_output);
  }
}

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix row property
 * variables.
 *
 * This version fills only a subset of graph adjacency matrix row property values. [@p vertex_first,
 * @p vertex_last) specifies the vertices with new values to be copied to graph adjacency matrix row
 * property variables.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex with new values to be
 * copied. v in [vertex_first, vertex_last) should be distinct (and should belong to this process in
 * multi-GPU), otherwise undefined behavior
 * @param vertex_last Iterator pointing to the last (exclusive) vertex with new values to be copied.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_row_value_output Wrapper used to access data storage to copy row properties
 * (for the rows assigned to this process in multi-GPU).
 */
template <typename GraphViewType, typename VertexIterator, typename VertexValueInputIterator>
void copy_to_adj_matrix_row(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexValueInputIterator vertex_value_input_first,
  row_properties_t<GraphViewType,
                   typename std::iterator_traits<VertexValueInputIterator>::value_type>&
    adj_matrix_row_value_output)
{
  if constexpr (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_minor(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_row_value_output);
  } else {
    copy_to_matrix_major(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_row_value_output);
  }
}

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix column property
 * variables.
 *
 * This version fills the entire set of graph adjacency matrix column property values.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_col_value_output Wrapper used to access data storage to copy column properties
 * (for the columns assigned to this process in multi-GPU).
 */
template <typename GraphViewType, typename VertexValueInputIterator>
void copy_to_adj_matrix_col(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexValueInputIterator vertex_value_input_first,
  col_properties_t<GraphViewType,
                   typename std::iterator_traits<VertexValueInputIterator>::value_type>&
    adj_matrix_col_value_output)
{
  if constexpr (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_major(handle, graph_view, vertex_value_input_first, adj_matrix_col_value_output);
  } else {
    copy_to_matrix_minor(handle, graph_view, vertex_value_input_first, adj_matrix_col_value_output);
  }
}

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix column property
 * variables.
 *
 * This version fills only a subset of graph adjacency matrix column property values. [@p
 * vertex_first, @p vertex_last) specifies the vertices with new values to be copied to graph
 * adjacency matrix column property variables.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex with new values to be
 * copied. v in [vertex_first, vertex_last) should be distinct (and should belong to this process in
 * multi-GPU), otherwise undefined behavior
 * @param vertex_last Iterator pointing to the last (exclusive) vertex with new values to be copied.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_col_value_output Wrapper used to access data storage to copy column properties
 * (for the columns assigned to this process in multi-GPU).
 */
template <typename GraphViewType, typename VertexIterator, typename VertexValueInputIterator>
void copy_to_adj_matrix_col(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  VertexValueInputIterator vertex_value_input_first,
  col_properties_t<GraphViewType,
                   typename std::iterator_traits<VertexValueInputIterator>::value_type>&
    adj_matrix_col_value_output)
{
  if constexpr (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_major(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_col_value_output);
  } else {
    copy_to_matrix_minor(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_col_value_output);
  }
}

}  // namespace cugraph

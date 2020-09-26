/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <experimental/graph_view.hpp>
#include <matrix_partition_device.cuh>
#include <partition_manager.hpp>
#include <utilities/comm_utils.cuh>
#include <utilities/error.hpp>
#include <utilities/thrust_tuple_utils.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>

#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {
namespace experimental {

namespace detail {

template <typename TupleType, size_t I>
auto allocate_buffer_tuple_element_impl(size_t buffer_size, cudaStream_t stream)
{
  using element_t = typename thrust::tuple_element<I, TupleType>::type;
  return rmm::device_uvector<element_t>(buffer_size, stream);
}

template <typename TupleType, size_t... Is>
auto allocate_buffer_tuple_impl(std::index_sequence<Is...>,
                                    size_t buffer_size,
                                    cudaStream_t stream)
{
  return thrust::make_tuple(
    allocate_buffer_tuple_element_impl<TupleType, Is>(buffer_size, stream)...);
}

template <typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto allocate_buffer(size_t buffer_size, cudaStream_t stream)
{
  return rmm::device_uvector<T>(buffer_size, stream);
}

template <typename T, typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto allocate_buffer(size_t buffer_size, cudaStream_t stream)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return allocate_buffer_tuple_impl<T>(
    std::make_index_sequence<tuple_size>(), buffer_size, stream);
}

template <typename TupleType, size_t I, typename BufferType>
auto get_buffer_begin_tuple_element_impl(BufferType& buffer)
{
  using element_t = typename thrust::tuple_element<I, TupleType>::type;
  return thrust::get<I>(buffer).begin();
}

template <typename TupleType, size_t... Is, typename BufferType>
auto get_buffer_begin_tuple_impl(std::index_sequence<Is...>, BufferType& buffer)
{
  return thrust::make_tuple(get_buffer_begin_tuple_element_impl<TupleType, Is>(buffer)...);
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
auto get_buffer_begin(BufferType& buffer)
{
  return buffer.begin();
}

template <typename T,
          typename BufferType,
          typename std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value>* = nullptr>
auto get_buffer_begin(BufferType& buffer)
{
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  return thrust::make_zip_iterator(
    get_buffer_begin_tuple_impl<T>(std::make_index_sequence<tuple_size>(), buffer));
}

template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename MatrixMajorValueOutputIterator>
void copy_to_matrix_major(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMajorValueOutputIterator matrix_major_value_output_first)
{
  if (GraphViewType::is_multi_gpu) {
    if (graph_view.is_hypergraph_partitioned()) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      std::vector<size_t> rx_counts(row_comm_size, size_t{0});
      std::vector<size_t> displacements(row_comm_size, size_t{0});
      for (int i = 0; i < row_comm_size; ++i) {
        rx_counts[i] = graph_view.get_vertex_partition_last(col_comm_rank * row_comm_size + i) -
                       graph_view.get_vertex_partition_first(col_comm_rank * row_comm_size + i);
        if (i == 0) {
          displacements[i] = 0;
        } else {
          displacements[i] = displacements[i - 1] + rx_counts[i - 1];
        }
      }
      device_allgatherv(row_comm,
                        vertex_value_input_first,
                        matrix_major_value_output_first,
                        rx_counts,
                        displacements,
                        handle.get_stream());
    }
  } else {
    assert(graph_view.get_number_of_local_vertices() == GraphViewType::is_adj_matrix_transposed
             ? graph_view.get_number_of_adj_matrix_local_cols()
             : graph_view.get_number_of_adj_matrix_local_rows());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 vertex_value_input_first,
                 vertex_value_input_first + graph_view.get_number_of_local_vertices(),
                 matrix_major_value_output_first);
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename MatrixMajorValueOutputIterator>
void copy_to_matrix_major(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexIterator vertex_first,
                          VertexIterator vertex_last,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMajorValueOutputIterator matrix_major_value_output_first)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (GraphViewType::is_multi_gpu) {
    if (graph_view.is_hypergraph_partitioned()) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      auto rx_counts =
        host_scalar_allgatherv(row_comm,
                               static_cast<size_t>(thrust::distance(vertex_first, vertex_last)),
                               handle.get_stream());
      std::vector<size_t> displacements(row_comm_size, size_t{0});
      std::partial_sum(rx_counts.begin(), rx_counts.end() - 1, displacements.begin() + 1);

      rmm::device_uvector<vertex_t> vertices(
        std::accumulate(rx_counts.begin(), rx_counts.end(), vertex_t{0}), handle.get_stream());
      auto tmp_buffer = detail::allocate_buffer<
        typename std::iterator_traits<VertexValueInputIterator>::value_type>(vertices.size(),
                                                                             handle.get_stream());
      auto value_first = detail::get_buffer_begin<
        typename std::iterator_traits<VertexValueInputIterator>::value_type>(tmp_buffer);

      // FIXME: this gather is unnecessary if NCCL directly takes a permutation iterator (and
      // directly gathers to the internal buffer)
      thrust::gather(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     vertex_first,
                     vertex_last,
                     vertex_value_input_first,
                     value_first + displacements[row_comm_rank]);

      device_allgatherv(
        row_comm,
        thrust::make_zip_iterator(thrust::make_tuple(vertex_first, vertex_value_input_first)),
        thrust::make_zip_iterator(thrust::make_tuple(vertices.begin(), value_first)),
        rx_counts,
        displacements,
        handle.get_stream());

      matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, 0);
      for (int i = 0; i < row_comm_size; ++i) {
        auto map_first = thrust::make_transform_iterator(
          vertices.begin() + displacements[i], [matrix_partition] __device__(auto v) {
            return matrix_partition.get_major_offset_from_major_nocheck(v);
          });
        // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
        // directly scatters from the internal buffer)
        thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        value_first + displacements[i],
                        value_first + displacements[i] + rx_counts[i],
                        map_first,
                        matrix_major_value_output_first);
      }
    }
  } else {
    assert(graph_view.get_number_of_local_vertices() == GraphViewType::is_adj_matrix_transposed
             ? graph_view.get_number_of_adj_matrix_local_cols()
             : graph_view.get_number_of_adj_matrix_local_rows());
    auto val_first = thrust::make_permutation_iterator(vertex_value_input_first, vertex_first);
    thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    matrix_major_value_output_first);
  }
}

template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename MatrixMinorValueOutputIterator>
void copy_to_matrix_minor(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMinorValueOutputIterator matrix_minor_value_output_first)
{
  if (GraphViewType::is_multi_gpu) {
    if (graph_view.is_hypergraph_partitioned()) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      // FIXME: this P2P is unnecessary if apply the partitioning scheme with hypergraph
      // partitioning
      auto comm_src_rank = row_comm_rank * col_comm_size + col_comm_rank;
      auto comm_dst_rank = (comm_rank % col_comm_size) * row_comm_size + comm_rank / col_comm_size;
      auto constexpr tuple_size = thrust_tuple_size_or_one<
        typename std::iterator_traits<VertexValueInputIterator>::value_type>::value;
      std::vector<raft::comms::request_t> requests(2 * tuple_size);
      device_isend<VertexValueInputIterator, MatrixMinorValueOutputIterator>(
        comm,
        vertex_value_input_first,
        static_cast<size_t>(graph_view.get_local_vertex_last() -
                            graph_view.get_local_vertex_first()),
        comm_dst_rank,
        int{0} /* base_tag */,
        requests.data());
      device_irecv<VertexValueInputIterator, MatrixMinorValueOutputIterator>(
        comm,
        matrix_minor_value_output_first +
          (graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + col_comm_rank) -
           graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size)),
        static_cast<size_t>(graph_view.get_vertex_partition_last(comm_src_rank) -
                            graph_view.get_vertex_partition_first(comm_src_rank)),
        comm_src_rank,
        int{0} /* base_tag */,
        requests.data() + tuple_size);
      comm.waitall(requests.size(), requests.data());

      for (int i = 0; i < col_comm_size; ++i) {
        auto offset = graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + i) -
                      graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size);
        auto count = graph_view.get_vertex_partition_last(row_comm_rank * col_comm_size + i) -
                     graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + i);
        device_bcast(col_comm,
                     matrix_minor_value_output_first + offset,
                     matrix_minor_value_output_first + offset,
                     count,
                     i,
                     handle.get_stream());
      }
    }
  } else {
    assert(graph_view.get_number_of_local_vertices() == GraphViewType::is_adj_matrix_transposed
             ? graph_view.get_number_of_adj_matrix_local_rows()
             : graph_view.get_number_of_adj_matrix_local_cols());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 vertex_value_input_first,
                 vertex_value_input_first + graph_view.get_number_of_local_vertices(),
                 matrix_minor_value_output_first);
  }
}

template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename MatrixMinorValueOutputIterator>
void copy_to_matrix_minor(raft::handle_t const& handle,
                          GraphViewType const& graph_view,
                          VertexIterator vertex_first,
                          VertexIterator vertex_last,
                          VertexValueInputIterator vertex_value_input_first,
                          MatrixMinorValueOutputIterator matrix_minor_value_output_first)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (GraphViewType::is_multi_gpu) {
    if (graph_view.is_hypergraph_partitioned()) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      auto& comm           = handle.get_comms();
      auto const comm_rank = comm.get_rank();
      auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      // FIXME: this P2P is unnecessary if apply the same partitioning scheme regardless of
      // hypergraph partitioning is applied or not
      auto comm_src_rank = row_comm_rank * col_comm_size + col_comm_rank;
      auto comm_dst_rank = (comm_rank % col_comm_size) * row_comm_size + comm_rank / col_comm_size;
      auto constexpr tuple_size = thrust_tuple_size_or_one<
        typename std::iterator_traits<VertexValueInputIterator>::value_type>::value;
      std::vector<raft::comms::request_t> requests(2 * tuple_size);
      device_isend<VertexValueInputIterator, MatrixMinorValueOutputIterator>(
        comm,
        vertex_value_input_first,
        static_cast<size_t>(graph_view.get_local_vertex_last() -
                            graph_view.get_local_vertex_first()),
        comm_dst_rank,
        int{0} /* base_tag */,
        requests.data());
      device_irecv<VertexValueInputIterator, MatrixMinorValueOutputIterator>(
        comm,
        matrix_minor_value_output_first +
          (graph_view.get_vertex_partition_first(comm_src_rank) -
           graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size)),
        static_cast<size_t>(graph_view.get_vertex_partition_last(comm_src_rank) -
                            graph_view.get_vertex_partition_first(comm_src_rank)),
        comm_src_rank,
        int{0} /* base_tag */,
        requests.data() + tuple_size);
      comm.waitall(requests.size(), requests.data());

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      for (int i = 0; i < col_comm_size; ++i) {
        auto offset = graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + i) -
                      graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size);
        auto count = graph_view.get_vertex_partition_last(row_comm_rank * col_comm_size + i) -
                     graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + i);
        device_bcast(col_comm,
                     matrix_minor_value_output_first + offset,
                     matrix_minor_value_output_first + offset,
                     count,
                     i,
                     handle.get_stream());
      }
    }
  } else {
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_adj_matrix_local_rows());
    auto val_first = thrust::make_permutation_iterator(vertex_value_input_first, vertex_first);
    thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    val_first,
                    val_first + thrust::distance(vertex_first, vertex_last),
                    vertex_first,
                    matrix_minor_value_output_first);
  }
}

}  // namespace detail

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix row property
 * variables.
 *
 * This version fills the entire set of graph adjacency matrix row property values. This function is
 * inspired by thrust::copy().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam AdjMatrixRowValueOutputIterator Type of the iterator for graph adjacency matrix row
 * output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_row_value_output_first Iterator pointing to the adjacency matrix row output
 * property variables for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_output_last` (exclusive) is deduced as @p adj_matrix_row_value_output_first
 * + @p graph_view.get_number_of_adj_matrix_local_rows().
 */
template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first)
{
  if (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_minor(
      handle, graph_view, vertex_value_input_first, adj_matrix_row_value_output_first);
  } else {
    copy_to_matrix_major(
      handle, graph_view, vertex_value_input_first, adj_matrix_row_value_output_first);
  }
}

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix row property
 * variables.
 *
 * This version fills only a subset of graph adjacency matrix row property values. [@p vertex_first,
 * @p vertex_last) specifies the vertices with new values to be copied to graph adjacency matrix row
 * property variables. This function is inspired by thrust::copy().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam AdjMatrixRowValueOutputIterator Type of the iterator for graph adjacency matrix row
 * output property variables.
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
 * @param adj_matrix_row_value_output_first Iterator pointing to the adjacency matrix row output
 * property variables for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_output_last` (exclusive) is deduced as @p adj_matrix_row_value_output_first
 * + @p graph_view.get_number_of_adj_matrix_local_rows().
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueOutputIterator>
void copy_to_adj_matrix_row(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first)
{
  if (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_minor(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_row_value_output_first);
  } else {
    copy_to_matrix_major(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_row_value_output_first);
  }
}

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix column property
 * variables.
 *
 * This version fills the entire set of graph adjacency matrix column property values. This function
 * is inspired by thrust::copy().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam AdjMatrixColValueOutputIterator Type of the iterator for graph adjacency matrix column
 * output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param adj_matrix_col_value_output_first Iterator pointing to the adjacency matrix column output
 * property variables for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_adj_matrix_local_cols().
 */
template <typename GraphViewType,
          typename VertexValueInputIterator,
          typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first)
{
  if (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_major(
      handle, graph_view, vertex_value_input_first, adj_matrix_col_value_output_first);
  } else {
    copy_to_matrix_minor(
      handle, graph_view, vertex_value_input_first, adj_matrix_col_value_output_first);
  }
}

/**
 * @brief Copy vertex property values to the corresponding graph adjacency matrix column property
 * variables.
 *
 * This version fills only a subset of graph adjacency matrix column property values. [@p
 * vertex_first, @p vertex_last) specifies the vertices with new values to be copied to graph
 * adjacency matrix column property variables. This function is inspired by thrust::copy().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam AdjMatrixColValueOutputIterator Type of the iterator for graph adjacency matrix column
 * output property variables.
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
 * @param adj_matrix_col_value_output_first Iterator pointing to the adjacency matrix column output
 * property variables for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_adj_matrix_local_cols().
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename VertexValueInputIterator,
          typename AdjMatrixColValueOutputIterator>
void copy_to_adj_matrix_col(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            VertexIterator vertex_first,
                            VertexIterator vertex_last,
                            VertexValueInputIterator vertex_value_input_first,
                            AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first)
{
  if (GraphViewType::is_adj_matrix_transposed) {
    copy_to_matrix_major(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_col_value_output_first);
  } else {
    copy_to_matrix_minor(handle,
                         graph_view,
                         vertex_first,
                         vertex_last,
                         vertex_value_input_first,
                         adj_matrix_col_value_output_first);
  }
}

}  // namespace experimental
}  // namespace cugraph

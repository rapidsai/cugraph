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
#include <vertex_partition_device.cuh>

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
        rx_counts[i]     = graph_view.get_vertex_partition_size(col_comm_rank * row_comm_size + i);
        displacements[i] = (i == 0) ? 0 : displacements[i - 1] + rx_counts[i - 1];
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
             ? graph_view.get_number_of_local_adj_matrix_partition_cols()
             : graph_view.get_number_of_local_adj_matrix_partition_rows());
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
        host_scalar_allgather(row_comm,
                              static_cast<size_t>(thrust::distance(vertex_first, vertex_last)),
                              handle.get_stream());

      matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, 0);
      for (int i = 0; i < row_comm_size; ++i) {
        rmm::device_uvector<vertex_t> rx_vertices(row_comm_rank == i ? size_t{0} : rx_counts[i],
                                                  handle.get_stream());
        auto rx_tmp_buffer =
          allocate_comm_buffer<typename std::iterator_traits<VertexValueInputIterator>::value_type>(
            rx_counts[i], handle.get_stream());
        auto rx_value_first = get_comm_buffer_begin<
          typename std::iterator_traits<VertexValueInputIterator>::value_type>(rx_tmp_buffer);

        if (row_comm_rank == i) {
          vertex_partition_device_t<GraphViewType> vertex_partition(graph_view);
          auto map_first =
            thrust::make_transform_iterator(vertex_first, [vertex_partition] __device__(auto v) {
              return vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v);
            });
          // FIXME: this gather (and temporary buffer) is unnecessary if NCCL directly takes a
          // permutation iterator (and directly gathers to the internal buffer)
          thrust::gather(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                         map_first,
                         map_first + thrust::distance(vertex_first, vertex_last),
                         vertex_value_input_first,
                         rx_value_first);
        }

        // FIXME: these broadcast operations can be placed between ncclGroupStart() and
        // ncclGroupEnd()
        device_bcast(
          row_comm, vertex_first, rx_vertices.begin(), rx_counts[i], i, handle.get_stream());
        device_bcast(
          row_comm, rx_value_first, rx_value_first, rx_counts[i], i, handle.get_stream());

        if (row_comm_rank == i) {
          auto map_first =
            thrust::make_transform_iterator(vertex_first, [matrix_partition] __device__(auto v) {
              return matrix_partition.get_major_offset_from_major_nocheck(v);
            });
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          rx_value_first,
                          rx_value_first + rx_counts[i],
                          map_first,
                          matrix_major_value_output_first);
        } else {
          auto map_first = thrust::make_transform_iterator(
            rx_vertices.begin(), [matrix_partition] __device__(auto v) {
              return matrix_partition.get_major_offset_from_major_nocheck(v);
            });
          // FIXME: this scatter is unnecessary if NCCL directly takes a permutation iterator (and
          // directly scatters from the internal buffer)
          thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          rx_value_first,
                          rx_value_first + rx_counts[i],
                          map_first,
                          matrix_major_value_output_first);
        }

        CUDA_TRY(cudaStreamSynchronize(
          handle.get_stream()));  // this is as necessary rx_tmp_buffer will become out-of-scope
                                  // once control flow exits this block (FIXME: we can reduce stream
                                  // synchronization if we compute the maximum rx_counts and
                                  // allocate rx_tmp_buffer outside the loop)
      }
    }
  } else {
    assert(graph_view.get_number_of_local_vertices() == GraphViewType::is_adj_matrix_transposed
             ? graph_view.get_number_of_local_adj_matrix_partition_cols()
             : graph_view.get_number_of_local_adj_matrix_partition_rows());
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

      // FIXME: this P2P is unnecessary if we apply the partitioning scheme used with hypergraph
      // partitioning
      auto comm_src_rank = row_comm_rank * col_comm_size + col_comm_rank;
      auto comm_dst_rank = (comm_rank % col_comm_size) * row_comm_size + comm_rank / col_comm_size;
      // FIXME: this branch may no longer necessary with NCCL backend
      if (comm_src_rank == comm_rank) {
        assert(comm_dst_rank == comm_rank);
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     vertex_value_input_first,
                     vertex_value_input_first + graph_view.get_number_of_local_vertices(),
                     matrix_minor_value_output_first +
                       (graph_view.get_vertex_partition_first(comm_src_rank) -
                        graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size)));
      } else {
        device_sendrecv<VertexValueInputIterator, MatrixMinorValueOutputIterator>(
          comm,
          vertex_value_input_first,
          static_cast<size_t>(graph_view.get_number_of_local_vertices()),
          comm_dst_rank,
          matrix_minor_value_output_first +
            (graph_view.get_vertex_partition_first(comm_src_rank) -
             graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size)),
          static_cast<size_t>(graph_view.get_vertex_partition_size(comm_src_rank)),
          comm_src_rank,
          handle.get_stream());
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      for (int i = 0; i < col_comm_size; ++i) {
        auto offset = graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + i) -
                      graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size);
        auto count = graph_view.get_vertex_partition_size(row_comm_rank * col_comm_size + i);
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
             ? graph_view.get_number_of_local_adj_matrix_partition_rows()
             : graph_view.get_number_of_local_adj_matrix_partition_cols());
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
      size_t tx_count    = thrust::distance(vertex_first, vertex_last);
      size_t rx_count{};
      // FIXME: it seems like raft::isend and raft::irecv do not properly handle the destination (or
      // source) == self case. Need to double check and fix this if this is indeed the case (or RAFT
      // may use ncclSend/ncclRecv instead of UCX for device data).
      if (comm_src_rank == comm_rank) {
        assert(comm_dst_rank == comm_rank);
        rx_count = tx_count;
      } else {
        std::vector<raft::comms::request_t> count_requests(2);
        comm.isend(&tx_count, 1, comm_dst_rank, 0 /* tag */, count_requests.data());
        comm.irecv(&rx_count, 1, comm_src_rank, 0 /* tag */, count_requests.data() + 1);
        comm.waitall(count_requests.size(), count_requests.data());
      }

      vertex_partition_device_t<GraphViewType> vertex_partition(graph_view);
      rmm::device_uvector<vertex_t> dst_vertices(rx_count, handle.get_stream());
      auto dst_tmp_buffer =
        allocate_comm_buffer<typename std::iterator_traits<VertexValueInputIterator>::value_type>(
          rx_count, handle.get_stream());
      auto dst_value_first =
        get_comm_buffer_begin<typename std::iterator_traits<VertexValueInputIterator>::value_type>(
          dst_tmp_buffer);
      if (comm_src_rank == comm_rank) {
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     vertex_first,
                     vertex_last,
                     dst_vertices.begin());
        auto map_first =
          thrust::make_transform_iterator(vertex_first, [vertex_partition] __device__(auto v) {
            return vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v);
          });
        thrust::gather(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       map_first,
                       map_first + thrust::distance(vertex_first, vertex_last),
                       vertex_value_input_first,
                       dst_value_first);
      } else {
        auto src_tmp_buffer =
          allocate_comm_buffer<typename std::iterator_traits<VertexValueInputIterator>::value_type>(
            tx_count, handle.get_stream());
        auto src_value_first = get_comm_buffer_begin<
          typename std::iterator_traits<VertexValueInputIterator>::value_type>(src_tmp_buffer);

        auto map_first =
          thrust::make_transform_iterator(vertex_first, [vertex_partition] __device__(auto v) {
            return vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v);
          });
        thrust::gather(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       map_first,
                       map_first + thrust::distance(vertex_first, vertex_last),
                       vertex_value_input_first,
                       src_value_first);

        device_sendrecv<decltype(vertex_first), decltype(dst_vertices.begin())>(
          comm,
          vertex_first,
          tx_count,
          comm_dst_rank,
          dst_vertices.begin(),
          rx_count,
          comm_src_rank,
          handle.get_stream());

        device_sendrecv<decltype(src_value_first), decltype(dst_value_first)>(comm,
                                                                              src_value_first,
                                                                              tx_count,
                                                                              comm_dst_rank,
                                                                              dst_value_first,
                                                                              rx_count,
                                                                              comm_src_rank,
                                                                              handle.get_stream());

        CUDA_TRY(cudaStreamSynchronize(
          handle.get_stream()));  // this is as necessary src_tmp_buffer will become out-of-scope
                                  // once control flow exits this block
      }

      // FIXME: now we can clear tx_tmp_buffer

      auto rx_counts = host_scalar_allgather(col_comm, rx_count, handle.get_stream());

      matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, 0);
      for (int i = 0; i < col_comm_size; ++i) {
        rmm::device_uvector<vertex_t> rx_vertices(col_comm_rank == i ? size_t{0} : rx_counts[i],
                                                  handle.get_stream());
        auto rx_tmp_buffer =
          allocate_comm_buffer<typename std::iterator_traits<VertexValueInputIterator>::value_type>(
            rx_counts[i], handle.get_stream());
        auto rx_value_first = get_comm_buffer_begin<
          typename std::iterator_traits<VertexValueInputIterator>::value_type>(rx_tmp_buffer);

        // FIXME: these broadcast operations can be placed between ncclGroupStart() and
        // ncclGroupEnd()
        device_bcast(col_comm,
                     dst_vertices.begin(),
                     rx_vertices.begin(),
                     rx_counts[i],
                     i,
                     handle.get_stream());
        device_bcast(
          col_comm, dst_value_first, rx_value_first, rx_counts[i], i, handle.get_stream());

        if (col_comm_rank == i) {
          auto map_first = thrust::make_transform_iterator(
            dst_vertices.begin(), [matrix_partition] __device__(auto v) {
              return matrix_partition.get_minor_offset_from_minor_nocheck(v);
            });

          thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          dst_value_first,
                          dst_value_first + rx_counts[i],
                          map_first,
                          matrix_minor_value_output_first);
        } else {
          auto map_first = thrust::make_transform_iterator(
            rx_vertices.begin(), [matrix_partition] __device__(auto v) {
              return matrix_partition.get_minor_offset_from_minor_nocheck(v);
            });

          thrust::scatter(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          rx_value_first,
                          rx_value_first + rx_counts[i],
                          map_first,
                          matrix_minor_value_output_first);
        }

        CUDA_TRY(cudaStreamSynchronize(
          handle.get_stream()));  // this is as necessary rx_tmp_buffer will become out-of-scope
                                  // once control flow exits this block (FIXME: we can reduce stream
                                  // synchronization if we compute the maximum rx_counts and
                                  // allocate rx_tmp_buffer outside the loop)
      }

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // this is as necessary dst_tmp_buffer will become out-of-scope once
                                // control flow exits this block
    }
  } else {
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_local_adj_matrix_partition_rows());
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
 * + @p graph_view.get_number_of_local_adj_matrix_partition_rows().
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
 * + @p graph_view.get_number_of_local_adj_matrix_partition_rows().
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
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
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
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
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

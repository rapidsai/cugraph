/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/handle.hpp>

#include <numeric>
#include <thrust/distance.h>
#include <utility>

namespace cugraph {

namespace detail {

/**
 * @brief Calculate local out degrees of the sources belonging to the adjacency matrices
 * stored on each gpu
 *
 * Iterate through partitions and store their local degrees
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return A single vector containing the local out degrees of the sources belong to the adjacency
 * matrices
 */
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> calculate_local_degrees(
  raft::handle_t const& handle, GraphViewType const& graph_view)
{
  static_assert(GraphViewType::is_adj_matrix_transposed == false);
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  rmm::device_uvector<edge_t> local_degrees(
    graph_view.get_number_of_local_adj_matrix_partition_rows(), handle.get_stream());

  // get_number_of_local_adj_matrix_partition_rows == summation of get_major_size() of all
  // partitions belonging to the gpu
  vertex_t partial_offset{0};
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));
    thrust::tabulate(handle.get_thrust_policy(),
                     local_degrees.begin() + partial_offset,
                     local_degrees.begin() + partial_offset + matrix_partition.get_major_size(),
                     [offsets = matrix_partition.get_offsets()] __device__(auto i) {
                       return offsets[i + 1] - offsets[i];
                     });
    partial_offset += matrix_partition.get_major_size();
  }
  return local_degrees;
}

/**
 * @brief Return partition information of all vertex ids of all the partitions belonging to a gpu
 *
 * Iterate through partitions and store the starting vertex ids, exclusive scan of vertex counts,
 * offsets and indices of the partitions csr structure
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return Tuple of device vectors. The first vector denotes the starting vertex ids belonging to
 * the gpu. The second vector denotes the vertex count offset (how many vertices are dealt with by
 * the previous partitions. The third vector is the pointer of the offset array of each partition.
 * The fourth vector is the pointer of the indices array of each partition.
 */
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::edge_type const*>,
           rmm::device_uvector<typename GraphViewType::vertex_type const*>>
partition_information(raft::handle_t const& handle, GraphViewType const& graph_view)
{
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using partition_t = matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                                     typename GraphViewType::edge_type,
                                                     typename GraphViewType::weight_type,
                                                     GraphViewType::is_multi_gpu>;

  std::vector<vertex_t> id_segments;
  std::vector<vertex_t> vertex_count_offsets;
  std::vector<edge_t const*> adjacency_list_offsets;
  std::vector<vertex_t const*> adjacency_list_indices;

  id_segments.reserve(1 + graph_view.get_number_of_local_adj_matrix_partitions());
  vertex_count_offsets.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
  adjacency_list_offsets.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
  adjacency_list_indices.reserve(graph_view.get_number_of_local_adj_matrix_partitions());

  id_segments.push_back(0);
  vertex_t counter{0};
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition = partition_t(graph_view.get_matrix_partition_view(i));

    // Starting vertex ids of each partition
    id_segments.push_back(graph_view.get_local_adj_matrix_partition_row_last(i));
    // Count of relative position of the vertices
    vertex_count_offsets.push_back(counter);
    // Adjacency list offset pointer of each partition
    adjacency_list_offsets.push_back(matrix_partition.get_offsets());
    // Adjacency list indices pointer of each partition
    adjacency_list_indices.push_back(matrix_partition.get_indices());

    counter += matrix_partition.get_major_size();
  }

  // Allocate device memory for transfer
  rmm::device_uvector<vertex_t> r_segments(id_segments.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> vc_offsets(vertex_count_offsets.size(), handle.get_stream());
  rmm::device_uvector<edge_t const*> al_offsets(adjacency_list_offsets.size(), handle.get_stream());
  rmm::device_uvector<vertex_t const*> al_indices(adjacency_list_indices.size(),
                                                  handle.get_stream());

  // Transfer data
  raft::update_device(
    r_segments.data(), id_segments.data(), id_segments.size(), handle.get_stream());
  raft::update_device(vc_offsets.data(),
                      vertex_count_offsets.data(),
                      vertex_count_offsets.size(),
                      handle.get_stream());
  raft::update_device(al_offsets.data(),
                      adjacency_list_offsets.data(),
                      adjacency_list_offsets.size(),
                      handle.get_stream());
  raft::update_device(al_indices.data(),
                      adjacency_list_indices.data(),
                      adjacency_list_indices.size(),
                      handle.get_stream());

  return std::make_tuple(
    std::move(r_segments), std::move(vc_offsets), std::move(al_offsets), std::move(al_indices));
}

}  // namespace detail

/**
 * @brief Return global out degrees of active sources
 *
 * Get partition information of all graph partitions on the gpu and select
 * global degrees of all active sources
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param active_sources Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param global_out_degrees Global out degrees for every source represented by current gpu
 * @return Global out degrees of all sources in active_sources
 */
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> get_active_source_global_degrees(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  rmm::device_uvector<typename GraphViewType::vertex_type>& active_sources,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_out_degrees)
{
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using partition_t = matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                                     typename GraphViewType::edge_type,
                                                     typename GraphViewType::weight_type,
                                                     GraphViewType::is_multi_gpu>;
  rmm::device_uvector<edge_t> active_source_degrees(active_sources.size(), handle.get_stream());

  std::vector<vertex_t> id_segments;
  std::vector<vertex_t> count_offsets;
  id_segments.reserve(1 + graph_view.get_number_of_local_adj_matrix_partitions());
  count_offsets.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
  id_segments.push_back(0);
  vertex_t counter{0};
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition = partition_t(graph_view.get_matrix_partition_view(i));
    // Starting vertex ids of each partition
    id_segments.push_back(graph_view.get_local_adj_matrix_partition_row_last(i));
    count_offsets.push_back(counter);
    counter += matrix_partition.get_major_size();
  }
  rmm::device_uvector<vertex_t> vertex_id_segments(id_segments.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> vertex_count_offsets(count_offsets.size(), handle.get_stream());
  raft::update_device(
    vertex_id_segments.data(), id_segments.data(), id_segments.size(), handle.get_stream());
  raft::update_device(
    vertex_count_offsets.data(), count_offsets.data(), count_offsets.size(), handle.get_stream());

  thrust::transform(
    handle.get_thrust_policy(),
    active_sources.begin(),
    active_sources.end(),
    active_source_degrees.begin(),
    [seg                  = vertex_id_segments.data(),
     global_out_degrees   = global_out_degrees.data(),
     vertex_count_offsets = vertex_count_offsets.data(),
     count                = vertex_id_segments.size()] __device__(auto v) {
      // Find which partition id did the source belong to
      auto partition_id =
        thrust::distance(seg, thrust::upper_bound(thrust::seq, seg, seg + count, v)) - 1;
      // starting position of the segment within global_degree_offset
      // where the information for partition (partition_id) starts
      //  vertex_count_offsets[partition_id]
      // The relative location of offset information for vertex id v within
      // the segment
      //  v - seg[partition_id]
      auto location_in_segment = v - seg[partition_id];
      // read location of global_degree_offset needs to take into account the
      // partition offsets because it is a concatenation of all the offsets
      // across all partitions
      auto location = location_in_segment + vertex_count_offsets[partition_id];

      return global_out_degrees[location];
    });
  return active_source_degrees;
}

/**
 * @brief Calculate global degree information for all vertices represented by current gpu
 *
 * Calculate local degree and perform row wise exclusive scan over all gpus in column
 * communicator.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return Tuple of two device vectors. The first one contains per source edge-count encountered
 * by gpus in the column communicator before current gpu. The second device vector contains the
 * global out degree for every source represented by current gpu
 */
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
get_global_degree_information(raft::handle_t const& handle, GraphViewType const& graph_view)
{
  using edge_t       = typename GraphViewType::edge_type;
  auto local_degrees = detail::calculate_local_degrees(handle, graph_view);

  auto& col_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_size = col_comm.get_size();
  auto const col_rank = col_comm.get_rank();
  auto& row_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_size = row_comm.get_size();

  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();
  auto const comm_rank = comm.get_rank();

  rmm::device_uvector<edge_t> temp_input(local_degrees.size(), handle.get_stream());
  raft::update_device(
    temp_input.data(), local_degrees.data(), local_degrees.size(), handle.get_stream());

  rmm::device_uvector<edge_t> recv_data(local_degrees.size(), handle.get_stream());
  if (col_rank == 0) {
    thrust::fill(handle.get_thrust_policy(), recv_data.begin(), recv_data.end(), edge_t{0});
  }
  for (int i = 0; i < col_size - 1; ++i) {
    if (col_rank == i) {
      comm.device_send(
        temp_input.begin(), temp_input.size(), comm_rank + row_size, handle.get_stream());
    }
    if (col_rank == i + 1) {
      comm.device_recv(
        recv_data.begin(), recv_data.size(), comm_rank - row_size, handle.get_stream());
      thrust::transform(handle.get_thrust_policy(),
                        temp_input.begin(),
                        temp_input.end(),
                        recv_data.begin(),
                        temp_input.begin(),
                        thrust::plus<edge_t>());
    }
    handle.get_comms().barrier();
  }
  // Get global degrees
  device_bcast(col_comm,
               temp_input.begin(),
               temp_input.begin(),
               temp_input.size(),
               col_size - 1,
               handle.get_stream());

  return std::make_tuple(std::move(recv_data), std::move(temp_input));
}

/**
 * @brief Gather active sources and associated client gpu ids across gpus in a
 * column communicator
 *
 * Collect all the vertex ids and client gpu ids to be processed by every gpu in
 * the column communicator and call sort on the list.
 *
 * @tparam vertex_t Type of vertex indices.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @tparam GPUIdIterator  Type of the iterator for gpu id identifiers.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertex_input_first Iterator pointing to the first vertex id to be processed
 * @param vertex_input_last Iterator pointing to the last (exclusive) vertex id to be processed
 * @param gpu_id_first Iterator pointing to the first gpu id to be processed
 * @return Device vector containing all the vertices that are to be processed by every gpu
 * in the column communicator
 */
template <typename GraphViewType, typename VertexIterator, typename GPUIdIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename std::iterator_traits<GPUIdIterator>::value_type>>
gather_active_sources_in_row(raft::handle_t const& handle,
                             GraphViewType const& graph_view,
                             VertexIterator vertex_input_first,
                             VertexIterator vertex_input_last,
                             GPUIdIterator gpu_id_first)
{
  static_assert(GraphViewType::is_adj_matrix_transposed == false);
  using vertex_t = typename GraphViewType::vertex_type;
  using gpu_t    = typename std::iterator_traits<GPUIdIterator>::value_type;

  auto& col_comm      = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  size_t source_count = thrust::distance(vertex_input_first, vertex_input_last);
  auto external_source_counts = host_scalar_allgather(col_comm, source_count, handle.get_stream());
  auto total_external_source_count =
    std::accumulate(external_source_counts.begin(), external_source_counts.end(), size_t{0});
  std::vector<size_t> displacements(external_source_counts.size(), size_t{0});
  std::exclusive_scan(
    external_source_counts.begin(), external_source_counts.end(), displacements.begin(), size_t{0});

  rmm::device_uvector<vertex_t> active_sources(total_external_source_count, handle.get_stream());
  rmm::device_uvector<gpu_t> active_source_gpu_ids(total_external_source_count,
                                                   handle.get_stream());
  // Get the sources other gpus on the same row are working on
  // TODO : replace with device_bcast for better scaling
  device_allgatherv(col_comm,
                    vertex_input_first,
                    active_sources.data(),
                    external_source_counts,
                    displacements,
                    handle.get_stream());
  device_allgatherv(col_comm,
                    gpu_id_first,
                    active_source_gpu_ids.data(),
                    external_source_counts,
                    displacements,
                    handle.get_stream());
  thrust::sort_by_key(handle.get_thrust_policy(),
                      active_sources.begin(),
                      active_sources.end(),
                      active_source_gpu_ids.begin());
  return std::make_tuple(std::move(active_sources), std::move(active_source_gpu_ids));
}

/**
 * @brief Gather valid edges present on the current gpu
 *
 * Collect all the edges that are present in the adjacency lists on the current gpu
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeIndexIterator Type of the iterator for edge indices.
 * @tparam GPUIdIterator  Type of the iterator for gpu id identifiers.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param active_sources_in_row Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param active_source_gpu_ids Device vector containing the gpu id associated by every vertex
 * present in active_sources_in_row
 * @param edge_index_first Iterator pointing to the first destination index
 * @param invalid_vertex_id Vertex id to fill in result if destination index is not accessible
 * on current gpu
 * @param indices_per_source Number of indices supplied for every source in the range
 * [vertex_input_first, vertex_input_last)
 * @param global_degree_offset Global degree offset to local adjacency list for every source
 * represented by current gpu
 * @return A tuple of device vector containing the sources and destinations gathered locally
 */
template <typename GraphViewType, typename EdgeIndexIterator, typename gpu_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<gpu_t>>
gather_local_edges(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  rmm::device_uvector<typename GraphViewType::vertex_type>& active_sources_in_row,
  rmm::device_uvector<gpu_t>& active_source_gpu_ids,
  EdgeIndexIterator edge_index_first,
  typename GraphViewType::vertex_type invalid_vertex_id,
  int indices_per_source,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_degree_offsets)
{
  using vertex_t  = typename GraphViewType::vertex_type;
  using edge_t    = typename GraphViewType::edge_type;
  auto edge_count = active_sources_in_row.size() * indices_per_source;
  rmm::device_uvector<vertex_t> sources(edge_count, handle.get_stream());
  rmm::device_uvector<vertex_t> destinations(edge_count, handle.get_stream());
  rmm::device_uvector<gpu_t> destination_gpu_ids(edge_count, handle.get_stream());

  auto [vertex_id_seg, vertex_count_offsets, graph_offsets, graph_indices] =
    detail::partition_information(handle, graph_view);

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(edge_count),
    [edge_index_first,
     active_sources        = active_sources_in_row.data(),
     active_source_gpu_ids = active_source_gpu_ids.data(),
     vertex_id_seg         = vertex_id_seg.data(),
     vertex_id_seg_count   = vertex_id_seg.size(),
     vertex_count_offsets  = vertex_count_offsets.data(),
     global_degree_offsets = global_degree_offsets.data(),
     graph_offsets         = graph_offsets.data(),
     graph_indices         = graph_indices.data(),
     sources               = sources.data(),
     destinations          = destinations.data(),
     dst_gpu_ids           = destination_gpu_ids.data(),
     invalid_vertex_id,
     indices_per_source] __device__(auto index) {
      // source which this edge index refers to
      auto loc           = index / indices_per_source;
      auto source        = active_sources[loc];
      sources[index]     = source;
      dst_gpu_ids[index] = active_source_gpu_ids[loc];

      // Find which partition id did the source belong to
      auto partition_id =
        thrust::distance(
          vertex_id_seg,
          thrust::upper_bound(
            thrust::seq, vertex_id_seg, vertex_id_seg + vertex_id_seg_count, source)) -
        1;
      // starting position of the segment within global_degree_offset
      // where the information for partition (partition_id) starts
      //  vertex_count_offsets[partition_id]
      // The relative location of offset information for vertex id v within
      // the segment
      //  v - seg[partition_id]
      auto location_in_segment = source - vertex_id_seg[partition_id];

      // csr offset value for vertex v that belongs to partition (partition_id)
      auto csr_offset       = graph_offsets[partition_id][location_in_segment];
      auto local_out_degree = graph_offsets[partition_id][location_in_segment + 1] - csr_offset;
      edge_t const* adjacency_list = graph_indices[partition_id] + csr_offset;
      // read location of global_degree_offset needs to take into account the
      // partition offsets because it is a concatenation of all the offsets
      // across all partitions
      auto location        = location_in_segment + vertex_count_offsets[partition_id];
      auto g_degree_offset = global_degree_offsets[location];
      auto g_dst_index     = edge_index_first[index];
      if ((g_dst_index >= g_degree_offset) && (g_dst_index < g_degree_offset + local_out_degree)) {
        destinations[index] = adjacency_list[g_dst_index - g_degree_offset];
      } else {
        destinations[index] = invalid_vertex_id;
      }
    });
  auto input_iter = thrust::make_zip_iterator(
    thrust::make_tuple(sources.begin(), destinations.begin(), destination_gpu_ids.begin()));

  auto compacted_length = thrust::distance(
    input_iter,
    thrust::remove_if(
      handle.get_thrust_policy(),
      input_iter,
      input_iter + destinations.size(),
      destinations.begin(),
      [invalid_vertex_id] __device__(auto dst) { return (dst == invalid_vertex_id); }));
  sources.resize(compacted_length, handle.get_stream());
  destinations.resize(compacted_length, handle.get_stream());
  destination_gpu_ids.resize(compacted_length, handle.get_stream());
  return std::make_tuple(
    std::move(sources), std::move(destinations), std::move(destination_gpu_ids));
}

}  // namespace cugraph

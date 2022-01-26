/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/handle.hpp>

#include <thrust/distance.h>
#include <utility>
#include <numeric>

namespace cugraph {

namespace detail {

/**
 * @brief Row wise exclusive scan of a local buffer
 *
 * Iteratively exchange buffers with neighbor gpu in column communicator and add them up.
 *
 * @tparam Type Data type of the buffer
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param input Reference to device buffer
 * @return A single vector containing the row wise exclusive scan
 */
  template <typename Type>
  rmm::device_uvector<Type> device_exclusive_scan(
  raft::handle_t const& handle,
  const rmm::device_uvector<Type> &input) {

    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_size = col_comm.get_size();
    auto const col_rank = col_comm.get_rank();
    auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_size = row_comm.get_size();
    auto const row_rank = row_comm.get_rank();

    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();

    rmm::device_uvector<Type> temp_input(input.size(), handle.get_stream());
    raft::update_device(temp_input.data(), input.data(), input.size(), handle.get_stream());

    rmm::device_uvector<Type> recv_data(input.size(), handle.get_stream());
    if ( col_rank == 0) {
      thrust::fill(handle.get_thrust_policy(), recv_data.begin(), recv_data.end(), Type{0});
    }
    for (int i = 0; i < col_size - 1; ++i) {
      if ( col_rank == i) {
        comm.device_send(
          temp_input.begin(), temp_input.size(), comm_rank + row_size, handle.get_stream());
      }
      if ( col_rank == i + 1) {
        comm.device_recv(
          recv_data.begin(), recv_data.size(), comm_rank - row_size, handle.get_stream());
        thrust::transform(handle.get_thrust_policy(),
                          temp_input.begin(), temp_input.end(),
                          recv_data.begin(),
                          temp_input.begin(),
                          thrust::plus<Type>());
      }
      handle.get_comms().barrier();
    }
    return recv_data;
  }

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
 * @return A single vector containing the local out degrees of the sources belong to the adjacency matrices
 */
  template <typename GraphViewType>
  rmm::device_uvector<typename GraphViewType::edge_type>
  calculate_local_degrees(raft::handle_t const& handle,
           GraphViewType const& graph_view) {

    static_assert(GraphViewType::is_adj_matrix_transposed == false);
    using vertex_t = typename GraphViewType::vertex_type;
    using edge_t = typename GraphViewType::edge_type;
    using weight_t = typename GraphViewType::weight_type;

    rmm::device_uvector<edge_t> local_degrees(graph_view.get_number_of_local_adj_matrix_partition_rows(),
                                              handle.get_stream());

    //get_number_of_local_adj_matrix_partition_rows == summation of get_major_size() of all
    //partitions belonging to the gpu
    vertex_t partial_offset{0};
    for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
      auto matrix_partition =
        matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
          graph_view.get_matrix_partition_view(i));
      thrust::tabulate(
        handle.get_thrust_policy(),
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
 * @return Tuple of device vectors. The first vector denotes the starting vertex ids belonging to the
 * gpu. The second vector denotes the vertex count offset (how many vertices are dealt with by the
 * previous partitions. The third vector is the pointer of the offset array of each partition. The
 * fourth vector is the pointer of the indices array of each partition.
 */
  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
            rmm::device_uvector<typename GraphViewType::vertex_type>,
            rmm::device_uvector<typename GraphViewType::edge_type const*>,
            rmm::device_uvector<typename GraphViewType::vertex_type const*>>
  partition_information(raft::handle_t const& handle,
           GraphViewType const& graph_view) {
    using vertex_t = typename GraphViewType::vertex_type;
    using edge_t = typename GraphViewType::edge_type;
    using partition_t = matrix_partition_device_view_t<typename GraphViewType::vertex_type, typename GraphViewType::edge_type, typename GraphViewType::weight_type, GraphViewType::is_multi_gpu>;

    std::vector<vertex_t> id_firsts;
    std::vector<vertex_t> vertex_count_offsets;
    std::vector<edge_t const*> adjacency_list_offsets;
    std::vector<vertex_t const*> adjacency_list_indices;

    id_firsts.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
    vertex_count_offsets.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
    adjacency_list_offsets.reserve(graph_view.get_number_of_local_adj_matrix_partitions());
    adjacency_list_indices.reserve(graph_view.get_number_of_local_adj_matrix_partitions());

    vertex_t counter{0};
    for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
      auto matrix_partition =
        partition_t(graph_view.get_matrix_partition_view(i));

      //Starting vertex ids of each partition
      id_firsts.push_back(graph_view.get_local_adj_matrix_partition_row_first(i));
      //Count of relative position of the vertices
      vertex_count_offsets.push_back(counter);
      //Adjacency list offset pointer of each partition
      adjacency_list_offsets.push_back(matrix_partition.get_offsets());
      //Adjacency list indices pointer of each partition
      adjacency_list_indices.push_back(matrix_partition.get_indices());

      counter += matrix_partition.get_major_size();
    }

    //Allocate device memory for transfer
    rmm::device_uvector<vertex_t> r_firsts(id_firsts.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> vc_offsets(vertex_count_offsets.size(), handle.get_stream());
    rmm::device_uvector<edge_t const*> al_offsets(adjacency_list_offsets.size(), handle.get_stream());
    rmm::device_uvector<vertex_t const*> al_indices(adjacency_list_indices.size(), handle.get_stream());

    //Transfer data
    raft::update_device(r_firsts.data(),
                        id_firsts.data(), id_firsts.size(),
                        handle.get_stream());
    raft::update_device(vc_offsets.data(),
                        vertex_count_offsets.data(), vertex_count_offsets.size(),
                        handle.get_stream());
    raft::update_device(al_offsets.data(),
                        adjacency_list_offsets.data(), adjacency_list_offsets.size(),
                        handle.get_stream());
    raft::update_device(al_indices.data(),
                        adjacency_list_indices.data(), adjacency_list_indices.size(),
                        handle.get_stream());

    return std::make_tuple(r_firsts, vc_offsets, al_offsets, al_indices);
  }

/**
 * @brief Gather active source across gpus in a column communicator
 *
 * Collect all the vertex ids to be processed by every gpu in the column communicator and
 * call sort and unique on the list.
 *
 * @tparam vertex_t Type of vertex indices.
 * @tparam VertexIterator  Type of the iterator for vertex identifiers.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertex_input_first Iterator pointing to the first vertex id to be processed
 * @param vertex_input_last Iterator pointing to the last (exclusive) vertex id to be processed
 * @return Device vector containing all the vertices that are to be processed by every gpu
 * in the column communicator
 */
  template <typename vertex_t, typename VertexIterator>
  rmm::device_uvector<vertex_t>
  gather_active_sources_in_row(raft::handle_t const& handle,
           VertexIterator vertex_input_first,
           VertexIterator vertex_input_last) {
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto source_count = thrust::distance(vertex_input_first, vertex_input_last);
    auto external_source_counts = host_scalar_allgather(col_comm,
                            source_count,
                            handle.get_stream());
    auto total_external_source_count =
      std::accumulate(external_source_counts.begin(), external_source_counts.end(), size_t{0});
    std::vector<size_t> displacements(external_source_counts.size(), size_t{0});
    std::exclusive_scan(external_source_counts.begin(), external_source_counts.end(),
                        displacements.begin(), size_t{0});

    rmm::device_uvector<vertex_t> active_sources(total_external_source_count, handle.get_stream());
    //Get the sources other gpus on the same row are working on
    //TODO : replace with device_bcast for better scaling
    device_allgatherv(col_comm,
                      vertex_input_first,
                      active_sources.begin(),
                      external_source_counts,
                      displacements,
                      handle.get_stream());
    thrust::sort(handle.get_thrust_policy(),
                 active_sources.begin(),
                 active_sources.end());
    active_sources.resize(
      thrust::distance(
        active_sources.begin(),
        thrust::unique(handle.get_thrust_policy(),
                       active_sources.begin(),
                       active_sources.end())),
      handle.get_stream());

    return active_sources;
  }

/**
 * @brief Gather active source across gpus in a column communicator
 *
 * Collect all the vertex ids to be processed by every gpu in the column communicator and
 * call sort and unique on the list.
 *
 * @tparam vertex_t Type of vertex indices.
 * @tparam VertexIterator Type of the iterator for vertex identifiers.
 * @tparam EdgeIndexIterator Type of the iterator for edge indices.
 * @tparam edge_t type of edge indices.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertex_input_first Iterator pointing to the first vertex id to be processed
 * @param vertex_input_last Iterator pointing to the last (exclusive) vertex id to be processed
 * @param edge_index_first Iterator pointing to the first destination index
 * @param active_sources_in_row Device vector containing all the vertex id that are processed by
 * gpus in the column communicator
 * @param active_sources_adjacency_lists Device vector containing adjacency list pointers of the
 * vertices in active_sources_in_row
 * @param active_sources_local_degrees Device vector containing local out degrees of the
 * vertices in active_sources_in_row
 * @param active_sources_global_degree_offsets Device vector containing out degrees offsets
 * of the vertices in active_sources_in_row
 * @param invalid_vertex_id Vertex id to fill in result if destination index is not accessible
 * on current gpu
 * @param indices_per_source Number of indices supplied for every source in the range
 * [vertex_input_first, vertex_input_last)
 * @return A tuple of device vector containing the sources and destinations gathered locally
 */
  template <typename VertexIterator, typename vertex_t,
           typename EdgeIndexIterator, typename edge_t>
  std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
  gather_local_edges(raft::handle_t const& handle,
           VertexIterator vertex_input_first,
           VertexIterator vertex_input_last,
           EdgeIndexIterator edge_index_first,
           rmm::device_uvector<vertex_t> &active_sources_in_row,
           rmm::device_uvector<const vertex_t*> &active_sources_adjacency_lists,
           rmm::device_uvector<edge_t> &active_sources_local_degrees,
           rmm::device_uvector<edge_t> &active_sources_global_degree_offsets,
           vertex_t invalid_vertex_id,
           int indices_per_source) {
    auto source_count = thrust::distance(vertex_input_first, vertex_input_last);
    auto edge_count = source_count*indices_per_source;
    rmm::device_uvector<vertex_t> sources(edge_count, handle.get_stream());
    rmm::device_uvector<vertex_t> destinations(edge_count, handle.get_stream());
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(edge_count),
                     [vertex_input_first,
                      indices_per_source,
                      edge_index_first,
                      active_sources = active_sources_in_row.data(),
                      active_source_count = active_sources_in_row.size(),
                      adjacency_lists = active_sources_adjacency_lists.data(),
                      degrees = active_sources_local_degrees.data(),
                      degree_offsets = active_sources_global_degree_offsets.data(),
                      sources = sources.data(),
                      destinations = destinations.data(),
                      invalid_vertex_id
                     ] __device__ (auto index) {
                       //source which this edge index refers to
                       auto source = vertex_input_first[index/indices_per_source];
                       sources[index] = source;

                       //location of source in active_sources
                       auto loc =
                         thrust::distance(active_sources.begin(),
                                          thrust::lower_bound(thrust::seq,
                                                              active_sources,
                                                              active_sources + active_source_count,
                                                              source));
                       auto global_dst_index = edge_index_first[index];
                       if ((global_dst_index >= degree_offsets[loc]) &&
                         (global_dst_index < degree_offsets[loc] + degrees[loc])) {
                         destinations[index] = adjacency_lists[loc][global_dst_index - degree_offsets[loc]];
                       } else {
                         destinations[index] = invalid_vertex_id;
                       }
                     });

    return std::make_tuple(sources, destinations);
  }

/**
 * @brief Return local out degrees, global out degrees offsets
 * @brief Return partition information of all vertex ids of all the partitions belonging to a gpu
 *
 * Iterate through partitions and store the starting vertex ids, exclusive scan of vertex counts,
 * offsets and indices of the partitions csr structure
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return Tuple of device vectors. The first vector denotes the starting vertex ids belonging to the
 * gpu. The second vector denotes the vertex count offset (how many vertices are dealt with by the
 * previous partitions. The third vector is the pointer of the offset array of each partition. The
 * fourth vector is the pointer of the indices array of each partition.
 */
  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
  rmm::device_uvector<typename GraphViewType::edge_type>,
  rmm::device_uvector<typename GraphViewType::vertex_type const*>>
    get_active_sources_information(raft::handle_t const& handle,
                                   GraphViewType const& graph_view,
                                   rmm::device_uvector<typename GraphViewType::vertex_type> &active_sources,
                                   const rmm::device_uvector<typename GraphViewType::edge_type> &local_degree_offset) {
      using vertex_t = typename GraphViewType::vertex_type;
      using edge_t = typename GraphViewType::edge_type;
      auto [vertex_id_first, vertex_count_offsets, graph_offsets, graph_indices] =
        partition_information(handle, graph_view);
      rmm::device_uvector<edge_t> active_source_degree_offsets(
        active_sources.size(), handle.get_stream());
      rmm::device_uvector<edge_t> active_source_degrees(
        active_sources.size(), handle.get_stream());
      rmm::device_uvector<vertex_t const*> active_source_adjacency_lists(
        active_sources.size(), handle.get_stream());
      thrust::transform(handle.get_stream(),
                        active_sources.begin(),
                        active_sources.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                        active_source_degree_offsets.begin(),
                        active_source_degrees.begin(),
                        active_source_adjacency_lists.begin())),
                        [first = vertex_id_first.data(),
                        local_degree_offset = local_degree_offset.data(),
                        vertex_count_offsets = vertex_count_offsets.data(),
                        graph_offsets = graph_offsets.data(),
                        graph_indices = graph_indices.data(),
                        count = vertex_id_first.size()] __device__ (auto v) {
                          //Find which partition id did the source belong to
                          auto partition_id = thrust::lower_bound(thrust::seq,
                                                                  first, first + count, v);
                          //starting position of the segment within local_degree_offset
                          //where the information for partition (partition_id) starts
                          //  vertex_count_offsets[partition_id]
                          //The relative location of offset information for vertex id v within
                          //the segment
                          //  v - first[partition_id]
                          auto location_in_segment = v - first[partition_id];
                          //csr offset value for vertex v that belongs to partition (partition_id)
                          auto csr_offset = graph_offsets[partition_id][location_in_segment];
                          auto local_out_degree = graph_offsets[partition_id][location_in_segment + 1] -
                                                  csr_offset;
                          //read location of local_degree_offset needs to take into account the
                          //partition offsets because it is a concatenation of all the offsets
                          //across all partitions
                          auto location = location_in_segment + vertex_count_offsets[partition_id];
                          return thrust::make_tuple(local_degree_offset[location], local_out_degree, graph_indices[partition_id] + csr_offset);
                        });
      return std::make_tuple(active_source_degrees, active_source_degree_offsets, active_source_adjacency_lists);
    }



}

  template <typename GraphViewType>
  rmm::device_uvector<typename GraphViewType::edge_type>
  get_local_degree_offset(raft::handle_t const& handle,
           GraphViewType const& graph_view) {
    auto local_degrees = detail::calculate_local_degrees(handle, graph_view);
    return detail::device_exclusive_scan(handle, local_degrees);
  }

/**
 * @brief Sample edges based on destination indices
 *
 * Assume supplied vertex ids input belongs to target gpu
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @return 
 */
  template <typename GraphViewType, typename VertexIterator, typename EdgeIndexIterator, typename T>
  void gather_edges(raft::handle_t const& handle,
           GraphViewType const& graph_view,
           VertexIterator vertex_input_first,
           VertexIterator vertex_input_last,
           EdgeIndexIterator edge_index_first,
           T indices_per_source,
           const rmm::device_uvector<typename GraphViewType::edge_type> &local_degree_offset) {
    static_assert(GraphViewType::is_adj_matrix_transposed == false);
    using vertex_t = typename GraphViewType::vertex_type;
    using edge_t = typename GraphViewType::edge_type;
    using weight_t = typename GraphViewType::weight_type;

    //active_sources_in_row is sorted and unique
    auto active_sources_in_row = detail::gather_active_sources_in_row<vertex_t>(handle, vertex_input_first, vertex_input_last);

    //Fill local_active_source_degree_offsets with relevant values from local_degree_offset
    //A lower_bound is required to translate the user supplied vertex ids to the
    //relevant position of the information of those ids in local_degree_offset.
    auto [vertex_id_first, vertex_count_offsets, graph_offsets, graph_indices] =
      partition_information(handle, graph_view);
    rmm::device_uvector<edge_t> active_source_degree_offsets(
      active_sources_in_row.size(), handle.get_stream());
    rmm::device_uvector<edge_t> active_source_degrees(
      active_sources_in_row.size(), handle.get_stream());
    rmm::device_uvector<vertex_t const*> active_source_adjacency_lists(
      active_sources_in_row.size(), handle.get_stream());
    thrust::transform(handle.get_stream(),
                      active_sources_in_row.begin(),
                      active_sources_in_row.end(),
                      thrust::make_zip_iterator(thrust::make_tuple(
                      active_source_degree_offsets.begin(),
                      active_source_degrees.begin(),
                      active_source_adjacency_lists.begin())),
                      [first = vertex_id_first.data(),
                       local_degree_offset = local_degree_offset.data(),
                       vertex_count_offsets = vertex_count_offsets.data(),
                       graph_offsets = graph_offsets.data(),
                       graph_indices = graph_indices.data(),
                       count = vertex_id_first.size()] __device__ (auto v) {
                        //Find which partition id did the source belong to
                        auto partition_id = thrust::lower_bound(thrust::seq,
                                                                first, first + count, v);
                        //starting position of the segment within local_degree_offset
                        //where the information for partition (partition_id) starts
                        //  vertex_count_offsets[partition_id]
                        //The relative location of offset information for vertex id v within
                        //the segment
                        //  v - first[partition_id]
                        auto location_in_segment = v - first[partition_id];
                        //csr offset value for vertex v that belongs to partition (partition_id)
                        auto csr_offset = graph_offsets[partition_id][location_in_segment];
                        auto local_out_degree = graph_offsets[partition_id][location_in_segment + 1] -
                                                csr_offset;
                        //read location of local_degree_offset needs to take into account the
                        //partition offsets because it is a concatenation of all the offsets
                        //across all partitions
                        auto location = location_in_segment + vertex_count_offsets[partition_id];
                        return thrust::make_tuple(local_degree_offset[location], local_out_degree, graph_indices[partition_id] + csr_offset);
                      });

    //Gather valid destinations
    auto [sources, destinations] =
      detail::gather_local_edges(handle,
                                 vertex_input_first, vertex_input_last,
                                 edge_index_first,
                                 active_sources_in_row,
                                 active_source_adjacency_lists,
                                 active_source_degrees,
                                 active_source_degree_offsets,
                                 graph_view.get_number_of_vertices(),
                                 indices_per_source);

  }


}  // namespace cugraph

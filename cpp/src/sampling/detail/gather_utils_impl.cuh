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

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/detail/graph_functions.cuh>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/handle.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <rmm/device_uvector.hpp>

#include <numeric>
#include <vector>

namespace cugraph {

namespace detail {

template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> compute_local_major_degrees(
  raft::handle_t const& handle, GraphViewType const& graph_view)
{
  static_assert(GraphViewType::is_storage_transposed == false);
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  rmm::device_uvector<edge_t> local_degrees(GraphViewType::is_storage_transposed
                                              ? graph_view.local_edge_partition_dst_range_size()
                                              : graph_view.local_edge_partition_src_range_size(),
                                            handle.get_stream());

  // FIXME optimize for communication
  // local_edge_partition_src_range_size == summation of major_range_size() of all partitions
  // belonging to the gpu
  vertex_t partial_offset{0};
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    // Check if hypersparse segment is present in the partition
    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    auto use_dcs         = segment_offsets
                             ? ((*segment_offsets).size() > (num_sparse_segments_per_vertex_partition + 1))
                             : false;

    if (use_dcs) {
      auto major_hypersparse_first = edge_partition.major_range_first() +
                                     (*segment_offsets)[num_sparse_segments_per_vertex_partition];
      // Calculate degrees in sparse region
      auto sparse_begin = local_degrees.begin() + partial_offset;
      auto sparse_end   = local_degrees.begin() + partial_offset +
                        (major_hypersparse_first - edge_partition.major_range_first());
      ;

      thrust::tabulate(handle.get_thrust_policy(),
                       sparse_begin,
                       sparse_end,
                       [offsets = edge_partition.offsets()] __device__(auto i) {
                         return offsets[i + 1] - offsets[i];
                       });

      // Calculate degrees in hypersparse region
      auto dcs_nzd_vertex_count = *(edge_partition.dcs_nzd_vertex_count());
      // Initialize hypersparse region degrees as 0
      thrust::fill(handle.get_thrust_policy(),
                   sparse_end,
                   sparse_begin + edge_partition.major_range_size(),
                   edge_t{0});
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(vertex_t{0}),
                       thrust::make_counting_iterator(dcs_nzd_vertex_count),
                       [major_hypersparse_first,
                        major_range_first = edge_partition.major_range_first(),
                        vertex_ids        = *(edge_partition.dcs_nzd_vertices()),
                        offsets           = edge_partition.offsets(),
                        local_degrees = thrust::raw_pointer_cast(sparse_begin)] __device__(auto i) {
                         auto d = offsets[(major_hypersparse_first - major_range_first) + i + 1] -
                                  offsets[(major_hypersparse_first - major_range_first) + i];
                         auto v                               = vertex_ids[i];
                         local_degrees[v - major_range_first] = d;
                       });
    } else {
      auto sparse_begin = local_degrees.begin() + partial_offset;
      auto sparse_end = local_degrees.begin() + partial_offset + edge_partition.major_range_size();
      thrust::tabulate(handle.get_thrust_policy(),
                       sparse_begin,
                       sparse_end,
                       [offsets = edge_partition.offsets()] __device__(auto i) {
                         return offsets[i + 1] - offsets[i];
                       });
    }
    partial_offset += edge_partition.major_range_size();
  }
  return local_degrees;
}

template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> get_global_adjacency_offset(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_degree_offsets,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_out_degrees)
{
  static_assert(GraphViewType::is_multi_gpu == true);
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  // auto const& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();
  auto const comm_rank = comm.get_rank();
  auto& col_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_size  = col_comm.get_size();
  auto& row_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_rank  = row_comm.get_rank();
  auto const row_size  = row_comm.get_size();

  rmm::device_uvector<edge_t> global_adjacency_list_offsets(global_degree_offsets.size(),
                                                            handle.get_stream());

  edge_t edge_count_in_all_previous_partitions{0};
  vertex_t vertex_offset{0};
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_counts =
      cugraph::host_scalar_allgather(comm, edge_partition.number_of_edges(), handle.get_stream());
    edge_t partial_edge_count{0};
    for (int r = 0; r < row_rank; ++r) {
      for (int c = 0; c < col_size; ++c) {
        partial_edge_count += edge_counts[r + c * row_size];
      }
    }
    thrust::exclusive_scan(
      handle.get_thrust_policy(),
      global_out_degrees.cbegin() + vertex_offset,
      global_out_degrees.cbegin() + vertex_offset + edge_partition.major_range_size(),
      global_adjacency_list_offsets.begin() + vertex_offset);

    thrust::transform(
      handle.get_thrust_policy(),
      global_adjacency_list_offsets.cbegin() + vertex_offset,
      global_adjacency_list_offsets.cbegin() + vertex_offset + edge_partition.major_range_size(),
      global_degree_offsets.cbegin() + vertex_offset,
      global_adjacency_list_offsets.begin() + vertex_offset,
      [offset = edge_count_in_all_previous_partitions + partial_edge_count] __device__(
        auto val0, auto val1) { return val0 + val1 + offset; });

    edge_count_in_all_previous_partitions +=
      std::accumulate(edge_counts.begin(), edge_counts.end(), edge_t{0});
    vertex_offset += edge_partition.major_range_size();
  }
  return global_adjacency_list_offsets;
}

template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
get_global_degree_information(raft::handle_t const& handle, GraphViewType const& graph_view)
{
  static_assert(GraphViewType::is_multi_gpu == true);
  using edge_t       = typename GraphViewType::edge_type;
  auto local_degrees = compute_local_major_degrees(handle, graph_view);

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
    col_comm.barrier();
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

template <typename GraphViewType, typename VertexIterator, typename GPUIdIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename std::iterator_traits<GPUIdIterator>::value_type>>
gather_active_majors(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     VertexIterator vertex_input_first,
                     VertexIterator vertex_input_last,
                     GPUIdIterator gpu_id_first)
{
  static_assert(GraphViewType::is_multi_gpu == true);
  static_assert(GraphViewType::is_storage_transposed == false);
  using gpu_t    = typename std::iterator_traits<GPUIdIterator>::value_type;
  using vertex_t = typename GraphViewType::vertex_type;

  auto const& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  size_t source_count  = thrust::distance(vertex_input_first, vertex_input_last);
  auto external_source_counts =
    cugraph::host_scalar_allgather(col_comm, source_count, handle.get_stream());
  auto total_external_source_count =
    std::accumulate(external_source_counts.begin(), external_source_counts.end(), size_t{0});
  std::vector<size_t> displacements(external_source_counts.size(), size_t{0});
  std::exclusive_scan(
    external_source_counts.begin(), external_source_counts.end(), displacements.begin(), size_t{0});

  rmm::device_uvector<vertex_t> active_majors(total_external_source_count, handle.get_stream());
  rmm::device_uvector<gpu_t> active_major_gpu_ids(total_external_source_count, handle.get_stream());
  // Get the sources other gpus on the same row are working on
  // FIXME : replace with device_bcast for better scaling
  device_allgatherv(col_comm,
                    vertex_input_first,
                    active_majors.data(),
                    external_source_counts,
                    displacements,
                    handle.get_stream());
  device_allgatherv(col_comm,
                    gpu_id_first,
                    active_major_gpu_ids.data(),
                    external_source_counts,
                    displacements,
                    handle.get_stream());
  thrust::sort_by_key(handle.get_thrust_policy(),
                      active_majors.begin(),
                      active_majors.end(),
                      active_major_gpu_ids.begin());
  return std::make_tuple(std::move(active_majors), std::move(active_major_gpu_ids));
}

template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::edge_type> get_active_major_global_degrees(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_out_degrees)
{
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using partition_t = edge_partition_device_view_t<typename GraphViewType::vertex_type,
                                                   typename GraphViewType::edge_type,
                                                   typename GraphViewType::weight_type,
                                                   GraphViewType::is_multi_gpu>;
  rmm::device_uvector<edge_t> active_major_degrees(active_majors.size(), handle.get_stream());

  std::vector<vertex_t> id_begin;
  std::vector<vertex_t> id_end;
  std::vector<vertex_t> count_offsets;
  id_begin.reserve(graph_view.number_of_local_edge_partitions());
  id_end.reserve(graph_view.number_of_local_edge_partitions());
  count_offsets.reserve(graph_view.number_of_local_edge_partitions());
  vertex_t counter{0};
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition = partition_t(graph_view.local_edge_partition_view(i));
    // Starting vertex ids of each partition
    id_begin.push_back(edge_partition.major_range_first());
    id_end.push_back(edge_partition.major_range_last());
    count_offsets.push_back(counter);
    counter += edge_partition.major_range_size();
  }
  rmm::device_uvector<vertex_t> vertex_id_begin(id_begin.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> vertex_id_end(id_end.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> vertex_count_offsets(count_offsets.size(), handle.get_stream());
  raft::update_device(
    vertex_id_begin.data(), id_begin.data(), id_begin.size(), handle.get_stream());
  raft::update_device(vertex_id_end.data(), id_end.data(), id_end.size(), handle.get_stream());
  raft::update_device(
    vertex_count_offsets.data(), count_offsets.data(), count_offsets.size(), handle.get_stream());

  thrust::transform(handle.get_thrust_policy(),
                    active_majors.begin(),
                    active_majors.end(),
                    active_major_degrees.begin(),
                    [id_begin             = vertex_id_begin.data(),
                     id_end               = vertex_id_end.data(),
                     global_out_degrees   = global_out_degrees.data(),
                     vertex_count_offsets = vertex_count_offsets.data(),
                     count                = vertex_id_end.size()] __device__(auto v) {
                      // Find which partition id did the vertex belong to
                      auto partition_id = thrust::distance(
                        id_end, thrust::upper_bound(thrust::seq, id_end, id_end + count, v));
                      // starting position of the segment within global_degree_offset
                      // where the information for partition (partition_id) starts
                      //  vertex_count_offsets[partition_id]
                      // The relative location of offset information for vertex id v within
                      // the segment
                      //  v - id_end[partition_id]
                      auto location_in_segment = v - id_begin[partition_id];
                      // read location of global_degree_offset needs to take into account the
                      // partition offsets because it is a concatenation of all the offsets
                      // across all partitions
                      auto location = location_in_segment + vertex_count_offsets[partition_id];
                      return global_out_degrees[location];
                    });
  return active_major_degrees;
}

template <typename GraphViewType>
std::tuple<rmm::device_uvector<edge_partition_device_view_t<typename GraphViewType::vertex_type,
                                                            typename GraphViewType::edge_type,
                                                            typename GraphViewType::weight_type,
                                                            GraphViewType::is_multi_gpu>>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
partition_information(raft::handle_t const& handle, GraphViewType const& graph_view)
{
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using partition_t = edge_partition_device_view_t<typename GraphViewType::vertex_type,
                                                   typename GraphViewType::edge_type,
                                                   typename GraphViewType::weight_type,
                                                   GraphViewType::is_multi_gpu>;

  std::vector<partition_t> partitions;
  std::vector<vertex_t> id_begin;
  std::vector<vertex_t> id_end;
  std::vector<vertex_t> hypersparse_begin;
  std::vector<vertex_t> vertex_count_offsets;

  partitions.reserve(graph_view.number_of_local_edge_partitions());
  id_begin.reserve(graph_view.number_of_local_edge_partitions());
  id_end.reserve(graph_view.number_of_local_edge_partitions());
  hypersparse_begin.reserve(graph_view.number_of_local_edge_partitions());
  vertex_count_offsets.reserve(graph_view.number_of_local_edge_partitions());

  vertex_t counter{0};
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    partitions.emplace_back(graph_view.local_edge_partition_view(i));
    auto& edge_partition = partitions.back();

    // Starting vertex ids of each partition
    id_begin.push_back(edge_partition.major_range_first());
    id_end.push_back(edge_partition.major_range_last());

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    auto use_dcs         = segment_offsets
                             ? ((*segment_offsets).size() > (num_sparse_segments_per_vertex_partition + 1))
                             : false;
    if (use_dcs) {
      auto major_hypersparse_first = edge_partition.major_range_first() +
                                     (*segment_offsets)[num_sparse_segments_per_vertex_partition];
      hypersparse_begin.push_back(major_hypersparse_first);
    } else {
      hypersparse_begin.push_back(edge_partition.major_range_last());
    }

    // Count of relative position of the vertices
    vertex_count_offsets.push_back(counter);

    counter += edge_partition.major_range_size();
  }

  // Allocate device memory for transfer
  rmm::device_uvector<partition_t> edge_partitions(graph_view.number_of_local_edge_partitions(),
                                                   handle.get_stream());

  rmm::device_uvector<vertex_t> major_begin(id_begin.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> minor_end(id_end.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> hs_begin(hypersparse_begin.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> vc_offsets(vertex_count_offsets.size(), handle.get_stream());

  // Transfer data
  raft::update_device(
    edge_partitions.data(), partitions.data(), partitions.size(), handle.get_stream());
  raft::update_device(major_begin.data(), id_begin.data(), id_begin.size(), handle.get_stream());
  raft::update_device(minor_end.data(), id_end.data(), id_end.size(), handle.get_stream());
  raft::update_device(vc_offsets.data(),
                      vertex_count_offsets.data(),
                      vertex_count_offsets.size(),
                      handle.get_stream());
  raft::update_device(
    hs_begin.data(), hypersparse_begin.data(), hypersparse_begin.size(), handle.get_stream());

  return std::make_tuple(std::move(edge_partitions),
                         std::move(major_begin),
                         std::move(minor_end),
                         std::move(hs_begin),
                         std::move(vc_offsets));
}

template <typename GraphViewType, typename gpu_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<gpu_t>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
gather_local_edges(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  const rmm::device_uvector<gpu_t>& active_major_gpu_ids,
  rmm::device_uvector<typename GraphViewType::edge_type>&& minor_map,
  typename GraphViewType::edge_type indices_per_major,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_degree_offsets,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_adjacency_list_offsets)
{
  static_assert(GraphViewType::is_multi_gpu == true);
  using vertex_t  = typename GraphViewType::vertex_type;
  using edge_t    = typename GraphViewType::edge_type;
  auto edge_count = active_majors.size() * indices_per_major;
  rmm::device_uvector<vertex_t> majors(edge_count, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(edge_count, handle.get_stream());
  rmm::device_uvector<gpu_t> minor_gpu_ids(edge_count, handle.get_stream());
  vertex_t invalid_vertex_id = graph_view.number_of_vertices();

  auto [partitions, id_begin, id_end, hypersparse_begin, vertex_count_offsets] =
    partition_information(handle, graph_view);

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(edge_count),
    [edge_index_first      = minor_map.begin(),
     active_majors         = active_majors.data(),
     active_major_gpu_ids  = active_major_gpu_ids.data(),
     id_begin              = id_begin.data(),
     id_end                = id_end.data(),
     id_seg_count          = id_begin.size(),
     vertex_count_offsets  = vertex_count_offsets.data(),
     glbl_degree_offsets   = global_degree_offsets.data(),
     glbl_adj_list_offsets = global_adjacency_list_offsets.data(),
     majors                = majors.data(),
     minors                = minors.data(),
     dst_gpu_ids           = minor_gpu_ids.data(),
     partitions            = partitions.data(),
     hypersparse_begin     = hypersparse_begin.data(),
     invalid_vertex_id,
     indices_per_major] __device__(auto index) {
      // major which this edge index refers to
      auto loc           = index / indices_per_major;
      auto major         = active_majors[loc];
      majors[index]      = major;
      dst_gpu_ids[index] = active_major_gpu_ids[loc];

      // Find which partition id did the major belong to
      auto partition_id = thrust::distance(
        id_end, thrust::upper_bound(thrust::seq, id_end, id_end + id_seg_count, major));
      // starting position of the segment within global_degree_offset
      // where the information for partition (partition_id) starts
      //  vertex_count_offsets[partition_id]
      // The relative location of offset information for vertex id v within
      // the segment
      //  v - seg[partition_id]
      vertex_t location_in_segment;
      if (major < hypersparse_begin[partition_id]) {
        location_in_segment = major - id_begin[partition_id];
      } else {
        auto row_hypersparse_idx =
          partitions[partition_id].major_hypersparse_idx_from_major_nocheck(major);
        if (row_hypersparse_idx) {
          location_in_segment = *(row_hypersparse_idx)-id_begin[partition_id];
        } else {
          minors[index] = invalid_vertex_id;
          return;
        }
      }

      // csr offset value for vertex v that belongs to partition (partition_id)
      auto offset_ptr                = partitions[partition_id].offsets();
      auto sparse_offset             = offset_ptr[location_in_segment];
      auto local_out_degree          = offset_ptr[location_in_segment + 1] - sparse_offset;
      vertex_t const* adjacency_list = partitions[partition_id].indices() + sparse_offset;
      // read location of global_degree_offset needs to take into account the
      // partition offsets because it is a concatenation of all the offsets
      // across all partitions
      auto location        = location_in_segment + vertex_count_offsets[partition_id];
      auto g_degree_offset = glbl_degree_offsets[location];
      auto g_dst_index     = edge_index_first[index];
      if ((g_dst_index >= g_degree_offset) && (g_dst_index < g_degree_offset + local_out_degree)) {
        minors[index]           = adjacency_list[g_dst_index - g_degree_offset];
        edge_index_first[index] = g_dst_index - g_degree_offset + glbl_adj_list_offsets[location];
      } else {
        minors[index] = invalid_vertex_id;
      }
    });
  auto input_iter = thrust::make_zip_iterator(
    thrust::make_tuple(majors.begin(), minors.begin(), minor_gpu_ids.begin(), minor_map.begin()));

  auto compacted_length = thrust::distance(
    input_iter,
    thrust::remove_if(
      handle.get_thrust_policy(),
      input_iter,
      input_iter + minors.size(),
      minors.begin(),
      [invalid_vertex_id] __device__(auto dst) { return (dst == invalid_vertex_id); }));
  majors.resize(compacted_length, handle.get_stream());
  minors.resize(compacted_length, handle.get_stream());
  minor_gpu_ids.resize(compacted_length, handle.get_stream());
  minor_map.resize(compacted_length, handle.get_stream());
  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(minor_gpu_ids), std::move(minor_map));
}

template <typename GraphViewType, typename VertexIterator>
typename GraphViewType::edge_type edgelist_count(raft::handle_t const& handle,
                                                 GraphViewType const& graph_view,
                                                 VertexIterator vertex_input_first,
                                                 VertexIterator vertex_input_last)
{
  using edge_t = typename GraphViewType::edge_type;
  // Expect that vertex input list is sorted
  auto [partitions, id_begin, id_end, hypersparse_begin, vertex_count_offsets] =
    partition_information(handle, graph_view);
  return thrust::transform_reduce(
    handle.get_thrust_policy(),
    vertex_input_first,
    vertex_input_last,
    [partitions           = partitions.data(),
     id_begin             = id_begin.data(),
     id_end               = id_end.data(),
     id_seg_count         = id_begin.size(),
     hypersparse_begin    = hypersparse_begin.data(),
     vertex_count_offsets = vertex_count_offsets.data()] __device__(auto major) {
      // Find which partition id did the vertex belong to
      auto partition_id = thrust::distance(
        id_end, thrust::upper_bound(thrust::seq, id_end, id_end + id_seg_count, major));
      auto edge_partition          = partitions[partition_id];
      auto major_hypersparse_first = hypersparse_begin[partition_id];
      if (major < major_hypersparse_first) {
        auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
        return edge_partition.local_degree(major_offset);
      } else {
        auto major_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
        return major_hypersparse_idx
                 ? edge_partition.local_degree(
                     edge_partition.major_offset_from_major_nocheck(major_hypersparse_first) +
                     *major_hypersparse_idx)
                 : edge_t{0};
      }
    },
    edge_t{0},
    thrust::plus<edge_t>());
}

template <typename vertex_t>
std::vector<vertex_t> get_active_major_segments(raft::handle_t const& handle,
                                                vertex_t major_range_first,
                                                vertex_t major_range_last,
                                                std::vector<vertex_t> const& partition_segments,
                                                const rmm::device_uvector<vertex_t>& active_majors)
{
  std::vector<vertex_t> segments(partition_segments.size());
  std::transform(partition_segments.begin(),
                 partition_segments.end(),
                 segments.begin(),
                 [major_range_first](auto s) { return s + major_range_first; });
  segments.push_back(major_range_last);

  rmm::device_uvector<vertex_t> p_segments(segments.size(), handle.get_stream());
  raft::update_device(p_segments.data(), segments.data(), segments.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> majors_segments(segments.size(), handle.get_stream());
  thrust::lower_bound(handle.get_thrust_policy(),
                      active_majors.cbegin(),
                      active_majors.cend(),
                      p_segments.begin(),
                      p_segments.end(),
                      majors_segments.begin());
  std::vector<vertex_t> active_majors_segments(majors_segments.size());
  raft::update_host(active_majors_segments.data(),
                    majors_segments.data(),
                    majors_segments.size(),
                    handle.get_stream());
  return active_majors_segments;
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void local_major_degree(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> partition,
  rmm::device_uvector<vertex_t> const& active_majors,
  std::vector<vertex_t> const& majors_segments,
  std::vector<vertex_t> const& partition_segments,
  edge_t* out_degrees)
{
  auto active_major_count = majors_segments.back() - majors_segments.front();
  // Sparse region
  if (majors_segments[3] - majors_segments[0] > 0) {
    thrust::transform(handle.get_thrust_policy(),
                      active_majors.cbegin() + majors_segments[0],
                      active_majors.cbegin() + majors_segments[3],
                      out_degrees,
                      [partition] __device__(auto major) {
                        auto major_offset = partition.major_offset_from_major_nocheck(major);
                        return partition.local_degree(major_offset);
                      });
  }
  // Hypersparse region
  if (majors_segments[4] - majors_segments[3] > 0) {
    auto major_hypersparse_first =
      partition.major_range_first() +
      partition_segments[detail::num_sparse_segments_per_vertex_partition];
    auto major_offset =
      static_cast<size_t>(major_hypersparse_first - partition.major_range_first());
    thrust::transform(handle.get_thrust_policy(),
                      active_majors.cbegin() + majors_segments[3],
                      active_majors.cbegin() + majors_segments[4],
                      out_degrees + majors_segments[3] - majors_segments[0],
                      [partition, major_offset] __device__(auto major) {
                        auto major_idx = partition.major_hypersparse_idx_from_major_nocheck(major);
                        if (major_idx) {
                          return partition.local_degree(major_offset + *major_idx);
                        } else {
                          return edge_t{0};
                        }
                      });
  }
}

template <typename GraphViewType, typename prop_t>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<prop_t>,
           rmm::device_uvector<typename GraphViewType::edge_type>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  const rmm::device_uvector<prop_t>& active_major_property,
  const rmm::device_uvector<typename GraphViewType::edge_type>& global_adjacency_list_offsets)
{
  // Assumes active_majors is sorted

  static_assert(GraphViewType::is_multi_gpu == true);
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  std::vector<std::vector<vertex_t>> active_majors_segments;
  vertex_t max_active_major_count{0};
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    // Identify segments of active_majors
    active_majors_segments.emplace_back(
      get_active_major_segments(handle,
                                partition.major_range_first(),
                                partition.major_range_last(),
                                *(graph_view.local_edge_partition_segment_offsets(i)),
                                active_majors));
    auto& majors_segments = active_majors_segments.back();
    // Count of active majors belonging to this partition
    max_active_major_count =
      std::max(max_active_major_count, majors_segments.back() - majors_segments.front());
  }

  auto& comm           = handle.get_comms();
  auto const comm_rank = comm.get_rank();
  rmm::device_uvector<edge_t> active_majors_out_offsets(1 + max_active_major_count,
                                                        handle.get_stream());
  auto edge_count = edgelist_count(handle, graph_view, active_majors.begin(), active_majors.end());
  rmm::device_uvector<vertex_t> majors(edge_count, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(edge_count, handle.get_stream());
  rmm::device_uvector<prop_t> minor_prop_ids(edge_count, handle.get_stream());
  rmm::device_uvector<edge_t> minor_map(edge_count, handle.get_stream());

  edge_t output_offset = 0;
  vertex_t vertex_offset{0};
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto& majors_segments = active_majors_segments[i];
    // Calculate local degree offsets
    auto active_major_count = majors_segments.back() - majors_segments.front();
    active_majors_out_offsets.set_element_to_zero_async(0, handle.get_stream());
    local_major_degree(handle,
                       partition,
                       active_majors,
                       majors_segments,
                       *(graph_view.local_edge_partition_segment_offsets(i)),
                       1 + active_majors_out_offsets.data());
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           active_majors_out_offsets.begin() + 1,
                           active_majors_out_offsets.begin() + 1 + active_major_count,
                           active_majors_out_offsets.begin() + 1);
    active_majors_out_offsets.resize(1 + active_major_count, handle.get_stream());
    partially_decompress_edge_partition_to_fill_edgelist(
      handle,
      partition,
      active_majors.cbegin(),
      active_majors_out_offsets.cbegin(),
      majors_segments,
      output_offset + majors.data(),
      output_offset + minors.data(),
      thrust::make_optional(
        thrust::make_tuple(active_major_property.cbegin(), output_offset + minor_prop_ids.data())),
      thrust::make_optional(
        thrust::make_tuple(global_adjacency_list_offsets.cbegin() + vertex_offset,
                           output_offset + minor_map.begin())));
    output_offset += active_majors_out_offsets.back_element(handle.get_stream());
    vertex_offset += partition.major_range_size();
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(minor_prop_ids), std::move(minor_map));
}

}  // namespace detail

}  // namespace cugraph

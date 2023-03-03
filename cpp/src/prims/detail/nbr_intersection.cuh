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

#include <prims/kv_store.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <array>
#include <type_traits>

namespace cugraph {

namespace detail {

// check vertices in the pair are valid and first element of the pair is within the local vertex
// partition range
template <typename vertex_t>
struct is_invalid_input_vertex_pair_t {
  vertex_t num_vertices{};
  raft::device_span<vertex_t const> edge_partition_major_range_firsts{};
  raft::device_span<vertex_t const> edge_partition_major_range_lasts{};
  vertex_t edge_partition_minor_range_first{};
  vertex_t edge_partition_minor_range_last{};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> pair) const
  {
    auto major = thrust::get<0>(pair);
    auto minor = thrust::get<1>(pair);
    if (!is_valid_vertex(num_vertices, major) || !is_valid_vertex(num_vertices, minor)) {
      return true;
    }
    auto it = thrust::upper_bound(thrust::seq,
                                  edge_partition_major_range_lasts.begin(),
                                  edge_partition_major_range_lasts.end(),
                                  major);
    if (it == edge_partition_major_range_lasts.end()) { return true; }
    auto edge_partition_idx =
      static_cast<size_t>(thrust::distance(edge_partition_major_range_lasts.begin(), it));
    if (major < edge_partition_major_range_firsts[edge_partition_idx]) { return true; }
    return (minor < edge_partition_minor_range_first) || (minor >= edge_partition_minor_range_last);
  }
};

// group index determined by row_comm_rank (primary key) and local edge partition index (secondary
// key)
template <typename vertex_t>
struct major_to_group_idx_t {
  vertex_t const* vertex_partition_range_lasts{nullptr};
  int comm_size{};
  int row_comm_size{};
  int col_comm_size{};

  __device__ int operator()(vertex_t major) const
  {
    auto it = thrust::upper_bound(
      thrust::seq, vertex_partition_range_lasts, vertex_partition_range_lasts + comm_size, major);
    auto comm_rank = static_cast<int>(thrust::distance(vertex_partition_range_lasts, it));
    return (comm_rank % row_comm_size) * col_comm_size + (comm_rank / row_comm_size);
  }
};

// primary key: row_comm_rank secondary key: local partition index => primary key: local partition
// index secondary key: row_comm_rank
struct reorder_group_count_t {
  int row_comm_size{};
  int col_comm_size{};
  size_t const* group_counts{nullptr};

  __device__ size_t operator()(size_t i) const
  {
    return group_counts[(static_cast<int>(i) % row_comm_size) * col_comm_size +
                        static_cast<int>(i) / row_comm_size];
  }
};

template <typename vertex_t, typename edge_t, bool multi_gpu>
struct update_rx_major_local_degree_t {
  int row_comm_size{};
  int col_comm_size{};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};

  size_t reordered_idx_first{};
  size_t local_partition_idx{};

  size_t const* rx_reordered_group_lasts{nullptr};
  size_t const* rx_group_firsts{nullptr};
  vertex_t const* rx_majors{nullptr};

  edge_t* local_degrees_for_rx_majors{nullptr};

  __device__ void operator()(size_t idx) const
  {
    auto it = thrust::upper_bound(
      thrust::seq, rx_reordered_group_lasts, rx_reordered_group_lasts + row_comm_size, idx);
    auto row_comm_rank = static_cast<int>(thrust::distance(rx_reordered_group_lasts, it));
    auto offset_in_local_edge_partition =
      idx - (row_comm_rank == int{0} ? reordered_idx_first
                                     : rx_reordered_group_lasts[row_comm_rank - int{1}]);
    auto major = rx_majors[rx_group_firsts[row_comm_rank * col_comm_size + local_partition_idx] +
                           offset_in_local_edge_partition];
    edge_t local_degree{};
    if (multi_gpu && (edge_partition.major_hypersparse_first() &&
                      (major >= *(edge_partition.major_hypersparse_first())))) {
      auto major_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
      local_degree               = major_hypersparse_idx
                                     ? edge_partition.local_degree((*(edge_partition.major_hypersparse_first()) -
                                                      edge_partition.major_range_first()) +
                                                     *major_hypersparse_idx)
                                     : edge_t{0};
    } else {
      local_degree =
        edge_partition.local_degree(edge_partition.major_offset_from_major_nocheck(major));
    }
    local_degrees_for_rx_majors[rx_group_firsts[row_comm_rank * col_comm_size +
                                                local_partition_idx] +
                                offset_in_local_edge_partition] = local_degree;
  }
};

template <typename vertex_t, typename edge_t, bool multi_gpu>
struct update_rx_major_local_nbrs_t {
  int row_comm_size{};
  int col_comm_size{};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};

  size_t reordered_idx_first{};
  size_t local_partition_idx{};

  size_t const* rx_reordered_group_lasts{nullptr};
  size_t const* rx_group_firsts{nullptr};
  vertex_t const* rx_majors{nullptr};
  size_t const* local_nbr_offsets_for_rx_majors{nullptr};

  vertex_t* local_nbrs_for_rx_majors{nullptr};

  __device__ void operator()(size_t idx) const
  {
    auto it = thrust::upper_bound(
      thrust::seq, rx_reordered_group_lasts, rx_reordered_group_lasts + row_comm_size, idx);
    auto row_comm_rank = static_cast<int>(thrust::distance(rx_reordered_group_lasts, it));
    auto offset_in_local_edge_partition =
      idx - (row_comm_rank == int{0} ? reordered_idx_first
                                     : rx_reordered_group_lasts[row_comm_rank - int{1}]);
    auto major = rx_majors[rx_group_firsts[row_comm_rank * col_comm_size + local_partition_idx] +
                           offset_in_local_edge_partition];
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{0};
    edge_t local_degree{0};
    if (multi_gpu && (edge_partition.major_hypersparse_first() &&
                      (major >= *(edge_partition.major_hypersparse_first())))) {
      auto major_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
      if (major_hypersparse_idx) {
        thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(
          (*(edge_partition.major_hypersparse_first()) - edge_partition.major_range_first()) +
          *major_hypersparse_idx);
      }
    } else {
      thrust::tie(indices, edge_offset, local_degree) =
        edge_partition.local_edges(edge_partition.major_offset_from_major_nocheck(major));
    }
    // FIXME: this can lead to thread-divergence with a mix of high-degree and low-degree
    // vertices in a single warp (better optimize if this becomes a performance
    // bottleneck)
    thrust::copy(thrust::seq,
                 indices,
                 indices + local_degree,
                 local_nbrs_for_rx_majors +
                   local_nbr_offsets_for_rx_majors[rx_group_firsts[row_comm_rank * col_comm_size +
                                                                   local_partition_idx] +
                                                   offset_in_local_edge_partition]);
  }
};

struct compute_local_nbr_count_per_rank_t {
  size_t const* rx_offsets{nullptr};
  size_t const* local_nbr_offsets_for_rx_majors{nullptr};

  __device__ size_t operator()(size_t i) const
  {
    return *(local_nbr_offsets_for_rx_majors + *(rx_offsets + (i + 1))) -
           *(local_nbr_offsets_for_rx_majors + *(rx_offsets + i));
  }
};

template <typename FirstElementToIdxMap,
          typename SecondElementToIdxMap,
          typename vertex_t,
          typename edge_t,
          bool multi_gpu>
struct pick_min_degree_t {
  FirstElementToIdxMap first_element_to_idx_map{};
  size_t const* first_element_offsets{nullptr};

  SecondElementToIdxMap second_element_to_idx_map{};
  size_t const* second_element_offsets{nullptr};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};

  __device__ edge_t operator()(thrust::tuple<vertex_t, vertex_t> pair) const
  {
    edge_t local_degree0{0};
    vertex_t major0 = thrust::get<0>(pair);
    if constexpr (std::is_same_v<FirstElementToIdxMap, void*>) {
      if constexpr (multi_gpu) {
        if (edge_partition.major_hypersparse_first() &&
            (major0 >= *(edge_partition.major_hypersparse_first()))) {
          auto major_hypersparse_idx =
            edge_partition.major_hypersparse_idx_from_major_nocheck(major0);
          local_degree0 =
            major_hypersparse_idx
              ? edge_partition.local_degree((*(edge_partition.major_hypersparse_first()) -
                                             edge_partition.major_range_first()) +
                                            *major_hypersparse_idx)
              : edge_t{0};
        } else {
          local_degree0 =
            edge_partition.local_degree(edge_partition.major_offset_from_major_nocheck(major0));
        }
      } else {
        local_degree0 =
          edge_partition.local_degree(edge_partition.major_offset_from_major_nocheck(major0));
      }
    } else {
      auto idx = first_element_to_idx_map.find(major0);
      local_degree0 =
        static_cast<edge_t>(first_element_offsets[idx + 1] - first_element_offsets[idx]);
    }

    edge_t local_degree1{0};
    vertex_t major1 = thrust::get<1>(pair);
    if constexpr (std::is_same_v<SecondElementToIdxMap, void*>) {
      if constexpr (multi_gpu) {
        if (edge_partition.major_hypersparse_first() &&
            (major1 >= *(edge_partition.major_hypersparse_first()))) {
          auto major_hypersparse_idx =
            edge_partition.major_hypersparse_idx_from_major_nocheck(major1);
          local_degree1 =
            major_hypersparse_idx
              ? edge_partition.local_degree((*(edge_partition.major_hypersparse_first()) -
                                             edge_partition.major_range_first()) +
                                            *major_hypersparse_idx)
              : edge_t{0};
        } else {
          local_degree1 =
            edge_partition.local_degree(edge_partition.major_offset_from_major_nocheck(major1));
        }
      } else {
        local_degree1 =
          edge_partition.local_degree(edge_partition.major_offset_from_major_nocheck(major1));
      }
    } else {
      auto idx = second_element_to_idx_map.find(major1);
      local_degree1 =
        static_cast<edge_t>(second_element_offsets[idx + 1] - second_element_offsets[idx]);
    }

    return thrust::minimum<edge_t>{}(local_degree0, local_degree1);
  }
};

template <typename FirstElementToIdxMap,
          typename SecondElementToIdxMap,
          typename VertexPairIterator,
          typename vertex_t,
          typename edge_t,
          bool multi_gpu>
struct copy_intersecting_nbrs_and_update_intersection_size_t {
  FirstElementToIdxMap first_element_to_idx_map{};
  size_t const* first_element_offsets{nullptr};
  vertex_t const* first_element_indices{nullptr};

  SecondElementToIdxMap second_element_to_idx_map{};
  size_t const* second_element_offsets{nullptr};
  vertex_t const* second_element_indices{nullptr};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};

  VertexPairIterator vertex_pair_first;
  size_t const* nbr_intersection_offsets{nullptr};
  vertex_t* nbr_intersection_indices{nullptr};

  vertex_t invalid_id{};

  __device__ edge_t operator()(size_t i) const
  {
    auto pair = *(vertex_pair_first + i);

    vertex_t const* indices0{nullptr};
    [[maybe_unused]] edge_t local_edge_offset0{0};
    edge_t local_degree0{0};
    if constexpr (std::is_same_v<FirstElementToIdxMap, void*>) {
      vertex_t major = thrust::get<0>(pair);
      if constexpr (multi_gpu) {
        if (edge_partition.major_hypersparse_first() &&
            (major >= *(edge_partition.major_hypersparse_first()))) {
          auto major_hypersparse_idx =
            edge_partition.major_hypersparse_idx_from_major_nocheck(major);
          if (major_hypersparse_idx) {
            thrust::tie(indices0, local_edge_offset0, local_degree0) = edge_partition.local_edges(
              (*(edge_partition.major_hypersparse_first()) - edge_partition.major_range_first()) +
              *major_hypersparse_idx);
          }
        } else {
          thrust::tie(indices0, local_edge_offset0, local_degree0) =
            edge_partition.local_edges(edge_partition.major_offset_from_major_nocheck(major));
        }
      } else {
        thrust::tie(indices0, local_edge_offset0, local_degree0) =
          edge_partition.local_edges(edge_partition.major_offset_from_major_nocheck(major));
      }
    } else {
      auto idx = first_element_to_idx_map.find(thrust::get<0>(pair));
      local_degree0 =
        static_cast<edge_t>(first_element_offsets[idx + 1] - first_element_offsets[idx]);
      indices0 = first_element_indices + first_element_offsets[idx];
    }

    vertex_t const* indices1{nullptr};
    [[maybe_unused]] edge_t local_edge_offset1{0};
    edge_t local_degree1{0};
    if constexpr (std::is_same_v<SecondElementToIdxMap, void*>) {
      vertex_t major = thrust::get<1>(pair);
      if constexpr (multi_gpu) {
        if (edge_partition.major_hypersparse_first() &&
            (major >= *(edge_partition.major_hypersparse_first()))) {
          auto major_hypersparse_idx =
            edge_partition.major_hypersparse_idx_from_major_nocheck(major);
          if (major_hypersparse_idx) {
            thrust::tie(indices1, local_edge_offset1, local_degree1) = edge_partition.local_edges(
              (*(edge_partition.major_hypersparse_first()) - edge_partition.major_range_first()) +
              *major_hypersparse_idx);
          }
        } else {
          thrust::tie(indices1, local_edge_offset1, local_degree1) =
            edge_partition.local_edges(edge_partition.major_offset_from_major_nocheck(major));
        }
      } else {
        thrust::tie(indices1, local_edge_offset1, local_degree1) =
          edge_partition.local_edges(edge_partition.major_offset_from_major_nocheck(major));
      }
    } else {
      auto idx = second_element_to_idx_map.find(thrust::get<1>(pair));
      local_degree1 =
        static_cast<edge_t>(second_element_offsets[idx + 1] - second_element_offsets[idx]);
      indices1 = second_element_indices + second_element_offsets[idx];
    }

    // FIXME: this can lead to thread-divergence with a mix of high-degree and low-degree
    // vertices in a single warp (better optimize if this becomes a performance
    // bottleneck)

    auto it = thrust::set_intersection(thrust::seq,
                                       indices0,
                                       indices0 + local_degree0,
                                       indices1,
                                       indices1 + local_degree1,
                                       nbr_intersection_indices + nbr_intersection_offsets[i]);
    thrust::fill(
      thrust::seq, it, nbr_intersection_indices + nbr_intersection_offsets[i + 1], invalid_id);

    return static_cast<size_t>(
      thrust::distance(nbr_intersection_indices + nbr_intersection_offsets[i], it));
  }
};

template <typename edge_t>
struct strided_accumulate_t {
  edge_t const* rx_nbr_intersection_sizes{nullptr};
  size_t edge_partition_input_size{};
  int col_comm_size{};

  __device__ edge_t operator()(size_t i) const
  {
    edge_t accumulated_size{0};
    for (int j = 0; j < col_comm_size; ++j) {
      accumulated_size += *(rx_nbr_intersection_sizes + edge_partition_input_size * j + i);
    }
    return accumulated_size;
  }
};

template <typename vertex_t>
struct gatherv_indices_t {
  size_t output_size{};
  int col_comm_size{};

  size_t const* gathered_intersection_offsets{nullptr};
  vertex_t const* gathered_intersection_indices{nullptr};
  size_t const* combined_nbr_intersection_offsets{nullptr};

  vertex_t* combined_nbr_intersection_indices{nullptr};

  __device__ void operator()(size_t i) const
  {
    auto output_offset = *(combined_nbr_intersection_offsets + i);

    // FIXME: this can lead to thread-divergence with a mix of high-degree and low-degree vertices
    // in a single warp (better optimize if this becomes a performance bottleneck)

    for (int j = 0; j < col_comm_size; ++j) {
      thrust::copy(
        thrust::seq,
        gathered_intersection_indices + gathered_intersection_offsets[output_size * j + i],
        gathered_intersection_indices + gathered_intersection_offsets[output_size * j + i + 1],
        combined_nbr_intersection_indices + output_offset);
      output_offset += gathered_intersection_offsets[output_size * j + i + 1] -
                       gathered_intersection_offsets[output_size * j + i];
    }
  }
};

template <typename GraphViewType, typename VertexPairIterator>
size_t count_invalid_vertex_pairs(raft::handle_t const& handle,
                                  GraphViewType const& graph_view,
                                  VertexPairIterator vertex_pair_first,
                                  VertexPairIterator vertex_pair_last)
{
  using vertex_t = typename GraphViewType::vertex_type;

  std::vector<vertex_t> h_edge_partition_major_range_firsts(
    graph_view.number_of_local_edge_partitions());
  std::vector<vertex_t> h_edge_partition_major_range_lasts(
    h_edge_partition_major_range_firsts.size());
  vertex_t edge_partition_minor_range_first{};
  vertex_t edge_partition_minor_range_last{};
  if constexpr (GraphViewType::is_multi_gpu) {
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); i++) {
      if constexpr (GraphViewType::is_storage_transposed) {
        h_edge_partition_major_range_firsts[i] = graph_view.local_edge_partition_dst_range_first(i);
        h_edge_partition_major_range_lasts[i]  = graph_view.local_edge_partition_dst_range_last(i);
      } else {
        h_edge_partition_major_range_firsts[i] = graph_view.local_edge_partition_src_range_first(i);
        h_edge_partition_major_range_lasts[i]  = graph_view.local_edge_partition_src_range_last(i);
      }
    }
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_minor_range_first = graph_view.local_edge_partition_src_range_first();
      edge_partition_minor_range_last  = graph_view.local_edge_partition_src_range_last();
    } else {
      edge_partition_minor_range_first = graph_view.local_edge_partition_dst_range_first();
      edge_partition_minor_range_last  = graph_view.local_edge_partition_dst_range_last();
    }
  } else {
    h_edge_partition_major_range_firsts[0] = vertex_t{0};
    h_edge_partition_major_range_lasts[0]  = graph_view.number_of_vertices();
    edge_partition_minor_range_first       = vertex_t{0};
    edge_partition_minor_range_last        = graph_view.number_of_vertices();
  }
  rmm::device_uvector<vertex_t> d_edge_partition_major_range_firsts(
    h_edge_partition_major_range_firsts.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_edge_partition_major_range_lasts(
    h_edge_partition_major_range_lasts.size(), handle.get_stream());
  raft::update_device(d_edge_partition_major_range_firsts.data(),
                      h_edge_partition_major_range_firsts.data(),
                      h_edge_partition_major_range_firsts.size(),
                      handle.get_stream());
  raft::update_device(d_edge_partition_major_range_lasts.data(),
                      h_edge_partition_major_range_lasts.data(),
                      h_edge_partition_major_range_lasts.size(),
                      handle.get_stream());

  auto num_invalid_pairs = thrust::count_if(
    handle.get_thrust_policy(),
    vertex_pair_first,
    vertex_pair_last,
    is_invalid_input_vertex_pair_t<vertex_t>{
      graph_view.number_of_vertices(),
      raft::device_span<vertex_t const>(d_edge_partition_major_range_firsts.begin(),
                                        d_edge_partition_major_range_firsts.end()),
      raft::device_span<vertex_t const>(d_edge_partition_major_range_lasts.begin(),
                                        d_edge_partition_major_range_lasts.end()),
      edge_partition_minor_range_first,
      edge_partition_minor_range_last});
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm = handle.get_comms();
    num_invalid_pairs =
      host_scalar_allreduce(comm, num_invalid_pairs, raft::comms::op_t::SUM, handle.get_stream());
  }

  return num_invalid_pairs;
}

// In multi-GPU, the first element of every vertex pair in [vertex_pair_first, vertex_pair) should
// be within the valid edge partition major range assigned to this process and the second element
// should be within the valid edge partition minor range assigned to this process.
// [vertex_pair_first, vertex_pair_last) should be sorted using the first element of each pair as
// the primary key and the second element of each pair as the secondary key.
// Calling this function in multiple groups can reduce the peak memory usage when the caller wants
// to compute neighbor intersections for a large number of vertex pairs. This is especially true if
// one can limit the number of unique vertices (aggregated over column communicator in multi-GPU) to
// build neighbor list; we need to bulid neighbor lists for the first element of every input vertex
// pair if intersect_dst_nbr[0] == GraphViewType::is_storage_transposed and build neighbor lists for
// the second element of every input vertex pair if single-GPU and intersect_dst_nbr[1] ==
// GraphViewType::is_storage_transposed or multi-GPU. For load balancing,
// thrust::distance(vertex_pair_first, vertex_pair_last) should be comparable across the global
// communicator. If we need to build the neighbor lists, grouping based on applying "vertex ID %
// number of groups"  is recommended for load-balancing.
template <typename GraphViewType, typename VertexPairIterator>
std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<typename GraphViewType::vertex_type>>
nbr_intersection(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 VertexPairIterator vertex_pair_first,
                 VertexPairIterator vertex_pair_last,
                 std::array<bool, 2> intersect_dst_nbr,
                 bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(std::is_same_v<typename thrust::iterator_traits<VertexPairIterator>::value_type,
                               thrust::tuple<vertex_t, vertex_t>>);

  size_t input_size = static_cast<size_t>(thrust::distance(vertex_pair_first, vertex_pair_last));

  std::array<bool, 2> intersect_minor_nbr = {
    intersect_dst_nbr[0] != GraphViewType::is_storage_transposed,
    intersect_dst_nbr[1] != GraphViewType::is_storage_transposed};

  // 1. Check input arguments

  if (do_expensive_check) {
    auto is_sorted =
      thrust::is_sorted(handle.get_thrust_policy(), vertex_pair_first, vertex_pair_last);
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& comm = handle.get_comms();
      is_sorted  = static_cast<bool>(host_scalar_allreduce(
        comm, static_cast<int>(is_sorted), raft::comms::op_t::MIN, handle.get_stream()));
    }
    CUGRAPH_EXPECTS(is_sorted, "Invalid input arguments: input vertex pairs should be sorted.");

    auto num_invalid_pairs =
      count_invalid_vertex_pairs(handle, graph_view, vertex_pair_first, vertex_pair_last);
    CUGRAPH_EXPECTS(num_invalid_pairs == 0,
                    "Invalid input arguments: there are invalid input vertex pairs.");
  }

  // 2. Collect neighbor lists for unique second pair elements (for the neighbors within the minor
  // range for this GPU); Note that no need to collect for first pair elements as they already
  // locally reside.

  std::optional<std::unique_ptr<kv_store_t<vertex_t, vertex_t, false>>> major_to_idx_map_ptr{
    std::nullopt};
  std::optional<rmm::device_uvector<size_t>> major_nbr_offsets{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> major_nbr_indices{std::nullopt};

  if constexpr (GraphViewType::is_multi_gpu) {
    if (intersect_minor_nbr[1]) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_size = row_comm.get_size();
      auto const row_comm_rank = row_comm.get_rank();

      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_size = col_comm.get_size();

      // 2.1 Find unique second pair element majors

      rmm::device_uvector<vertex_t> unique_majors(input_size, handle.get_stream());
      {
        auto second_element_first = thrust::make_transform_iterator(
          vertex_pair_first, thrust_tuple_get<thrust::tuple<vertex_t, vertex_t>, size_t{1}>{});
        thrust::copy(handle.get_thrust_policy(),
                     second_element_first,
                     second_element_first + input_size,
                     unique_majors.begin());

        thrust::sort(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end());
        unique_majors.resize(
          thrust::distance(
            unique_majors.begin(),
            thrust::unique(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end())),
          handle.get_stream());

        unique_majors.shrink_to_fit(handle.get_stream());

        if (col_comm_size > 1) {
          // FIXME: We may refactor this code to improve scalability. We may call multiple gatherv
          // calls, perform local sort and unique, and call multiple broadcasts rather than
          // performing sort and unique for the entire range in every GPU in col_comm.
          auto rx_counts =
            host_scalar_allgather(col_comm, unique_majors.size(), handle.get_stream());
          std::vector<size_t> rx_displacements(rx_counts.size());
          std::exclusive_scan(
            rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
          rmm::device_uvector<vertex_t> rx_unique_majors(rx_displacements.back() + rx_counts.back(),
                                                         handle.get_stream());
          device_allgatherv(col_comm,
                            unique_majors.begin(),
                            rx_unique_majors.begin(),
                            rx_counts,
                            rx_displacements,
                            handle.get_stream());
          unique_majors = std::move(rx_unique_majors);

          thrust::sort(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end());
          unique_majors.resize(thrust::distance(unique_majors.begin(),
                                                thrust::unique(handle.get_thrust_policy(),
                                                               unique_majors.begin(),
                                                               unique_majors.end())),
                               handle.get_stream());

          unique_majors.shrink_to_fit(handle.get_stream());
        }
      }

      // 2.2 Send majors and group (row_comm_rank, edge_partition_idx) counts

      rmm::device_uvector<vertex_t> rx_majors(0, handle.get_stream());
      std::vector<size_t> rx_major_counts{};
      rmm::device_uvector<size_t> rx_group_counts(size_t{0}, handle.get_stream());
      {
        auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
        rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
          h_vertex_partition_range_lasts.size(), handle.get_stream());
        raft::update_device(d_vertex_partition_range_lasts.data(),
                            h_vertex_partition_range_lasts.data(),
                            h_vertex_partition_range_lasts.size(),
                            handle.get_stream());

        auto d_tx_group_counts = groupby_and_count(
          unique_majors.begin(),
          unique_majors.end(),
          major_to_group_idx_t<vertex_t>{
            d_vertex_partition_range_lasts.data(), comm_size, row_comm_size, col_comm_size},
          comm_size,
          std::numeric_limits<size_t>::max(),
          handle.get_stream());
        std::vector<size_t> h_tx_group_counts(d_tx_group_counts.size());
        raft::update_host(h_tx_group_counts.data(),
                          d_tx_group_counts.data(),
                          d_tx_group_counts.size(),
                          handle.get_stream());
        handle.sync_stream();

        std::vector<size_t> tx_counts(row_comm_size, size_t{0});
        for (size_t i = 0; i < tx_counts.size(); ++i) {
          tx_counts[i] = std::reduce(h_tx_group_counts.begin() + col_comm_size * i,
                                     h_tx_group_counts.begin() + col_comm_size * (i + 1),
                                     size_t{0});
        }

        std::tie(rx_majors, rx_major_counts) =
          shuffle_values(row_comm, unique_majors.begin(), tx_counts, handle.get_stream());

        std::tie(rx_group_counts, std::ignore) =
          shuffle_values(row_comm,
                         d_tx_group_counts.begin(),
                         std::vector<size_t>(row_comm_size, col_comm_size),
                         handle.get_stream());
      }

      // 2.3. Enumerate degrees and neighbors for the received majors

      rmm::device_uvector<edge_t> local_degrees_for_rx_majors(size_t{0}, handle.get_stream());
      rmm::device_uvector<vertex_t> local_nbrs_for_rx_majors(size_t{0}, handle.get_stream());
      std::vector<size_t> local_nbr_counts{};
      {
        rmm::device_uvector<size_t> rx_reordered_group_counts(
          rx_group_counts.size(),
          handle.get_stream());  // reorder using local edge partition index as the primary key and
                                 // row_comm_rank as the secondary key
        thrust::tabulate(
          handle.get_thrust_policy(),
          rx_reordered_group_counts.begin(),
          rx_reordered_group_counts.end(),
          reorder_group_count_t{row_comm_size, col_comm_size, rx_group_counts.data()});

        rmm::device_uvector<size_t> d_rx_reordered_group_lasts(rx_reordered_group_counts.size(),
                                                               handle.get_stream());
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               rx_reordered_group_counts.begin(),
                               rx_reordered_group_counts.end(),
                               d_rx_reordered_group_lasts.begin());
        std::vector<size_t> h_rx_reordered_group_lasts(d_rx_reordered_group_lasts.size());
        raft::update_host(h_rx_reordered_group_lasts.data(),
                          d_rx_reordered_group_lasts.data(),
                          d_rx_reordered_group_lasts.size(),
                          handle.get_stream());
        handle.sync_stream();

        rmm::device_uvector<size_t> rx_group_firsts(rx_group_counts.size(), handle.get_stream());
        thrust::exclusive_scan(handle.get_thrust_policy(),
                               rx_group_counts.begin(),
                               rx_group_counts.end(),
                               rx_group_firsts.begin());

        local_degrees_for_rx_majors.resize(rx_majors.size(), handle.get_stream());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          auto edge_partition =
            edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
              graph_view.local_edge_partition_view(i));
          auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
          auto reordered_idx_first =
            (i == size_t{0}) ? size_t{0} : h_rx_reordered_group_lasts[i * row_comm_size - 1];
          auto reordered_idx_last = h_rx_reordered_group_lasts[(i + 1) * row_comm_size - 1];
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(reordered_idx_first),
            thrust::make_counting_iterator(reordered_idx_last),
            update_rx_major_local_degree_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>{
              row_comm_size,
              col_comm_size,
              edge_partition,
              reordered_idx_first,
              i,
              d_rx_reordered_group_lasts.data() + i * row_comm_size,
              rx_group_firsts.data(),
              rx_majors.data(),
              local_degrees_for_rx_majors.data()});
        }

        rmm::device_uvector<size_t> local_nbr_offsets_for_rx_majors(
          local_degrees_for_rx_majors.size() + 1, handle.get_stream());
        local_nbr_offsets_for_rx_majors.set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto degree_first = thrust::make_transform_iterator(local_degrees_for_rx_majors.begin(),
                                                            detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               degree_first,
                               degree_first + local_degrees_for_rx_majors.size(),
                               local_nbr_offsets_for_rx_majors.begin() + 1);

        local_nbrs_for_rx_majors.resize(
          local_nbr_offsets_for_rx_majors.back_element(handle.get_stream()), handle.get_stream());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          auto edge_partition =
            edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
              graph_view.local_edge_partition_view(i));
          auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
          auto reordered_idx_first =
            (i == size_t{0}) ? size_t{0} : h_rx_reordered_group_lasts[i * row_comm_size - 1];
          auto reordered_idx_last = h_rx_reordered_group_lasts[(i + 1) * row_comm_size - 1];

          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(reordered_idx_first),
            thrust::make_counting_iterator(reordered_idx_last),
            update_rx_major_local_nbrs_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>{
              row_comm_size,
              col_comm_size,
              edge_partition,
              reordered_idx_first,
              i,
              d_rx_reordered_group_lasts.data() + i * row_comm_size,
              rx_group_firsts.data(),
              rx_majors.data(),
              local_nbr_offsets_for_rx_majors.data(),
              local_nbrs_for_rx_majors.data()});
        }

        std::vector<size_t> h_rx_offsets(rx_major_counts.size() + size_t{1}, size_t{0});
        std::inclusive_scan(
          rx_major_counts.begin(), rx_major_counts.end(), h_rx_offsets.begin() + 1);
        rmm::device_uvector<size_t> d_rx_offsets(h_rx_offsets.size(), handle.get_stream());
        raft::update_device(
          d_rx_offsets.data(), h_rx_offsets.data(), h_rx_offsets.size(), handle.get_stream());
        rmm::device_uvector<size_t> d_local_nbr_counts(rx_major_counts.size(), handle.get_stream());
        thrust::tabulate(handle.get_thrust_policy(),
                         d_local_nbr_counts.begin(),
                         d_local_nbr_counts.end(),
                         compute_local_nbr_count_per_rank_t{
                           d_rx_offsets.data(), local_nbr_offsets_for_rx_majors.data()});
        local_nbr_counts.resize(d_local_nbr_counts.size());
        raft::update_host(local_nbr_counts.data(),
                          d_local_nbr_counts.data(),
                          d_local_nbr_counts.size(),
                          handle.get_stream());
        handle.sync_stream();
      }

      // 2.4 Send the degrees and neighbors back

      {
        rmm::device_uvector<edge_t> local_degrees_for_unique_majors(size_t{0}, handle.get_stream());
        std::tie(local_degrees_for_unique_majors, std::ignore) = shuffle_values(
          row_comm, local_degrees_for_rx_majors.begin(), rx_major_counts, handle.get_stream());
        major_nbr_offsets = rmm::device_uvector<size_t>(local_degrees_for_unique_majors.size() + 1,
                                                        handle.get_stream());
        (*major_nbr_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto degree_first = thrust::make_transform_iterator(local_degrees_for_unique_majors.begin(),
                                                            detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               degree_first,
                               degree_first + local_degrees_for_unique_majors.size(),
                               (*major_nbr_offsets).begin() + 1);
      }

      std::tie(*major_nbr_indices, std::ignore) = shuffle_values(
        row_comm, local_nbrs_for_rx_majors.begin(), local_nbr_counts, handle.get_stream());

      major_to_idx_map_ptr = std::make_unique<kv_store_t<vertex_t, vertex_t, false>>(
        unique_majors.begin(),
        unique_majors.end(),
        thrust::make_counting_iterator(vertex_t{0}),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value,
        handle.get_stream());
    }
  }

  // 3. Collect neighbor list for minors (for the neighbors within the minor range for this GPU)

  std::optional<std::unique_ptr<kv_store_t<vertex_t, vertex_t, false>>> minor_to_idx_map_ptr{
    std::nullopt};
  std::optional<rmm::device_uvector<size_t>> minor_nbr_offsets{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> minor_nbr_indices{std::nullopt};

  if (!intersect_minor_nbr[0] || !intersect_minor_nbr[1]) {
    // FIXME: currently no use case, but this can be necessary to supporting triangle counting for
    // directed graphs
    CUGRAPH_FAIL("unimplemented.");
  }

  // 4. Intersect

  rmm::device_uvector<size_t> nbr_intersection_offsets(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> nbr_intersection_indices(size_t{0}, handle.get_stream());
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    std::vector<size_t> input_counts(col_comm_size);
    std::vector<size_t> input_lasts(input_counts.size());
    {
      std::vector<vertex_t> h_edge_partition_major_range_lasts(
        graph_view.number_of_local_edge_partitions());
      for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); i++) {
        if constexpr (GraphViewType::is_storage_transposed) {
          h_edge_partition_major_range_lasts[i] = graph_view.local_edge_partition_dst_range_last(i);
        } else {
          h_edge_partition_major_range_lasts[i] = graph_view.local_edge_partition_src_range_last(i);
        }
      }
      rmm::device_uvector<vertex_t> d_edge_partition_major_range_lasts(
        h_edge_partition_major_range_lasts.size(), handle.get_stream());
      raft::update_device(d_edge_partition_major_range_lasts.data(),
                          h_edge_partition_major_range_lasts.data(),
                          h_edge_partition_major_range_lasts.size(),
                          handle.get_stream());

      rmm::device_uvector<size_t> d_lasts(col_comm_size, handle.get_stream());
      auto first_element_first = thrust::make_transform_iterator(
        vertex_pair_first, thrust_tuple_get<thrust::tuple<vertex_t, vertex_t>, size_t{0}>{});
      thrust::lower_bound(handle.get_thrust_policy(),
                          first_element_first,
                          first_element_first + input_size,
                          d_edge_partition_major_range_lasts.begin(),
                          d_edge_partition_major_range_lasts.end(),
                          d_lasts.begin());
      raft::update_host(input_lasts.data(), d_lasts.data(), d_lasts.size(), handle.get_stream());
      handle.sync_stream();
      std::adjacent_difference(input_lasts.begin(), input_lasts.end(), input_counts.begin());
    }

    std::vector<rmm::device_uvector<edge_t>> edge_partition_nbr_intersection_sizes{};
    std::vector<rmm::device_uvector<vertex_t>> edge_partition_nbr_intersection_indices{};
    edge_partition_nbr_intersection_sizes.reserve(graph_view.number_of_local_edge_partitions());
    edge_partition_nbr_intersection_indices.reserve(graph_view.number_of_local_edge_partitions());
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      auto rx_v_pair_counts = host_scalar_allgather(col_comm, input_counts[i], handle.get_stream());
      std::vector<size_t> rx_v_pair_displacements(rx_v_pair_counts.size());
      std::exclusive_scan(rx_v_pair_counts.begin(),
                          rx_v_pair_counts.end(),
                          rx_v_pair_displacements.begin(),
                          size_t{0});
      auto aggregate_rx_v_pair_size = rx_v_pair_displacements.back() + rx_v_pair_counts.back();

      // 4.1. All-gather vertex pairs & locally intersect

      rmm::device_uvector<edge_t> rx_v_pair_nbr_intersection_sizes(size_t{0}, handle.get_stream());
      rmm::device_uvector<vertex_t> rx_v_pair_nbr_intersection_indices(size_t{0},
                                                                       handle.get_stream());
      std::vector<size_t> rx_v_pair_nbr_intersection_index_tx_counts(size_t{0});
      {
        auto vertex_pair_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, vertex_t>>(
          aggregate_rx_v_pair_size, handle.get_stream());

        thrust::copy(
          handle.get_thrust_policy(),
          vertex_pair_first + (i == size_t{0} ? size_t{0} : input_lasts[i - 1]),
          vertex_pair_first + input_lasts[i],
          get_dataframe_buffer_begin(vertex_pair_buffer) + rx_v_pair_displacements[col_comm_rank]);

        device_allgatherv(
          col_comm,
          get_dataframe_buffer_begin(vertex_pair_buffer) + rx_v_pair_displacements[col_comm_rank],
          get_dataframe_buffer_begin(vertex_pair_buffer),
          rx_v_pair_counts,
          rx_v_pair_displacements,
          handle.get_stream());

        auto edge_partition =
          edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
            graph_view.local_edge_partition_view(i));
        auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);

        rx_v_pair_nbr_intersection_sizes.resize(
          aggregate_rx_v_pair_size,
          handle
            .get_stream());  // initially store minimum degrees (upper bound for intersection sizes)
        if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
          auto second_element_to_idx_map =
            detail::kv_cuco_store_device_view_t((*major_to_idx_map_ptr)->view());
          thrust::transform(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer),
            get_dataframe_buffer_end(vertex_pair_buffer),
            rx_v_pair_nbr_intersection_sizes.begin(),
            pick_min_degree_t<void*, decltype(second_element_to_idx_map), vertex_t, edge_t, true>{
              nullptr,
              nullptr,
              second_element_to_idx_map,
              (*major_nbr_offsets).data(),
              edge_partition});
        } else {
          CUGRAPH_FAIL("unimplemented.");
        }

        rmm::device_uvector<size_t> rx_v_pair_nbr_intersection_offsets(
          rx_v_pair_nbr_intersection_sizes.size() + 1, handle.get_stream());
        rx_v_pair_nbr_intersection_offsets.set_element_to_zero_async(size_t{0},
                                                                     handle.get_stream());
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               rx_v_pair_nbr_intersection_sizes.begin(),
                               rx_v_pair_nbr_intersection_sizes.end(),
                               rx_v_pair_nbr_intersection_offsets.begin() + 1);

        rx_v_pair_nbr_intersection_indices.resize(
          rx_v_pair_nbr_intersection_offsets.back_element(handle.get_stream()),
          handle.get_stream());
        if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
          auto second_element_to_idx_map =
            detail::kv_cuco_store_device_view_t((*major_to_idx_map_ptr)->view());
          thrust::tabulate(handle.get_thrust_policy(),
                           rx_v_pair_nbr_intersection_sizes.begin(),
                           rx_v_pair_nbr_intersection_sizes.end(),
                           copy_intersecting_nbrs_and_update_intersection_size_t<
                             void*,
                             decltype(second_element_to_idx_map),
                             decltype(get_dataframe_buffer_begin(vertex_pair_buffer)),
                             vertex_t,
                             edge_t,
                             true>{nullptr,
                                   nullptr,
                                   nullptr,
                                   second_element_to_idx_map,
                                   (*major_nbr_offsets).data(),
                                   (*major_nbr_indices).data(),
                                   edge_partition,
                                   get_dataframe_buffer_begin(vertex_pair_buffer),
                                   rx_v_pair_nbr_intersection_offsets.data(),
                                   rx_v_pair_nbr_intersection_indices.data(),
                                   invalid_vertex_id<vertex_t>::value});
        } else {
          CUGRAPH_FAIL("unimplemented.");
        }

        rx_v_pair_nbr_intersection_indices.resize(
          thrust::distance(rx_v_pair_nbr_intersection_indices.begin(),
                           thrust::remove(handle.get_thrust_policy(),
                                          rx_v_pair_nbr_intersection_indices.begin(),
                                          rx_v_pair_nbr_intersection_indices.end(),
                                          invalid_vertex_id<vertex_t>::value)),
          handle.get_stream());
        rx_v_pair_nbr_intersection_indices.shrink_to_fit(handle.get_stream());

        thrust::inclusive_scan(handle.get_thrust_policy(),
                               rx_v_pair_nbr_intersection_sizes.begin(),
                               rx_v_pair_nbr_intersection_sizes.end(),
                               rx_v_pair_nbr_intersection_offsets.begin() + 1);

        std::vector<size_t> h_rx_v_pair_lasts(rx_v_pair_counts.size());
        std::inclusive_scan(
          rx_v_pair_counts.begin(), rx_v_pair_counts.end(), h_rx_v_pair_lasts.begin());
        rmm::device_uvector<size_t> d_rx_v_pair_lasts(h_rx_v_pair_lasts.size(),
                                                      handle.get_stream());
        raft::update_device(d_rx_v_pair_lasts.data(),
                            h_rx_v_pair_lasts.data(),
                            h_rx_v_pair_lasts.size(),
                            handle.get_stream());
        rmm::device_uvector<size_t> d_rx_v_pair_nbr_intersection_index_tx_lasts(
          d_rx_v_pair_lasts.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       d_rx_v_pair_lasts.begin(),
                       d_rx_v_pair_lasts.end(),
                       rx_v_pair_nbr_intersection_offsets.begin(),
                       d_rx_v_pair_nbr_intersection_index_tx_lasts.begin());
        std::vector<size_t> h_rx_v_pair_nbr_intersection_index_tx_lasts(
          d_rx_v_pair_nbr_intersection_index_tx_lasts.size());
        raft::update_host(h_rx_v_pair_nbr_intersection_index_tx_lasts.data(),
                          d_rx_v_pair_nbr_intersection_index_tx_lasts.data(),
                          d_rx_v_pair_nbr_intersection_index_tx_lasts.size(),
                          handle.get_stream());
        handle.sync_stream();
        rx_v_pair_nbr_intersection_index_tx_counts.resize(
          h_rx_v_pair_nbr_intersection_index_tx_lasts.size());
        std::adjacent_difference(h_rx_v_pair_nbr_intersection_index_tx_lasts.begin(),
                                 h_rx_v_pair_nbr_intersection_index_tx_lasts.end(),
                                 rx_v_pair_nbr_intersection_index_tx_counts.begin());
      }

      // 4.2. All-to-all intersection outputs

      rmm::device_uvector<edge_t> combined_nbr_intersection_sizes(size_t{0}, handle.get_stream());
      rmm::device_uvector<size_t> combined_nbr_intersection_offsets(size_t{0}, handle.get_stream());
      rmm::device_uvector<size_t> gathered_nbr_intersection_offsets(size_t{0}, handle.get_stream());
      std::vector<size_t> gathered_nbr_intersection_index_rx_counts(size_t{0});
      {
        std::vector<int> ranks(col_comm_size);
        std::iota(ranks.begin(), ranks.end(), int{0});
        std::vector<size_t> displacements(col_comm_size);
        for (int i = 0; i < col_comm_size; ++i) {
          displacements[i] = rx_v_pair_counts[col_comm_rank] * i;
        }
        rmm::device_uvector<edge_t> gathered_nbr_intersection_sizes(
          rx_v_pair_counts[col_comm_rank] * col_comm_size, handle.get_stream());
        device_multicast_sendrecv(
          col_comm,
          rx_v_pair_nbr_intersection_sizes.begin(),
          rx_v_pair_counts,
          rx_v_pair_displacements,
          ranks,
          gathered_nbr_intersection_sizes.begin(),
          std::vector<size_t>(col_comm_size, rx_v_pair_counts[col_comm_rank]),
          displacements,
          ranks,
          handle.get_stream());
        rx_v_pair_nbr_intersection_sizes.resize(size_t{0}, handle.get_stream());
        rx_v_pair_nbr_intersection_sizes.shrink_to_fit(handle.get_stream());

        combined_nbr_intersection_sizes.resize(rx_v_pair_counts[col_comm_rank],
                                               handle.get_stream());

        thrust::tabulate(handle.get_thrust_policy(),
                         combined_nbr_intersection_sizes.begin(),
                         combined_nbr_intersection_sizes.end(),
                         strided_accumulate_t<edge_t>{gathered_nbr_intersection_sizes.data(),
                                                      rx_v_pair_counts[col_comm_rank],
                                                      col_comm_size});

        combined_nbr_intersection_offsets.resize(rx_v_pair_counts[col_comm_rank] + 1,
                                                 handle.get_stream());
        combined_nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto combined_size_first = thrust::make_transform_iterator(
          combined_nbr_intersection_sizes.begin(), detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               combined_size_first,
                               combined_size_first + combined_nbr_intersection_sizes.size(),
                               combined_nbr_intersection_offsets.begin() + 1);

        gathered_nbr_intersection_offsets.resize(gathered_nbr_intersection_sizes.size() + 1,
                                                 handle.get_stream());
        gathered_nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto gathered_size_first = thrust::make_transform_iterator(
          gathered_nbr_intersection_sizes.begin(), detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               gathered_size_first,
                               gathered_size_first + gathered_nbr_intersection_sizes.size(),
                               gathered_nbr_intersection_offsets.begin() + 1);

        auto map_first = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{1}),
          detail::multiplier_t<size_t>{rx_v_pair_counts[col_comm_rank]});
        rmm::device_uvector<size_t> d_lasts(col_comm_size, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       map_first,
                       map_first + col_comm_size,
                       gathered_nbr_intersection_offsets.begin(),
                       d_lasts.begin());
        std::vector<size_t> h_lasts(d_lasts.size());
        raft::update_host(h_lasts.data(), d_lasts.data(), d_lasts.size(), handle.get_stream());
        handle.sync_stream();
        gathered_nbr_intersection_index_rx_counts.resize(h_lasts.size());
        std::adjacent_difference(
          h_lasts.begin(), h_lasts.end(), gathered_nbr_intersection_index_rx_counts.begin());
      }

      rmm::device_uvector<vertex_t> combined_nbr_intersection_indices(size_t{0},
                                                                      handle.get_stream());
      {
        std::vector<int> ranks(col_comm_size);
        std::iota(ranks.begin(), ranks.end(), int{0});

        std::vector<size_t> tx_displacements(rx_v_pair_nbr_intersection_index_tx_counts.size());
        std::exclusive_scan(rx_v_pair_nbr_intersection_index_tx_counts.begin(),
                            rx_v_pair_nbr_intersection_index_tx_counts.end(),
                            tx_displacements.begin(),
                            size_t{0});

        std::vector<size_t> rx_displacements(gathered_nbr_intersection_index_rx_counts.size());
        std::exclusive_scan(gathered_nbr_intersection_index_rx_counts.begin(),
                            gathered_nbr_intersection_index_rx_counts.end(),
                            rx_displacements.begin(),
                            size_t{0});

        rmm::device_uvector<vertex_t> gathered_nbr_intersection_indices(
          rx_displacements.back() + gathered_nbr_intersection_index_rx_counts.back(),
          handle.get_stream());
        device_multicast_sendrecv(col_comm,
                                  rx_v_pair_nbr_intersection_indices.begin(),
                                  rx_v_pair_nbr_intersection_index_tx_counts,
                                  tx_displacements,
                                  ranks,
                                  gathered_nbr_intersection_indices.begin(),
                                  gathered_nbr_intersection_index_rx_counts,
                                  rx_displacements,
                                  ranks,
                                  handle.get_stream());
        rx_v_pair_nbr_intersection_indices.resize(size_t{0}, handle.get_stream());
        rx_v_pair_nbr_intersection_indices.shrink_to_fit(handle.get_stream());

        combined_nbr_intersection_indices.resize(gathered_nbr_intersection_indices.size(),
                                                 handle.get_stream());

        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(rx_v_pair_counts[col_comm_rank]),
                         gatherv_indices_t<vertex_t>{rx_v_pair_counts[col_comm_rank],
                                                     col_comm_size,
                                                     gathered_nbr_intersection_offsets.data(),
                                                     gathered_nbr_intersection_indices.data(),
                                                     combined_nbr_intersection_offsets.data(),
                                                     combined_nbr_intersection_indices.data()});
      }

      edge_partition_nbr_intersection_sizes.push_back(std::move(combined_nbr_intersection_sizes));
      edge_partition_nbr_intersection_indices.push_back(
        std::move(combined_nbr_intersection_indices));
    }

    rmm::device_uvector<edge_t> nbr_intersection_sizes(input_size, handle.get_stream());
    size_t num_nbr_intersection_indices{0};
    for (size_t i = 0; i < edge_partition_nbr_intersection_indices.size(); ++i) {
      num_nbr_intersection_indices += edge_partition_nbr_intersection_indices[i].size();
    }
    nbr_intersection_indices.resize(num_nbr_intersection_indices, handle.get_stream());
    size_t size_offset{0};
    size_t index_offset{0};
    for (size_t i = 0; i < edge_partition_nbr_intersection_sizes.size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   edge_partition_nbr_intersection_sizes[i].begin(),
                   edge_partition_nbr_intersection_sizes[i].end(),
                   nbr_intersection_sizes.begin() + size_offset);
      size_offset += edge_partition_nbr_intersection_sizes[i].size();
      thrust::copy(handle.get_thrust_policy(),
                   edge_partition_nbr_intersection_indices[i].begin(),
                   edge_partition_nbr_intersection_indices[i].end(),
                   nbr_intersection_indices.begin() + index_offset);
      index_offset += edge_partition_nbr_intersection_indices[i].size();
    }
    nbr_intersection_offsets.resize(nbr_intersection_sizes.size() + size_t{1}, handle.get_stream());
    nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
    auto size_first = thrust::make_transform_iterator(nbr_intersection_sizes.begin(),
                                                      detail::typecast_t<edge_t, size_t>{});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           size_first,
                           size_first + nbr_intersection_sizes.size(),
                           nbr_intersection_offsets.begin() + 1);
  } else {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(size_t{0}));

    rmm::device_uvector<edge_t> nbr_intersection_sizes(
      input_size,
      handle.get_stream());  // initially store minimum degrees (upper bound for intersection sizes)
    if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
      thrust::transform(handle.get_thrust_policy(),
                        vertex_pair_first,
                        vertex_pair_first + input_size,
                        nbr_intersection_sizes.begin(),
                        pick_min_degree_t<void*, void*, vertex_t, edge_t, false>{
                          nullptr, nullptr, nullptr, nullptr, edge_partition});
    } else {
      CUGRAPH_FAIL("unimplemented.");
    }

    nbr_intersection_offsets.resize(nbr_intersection_sizes.size() + 1, handle.get_stream());
    nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
    auto size_first = thrust::make_transform_iterator(nbr_intersection_sizes.begin(),
                                                      detail::typecast_t<edge_t, size_t>{});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           size_first,
                           size_first + nbr_intersection_sizes.size(),
                           nbr_intersection_offsets.begin() + 1);

    nbr_intersection_indices.resize(nbr_intersection_offsets.back_element(handle.get_stream()),
                                    handle.get_stream());
    if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
      thrust::tabulate(
        handle.get_thrust_policy(),
        nbr_intersection_sizes.begin(),
        nbr_intersection_sizes.end(),
        copy_intersecting_nbrs_and_update_intersection_size_t<void*,
                                                              void*,
                                                              decltype(vertex_pair_first),
                                                              vertex_t,
                                                              edge_t,
                                                              false>{
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          edge_partition,
          vertex_pair_first,
          nbr_intersection_offsets.data(),
          nbr_intersection_indices.data(),
          invalid_vertex_id<vertex_t>::value});
    } else {
      CUGRAPH_FAIL("unimplemented.");
    }

#if 1  // FIXME: work-around for the 32 bit integer overflow issue in thrust::remove,
       // thrust::remove_if, and thrust::copy_if (https://github.com/NVIDIA/thrust/issues/1302)
    rmm::device_uvector<vertex_t> tmp_indices(
      thrust::count_if(handle.get_thrust_policy(),
                       nbr_intersection_indices.begin(),
                       nbr_intersection_indices.end(),
                       detail::not_equal_t<vertex_t>{invalid_vertex_id<vertex_t>::value}),
      handle.get_stream());
    size_t num_copied{0};
    size_t num_scanned{0};
    while (num_scanned < nbr_intersection_indices.size()) {
      size_t this_scan_size = std::min(
        size_t{1} << 30,
        static_cast<size_t>(thrust::distance(nbr_intersection_indices.begin() + num_scanned,
                                             nbr_intersection_indices.end())));
      num_copied += static_cast<size_t>(thrust::distance(
        tmp_indices.begin() + num_copied,
        thrust::copy_if(handle.get_thrust_policy(),
                        nbr_intersection_indices.begin() + num_scanned,
                        nbr_intersection_indices.begin() + num_scanned + this_scan_size,
                        tmp_indices.begin() + num_copied,
                        detail::not_equal_t<vertex_t>{invalid_vertex_id<vertex_t>::value})));
      num_scanned += this_scan_size;
    }
    nbr_intersection_indices = std::move(tmp_indices);
#else
    nbr_intersection_indices.resize(
      thrust::distance(nbr_intersection_indices.begin(),
                       thrust::remove(handle.get_thrust_policy(),
                                      nbr_intersection_indices.begin(),
                                      nbr_intersection_indices.end(),
                                      invalid_vertex_id<vertex_t>::value)),
      handle.get_stream());
#endif

    thrust::inclusive_scan(handle.get_thrust_policy(),
                           size_first,
                           size_first + nbr_intersection_sizes.size(),
                           nbr_intersection_offsets.begin() + 1);
  }

  // 5. Return

  return std::make_tuple(std::move(nbr_intersection_offsets), std::move(nbr_intersection_indices));
}

}  // namespace detail

}  // namespace cugraph

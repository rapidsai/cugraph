/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/export.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/detail/optional_dataframe_buffer.hpp>
#include <cugraph/prims/kv_store.cuh>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error_check_utils.cuh>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <array>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

// group index determined by major_comm_rank (primary key) and local edge partition index (secondary
// key)
template <typename vertex_t>
struct major_to_group_idx_t {
  raft::device_span<vertex_t const> vertex_partition_range_lasts{};
  int major_comm_size{};
  int minor_comm_size{};

  __device__ int operator()(vertex_t major) const
  {
    auto vertex_partition_id =
      compute_vertex_partition_id_from_int_vertex_t<vertex_t>{vertex_partition_range_lasts}(major);
    auto major_comm_rank         = vertex_partition_id % major_comm_size;
    auto local_edge_partition_id = vertex_partition_id / major_comm_size;
    return major_comm_rank * minor_comm_size + local_edge_partition_id;
  }
};

// primary key: major_comm_rank secondary key: local edge partition index => primary key: local edge
// partition index secondary key: major_comm_rank
struct reorder_group_count_t {
  int major_comm_size{};
  int minor_comm_size{};
  raft::device_span<size_t const> group_counts{};

  __device__ size_t operator()(size_t i) const
  {
    auto local_edge_partition_id = static_cast<int>(i) / major_comm_size;
    auto major_comm_rank         = static_cast<int>(i) % major_comm_size;
    return group_counts[major_comm_rank * minor_comm_size + local_edge_partition_id];
  }
};

template <typename vertex_t, typename edge_t, bool multi_gpu>
struct update_rx_major_local_degree_t {
  int major_comm_size{};
  int minor_comm_size{};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};
  cuda::std::optional<edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>
    edge_partition_e_mask{};

  size_t reordered_idx_first{};
  size_t local_edge_partition_idx{};

  raft::device_span<size_t const> rx_reordered_group_lasts{};
  raft::device_span<size_t const> rx_group_firsts{};
  raft::device_span<vertex_t const> rx_majors{};

  raft::device_span<edge_t> local_degrees_for_rx_majors{};

  __device__ void operator()(size_t idx) const
  {
    auto it = thrust::upper_bound(
      thrust::seq, rx_reordered_group_lasts.begin(), rx_reordered_group_lasts.end(), idx);
    auto major_comm_rank =
      static_cast<int>(cuda::std::distance(rx_reordered_group_lasts.begin(), it));
    auto offset_in_local_edge_partition =
      idx - (major_comm_rank == int{0} ? reordered_idx_first
                                       : rx_reordered_group_lasts[major_comm_rank - int{1}]);
    auto major =
      rx_majors[rx_group_firsts[major_comm_rank * minor_comm_size + local_edge_partition_idx] +
                offset_in_local_edge_partition];
    auto major_idx    = edge_partition.major_idx_from_major_nocheck(major);
    auto local_degree = major_idx ? edge_partition.local_degree(*major_idx) : edge_t{0};

    if (edge_partition_e_mask && (local_degree > edge_t{0})) {
      auto local_offset = edge_partition.local_offset(*major_idx);
      local_degree      = static_cast<edge_t>(
        count_set_bits((*edge_partition_e_mask).value_first(), local_offset, local_degree));
    }

    local_degrees_for_rx_majors[rx_group_firsts[major_comm_rank * minor_comm_size +
                                                local_edge_partition_idx] +
                                offset_in_local_edge_partition] = local_degree;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename edge_partition_e_input_device_view_t,
          typename optional_property_buffer_mutable_view_t,
          bool multi_gpu>
struct update_rx_major_local_nbrs_t {
  int major_comm_size{};
  int minor_comm_size{};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};
  edge_partition_e_input_device_view_t edge_partition_e_value_input{};
  cuda::std::optional<edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>
    edge_partition_e_mask{};

  size_t reordered_idx_first{};
  size_t local_edge_partition_idx{};

  raft::device_span<size_t const> rx_reordered_group_lasts{};
  raft::device_span<size_t const> rx_group_firsts{};
  raft::device_span<vertex_t const> rx_majors{};
  raft::device_span<size_t const> local_nbr_offsets_for_rx_majors{};
  raft::device_span<vertex_t> local_nbrs_for_rx_majors{};
  optional_property_buffer_mutable_view_t local_e_property_values_for_rx_majors{};

  __device__ void operator()(size_t idx)
  {
    using edge_property_value_t = typename edge_partition_e_input_device_view_t::value_type;

    auto it = thrust::upper_bound(
      thrust::seq, rx_reordered_group_lasts.begin(), rx_reordered_group_lasts.end(), idx);
    auto major_comm_rank =
      static_cast<int>(cuda::std::distance(rx_reordered_group_lasts.begin(), it));
    auto offset_in_local_edge_partition =
      idx - (major_comm_rank == int{0} ? reordered_idx_first
                                       : rx_reordered_group_lasts[major_comm_rank - int{1}]);
    auto major =
      rx_majors[rx_group_firsts[major_comm_rank * minor_comm_size + local_edge_partition_idx] +
                offset_in_local_edge_partition];

    edge_t edge_offset{0};
    edge_t local_degree{0};
    if (multi_gpu && (edge_partition.major_hypersparse_first() &&
                      (major >= *(edge_partition.major_hypersparse_first())))) {
      auto major_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
      if (major_hypersparse_idx) {
        auto major_idx =
          (*(edge_partition.major_hypersparse_first()) - edge_partition.major_range_first()) +
          *major_hypersparse_idx;
        edge_offset  = edge_partition.local_offset(major_idx);
        local_degree = edge_partition.local_degree(major_idx);
      }
    } else {
      auto major_idx = edge_partition.major_offset_from_major_nocheck(major);
      edge_offset    = edge_partition.local_offset(major_idx);
      local_degree   = edge_partition.local_degree(major_idx);
    }

    auto indices = edge_partition.indices();
    size_t output_start_offset =
      local_nbr_offsets_for_rx_majors[rx_group_firsts[major_comm_rank * minor_comm_size +
                                                      local_edge_partition_idx] +
                                      offset_in_local_edge_partition];

    // FIXME: this can lead to thread-divergence with a mix of high-degree and low-degree
    // vertices in a single warp (better optimize if this becomes a performance
    // bottleneck)

    static_assert(!edge_partition_e_input_device_view_t::has_packed_bool_element, "unimplemented.");
    if (local_degree > 0) {
      if (edge_partition_e_mask) {
        auto mask_first = (*edge_partition_e_mask).value_first();
        if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
          auto input_first =
            thrust::make_zip_iterator(indices, edge_partition_e_value_input.value_first());
          copy_if_mask_set(input_first,
                           mask_first,
                           thrust::make_zip_iterator(local_nbrs_for_rx_majors.begin(),
                                                     local_e_property_values_for_rx_majors),
                           edge_offset,
                           output_start_offset,
                           local_degree);
        } else {
          copy_if_mask_set(indices,
                           mask_first,
                           local_nbrs_for_rx_majors.begin(),
                           edge_offset,
                           output_start_offset,
                           local_degree);
        }
      } else {
        if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
          auto input_first =
            thrust::make_zip_iterator(indices, edge_partition_e_value_input.value_first()) +
            edge_offset;
          thrust::copy(thrust::seq,
                       input_first,
                       input_first + local_degree,
                       thrust::make_zip_iterator(local_nbrs_for_rx_majors.begin(),
                                                 local_e_property_values_for_rx_majors) +
                         output_start_offset);
        } else {
          thrust::copy(thrust::seq,
                       indices + edge_offset,
                       indices + (edge_offset + local_degree),
                       local_nbrs_for_rx_majors.begin() + output_start_offset);
        }
      }
    }
  }
};

struct compute_local_nbr_count_per_rank_t {
  raft::device_span<size_t const> rx_offsets{};
  raft::device_span<size_t const> local_nbr_offsets_for_rx_majors{};

  __device__ size_t operator()(size_t i) const
  {
    return local_nbr_offsets_for_rx_majors[rx_offsets[i + 1]] -
           local_nbr_offsets_for_rx_majors[rx_offsets[i]];
  }
};

template <typename FirstElementToIdxMap,
          typename SecondElementToIdxMap,
          typename vertex_t,
          typename edge_t,
          bool multi_gpu>
struct pick_min_degree_t {
  FirstElementToIdxMap first_element_to_idx_map{};
  raft::device_span<edge_t const> first_element_offsets{};

  SecondElementToIdxMap second_element_to_idx_map{};
  raft::device_span<edge_t const> second_element_offsets{};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};
  cuda::std::optional<edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>
    edge_partition_e_mask{};

  __device__ edge_t operator()(cuda::std::tuple<vertex_t, vertex_t> pair) const
  {
    edge_t local_degree0{0};
    vertex_t major0 = cuda::std::get<0>(pair);
    if constexpr (std::is_same_v<FirstElementToIdxMap, void*>) {
      auto major_idx = edge_partition.major_idx_from_major_nocheck(major0);
      local_degree0  = major_idx ? edge_partition.local_degree(*major_idx) : edge_t{0};

      if (edge_partition_e_mask && (local_degree0 > edge_t{0})) {
        auto local_offset = edge_partition.local_offset(*major_idx);
        local_degree0 =
          count_set_bits((*edge_partition_e_mask).value_first(), local_offset, local_degree0);
      }
    } else {
      auto idx = first_element_to_idx_map.find(major0);
      local_degree0 =
        static_cast<edge_t>(first_element_offsets[idx + 1] - first_element_offsets[idx]);
    }

    edge_t local_degree1{0};
    vertex_t major1 = cuda::std::get<1>(pair);
    if constexpr (std::is_same_v<SecondElementToIdxMap, void*>) {
      auto major_idx = edge_partition.major_idx_from_major_nocheck(major1);
      local_degree1  = major_idx ? edge_partition.local_degree(*major_idx) : edge_t{0};

      if (edge_partition_e_mask && (local_degree1 > edge_t{0})) {
        auto local_offset = edge_partition.local_offset(*major_idx);
        local_degree1 =
          count_set_bits((*edge_partition_e_mask).value_first(), local_offset, local_degree1);
      }
    } else {
      auto idx = second_element_to_idx_map.find(major1);
      local_degree1 =
        static_cast<edge_t>(second_element_offsets[idx + 1] - second_element_offsets[idx]);
    }

    return cuda::minimum<edge_t>{}(local_degree0, local_degree1);
  }
};

template <bool check_edge_mask,
          typename InputKeyIterator0,
          typename InputKeyIterator1,
          typename InputValueIterator0,  // should be void* if invalid
          typename InputValueIterator1,  // should be void* if invalid
          typename MaskIterator,         // should be packed bool
          typename OutputKeyIterator,
          typename OutputValueIterator0,
          typename OutputValueIterator1,
          typename edge_t>
__device__ edge_t set_intersection_by_key_with_mask(InputKeyIterator0 input_key_first0,
                                                    InputKeyIterator1 input_key_first1,
                                                    InputValueIterator0 input_value_first0,
                                                    InputValueIterator1 input_value_first1,
                                                    MaskIterator mask_first,
                                                    OutputKeyIterator output_key_first,
                                                    OutputValueIterator0 output_value_first0,
                                                    OutputValueIterator1 output_value_first1,
                                                    edge_t input_start_offset0,
                                                    edge_t input_size0,
                                                    bool apply_mask0,
                                                    edge_t input_start_offset1,
                                                    edge_t input_size1,
                                                    bool apply_mask1,
                                                    size_t output_start_offset)
{
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<MaskIterator>::value_type, uint32_t>);
  static_assert(std::is_same_v<InputValueIterator0, void*> ==
                std::is_same_v<InputValueIterator1, void*>);

  check_bit_set_t<MaskIterator, edge_t> check_bit_set{mask_first, edge_t{0}};

  auto idx0       = input_start_offset0;
  auto idx1       = input_start_offset1;
  auto output_idx = output_start_offset;
  while ((idx0 < (input_start_offset0 + input_size0)) &&
         (idx1 < (input_start_offset1 + input_size1))) {
    bool valid0 = true;
    bool valid1 = true;
    if constexpr (check_edge_mask) {
      valid0 = apply_mask0 ? check_bit_set(idx0) : true;
      valid1 = apply_mask1 ? check_bit_set(idx1) : true;
      if (!valid0) { ++idx0; }
      if (!valid1) { ++idx1; }
    }

    if (valid0 && valid1) {
      auto key0 = *(input_key_first0 + idx0);
      auto key1 = *(input_key_first1 + idx1);
      if (key0 < key1) {
        ++idx0;
      } else if (key0 > key1) {
        ++idx1;
      } else {
        *(output_key_first + output_idx) = key0;
        if constexpr (!std::is_same_v<InputValueIterator0, void*>) {
          *(output_value_first0 + output_idx) = *(input_value_first0 + idx0);
          *(output_value_first1 + output_idx) = *(input_value_first1 + idx1);
        }
        ++idx0;
        ++idx1;
        ++output_idx;
      }
    }
  }

  return (output_idx - output_start_offset);
}

// ============================================================================
// Non-materializing neighbor intersection.
//
// set_intersection_by_key_with_mask (above) writes every common neighbor it finds into an output
// array, so the full intersection is stored in memory (its size grows with the total number of
// common neighbors across all pairs). The kernels below never store the common neighbors. For each
// input pair (v0, v1) and each common neighbor w of v0 and v1, they call a user-provided device
// operator once and move on, so nothing that grows with the intersection size is allocated:
//
//   intersection_op(v0, v1, w, v0_v1_edge_offset, v0_w_edge_offset, v1_w_edge_offset);
//
// The three offsets are the positions, in this edge partition's neighbor array (the CSR indices[]
// array), of the edges (v0, v1), (v0, w) and (v1, w). Passing these positions lets the operator
// act on those edges without searching the graph itself.
//
// Pairs are degree-binned by min(degree(v0), degree(v1)) and dispatched to a thread-, warp-, or
// block-per-pair kernel.
// ============================================================================

constexpr size_t nbr_intersection_block_size = 256;

// Resolve a pair endpoint's neighbor list: the base indices pointer, the offset into it, and the
// degree. When the endpoint resides locally (ElementToIdxMap is void*), it is read from the edge
// partition (multi-GPU low-degree majors are looked up in the hypersparse region; a major absent
// from it has degree 0). Otherwise the endpoint's neighbor list was gathered from another rank and
// is read from element_indices / element_offsets via the idx-map.
template <typename vertex_t, typename edge_t, bool multi_gpu, typename ElementToIdxMap>
__device__ void nbr_intersection_nbr_list(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> const& edge_partition,
  ElementToIdxMap element_to_idx_map,
  raft::device_span<edge_t const> element_offsets,
  raft::device_span<vertex_t const> element_indices,
  vertex_t v,
  vertex_t const*& nbr_indices,
  edge_t& nbr_offset,
  edge_t& nbr_degree)
{
  if constexpr (std::is_same_v<ElementToIdxMap, void*>) {
    nbr_indices = edge_partition.indices();
    if constexpr (multi_gpu) {
      if (edge_partition.major_hypersparse_first() &&
          v >= *(edge_partition.major_hypersparse_first())) {
        auto hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(v);
        if (hypersparse_idx) {
          auto major_idx =
            (*(edge_partition.major_hypersparse_first()) - edge_partition.major_range_first()) +
            *hypersparse_idx;
          nbr_offset = edge_partition.local_offset(major_idx);
          nbr_degree = edge_partition.local_degree(major_idx);
        } else {
          nbr_offset = edge_t{0};
          nbr_degree = edge_t{0};
        }
        return;
      }
    }
    auto major_idx = edge_partition.major_offset_from_major_nocheck(v);
    nbr_offset     = edge_partition.local_offset(major_idx);
    nbr_degree     = edge_partition.local_degree(major_idx);
  } else {
    auto idx    = element_to_idx_map.find(v);
    nbr_indices = element_indices.begin();
    nbr_offset  = element_offsets[idx];
    nbr_degree  = static_cast<edge_t>(element_offsets[idx + 1] - element_offsets[idx]);
  }
}

// Single thread per pair: merge-walk the two endpoints' sorted neighbor lists.
template <bool check_edge_mask,
          typename vertex_t,
          typename edge_t,
          bool multi_gpu,
          typename VertexPairIterator,
          typename FirstElementToIdxMap,
          typename SecondElementToIdxMap,
          typename IntersectionOp>
__global__ static void nbr_intersection_low_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  VertexPairIterator vertex_pair_first,
  raft::device_span<size_t const> pair_indices,
  FirstElementToIdxMap first_element_to_idx_map,
  raft::device_span<edge_t const> first_element_offsets,
  raft::device_span<vertex_t const> first_element_indices,
  SecondElementToIdxMap second_element_to_idx_map,
  raft::device_span<edge_t const> second_element_offsets,
  raft::device_span<vertex_t const> second_element_indices,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask)
{
  constexpr bool v0_local = std::is_same_v<FirstElementToIdxMap, void*>;
  constexpr bool v1_local = std::is_same_v<SecondElementToIdxMap, void*>;

  auto const tid = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  size_t idx     = tid;

  check_bit_set_t<uint32_t const*, edge_t> check_bit_set{edge_mask, edge_t{0}};

  while (idx < pair_indices.size()) {
    auto i    = pair_indices[idx];
    auto pair = *(vertex_pair_first + i);
    auto v0   = cuda::std::get<0>(pair);
    auto v1   = cuda::std::get<1>(pair);

    vertex_t const* v0_indices{nullptr};
    vertex_t const* v1_indices{nullptr};
    edge_t v0_off{}, v0_deg{}, v1_off{}, v1_deg{};
    nbr_intersection_nbr_list(edge_partition,
                              first_element_to_idx_map,
                              first_element_offsets,
                              first_element_indices,
                              v0,
                              v0_indices,
                              v0_off,
                              v0_deg);
    nbr_intersection_nbr_list(edge_partition,
                              second_element_to_idx_map,
                              second_element_offsets,
                              second_element_indices,
                              v1,
                              v1_indices,
                              v1_off,
                              v1_deg);

    // Offset of edge (v0, v1): position of v1 in v0's neighbor list.
    auto v0_v1_itr =
      thrust::lower_bound(thrust::seq, v0_indices + v0_off, v0_indices + v0_off + v0_deg, v1);
    edge_t v0_v1_edge_offset = static_cast<edge_t>(v0_v1_itr - v0_indices);

    edge_t i0 = 0;
    edge_t i1 = 0;
    while (i0 < v0_deg && i1 < v1_deg) {
      if constexpr (check_edge_mask && v0_local) {
        if (!check_bit_set(v0_off + i0)) {
          ++i0;
          continue;
        }
      }
      if constexpr (check_edge_mask && v1_local) {
        if (!check_bit_set(v1_off + i1)) {
          ++i1;
          continue;
        }
      }
      auto n0 = v0_indices[v0_off + i0];
      auto n1 = v1_indices[v1_off + i1];
      if (n0 < n1) {
        ++i0;
      } else if (n0 > n1) {
        ++i1;
      } else {
        intersection_op(v0, v1, n0, v0_v1_edge_offset, v0_off + i0, v1_off + i1);
        ++i0;
        ++i1;
      }
    }

    idx += static_cast<size_t>(gridDim.x) * blockDim.x;
  }
}

// One warp per pair: each lane scans part of the shorter neighbor list and binary-searches the
// longer one.
template <bool check_edge_mask,
          typename vertex_t,
          typename edge_t,
          bool multi_gpu,
          typename VertexPairIterator,
          typename FirstElementToIdxMap,
          typename SecondElementToIdxMap,
          typename IntersectionOp>
__global__ static void nbr_intersection_mid_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  VertexPairIterator vertex_pair_first,
  raft::device_span<size_t const> pair_indices,
  FirstElementToIdxMap first_element_to_idx_map,
  raft::device_span<edge_t const> first_element_offsets,
  raft::device_span<vertex_t const> first_element_indices,
  SecondElementToIdxMap second_element_to_idx_map,
  raft::device_span<edge_t const> second_element_offsets,
  raft::device_span<vertex_t const> second_element_indices,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask)
{
  constexpr bool v0_local = std::is_same_v<FirstElementToIdxMap, void*>;
  constexpr bool v1_local = std::is_same_v<SecondElementToIdxMap, void*>;

  auto const tid     = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  auto const lane_id = static_cast<edge_t>(tid % raft::warp_size());
  size_t idx         = tid / raft::warp_size();

  check_bit_set_t<uint32_t const*, edge_t> check_bit_set{edge_mask, edge_t{0}};

  while (idx < pair_indices.size()) {
    auto i    = pair_indices[idx];
    auto pair = *(vertex_pair_first + i);
    auto v0   = cuda::std::get<0>(pair);
    auto v1   = cuda::std::get<1>(pair);

    vertex_t const* v0_indices{nullptr};
    vertex_t const* v1_indices{nullptr};
    edge_t v0_off{}, v0_deg{}, v1_off{}, v1_deg{};
    nbr_intersection_nbr_list(edge_partition,
                              first_element_to_idx_map,
                              first_element_offsets,
                              first_element_indices,
                              v0,
                              v0_indices,
                              v0_off,
                              v0_deg);
    nbr_intersection_nbr_list(edge_partition,
                              second_element_to_idx_map,
                              second_element_offsets,
                              second_element_indices,
                              v1,
                              v1_indices,
                              v1_off,
                              v1_deg);

    // Scan the shorter neighbor list, binary-search the longer one.
    bool v0_is_short              = (v0_deg <= v1_deg);
    vertex_t const* short_indices = v0_is_short ? v0_indices : v1_indices;
    vertex_t const* long_indices  = v0_is_short ? v1_indices : v0_indices;
    edge_t short_off              = v0_is_short ? v0_off : v1_off;
    edge_t short_deg              = v0_is_short ? v0_deg : v1_deg;
    edge_t long_off               = v0_is_short ? v1_off : v0_off;
    edge_t long_deg               = v0_is_short ? v1_deg : v0_deg;
    bool short_local              = v0_is_short ? v0_local : v1_local;
    bool long_local               = v0_is_short ? v1_local : v0_local;

    // Offset of edge (v0, v1): position of v1 in v0's neighbor list (computed once per warp).
    edge_t v0_v1_edge_offset{};
    if (lane_id == 0) {
      auto v0_v1_itr =
        thrust::lower_bound(thrust::seq, v0_indices + v0_off, v0_indices + v0_off + v0_deg, v1);
      v0_v1_edge_offset = static_cast<edge_t>(v0_v1_itr - v0_indices);
    }
    v0_v1_edge_offset = __shfl_sync(uint32_t{0xffffffff}, v0_v1_edge_offset, 0);

    for (edge_t s = lane_id; s < short_deg; s += static_cast<edge_t>(raft::warp_size())) {
      if constexpr (check_edge_mask) {
        if (short_local && !check_bit_set(short_off + s)) { continue; }
      }
      auto w = short_indices[short_off + s];

      edge_t lo = long_off;
      edge_t hi = long_off + long_deg;
      while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (long_indices[mid] < w) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (lo < long_off + long_deg && long_indices[lo] == w) {
        if constexpr (check_edge_mask) {
          if (long_local && !check_bit_set(lo)) { continue; }
        }
        edge_t v0_w_edge_offset = v0_is_short ? (short_off + s) : lo;
        edge_t v1_w_edge_offset = v0_is_short ? lo : (short_off + s);
        intersection_op(v0, v1, w, v0_v1_edge_offset, v0_w_edge_offset, v1_w_edge_offset);
      }
    }

    idx += static_cast<size_t>(gridDim.x) * (blockDim.x / raft::warp_size());
  }
}

// One thread block per pair: each thread scans part of the shorter neighbor list and
// binary-searches the longer one.
template <bool check_edge_mask,
          typename vertex_t,
          typename edge_t,
          bool multi_gpu,
          typename VertexPairIterator,
          typename FirstElementToIdxMap,
          typename SecondElementToIdxMap,
          typename IntersectionOp>
__global__ static void nbr_intersection_high_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  VertexPairIterator vertex_pair_first,
  raft::device_span<size_t const> pair_indices,
  FirstElementToIdxMap first_element_to_idx_map,
  raft::device_span<edge_t const> first_element_offsets,
  raft::device_span<vertex_t const> first_element_indices,
  SecondElementToIdxMap second_element_to_idx_map,
  raft::device_span<edge_t const> second_element_offsets,
  raft::device_span<vertex_t const> second_element_indices,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask)
{
  constexpr bool v0_local = std::is_same_v<FirstElementToIdxMap, void*>;
  constexpr bool v1_local = std::is_same_v<SecondElementToIdxMap, void*>;

  size_t idx = static_cast<size_t>(blockIdx.x);

  check_bit_set_t<uint32_t const*, edge_t> check_bit_set{edge_mask, edge_t{0}};

  __shared__ edge_t shared_v0_v1_edge_offset;

  while (idx < pair_indices.size()) {
    auto i    = pair_indices[idx];
    auto pair = *(vertex_pair_first + i);
    auto v0   = cuda::std::get<0>(pair);
    auto v1   = cuda::std::get<1>(pair);

    vertex_t const* v0_indices{nullptr};
    vertex_t const* v1_indices{nullptr};
    edge_t v0_off{}, v0_deg{}, v1_off{}, v1_deg{};
    nbr_intersection_nbr_list(edge_partition,
                              first_element_to_idx_map,
                              first_element_offsets,
                              first_element_indices,
                              v0,
                              v0_indices,
                              v0_off,
                              v0_deg);
    nbr_intersection_nbr_list(edge_partition,
                              second_element_to_idx_map,
                              second_element_offsets,
                              second_element_indices,
                              v1,
                              v1_indices,
                              v1_off,
                              v1_deg);

    bool v0_is_short              = (v0_deg <= v1_deg);
    vertex_t const* short_indices = v0_is_short ? v0_indices : v1_indices;
    vertex_t const* long_indices  = v0_is_short ? v1_indices : v0_indices;
    edge_t short_off              = v0_is_short ? v0_off : v1_off;
    edge_t short_deg              = v0_is_short ? v0_deg : v1_deg;
    edge_t long_off               = v0_is_short ? v1_off : v0_off;
    edge_t long_deg               = v0_is_short ? v1_deg : v0_deg;
    bool short_local              = v0_is_short ? v0_local : v1_local;
    bool long_local               = v0_is_short ? v1_local : v0_local;

    // Offset of edge (v0, v1): position of v1 in v0's neighbor list (computed once per block).
    if (threadIdx.x == 0) {
      auto v0_v1_itr =
        thrust::lower_bound(thrust::seq, v0_indices + v0_off, v0_indices + v0_off + v0_deg, v1);
      shared_v0_v1_edge_offset = static_cast<edge_t>(v0_v1_itr - v0_indices);
    }
    __syncthreads();
    edge_t v0_v1_edge_offset = shared_v0_v1_edge_offset;

    for (edge_t s = static_cast<edge_t>(threadIdx.x); s < short_deg;
         s += static_cast<edge_t>(blockDim.x)) {
      if constexpr (check_edge_mask) {
        if (short_local && !check_bit_set(short_off + s)) { continue; }
      }
      auto w = short_indices[short_off + s];

      edge_t lo = long_off;
      edge_t hi = long_off + long_deg;
      while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (long_indices[mid] < w) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (lo < long_off + long_deg && long_indices[lo] == w) {
        if constexpr (check_edge_mask) {
          if (long_local && !check_bit_set(lo)) { continue; }
        }
        edge_t v0_w_edge_offset = v0_is_short ? (short_off + s) : lo;
        edge_t v1_w_edge_offset = v0_is_short ? lo : (short_off + s);
        intersection_op(v0, v1, w, v0_v1_edge_offset, v0_w_edge_offset, v1_w_edge_offset);
      }
    }

    idx += static_cast<size_t>(gridDim.x);
    __syncthreads();
  }
}

// Multi-GPU only: gather the neighbor lists of the unique second pair elements so the
// non-materializing kernels can resolve a remote endpoint (the first element is always local).
// Returns the vertex -> idx map and the gathered (offsets, indices). Mirrors the second-element
// collection in the materializing nbr_intersection; intersection-edge values are omitted because
// this path carries none.
template <typename GraphViewType, typename VertexPairIterator>
std::tuple<
  std::unique_ptr<
    kv_store_t<typename GraphViewType::vertex_type, typename GraphViewType::vertex_type, false>>,
  rmm::device_uvector<typename GraphViewType::edge_type>,
  rmm::device_uvector<typename GraphViewType::vertex_type>>
nbr_intersection_collect_second_nbrs(raft::handle_t const& handle,
                                     GraphViewType const& graph_view,
                                     VertexPairIterator vertex_pair_first,
                                     VertexPairIterator vertex_pair_last)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_dummy_t     = detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>;

  auto input_size = static_cast<size_t>(cuda::std::distance(vertex_pair_first, vertex_pair_last));
  auto edge_mask_view = graph_view.edge_mask_view();

  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  // Unique second pair elements (all-gathered over minor_comm).
  rmm::device_uvector<vertex_t> unique_majors(input_size, handle.get_stream());
  {
    auto second_element_first = cuda::make_transform_iterator(
      vertex_pair_first, thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, size_t{1}>{});
    thrust::copy(handle.get_thrust_policy(),
                 second_element_first,
                 second_element_first + input_size,
                 unique_majors.begin());
    thrust::sort(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end());
    unique_majors.resize(
      cuda::std::distance(
        unique_majors.begin(),
        thrust::unique(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end())),
      handle.get_stream());
    unique_majors.shrink_to_fit(handle.get_stream());

    if (minor_comm_size > 1) {
      auto rx_counts = host_scalar_allgather(minor_comm, unique_majors.size(), handle.get_stream());
      std::vector<size_t> rx_displacements(rx_counts.size());
      std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
      rmm::device_uvector<vertex_t> rx_unique_majors(rx_displacements.back() + rx_counts.back(),
                                                     handle.get_stream());
      cugraph::device_allgatherv(minor_comm,
                                 unique_majors.begin(),
                                 rx_unique_majors.begin(),
                                 raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                                 raft::host_span<size_t const>(rx_displacements.data(),
                                                               rx_displacements.size()),
                                 handle.get_stream());
      unique_majors = std::move(rx_unique_majors);
      thrust::sort(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end());
      unique_majors.resize(
        cuda::std::distance(
          unique_majors.begin(),
          thrust::unique(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end())),
        handle.get_stream());
      unique_majors.shrink_to_fit(handle.get_stream());
    }
  }

  // Send majors and group (major_comm_rank, local edge_partition_idx) counts to owners.
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
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size},
      comm_size,
      std::numeric_limits<size_t>::max(),
      handle.get_stream());
    std::vector<size_t> h_tx_group_counts(d_tx_group_counts.size());
    raft::update_host(h_tx_group_counts.data(),
                      d_tx_group_counts.data(),
                      d_tx_group_counts.size(),
                      handle.get_stream());
    handle.sync_stream();

    std::vector<size_t> tx_counts(major_comm_size, size_t{0});
    for (size_t i = 0; i < tx_counts.size(); ++i) {
      tx_counts[i] = std::reduce(h_tx_group_counts.begin() + minor_comm_size * i,
                                 h_tx_group_counts.begin() + minor_comm_size * (i + 1),
                                 size_t{0});
    }

    std::tie(rx_majors, rx_major_counts) =
      shuffle_values(major_comm,
                     unique_majors.begin(),
                     raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                     handle.get_stream());

    std::vector<size_t> tmp_counts(major_comm_size, minor_comm_size);
    std::tie(rx_group_counts, std::ignore) =
      shuffle_values(major_comm,
                     d_tx_group_counts.begin(),
                     raft::host_span<size_t const>(tmp_counts.data(), tmp_counts.size()),
                     handle.get_stream());
  }

  // Enumerate degrees and neighbors for the received majors.
  rmm::device_uvector<edge_t> local_degrees_for_rx_majors(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> local_nbrs_for_rx_majors(size_t{0}, handle.get_stream());
  std::vector<size_t> local_nbr_counts{};
  {
    rmm::device_uvector<size_t> rx_reordered_group_counts(rx_group_counts.size(),
                                                          handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      rx_reordered_group_counts.begin(),
      rx_reordered_group_counts.end(),
      reorder_group_count_t{
        major_comm_size,
        minor_comm_size,
        raft::device_span<size_t const>(rx_group_counts.data(), rx_group_counts.size())});

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
      auto edge_partition_e_mask =
        edge_mask_view
          ? cuda::std::make_optional<
              detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
              *edge_mask_view, i)
          : cuda::std::nullopt;
      auto reordered_idx_first =
        (i == size_t{0}) ? size_t{0} : h_rx_reordered_group_lasts[i * major_comm_size - 1];
      auto reordered_idx_last = h_rx_reordered_group_lasts[(i + 1) * major_comm_size - 1];
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(reordered_idx_first),
        thrust::make_counting_iterator(reordered_idx_last),
        update_rx_major_local_degree_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>{
          major_comm_size,
          minor_comm_size,
          edge_partition,
          edge_partition_e_mask,
          reordered_idx_first,
          i,
          raft::device_span<size_t const>(
            d_rx_reordered_group_lasts.data() + i * major_comm_size, major_comm_size),
          raft::device_span<size_t const>(rx_group_firsts.data(), rx_group_firsts.size()),
          raft::device_span<vertex_t const>(rx_majors.data(), rx_majors.size()),
          raft::device_span<edge_t>(local_degrees_for_rx_majors.data(),
                                    local_degrees_for_rx_majors.size())});
    }

    rmm::device_uvector<size_t> local_nbr_offsets_for_rx_majors(
      local_degrees_for_rx_majors.size() + 1, handle.get_stream());
    local_nbr_offsets_for_rx_majors.set_element_to_zero_async(size_t{0}, handle.get_stream());
    auto degree_first = cuda::make_transform_iterator(local_degrees_for_rx_majors.begin(),
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
      auto edge_partition_e_mask =
        edge_mask_view
          ? cuda::std::make_optional<
              detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
              *edge_mask_view, i)
          : cuda::std::nullopt;
      auto reordered_idx_first =
        (i == size_t{0}) ? size_t{0} : h_rx_reordered_group_lasts[i * major_comm_size - 1];
      auto reordered_idx_last = h_rx_reordered_group_lasts[(i + 1) * major_comm_size - 1];
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(reordered_idx_first),
        thrust::make_counting_iterator(reordered_idx_last),
        update_rx_major_local_nbrs_t<vertex_t,
                                     edge_t,
                                     e_dummy_t,
                                     void*,
                                     GraphViewType::is_multi_gpu>{
          major_comm_size,
          minor_comm_size,
          edge_partition,
          e_dummy_t{},
          edge_partition_e_mask,
          reordered_idx_first,
          i,
          raft::device_span<size_t const>(
            d_rx_reordered_group_lasts.data() + i * major_comm_size, major_comm_size),
          raft::device_span<size_t const>(rx_group_firsts.data(), rx_group_firsts.size()),
          raft::device_span<vertex_t const>(rx_majors.data(), rx_majors.size()),
          raft::device_span<size_t const>(local_nbr_offsets_for_rx_majors.data(),
                                          local_nbr_offsets_for_rx_majors.size()),
          raft::device_span<vertex_t>(local_nbrs_for_rx_majors.data(),
                                      local_nbrs_for_rx_majors.size()),
          static_cast<void*>(nullptr)});
    }

    std::vector<size_t> h_rx_offsets(rx_major_counts.size() + size_t{1}, size_t{0});
    std::inclusive_scan(rx_major_counts.begin(), rx_major_counts.end(), h_rx_offsets.begin() + 1);
    rmm::device_uvector<size_t> d_rx_offsets(h_rx_offsets.size(), handle.get_stream());
    raft::update_device(
      d_rx_offsets.data(), h_rx_offsets.data(), h_rx_offsets.size(), handle.get_stream());
    rmm::device_uvector<size_t> d_local_nbr_counts(rx_major_counts.size(), handle.get_stream());
    thrust::tabulate(handle.get_thrust_policy(),
                     d_local_nbr_counts.begin(),
                     d_local_nbr_counts.end(),
                     compute_local_nbr_count_per_rank_t{
                       raft::device_span<size_t const>(d_rx_offsets.data(), d_rx_offsets.size()),
                       raft::device_span<size_t const>(local_nbr_offsets_for_rx_majors.data(),
                                                       local_nbr_offsets_for_rx_majors.size())});
    local_nbr_counts.resize(d_local_nbr_counts.size());
    raft::update_host(local_nbr_counts.data(),
                      d_local_nbr_counts.data(),
                      d_local_nbr_counts.size(),
                      handle.get_stream());
    handle.sync_stream();

    // Send the degrees and neighbors back.
    rmm::device_uvector<edge_t> local_degrees_for_unique_majors(size_t{0}, handle.get_stream());
    std::tie(local_degrees_for_unique_majors, std::ignore) =
      shuffle_values(major_comm,
                     local_degrees_for_rx_majors.begin(),
                     raft::host_span<size_t const>(rx_major_counts.data(), rx_major_counts.size()),
                     handle.get_stream());
    rmm::device_uvector<edge_t> major_nbr_offsets(local_degrees_for_unique_majors.size() + 1,
                                                  handle.get_stream());
    major_nbr_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
    auto out_degree_first = cuda::make_transform_iterator(local_degrees_for_unique_majors.begin(),
                                                          detail::typecast_t<edge_t, edge_t>{});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           out_degree_first,
                           out_degree_first + local_degrees_for_unique_majors.size(),
                           major_nbr_offsets.begin() + 1);

    rmm::device_uvector<vertex_t> major_nbr_indices(0, handle.get_stream());
    std::tie(major_nbr_indices, std::ignore) =
      shuffle_values(major_comm,
                     local_nbrs_for_rx_majors.begin(),
                     raft::host_span<size_t const>(local_nbr_counts.data(), local_nbr_counts.size()),
                     handle.get_stream());

    auto major_to_idx_map = std::make_unique<kv_store_t<vertex_t, vertex_t, false>>(
      unique_majors.begin(),
      unique_majors.end(),
      thrust::make_counting_iterator(vertex_t{0}),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());

    return std::make_tuple(
      std::move(major_to_idx_map), std::move(major_nbr_offsets), std::move(major_nbr_indices));
  }
}

// Degree bands for choosing the per-pair kernel from min(degree(v0), degree(v1)). Tunable.
constexpr size_t nbr_intersection_low_degree_threshold = 32;
constexpr size_t nbr_intersection_mid_degree_threshold = 1024;

// Non-materializing neighbor intersection. For every input pair, invokes intersection_op once per
// common neighbor (see the kernels above) instead of returning the intersection. The pairs are
// degree-binned by min(degree(v0), degree(v1)) and each bin is dispatched to the thread-, warp-, or
// block-per-pair kernel.
template <typename GraphViewType, typename VertexPairIterator, typename IntersectionOp>
void nbr_intersection(raft::handle_t const& handle,
                      GraphViewType const& graph_view,
                      edge_partition_device_view_t<typename GraphViewType::vertex_type,
                                                   typename GraphViewType::edge_type,
                                                   GraphViewType::is_multi_gpu> edge_partition,
                      VertexPairIterator vertex_pair_first,
                      VertexPairIterator vertex_pair_last,
                      IntersectionOp intersection_op,
                      uint32_t const* edge_mask)
{
  using vertex_t           = typename GraphViewType::vertex_type;
  using edge_t             = typename GraphViewType::edge_type;
  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;

  auto stream    = handle.get_stream();
  auto num_pairs = static_cast<size_t>(cuda::std::distance(vertex_pair_first, vertex_pair_last));
  if (num_pairs == 0) { return; }

  // Multi-GPU: gather the second pair elements' neighbor lists so the kernels can resolve a remote
  // endpoint without materializing the intersection. The first element is always local.
  std::optional<std::unique_ptr<kv_store_t<vertex_t, vertex_t, false>>> major_to_idx_map_ptr{
    std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> major_nbr_offsets{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> major_nbr_indices{std::nullopt};
  if constexpr (multi_gpu) {
    auto [m, o, n] = nbr_intersection_collect_second_nbrs(
      handle, graph_view, vertex_pair_first, vertex_pair_last);
    major_to_idx_map_ptr = std::move(m);
    major_nbr_offsets    = std::move(o);
    major_nbr_indices    = std::move(n);
  }

  // Per-vertex degrees (mask-aware) of the local (first) endpoints used to bin each pair.
  auto vertex_degrees = (edge_mask != nullptr)
                          ? edge_partition.compute_local_degrees_with_mask(edge_mask, stream)
                          : edge_partition.compute_local_degrees(stream);
  auto degrees = vertex_degrees.data();

  // min(degree(v0), degree(v1)) for each pair. v0 is local; v1's degree is local in single-GPU and
  // comes from the gathered offsets (via the map) in multi-GPU.
  rmm::device_uvector<edge_t> min_degrees(num_pairs, stream);
  if constexpr (multi_gpu) {
    auto map_view = detail::kv_cuco_store_find_device_view_t((*major_to_idx_map_ptr)->view());
    auto nbr_offsets =
      raft::device_span<edge_t const>((*major_nbr_offsets).data(), (*major_nbr_offsets).size());
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_pairs),
      min_degrees.begin(),
      cuda::proclaim_return_type<edge_t>(
        [vertex_pair_first, edge_partition, degrees, map_view, nbr_offsets] __device__(size_t i) {
          auto pair = *(vertex_pair_first + i);
          auto d0   = degrees[edge_partition.major_offset_from_major_nocheck(
            cuda::std::get<0>(pair))];
          auto idx  = map_view.find(cuda::std::get<1>(pair));
          auto d1   = static_cast<edge_t>(nbr_offsets[idx + 1] - nbr_offsets[idx]);
          return d0 < d1 ? d0 : d1;
        }));
  } else {
    thrust::transform(handle.get_thrust_policy(),
                      thrust::make_counting_iterator(size_t{0}),
                      thrust::make_counting_iterator(num_pairs),
                      min_degrees.begin(),
                      cuda::proclaim_return_type<edge_t>(
                        [vertex_pair_first, edge_partition, degrees] __device__(size_t i) {
                          auto pair = *(vertex_pair_first + i);
                          auto d0   = degrees[edge_partition.major_offset_from_major_nocheck(
                            cuda::std::get<0>(pair))];
                          auto d1   = degrees[edge_partition.major_offset_from_major_nocheck(
                            cuda::std::get<1>(pair))];
                          return d0 < d1 ? d0 : d1;
                        }));
  }
  auto min_degree_first = min_degrees.data();

  auto counting = thrust::make_counting_iterator(size_t{0});
  auto low_band = cuda::proclaim_return_type<bool>([min_degree_first] __device__(size_t i) {
    return min_degree_first[i] < static_cast<edge_t>(nbr_intersection_low_degree_threshold);
  });
  auto high_band = cuda::proclaim_return_type<bool>([min_degree_first] __device__(size_t i) {
    return min_degree_first[i] >= static_cast<edge_t>(nbr_intersection_mid_degree_threshold);
  });

  auto num_low  = static_cast<size_t>(
    thrust::count_if(handle.get_thrust_policy(), counting, counting + num_pairs, low_band));
  auto num_high = static_cast<size_t>(
    thrust::count_if(handle.get_thrust_policy(), counting, counting + num_pairs, high_band));
  auto num_mid  = num_pairs - num_low - num_high;

  rmm::device_uvector<size_t> low_indices(num_low, stream);
  rmm::device_uvector<size_t> mid_indices(num_mid, stream);
  rmm::device_uvector<size_t> high_indices(num_high, stream);

  if (num_low > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_pairs, low_indices.begin(), low_band);
  }
  if (num_mid > 0) {
    thrust::copy_if(handle.get_thrust_policy(),
                    counting,
                    counting + num_pairs,
                    mid_indices.begin(),
                    cuda::proclaim_return_type<bool>([min_degree_first] __device__(size_t i) {
                      auto d = min_degree_first[i];
                      return d >= static_cast<edge_t>(nbr_intersection_low_degree_threshold) &&
                             d < static_cast<edge_t>(nbr_intersection_mid_degree_threshold);
                    }));
  }
  if (num_high > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_pairs, high_indices.begin(), high_band);
  }

  auto max_grid_size = handle.get_device_properties().maxGridSize[0];

  // Launch the three degree-binned kernels. The first endpoint is always local (void* map); the
  // second endpoint is local in single-GPU (void* map) and resolved through the gathered arrays in
  // multi-GPU (the kv_store find view + major_nbr_offsets / major_nbr_indices).
  auto launch_all = [&](auto second_element_to_idx_map,
                        raft::device_span<edge_t const> second_element_offsets,
                        raft::device_span<vertex_t const> second_element_indices) {
    void* first_element_to_idx_map = nullptr;
    auto first_element_offsets     = raft::device_span<edge_t const>{};
    auto first_element_indices     = raft::device_span<vertex_t const>{};

    if (num_low > 0) {
      raft::grid_1d_thread_t grid(num_low, nbr_intersection_block_size, max_grid_size);
      auto pairs = raft::device_span<size_t const>(low_indices.data(), num_low);
      if (edge_mask != nullptr) {
        nbr_intersection_low_degree<true, vertex_t, edge_t, multi_gpu>
          <<<grid.num_blocks, grid.block_size, 0, stream>>>(
            edge_partition, vertex_pair_first, pairs, first_element_to_idx_map,
            first_element_offsets, first_element_indices, second_element_to_idx_map,
            second_element_offsets, second_element_indices, intersection_op, edge_mask);
      } else {
        nbr_intersection_low_degree<false, vertex_t, edge_t, multi_gpu>
          <<<grid.num_blocks, grid.block_size, 0, stream>>>(
            edge_partition, vertex_pair_first, pairs, first_element_to_idx_map,
            first_element_offsets, first_element_indices, second_element_to_idx_map,
            second_element_offsets, second_element_indices, intersection_op, nullptr);
      }
    }

    if (num_mid > 0) {
      raft::grid_1d_warp_t grid(num_mid, nbr_intersection_block_size, max_grid_size);
      auto pairs = raft::device_span<size_t const>(mid_indices.data(), num_mid);
      if (edge_mask != nullptr) {
        nbr_intersection_mid_degree<true, vertex_t, edge_t, multi_gpu>
          <<<grid.num_blocks, grid.block_size, 0, stream>>>(
            edge_partition, vertex_pair_first, pairs, first_element_to_idx_map,
            first_element_offsets, first_element_indices, second_element_to_idx_map,
            second_element_offsets, second_element_indices, intersection_op, edge_mask);
      } else {
        nbr_intersection_mid_degree<false, vertex_t, edge_t, multi_gpu>
          <<<grid.num_blocks, grid.block_size, 0, stream>>>(
            edge_partition, vertex_pair_first, pairs, first_element_to_idx_map,
            first_element_offsets, first_element_indices, second_element_to_idx_map,
            second_element_offsets, second_element_indices, intersection_op, nullptr);
      }
    }

    if (num_high > 0) {
      raft::grid_1d_block_t grid(num_high, nbr_intersection_block_size, max_grid_size);
      auto pairs = raft::device_span<size_t const>(high_indices.data(), num_high);
      if (edge_mask != nullptr) {
        nbr_intersection_high_degree<true, vertex_t, edge_t, multi_gpu>
          <<<grid.num_blocks, grid.block_size, 0, stream>>>(
            edge_partition, vertex_pair_first, pairs, first_element_to_idx_map,
            first_element_offsets, first_element_indices, second_element_to_idx_map,
            second_element_offsets, second_element_indices, intersection_op, edge_mask);
      } else {
        nbr_intersection_high_degree<false, vertex_t, edge_t, multi_gpu>
          <<<grid.num_blocks, grid.block_size, 0, stream>>>(
            edge_partition, vertex_pair_first, pairs, first_element_to_idx_map,
            first_element_offsets, first_element_indices, second_element_to_idx_map,
            second_element_offsets, second_element_indices, intersection_op, nullptr);
      }
    }
  };

  if constexpr (multi_gpu) {
    launch_all(detail::kv_cuco_store_find_device_view_t((*major_to_idx_map_ptr)->view()),
               raft::device_span<edge_t const>((*major_nbr_offsets).data(),
                                               (*major_nbr_offsets).size()),
               raft::device_span<vertex_t const>((*major_nbr_indices).data(),
                                                 (*major_nbr_indices).size()));
  } else {
    launch_all(static_cast<void*>(nullptr),
               raft::device_span<edge_t const>{},
               raft::device_span<vertex_t const>{});
  }
}

template <typename FirstElementToIdxMap,
          typename SecondElementToIdxMap,
          typename VertexPairIterator,
          typename vertex_t,
          typename edge_t,
          typename edge_partition_e_input_device_view_t,
          typename optional_property_buffer_view_t,
          typename optional_property_buffer_mutable_view_t,
          bool multi_gpu>
struct copy_intersecting_nbrs_and_update_intersection_size_t {
  FirstElementToIdxMap first_element_to_idx_map{};
  raft::device_span<edge_t const> first_element_offsets{};
  raft::device_span<vertex_t const> first_element_indices{};
  optional_property_buffer_view_t first_element_edge_property_values{};

  SecondElementToIdxMap second_element_to_idx_map{};
  raft::device_span<edge_t const> second_element_offsets{};
  raft::device_span<vertex_t const> second_element_indices{};
  optional_property_buffer_view_t second_element_edge_property_values{};

  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition{};
  edge_partition_e_input_device_view_t edge_partition_e_value_input{};
  cuda::std::optional<edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>
    edge_partition_e_mask{};

  VertexPairIterator vertex_pair_first;
  raft::device_span<size_t const> nbr_intersection_offsets{};
  raft::device_span<vertex_t> nbr_intersection_indices{};

  optional_property_buffer_mutable_view_t nbr_intersection_e_property_values0{};
  optional_property_buffer_mutable_view_t nbr_intersection_e_property_values1{};
  vertex_t invalid_id{};

  __device__ edge_t operator()(size_t i)
  {
    using edge_property_value_t = typename edge_partition_e_input_device_view_t::value_type;

    auto pair = *(vertex_pair_first + i);
    vertex_t const* indices0{};
    std::conditional_t<!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>,
                       edge_property_value_t const*,
                       void*>
      edge_property_values0{};

    edge_t local_edge_offset0{0};
    edge_t local_degree0{0};
    if constexpr (std::is_same_v<FirstElementToIdxMap, void*>) {
      indices0 = edge_partition.indices();
      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        edge_property_values0 = edge_partition_e_value_input.value_first();
      }

      vertex_t major = cuda::std::get<0>(pair);
      if constexpr (multi_gpu) {
        if (edge_partition.major_hypersparse_first() &&
            (major >= *(edge_partition.major_hypersparse_first()))) {
          auto major_hypersparse_idx =
            edge_partition.major_hypersparse_idx_from_major_nocheck(major);
          if (major_hypersparse_idx) {
            auto major_idx =
              (*(edge_partition.major_hypersparse_first()) - edge_partition.major_range_first()) +
              *major_hypersparse_idx;
            local_edge_offset0 = edge_partition.local_offset(major_idx);
            local_degree0      = edge_partition.local_degree(major_idx);
          }
        } else {
          auto major_idx     = edge_partition.major_offset_from_major_nocheck(major);
          local_edge_offset0 = edge_partition.local_offset(major_idx);
          local_degree0      = edge_partition.local_degree(major_idx);
        }
      } else {
        auto major_idx     = edge_partition.major_offset_from_major_nocheck(major);
        local_edge_offset0 = edge_partition.local_offset(major_idx);
        local_degree0      = edge_partition.local_degree(major_idx);
      }
    } else {
      indices0 = first_element_indices.begin();
      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        edge_property_values0 = first_element_edge_property_values;
      }

      auto idx           = first_element_to_idx_map.find(cuda::std::get<0>(pair));
      local_edge_offset0 = first_element_offsets[idx];
      local_degree0      = static_cast<edge_t>(first_element_offsets[idx + 1] - local_edge_offset0);
    }

    vertex_t const* indices1{};
    std::conditional_t<!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>,
                       edge_property_value_t const*,
                       void*>
      edge_property_values1{};

    edge_t local_edge_offset1{0};
    edge_t local_degree1{0};
    if constexpr (std::is_same_v<SecondElementToIdxMap, void*>) {
      indices1 = edge_partition.indices();
      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        edge_property_values1 = edge_partition_e_value_input.value_first();
      }

      vertex_t major = cuda::std::get<1>(pair);
      if constexpr (multi_gpu) {
        if (edge_partition.major_hypersparse_first() &&
            (major >= *(edge_partition.major_hypersparse_first()))) {
          auto major_hypersparse_idx =
            edge_partition.major_hypersparse_idx_from_major_nocheck(major);
          if (major_hypersparse_idx) {
            auto major_idx =
              (*(edge_partition.major_hypersparse_first()) - edge_partition.major_range_first()) +
              *major_hypersparse_idx;
            local_edge_offset1 = edge_partition.local_offset(major_idx);
            local_degree1      = edge_partition.local_degree(major_idx);
          }
        } else {
          auto major_idx     = edge_partition.major_offset_from_major_nocheck(major);
          local_edge_offset1 = edge_partition.local_offset(major_idx);
          local_degree1      = edge_partition.local_degree(major_idx);
        }
      } else {
        auto major_idx     = edge_partition.major_offset_from_major_nocheck(major);
        local_edge_offset1 = edge_partition.local_offset(major_idx);
        local_degree1      = edge_partition.local_degree(major_idx);
      }
    } else {
      indices1 = second_element_indices.begin();
      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        edge_property_values1 = second_element_edge_property_values;
      }

      auto idx           = second_element_to_idx_map.find(cuda::std::get<1>(pair));
      local_edge_offset1 = second_element_offsets[idx];
      local_degree1 = static_cast<edge_t>(second_element_offsets[idx + 1] - local_edge_offset1);
    }

    // FIXME: this can lead to thread-divergence with a mix of high-degree and low-degree
    // vertices in a single warp (better optimize if this becomes a performance
    // bottleneck)

    edge_t intersection_size{};
    if (edge_partition_e_mask) {
      intersection_size =
        set_intersection_by_key_with_mask<true>(indices0,
                                                indices1,
                                                edge_property_values0,
                                                edge_property_values1,
                                                (*edge_partition_e_mask).value_first(),
                                                nbr_intersection_indices.begin(),
                                                nbr_intersection_e_property_values0,
                                                nbr_intersection_e_property_values1,
                                                local_edge_offset0,
                                                local_degree0,
                                                std::is_same_v<FirstElementToIdxMap, void*>,
                                                local_edge_offset1,
                                                local_degree1,
                                                std::is_same_v<SecondElementToIdxMap, void*>,
                                                nbr_intersection_offsets[i]);
    } else {
      intersection_size =
        set_intersection_by_key_with_mask<false>(indices0,
                                                 indices1,
                                                 edge_property_values0,
                                                 edge_property_values1,
                                                 static_cast<uint32_t const*>(nullptr),
                                                 nbr_intersection_indices.begin(),
                                                 nbr_intersection_e_property_values0,
                                                 nbr_intersection_e_property_values1,
                                                 local_edge_offset0,
                                                 local_degree0,
                                                 false,
                                                 local_edge_offset1,
                                                 local_degree1,
                                                 false,
                                                 nbr_intersection_offsets[i]);
    }

    thrust::fill(
      thrust::seq,
      nbr_intersection_indices.begin() + (nbr_intersection_offsets[i] + intersection_size),
      nbr_intersection_indices.begin() + nbr_intersection_offsets[i + 1],
      invalid_id);

    return intersection_size;
  }
};

template <typename edge_t>
struct strided_accumulate_t {
  raft::device_span<edge_t const> rx_nbr_intersection_sizes{};
  size_t edge_partition_input_size{};
  int minor_comm_size{};

  __device__ edge_t operator()(size_t i) const
  {
    edge_t accumulated_size{0};
    for (int j = 0; j < minor_comm_size; ++j) {
      accumulated_size += rx_nbr_intersection_sizes[edge_partition_input_size * j + i];
    }
    return accumulated_size;
  }
};

template <typename vertex_t,
          typename edge_property_value_t,
          typename optional_property_buffer_view_t,
          typename optional_property_buffer_mutable_view_t>
struct gatherv_indices_t {
  size_t output_size{};
  int minor_comm_size{};

  raft::device_span<size_t const> gathered_intersection_offsets{};
  raft::device_span<vertex_t const> gathered_intersection_indices{};
  raft::device_span<size_t const> combined_nbr_intersection_offsets{};
  raft::device_span<vertex_t> combined_nbr_intersection_indices{};

  optional_property_buffer_view_t gathered_nbr_intersection_e_property_values0{};
  optional_property_buffer_view_t gathered_nbr_intersection_e_property_values1{};
  optional_property_buffer_mutable_view_t combined_nbr_intersection_e_property_values0{};
  optional_property_buffer_mutable_view_t combined_nbr_intersection_e_property_values1{};

  __device__ void operator()(size_t i) const
  {
    auto output_offset = combined_nbr_intersection_offsets[i];

    // FIXME: this can lead to thread-divergence with a mix of high-degree and low-degree vertices
    // in a single warp (better optimize if this becomes a performance bottleneck)

    for (int j = 0; j < minor_comm_size; ++j) {
      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        auto zipped_gathered_begin = thrust::make_zip_iterator(
          cuda::std::make_tuple(gathered_intersection_indices.begin(),
                                gathered_nbr_intersection_e_property_values0,
                                gathered_nbr_intersection_e_property_values1));

        auto zipped_combined_begin = thrust::make_zip_iterator(
          cuda::std::make_tuple(combined_nbr_intersection_indices.begin(),
                                combined_nbr_intersection_e_property_values0,
                                combined_nbr_intersection_e_property_values1));

        thrust::copy(thrust::seq,
                     zipped_gathered_begin + gathered_intersection_offsets[output_size * j + i],
                     zipped_gathered_begin + gathered_intersection_offsets[output_size * j + i + 1],
                     zipped_combined_begin + output_offset);
      } else {
        thrust::copy(thrust::seq,
                     gathered_intersection_indices.begin() +
                       gathered_intersection_offsets[output_size * j + i],
                     gathered_intersection_indices.begin() +
                       gathered_intersection_offsets[output_size * j + i + 1],
                     combined_nbr_intersection_indices.begin() + output_offset);
      }
      output_offset += gathered_intersection_offsets[output_size * j + i + 1] -
                       gathered_intersection_offsets[output_size * j + i];
    }
  }
};

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
// cuda::std::distance(vertex_pair_first, vertex_pair_last) should be comparable across the global
// communicator. If we need to build the neighbor lists, grouping based on applying "vertex ID %
// number of groups"  is recommended for load-balancing.
template <typename GraphViewType, typename VertexPairIterator, typename EdgeValueInputWrapper>
std::conditional_t<
  !std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
  std::tuple<rmm::device_uvector<size_t>,
             rmm::device_uvector<typename GraphViewType::vertex_type>,
             rmm::device_uvector<typename EdgeValueInputWrapper::value_type>,
             rmm::device_uvector<typename EdgeValueInputWrapper::value_type>>,
  std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<typename GraphViewType::vertex_type>>>
nbr_intersection(raft::handle_t const& handle,
                 GraphViewType const& graph_view,
                 EdgeValueInputWrapper edge_value_input,
                 VertexPairIterator vertex_pair_first,
                 VertexPairIterator vertex_pair_last,
                 std::array<bool, 2> intersect_minor_nbr,
                 bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  using edge_property_value_t = typename EdgeValueInputWrapper::value_type;

  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_iterator, void*>,
    std::conditional_t<
      std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
      detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
      detail::edge_partition_edge_multi_index_property_device_view_t<edge_t, vertex_t>>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;

  using optional_property_buffer_value_type =
    std::conditional_t<!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>,
                       edge_property_value_t,
                       void>;

  using optional_property_buffer_view_t =
    std::conditional_t<!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>,
                       edge_property_value_t const*,
                       void*>;
  using optional_property_buffer_mutable_view_t =
    std::conditional_t<!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>,
                       edge_property_value_t*,
                       void*>;

  static_assert(std::is_same_v<typename thrust::iterator_traits<VertexPairIterator>::value_type,
                               cuda::std::tuple<vertex_t, vertex_t>>);

  size_t input_size = static_cast<size_t>(cuda::std::distance(vertex_pair_first, vertex_pair_last));

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

  // 2. Collect neighbor lists (within the minor range for this GPU in multi-GPU) for unique second
  // pair elements (all-gathered over minor_comm in multi-GPU); Note that no need to collect for
  // first pair elements as they already locally reside.

  auto edge_mask_view = graph_view.edge_mask_view();

  std::optional<std::unique_ptr<kv_store_t<vertex_t, vertex_t, false>>> major_to_idx_map_ptr{
    std::nullopt};  // idx to major_nbr_offsets
  std::optional<rmm::device_uvector<edge_t>> major_nbr_offsets{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> major_nbr_indices{std::nullopt};

  [[maybe_unused]] auto major_e_property_values =
    cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
      0, handle.get_stream());

  if constexpr (GraphViewType::is_multi_gpu) {
    if (intersect_minor_nbr[1]) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto const major_comm_rank = major_comm.get_rank();

      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      // 2.1 Find unique second pair element majors

      rmm::device_uvector<vertex_t> unique_majors(input_size, handle.get_stream());
      {
        auto second_element_first = cuda::make_transform_iterator(
          vertex_pair_first, thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, size_t{1}>{});
        thrust::copy(handle.get_thrust_policy(),
                     second_element_first,
                     second_element_first + input_size,
                     unique_majors.begin());

        thrust::sort(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end());
        unique_majors.resize(
          cuda::std::distance(
            unique_majors.begin(),
            thrust::unique(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end())),
          handle.get_stream());

        unique_majors.shrink_to_fit(handle.get_stream());

        if (minor_comm_size > 1) {
          // FIXME: We may refactor this code to improve scalability. We may call multiple gatherv
          // calls, perform local sort and unique, and call multiple broadcasts rather than
          // performing sort and unique for the entire range in every GPU in minor_comm.
          auto rx_counts =
            host_scalar_allgather(minor_comm, unique_majors.size(), handle.get_stream());
          std::vector<size_t> rx_displacements(rx_counts.size());
          std::exclusive_scan(
            rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
          rmm::device_uvector<vertex_t> rx_unique_majors(rx_displacements.back() + rx_counts.back(),
                                                         handle.get_stream());
          cugraph::device_allgatherv(
            minor_comm,
            unique_majors.begin(),
            rx_unique_majors.begin(),
            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
            raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
            handle.get_stream());
          unique_majors = std::move(rx_unique_majors);

          thrust::sort(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end());
          unique_majors.resize(cuda::std::distance(unique_majors.begin(),
                                                   thrust::unique(handle.get_thrust_policy(),
                                                                  unique_majors.begin(),
                                                                  unique_majors.end())),
                               handle.get_stream());

          unique_majors.shrink_to_fit(handle.get_stream());
        }
      }

      // 2.2 Send majors and group (major_comm_rank, local edge_partition_idx) counts

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
            raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                              d_vertex_partition_range_lasts.size()),
            major_comm_size,
            minor_comm_size},
          comm_size,
          std::numeric_limits<size_t>::max(),
          handle.get_stream());
        std::vector<size_t> h_tx_group_counts(d_tx_group_counts.size());
        raft::update_host(h_tx_group_counts.data(),
                          d_tx_group_counts.data(),
                          d_tx_group_counts.size(),
                          handle.get_stream());
        handle.sync_stream();

        std::vector<size_t> tx_counts(major_comm_size, size_t{0});
        for (size_t i = 0; i < tx_counts.size(); ++i) {
          tx_counts[i] = std::reduce(h_tx_group_counts.begin() + minor_comm_size * i,
                                     h_tx_group_counts.begin() + minor_comm_size * (i + 1),
                                     size_t{0});
        }

        std::tie(rx_majors, rx_major_counts) =
          shuffle_values(major_comm,
                         unique_majors.begin(),
                         raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                         handle.get_stream());

        std::vector<size_t> tmp_counts(major_comm_size, minor_comm_size);
        std::tie(rx_group_counts, std::ignore) =
          shuffle_values(major_comm,
                         d_tx_group_counts.begin(),
                         raft::host_span<size_t const>(tmp_counts.data(), tmp_counts.size()),
                         handle.get_stream());
      }

      // 2.3. Enumerate degrees and neighbors for the received majors

      rmm::device_uvector<edge_t> local_degrees_for_rx_majors(size_t{0}, handle.get_stream());
      rmm::device_uvector<vertex_t> local_nbrs_for_rx_majors(size_t{0}, handle.get_stream());

      [[maybe_unused]] auto local_e_property_values_for_rx_majors =
        cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
          0, handle.get_stream());

      std::vector<size_t> local_nbr_counts{};
      {
        rmm::device_uvector<size_t> rx_reordered_group_counts(
          rx_group_counts.size(),
          handle.get_stream());  // reorder using local edge partition index as the primary key and
                                 // major_comm_rank as the secondary key
        thrust::tabulate(handle.get_thrust_policy(),
                         rx_reordered_group_counts.begin(),
                         rx_reordered_group_counts.end(),
                         reorder_group_count_t{major_comm_size,
                                               minor_comm_size,
                                               raft::device_span<size_t const>(
                                                 rx_group_counts.data(), rx_group_counts.size())});

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
          auto edge_partition_e_mask =
            edge_mask_view
              ? cuda::std::make_optional<
                  detail::
                    edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
                  *edge_mask_view, i)
              : cuda::std::nullopt;
          auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
          auto reordered_idx_first =
            (i == size_t{0}) ? size_t{0} : h_rx_reordered_group_lasts[i * major_comm_size - 1];
          auto reordered_idx_last = h_rx_reordered_group_lasts[(i + 1) * major_comm_size - 1];
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(reordered_idx_first),
            thrust::make_counting_iterator(reordered_idx_last),
            update_rx_major_local_degree_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>{
              major_comm_size,
              minor_comm_size,
              edge_partition,
              edge_partition_e_mask,
              reordered_idx_first,
              i,
              raft::device_span<size_t const>(
                d_rx_reordered_group_lasts.data() + i * major_comm_size, major_comm_size),
              raft::device_span<size_t const>(rx_group_firsts.data(), rx_group_firsts.size()),
              raft::device_span<vertex_t const>(rx_majors.data(), rx_majors.size()),
              raft::device_span<edge_t>(local_degrees_for_rx_majors.data(),
                                        local_degrees_for_rx_majors.size())});
        }

        rmm::device_uvector<size_t> local_nbr_offsets_for_rx_majors(
          local_degrees_for_rx_majors.size() + 1, handle.get_stream());
        local_nbr_offsets_for_rx_majors.set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto degree_first = cuda::make_transform_iterator(local_degrees_for_rx_majors.begin(),
                                                          detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               degree_first,
                               degree_first + local_degrees_for_rx_majors.size(),
                               local_nbr_offsets_for_rx_majors.begin() + 1);

        local_nbrs_for_rx_majors.resize(
          local_nbr_offsets_for_rx_majors.back_element(handle.get_stream()), handle.get_stream());

        optional_property_buffer_mutable_view_t optional_local_e_property_values{};

        if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
          local_e_property_values_for_rx_majors.resize(local_nbrs_for_rx_majors.size(),
                                                       handle.get_stream());
          optional_local_e_property_values = local_e_property_values_for_rx_majors.data();
        }

        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          auto edge_partition =
            edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
              graph_view.local_edge_partition_view(i));
          auto edge_partition_e_value_input =
            edge_partition_e_input_device_view_t(edge_value_input, i);
          auto edge_partition_e_mask =
            edge_mask_view
              ? cuda::std::make_optional<
                  detail::
                    edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
                  *edge_mask_view, i)
              : cuda::std::nullopt;

          auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
          auto reordered_idx_first =
            (i == size_t{0}) ? size_t{0} : h_rx_reordered_group_lasts[i * major_comm_size - 1];
          auto reordered_idx_last = h_rx_reordered_group_lasts[(i + 1) * major_comm_size - 1];

          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(reordered_idx_first),
            thrust::make_counting_iterator(reordered_idx_last),
            update_rx_major_local_nbrs_t<vertex_t,
                                         edge_t,
                                         edge_partition_e_input_device_view_t,
                                         optional_property_buffer_mutable_view_t,
                                         GraphViewType::is_multi_gpu>{
              major_comm_size,
              minor_comm_size,
              edge_partition,
              edge_partition_e_value_input,
              edge_partition_e_mask,
              reordered_idx_first,
              i,
              raft::device_span<size_t const>(
                d_rx_reordered_group_lasts.data() + i * major_comm_size, major_comm_size),
              raft::device_span<size_t const>(rx_group_firsts.data(), rx_group_firsts.size()),
              raft::device_span<vertex_t const>(rx_majors.data(), rx_majors.size()),
              raft::device_span<size_t const>(local_nbr_offsets_for_rx_majors.data(),
                                              local_nbr_offsets_for_rx_majors.size()),
              raft::device_span<vertex_t>(local_nbrs_for_rx_majors.data(),
                                          local_nbrs_for_rx_majors.size()),
              optional_local_e_property_values});
        }

        std::vector<size_t> h_rx_offsets(rx_major_counts.size() + size_t{1}, size_t{0});
        std::inclusive_scan(
          rx_major_counts.begin(), rx_major_counts.end(), h_rx_offsets.begin() + 1);
        rmm::device_uvector<size_t> d_rx_offsets(h_rx_offsets.size(), handle.get_stream());
        raft::update_device(
          d_rx_offsets.data(), h_rx_offsets.data(), h_rx_offsets.size(), handle.get_stream());
        rmm::device_uvector<size_t> d_local_nbr_counts(rx_major_counts.size(), handle.get_stream());
        thrust::tabulate(
          handle.get_thrust_policy(),
          d_local_nbr_counts.begin(),
          d_local_nbr_counts.end(),
          compute_local_nbr_count_per_rank_t{
            raft::device_span<size_t const>(d_rx_offsets.data(), d_rx_offsets.size()),
            raft::device_span<size_t const>(local_nbr_offsets_for_rx_majors.data(),
                                            local_nbr_offsets_for_rx_majors.size())});
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
          major_comm,
          local_degrees_for_rx_majors.begin(),
          raft::host_span<size_t const>(rx_major_counts.data(), rx_major_counts.size()),
          handle.get_stream());
        major_nbr_offsets = rmm::device_uvector<edge_t>(local_degrees_for_unique_majors.size() + 1,
                                                        handle.get_stream());
        (*major_nbr_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto degree_first = cuda::make_transform_iterator(local_degrees_for_unique_majors.begin(),
                                                          detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               degree_first,
                               degree_first + local_degrees_for_unique_majors.size(),
                               (*major_nbr_offsets).begin() + 1);
      }

      std::tie(major_nbr_indices, std::ignore) = shuffle_values(
        major_comm,
        local_nbrs_for_rx_majors.begin(),
        raft::host_span<size_t const>(local_nbr_counts.data(), local_nbr_counts.size()),
        handle.get_stream());

      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        std::tie(major_e_property_values, std::ignore) = shuffle_values(
          major_comm,
          local_e_property_values_for_rx_majors.begin(),
          raft::host_span<size_t const>(local_nbr_counts.data(), local_nbr_counts.size()),
          handle.get_stream());
      }

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
    std::nullopt};  // idx to minor_nbr_offsets
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

  [[maybe_unused]] auto nbr_intersection_e_property_values0 =
    cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
      0, handle.get_stream());

  [[maybe_unused]] auto nbr_intersection_e_property_values1 =
    cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
      0, handle.get_stream());

  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    std::vector<size_t> input_counts(minor_comm_size);
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

      rmm::device_uvector<size_t> d_lasts(minor_comm_size, handle.get_stream());
      auto first_element_first = cuda::make_transform_iterator(
        vertex_pair_first, thrust_tuple_get<cuda::std::tuple<vertex_t, vertex_t>, size_t{0}>{});
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

    [[maybe_unused]] std::conditional_t<
      !std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>,
      std::vector<rmm::device_uvector<edge_property_value_t>>,
      std::byte /* dummy */> edge_partition_nbr_intersection_e_property_values0{};
    [[maybe_unused]] std::conditional_t<
      !std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>,
      std::vector<rmm::device_uvector<edge_property_value_t>>,
      std::byte /* dummy */> edge_partition_nbr_intersection_e_property_values1{};

    if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
      edge_partition_nbr_intersection_e_property_values0.reserve(
        graph_view.number_of_local_edge_partitions());
      edge_partition_nbr_intersection_e_property_values1.reserve(
        graph_view.number_of_local_edge_partitions());
    }

    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      auto rx_v_pair_counts =
        host_scalar_allgather(minor_comm, input_counts[i], handle.get_stream());
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

      [[maybe_unused]] auto rx_v_pair_nbr_intersection_e_property_values0 =
        cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
          0, handle.get_stream());

      [[maybe_unused]] auto rx_v_pair_nbr_intersection_e_property_values1 =
        cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
          0, handle.get_stream());

      std::vector<size_t> rx_v_pair_nbr_intersection_index_tx_counts(size_t{0});
      {
        auto vertex_pair_buffer = allocate_dataframe_buffer<cuda::std::tuple<vertex_t, vertex_t>>(
          aggregate_rx_v_pair_size, handle.get_stream());

        thrust::copy(handle.get_thrust_policy(),
                     vertex_pair_first + (i == size_t{0} ? size_t{0} : input_lasts[i - 1]),
                     vertex_pair_first + input_lasts[i],
                     get_dataframe_buffer_begin(vertex_pair_buffer) +
                       rx_v_pair_displacements[minor_comm_rank]);

        cugraph::device_allgatherv(
          minor_comm,
          get_dataframe_buffer_begin(vertex_pair_buffer) + rx_v_pair_displacements[minor_comm_rank],
          get_dataframe_buffer_begin(vertex_pair_buffer),
          raft::host_span<size_t const>(rx_v_pair_counts.data(), rx_v_pair_counts.size()),
          raft::host_span<size_t const>(rx_v_pair_displacements.data(),
                                        rx_v_pair_displacements.size()),
          handle.get_stream());

        auto edge_partition =
          edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
            graph_view.local_edge_partition_view(i));
        auto edge_partition_e_value_input =
          edge_partition_e_input_device_view_t(edge_value_input, i);
        auto edge_partition_e_mask =
          edge_mask_view
            ? cuda::std::make_optional<
                detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
                *edge_mask_view, i)
            : cuda::std::nullopt;

        auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);

        rx_v_pair_nbr_intersection_sizes.resize(
          aggregate_rx_v_pair_size,
          handle
            .get_stream());  // initially store minimum degrees (upper bound for intersection sizes)
        if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
          auto second_element_to_idx_map =
            detail::kv_cuco_store_find_device_view_t((*major_to_idx_map_ptr)->view());
          thrust::transform(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(vertex_pair_buffer),
            get_dataframe_buffer_end(vertex_pair_buffer),
            rx_v_pair_nbr_intersection_sizes.begin(),
            pick_min_degree_t<void*, decltype(second_element_to_idx_map), vertex_t, edge_t, true>{
              nullptr,
              raft::device_span<edge_t const>(),
              second_element_to_idx_map,
              raft::device_span<edge_t const>((*major_nbr_offsets).data(),
                                              (*major_nbr_offsets).size()),
              edge_partition,
              edge_partition_e_mask});
        } else {
          CUGRAPH_FAIL("unimplemented.");
        }

        rmm::device_uvector<size_t> rx_v_pair_nbr_intersection_offsets(
          rx_v_pair_nbr_intersection_sizes.size() + 1, handle.get_stream());
        rx_v_pair_nbr_intersection_offsets.set_element_to_zero_async(size_t{0},
                                                                     handle.get_stream());
        auto size_first = cuda::make_transform_iterator(
          rx_v_pair_nbr_intersection_sizes.begin(), cugraph::detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               size_first,
                               size_first + rx_v_pair_nbr_intersection_sizes.size(),
                               rx_v_pair_nbr_intersection_offsets.begin() + 1);

        rx_v_pair_nbr_intersection_indices.resize(
          rx_v_pair_nbr_intersection_offsets.back_element(handle.get_stream()),
          handle.get_stream());

        optional_property_buffer_mutable_view_t
          rx_v_pair_optional_nbr_intersection_e_property_values0{};
        optional_property_buffer_mutable_view_t
          rx_v_pair_optional_nbr_intersection_e_property_values1{};

        if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
          rx_v_pair_nbr_intersection_e_property_values0.resize(
            rx_v_pair_nbr_intersection_indices.size(), handle.get_stream());
          rx_v_pair_nbr_intersection_e_property_values1.resize(
            rx_v_pair_nbr_intersection_indices.size(), handle.get_stream());

          rx_v_pair_optional_nbr_intersection_e_property_values0 =
            rx_v_pair_nbr_intersection_e_property_values0.data();

          rx_v_pair_optional_nbr_intersection_e_property_values1 =
            rx_v_pair_nbr_intersection_e_property_values1.data();
        }

        if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
          optional_property_buffer_view_t optional_major_e_property_values{};
          if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
            optional_major_e_property_values = major_e_property_values.data();
          }

          auto second_element_to_idx_map =
            detail::kv_cuco_store_find_device_view_t((*major_to_idx_map_ptr)->view());
          thrust::tabulate(
            handle.get_thrust_policy(),
            rx_v_pair_nbr_intersection_sizes.begin(),
            rx_v_pair_nbr_intersection_sizes.end(),
            copy_intersecting_nbrs_and_update_intersection_size_t<
              void*,
              decltype(second_element_to_idx_map),
              decltype(get_dataframe_buffer_begin(vertex_pair_buffer)),
              vertex_t,
              edge_t,
              edge_partition_e_input_device_view_t,
              optional_property_buffer_view_t,
              optional_property_buffer_mutable_view_t,
              true>{nullptr,
                    raft::device_span<edge_t const>(),
                    raft::device_span<vertex_t const>(),
                    optional_property_buffer_view_t{},
                    second_element_to_idx_map,
                    raft::device_span<edge_t const>((*major_nbr_offsets).data(),
                                                    (*major_nbr_offsets).size()),
                    raft::device_span<vertex_t const>((*major_nbr_indices).data(),
                                                      (*major_nbr_indices).size()),
                    optional_major_e_property_values,
                    edge_partition,
                    edge_partition_e_value_input,
                    edge_partition_e_mask,
                    get_dataframe_buffer_begin(vertex_pair_buffer),
                    raft::device_span<size_t const>(rx_v_pair_nbr_intersection_offsets.data(),
                                                    rx_v_pair_nbr_intersection_offsets.size()),
                    raft::device_span<vertex_t>(rx_v_pair_nbr_intersection_indices.data(),
                                                rx_v_pair_nbr_intersection_indices.size()),
                    rx_v_pair_optional_nbr_intersection_e_property_values0,
                    rx_v_pair_optional_nbr_intersection_e_property_values1,

                    invalid_vertex_id<vertex_t>::value});
        } else {
          CUGRAPH_FAIL("unimplemented.");
        }

        if constexpr (std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
          rx_v_pair_nbr_intersection_indices.resize(
            cuda::std::distance(rx_v_pair_nbr_intersection_indices.begin(),
                                thrust::remove(handle.get_thrust_policy(),
                                               rx_v_pair_nbr_intersection_indices.begin(),
                                               rx_v_pair_nbr_intersection_indices.end(),
                                               invalid_vertex_id<vertex_t>::value)),
            handle.get_stream());
          rx_v_pair_nbr_intersection_indices.shrink_to_fit(handle.get_stream());
        } else {
          auto common_nbr_and_e_property_values_begin = thrust::make_zip_iterator(
            cuda::std::make_tuple(rx_v_pair_nbr_intersection_indices.begin(),
                                  rx_v_pair_nbr_intersection_e_property_values0.begin(),
                                  rx_v_pair_nbr_intersection_e_property_values1.begin()));

          auto last = thrust::remove_if(
            handle.get_thrust_policy(),
            common_nbr_and_e_property_values_begin,
            common_nbr_and_e_property_values_begin + rx_v_pair_nbr_intersection_indices.size(),
            [] __device__(auto nbr_p0_p1) {
              return cuda::std::get<0>(nbr_p0_p1) == invalid_vertex_id<vertex_t>::value;
            });

          rx_v_pair_nbr_intersection_indices.resize(
            cuda::std::distance(common_nbr_and_e_property_values_begin, last), handle.get_stream());

          rx_v_pair_nbr_intersection_indices.shrink_to_fit(handle.get_stream());

          rx_v_pair_nbr_intersection_e_property_values0.resize(
            rx_v_pair_nbr_intersection_indices.size(), handle.get_stream());
          rx_v_pair_nbr_intersection_e_property_values0.shrink_to_fit(handle.get_stream());

          rx_v_pair_nbr_intersection_e_property_values1.resize(
            rx_v_pair_nbr_intersection_indices.size(), handle.get_stream());
          rx_v_pair_nbr_intersection_e_property_values1.shrink_to_fit(handle.get_stream());
        }

        thrust::inclusive_scan(handle.get_thrust_policy(),
                               size_first,
                               size_first + rx_v_pair_nbr_intersection_sizes.size(),
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
        std::vector<int> ranks(minor_comm_size);
        std::iota(ranks.begin(), ranks.end(), int{0});
        std::vector<size_t> displacements(minor_comm_size);
        for (int i = 0; i < minor_comm_size; ++i) {
          displacements[i] = rx_v_pair_counts[minor_comm_rank] * i;
        }
        rmm::device_uvector<edge_t> gathered_nbr_intersection_sizes(
          rx_v_pair_counts[minor_comm_rank] * minor_comm_size, handle.get_stream());
        std::vector<size_t> rx_counts(minor_comm_size, rx_v_pair_counts[minor_comm_rank]);
        device_multicast_sendrecv(
          minor_comm,
          rx_v_pair_nbr_intersection_sizes.begin(),
          raft::host_span<size_t const>(rx_v_pair_counts.data(), rx_v_pair_counts.size()),
          raft::host_span<size_t const>(rx_v_pair_displacements.data(),
                                        rx_v_pair_displacements.size()),
          raft::host_span<int const>(ranks.data(), ranks.size()),
          gathered_nbr_intersection_sizes.begin(),
          raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
          raft::host_span<size_t const>(displacements.data(), displacements.size()),
          raft::host_span<int const>(ranks.data(), ranks.size()),
          handle.get_stream());
        rx_v_pair_nbr_intersection_sizes.resize(size_t{0}, handle.get_stream());
        rx_v_pair_nbr_intersection_sizes.shrink_to_fit(handle.get_stream());

        combined_nbr_intersection_sizes.resize(rx_v_pair_counts[minor_comm_rank],
                                               handle.get_stream());

        thrust::tabulate(handle.get_thrust_policy(),
                         combined_nbr_intersection_sizes.begin(),
                         combined_nbr_intersection_sizes.end(),
                         strided_accumulate_t<edge_t>{
                           raft::device_span<edge_t const>(gathered_nbr_intersection_sizes.data(),
                                                           gathered_nbr_intersection_sizes.size()),
                           rx_v_pair_counts[minor_comm_rank],
                           minor_comm_size});

        combined_nbr_intersection_offsets.resize(rx_v_pair_counts[minor_comm_rank] + 1,
                                                 handle.get_stream());
        combined_nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto combined_size_first = cuda::make_transform_iterator(
          combined_nbr_intersection_sizes.begin(), detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               combined_size_first,
                               combined_size_first + combined_nbr_intersection_sizes.size(),
                               combined_nbr_intersection_offsets.begin() + 1);

        gathered_nbr_intersection_offsets.resize(gathered_nbr_intersection_sizes.size() + 1,
                                                 handle.get_stream());
        gathered_nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
        auto gathered_size_first = cuda::make_transform_iterator(
          gathered_nbr_intersection_sizes.begin(), detail::typecast_t<edge_t, size_t>{});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               gathered_size_first,
                               gathered_size_first + gathered_nbr_intersection_sizes.size(),
                               gathered_nbr_intersection_offsets.begin() + 1);

        auto map_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{1}),
          detail::multiplier_t<size_t>{rx_v_pair_counts[minor_comm_rank]});
        rmm::device_uvector<size_t> d_lasts(minor_comm_size, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       map_first,
                       map_first + minor_comm_size,
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

      [[maybe_unused]] auto combined_nbr_intersection_e_property_values0 =
        cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
          size_t{0}, handle.get_stream());

      [[maybe_unused]] auto combined_nbr_intersection_e_property_values1 =
        cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
          size_t{0}, handle.get_stream());

      {
        std::vector<int> ranks(minor_comm_size);
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
        device_multicast_sendrecv(
          minor_comm,
          rx_v_pair_nbr_intersection_indices.begin(),
          raft::host_span<size_t const>(rx_v_pair_nbr_intersection_index_tx_counts.data(),
                                        rx_v_pair_nbr_intersection_index_tx_counts.size()),
          raft::host_span<size_t const>(tx_displacements.data(), tx_displacements.size()),
          raft::host_span<int const>(ranks.data(), ranks.size()),
          gathered_nbr_intersection_indices.begin(),
          raft::host_span<size_t const>(gathered_nbr_intersection_index_rx_counts.data(),
                                        gathered_nbr_intersection_index_rx_counts.size()),
          raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
          raft::host_span<int const>(ranks.data(), ranks.size()),
          handle.get_stream());
        rx_v_pair_nbr_intersection_indices.resize(size_t{0}, handle.get_stream());
        rx_v_pair_nbr_intersection_indices.shrink_to_fit(handle.get_stream());

        combined_nbr_intersection_indices.resize(gathered_nbr_intersection_indices.size(),
                                                 handle.get_stream());

        [[maybe_unused]] auto gathered_nbr_intersection_e_property_values0 =
          cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
            rx_displacements.back() + gathered_nbr_intersection_index_rx_counts.back(),
            handle.get_stream());

        [[maybe_unused]] auto gathered_nbr_intersection_e_property_values1 =
          cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
            rx_displacements.back() + gathered_nbr_intersection_index_rx_counts.back(),
            handle.get_stream());

        if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
          device_multicast_sendrecv(
            minor_comm,
            rx_v_pair_nbr_intersection_e_property_values0.begin(),
            raft::host_span<size_t const>(rx_v_pair_nbr_intersection_index_tx_counts.data(),
                                          rx_v_pair_nbr_intersection_index_tx_counts.size()),
            raft::host_span<size_t const>(tx_displacements.data(), tx_displacements.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            gathered_nbr_intersection_e_property_values0.begin(),
            raft::host_span<size_t const>(gathered_nbr_intersection_index_rx_counts.data(),
                                          gathered_nbr_intersection_index_rx_counts.size()),
            raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            handle.get_stream());
          rx_v_pair_nbr_intersection_e_property_values0.resize(size_t{0}, handle.get_stream());
          rx_v_pair_nbr_intersection_e_property_values0.shrink_to_fit(handle.get_stream());

          combined_nbr_intersection_e_property_values0.resize(
            gathered_nbr_intersection_e_property_values0.size(), handle.get_stream());

          device_multicast_sendrecv(
            minor_comm,
            rx_v_pair_nbr_intersection_e_property_values1.begin(),
            raft::host_span<size_t const>(rx_v_pair_nbr_intersection_index_tx_counts.data(),
                                          rx_v_pair_nbr_intersection_index_tx_counts.size()),
            raft::host_span<size_t const>(tx_displacements.data(), tx_displacements.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            gathered_nbr_intersection_e_property_values1.begin(),
            raft::host_span<size_t const>(gathered_nbr_intersection_index_rx_counts.data(),
                                          gathered_nbr_intersection_index_rx_counts.size()),
            raft::host_span<size_t const>(rx_displacements.data(), rx_displacements.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            handle.get_stream());
          rx_v_pair_nbr_intersection_e_property_values1.resize(size_t{0}, handle.get_stream());
          rx_v_pair_nbr_intersection_e_property_values1.shrink_to_fit(handle.get_stream());
          combined_nbr_intersection_e_property_values1.resize(
            gathered_nbr_intersection_e_property_values1.size(), handle.get_stream());
        }

        if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(rx_v_pair_counts[minor_comm_rank]),
            gatherv_indices_t<vertex_t,
                              edge_property_value_t,
                              optional_property_buffer_view_t,
                              optional_property_buffer_mutable_view_t>{
              rx_v_pair_counts[minor_comm_rank],
              minor_comm_size,
              raft::device_span<size_t const>(gathered_nbr_intersection_offsets.data(),
                                              gathered_nbr_intersection_offsets.size()),
              raft::device_span<vertex_t const>(gathered_nbr_intersection_indices.data(),
                                                gathered_nbr_intersection_indices.size()),
              raft::device_span<size_t const>(combined_nbr_intersection_offsets.data(),
                                              combined_nbr_intersection_offsets.size()),
              raft::device_span<vertex_t>(combined_nbr_intersection_indices.data(),
                                          combined_nbr_intersection_indices.size()),
              gathered_nbr_intersection_e_property_values0.data(),
              gathered_nbr_intersection_e_property_values1.data(),
              combined_nbr_intersection_e_property_values0.data(),
              combined_nbr_intersection_e_property_values1.data()});
        } else {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(rx_v_pair_counts[minor_comm_rank]),
            gatherv_indices_t<vertex_t,
                              edge_property_value_t,
                              optional_property_buffer_view_t,
                              optional_property_buffer_mutable_view_t>{
              rx_v_pair_counts[minor_comm_rank],
              minor_comm_size,
              raft::device_span<size_t const>(gathered_nbr_intersection_offsets.data(),
                                              gathered_nbr_intersection_offsets.size()),
              raft::device_span<vertex_t const>(gathered_nbr_intersection_indices.data(),
                                                gathered_nbr_intersection_indices.size()),
              raft::device_span<size_t const>(combined_nbr_intersection_offsets.data(),
                                              combined_nbr_intersection_offsets.size()),
              raft::device_span<vertex_t>(combined_nbr_intersection_indices.data(),
                                          combined_nbr_intersection_indices.size())

            });
        }
      }

      edge_partition_nbr_intersection_sizes.push_back(std::move(combined_nbr_intersection_sizes));
      edge_partition_nbr_intersection_indices.push_back(
        std::move(combined_nbr_intersection_indices));
      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        edge_partition_nbr_intersection_e_property_values0.push_back(
          std::move(combined_nbr_intersection_e_property_values0));
        edge_partition_nbr_intersection_e_property_values1.push_back(
          std::move(combined_nbr_intersection_e_property_values1));
      }
    }

    rmm::device_uvector<edge_t> nbr_intersection_sizes(input_size, handle.get_stream());
    size_t num_nbr_intersection_indices{0};
    for (size_t i = 0; i < edge_partition_nbr_intersection_indices.size(); ++i) {
      num_nbr_intersection_indices += edge_partition_nbr_intersection_indices[i].size();
    }
    nbr_intersection_indices.resize(num_nbr_intersection_indices, handle.get_stream());
    if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
      nbr_intersection_e_property_values0.resize(nbr_intersection_indices.size(),
                                                 handle.get_stream());
      nbr_intersection_e_property_values1.resize(nbr_intersection_indices.size(),
                                                 handle.get_stream());
    }
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

      if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        thrust::copy(handle.get_thrust_policy(),
                     edge_partition_nbr_intersection_e_property_values0[i].begin(),
                     edge_partition_nbr_intersection_e_property_values0[i].end(),
                     nbr_intersection_e_property_values0.begin() + index_offset);

        thrust::copy(handle.get_thrust_policy(),
                     edge_partition_nbr_intersection_e_property_values1[i].begin(),
                     edge_partition_nbr_intersection_e_property_values1[i].end(),
                     nbr_intersection_e_property_values1.begin() + index_offset);
      }

      index_offset += edge_partition_nbr_intersection_indices[i].size();
    }
    nbr_intersection_offsets.resize(nbr_intersection_sizes.size() + size_t{1}, handle.get_stream());
    nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
    auto size_first = cuda::make_transform_iterator(nbr_intersection_sizes.begin(),
                                                    detail::typecast_t<edge_t, size_t>{});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           size_first,
                           size_first + nbr_intersection_sizes.size(),
                           nbr_intersection_offsets.begin() + 1);
  } else {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(size_t{0}));
    auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, 0);
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, 0)
        : cuda::std::nullopt;

    rmm::device_uvector<edge_t> nbr_intersection_sizes(
      input_size,
      handle.get_stream());  // initially store minimum degrees (upper bound for intersection sizes)
    if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
      thrust::transform(
        handle.get_thrust_policy(),
        vertex_pair_first,
        vertex_pair_first + input_size,
        nbr_intersection_sizes.begin(),
        pick_min_degree_t<void*, void*, vertex_t, edge_t, false>{nullptr,
                                                                 raft::device_span<edge_t const>(),
                                                                 nullptr,
                                                                 raft::device_span<edge_t const>(),
                                                                 edge_partition,
                                                                 edge_partition_e_mask});
    } else {
      CUGRAPH_FAIL("unimplemented.");
    }

    nbr_intersection_offsets.resize(nbr_intersection_sizes.size() + 1, handle.get_stream());
    nbr_intersection_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
    auto size_first = cuda::make_transform_iterator(nbr_intersection_sizes.begin(),
                                                    detail::typecast_t<edge_t, size_t>{});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           size_first,
                           size_first + nbr_intersection_sizes.size(),
                           nbr_intersection_offsets.begin() + 1);

    nbr_intersection_indices.resize(nbr_intersection_offsets.back_element(handle.get_stream()),
                                    handle.get_stream());

    optional_property_buffer_mutable_view_t optional_nbr_intersection_e_property_values0{};
    optional_property_buffer_mutable_view_t optional_nbr_intersection_e_property_values1{};

    if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
      nbr_intersection_e_property_values0.resize(nbr_intersection_indices.size(),
                                                 handle.get_stream());
      nbr_intersection_e_property_values1.resize(nbr_intersection_indices.size(),
                                                 handle.get_stream());

      optional_nbr_intersection_e_property_values0 = nbr_intersection_e_property_values0.data();
      optional_nbr_intersection_e_property_values1 = nbr_intersection_e_property_values1.data();
    }

    if (intersect_minor_nbr[0] && intersect_minor_nbr[1]) {
      thrust::tabulate(handle.get_thrust_policy(),
                       nbr_intersection_sizes.begin(),
                       nbr_intersection_sizes.end(),
                       copy_intersecting_nbrs_and_update_intersection_size_t<
                         void*,
                         void*,
                         decltype(vertex_pair_first),
                         vertex_t,
                         edge_t,
                         edge_partition_e_input_device_view_t,
                         optional_property_buffer_view_t,
                         optional_property_buffer_mutable_view_t,
                         false>{nullptr,
                                raft::device_span<edge_t const>(),
                                raft::device_span<vertex_t const>(),
                                optional_property_buffer_view_t{},
                                nullptr,
                                raft::device_span<edge_t const>(),
                                raft::device_span<vertex_t const>(),
                                optional_property_buffer_view_t{},
                                edge_partition,
                                edge_partition_e_value_input,
                                edge_partition_e_mask,
                                vertex_pair_first,
                                raft::device_span<size_t const>(nbr_intersection_offsets.data(),
                                                                nbr_intersection_offsets.size()),
                                raft::device_span<vertex_t>(nbr_intersection_indices.data(),
                                                            nbr_intersection_indices.size()),
                                optional_nbr_intersection_e_property_values0,
                                optional_nbr_intersection_e_property_values1,
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
                       detail::is_not_equal_t<vertex_t>{invalid_vertex_id<vertex_t>::value}),
      handle.get_stream());

    [[maybe_unused]] auto tmp_property_values0 =
      cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
        tmp_indices.size(), handle.get_stream());

    [[maybe_unused]] auto tmp_property_values1 =
      cugraph::detail::allocate_optional_dataframe_buffer<optional_property_buffer_value_type>(
        tmp_indices.size(), handle.get_stream());

    size_t num_copied{0};
    size_t num_scanned{0};

    while (num_scanned < nbr_intersection_indices.size()) {
      size_t this_scan_size = std::min(
        size_t{1} << 27,
        static_cast<size_t>(cuda::std::distance(nbr_intersection_indices.begin() + num_scanned,
                                                nbr_intersection_indices.end())));
      if constexpr (std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
        num_copied += static_cast<size_t>(cuda::std::distance(
          tmp_indices.begin() + num_copied,
          thrust::copy_if(handle.get_thrust_policy(),
                          nbr_intersection_indices.begin() + num_scanned,
                          nbr_intersection_indices.begin() + num_scanned + this_scan_size,
                          tmp_indices.begin() + num_copied,
                          detail::is_not_equal_t<vertex_t>{invalid_vertex_id<vertex_t>::value})));
      } else {
        auto zipped_itr_to_indices_and_e_property_values_begin = thrust::make_zip_iterator(
          cuda::std::make_tuple(nbr_intersection_indices.begin(),
                                nbr_intersection_e_property_values0.begin(),
                                nbr_intersection_e_property_values1.begin()));

        auto zipped_itr_to_tmps_begin = thrust::make_zip_iterator(
          tmp_indices.begin(), tmp_property_values0.begin(), tmp_property_values1.begin());

        num_copied += static_cast<size_t>(cuda::std::distance(
          zipped_itr_to_tmps_begin + num_copied,
          thrust::copy_if(
            handle.get_thrust_policy(),
            zipped_itr_to_indices_and_e_property_values_begin + num_scanned,
            zipped_itr_to_indices_and_e_property_values_begin + num_scanned + this_scan_size,
            zipped_itr_to_tmps_begin + num_copied,
            [] __device__(auto nbr_p0_p1) {
              auto nbr = cuda::std::get<0>(nbr_p0_p1);
              auto p0  = cuda::std::get<1>(nbr_p0_p1);
              auto p1  = cuda::std::get<2>(nbr_p0_p1);
              return cuda::std::get<0>(nbr_p0_p1) != invalid_vertex_id<vertex_t>::value;
            })));
      }
      num_scanned += this_scan_size;
    }
    nbr_intersection_indices = std::move(tmp_indices);
    if constexpr (!std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
      nbr_intersection_e_property_values0 = std::move(tmp_property_values0);
      nbr_intersection_e_property_values1 = std::move(tmp_property_values1);
    }
#else
    if constexpr (std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
      nbr_intersection_indices.resize(
        cuda::std::distance(nbr_intersection_indices.begin(),
                            thrust::remove(handle.get_thrust_policy(),
                                           nbr_intersection_indices.begin(),
                                           nbr_intersection_indices.end(),
                                           invalid_vertex_id<vertex_t>::value)),
        handle.get_stream());
    } else {
      nbr_intersection_indices.resize(
        cuda::std::distance(zipped_itr_to_indices_and_e_property_values_begin,
                            thrust::remove_if(handle.get_thrust_policy(),
                                              zipped_itr_to_indices_and_e_property_values_begin,
                                              zipped_itr_to_indices_and_e_property_values_begin +
                                                nbr_intersection_indices.size(),
                                              [] __device__(auto nbr_p0_p1) {
                                                return cuda::std::get<0>(nbr_p0_p1) ==
                                                       invalid_vertex_id<vertex_t>::value;
                                              })),
        handle.get_stream());

      nbr_intersection_e_property_values0.resize(nbr_intersection_indices.size(),
                                                 handle.get_stream());
      nbr_intersection_e_property_values1.resize(nbr_intersection_indices.size(),
                                                 handle.get_stream());
    }
#endif

    thrust::inclusive_scan(handle.get_thrust_policy(),
                           size_first,
                           size_first + nbr_intersection_sizes.size(),
                           nbr_intersection_offsets.begin() + 1);
  }

  // 5. Return

  if constexpr (std::is_same_v<edge_property_value_t, cuda::std::nullopt_t>) {
    return std::make_tuple(std::move(nbr_intersection_offsets),
                           std::move(nbr_intersection_indices));

  } else {
    return std::make_tuple(std::move(nbr_intersection_offsets),
                           std::move(nbr_intersection_indices),
                           std::move(nbr_intersection_e_property_values0),
                           std::move(nbr_intersection_e_property_values1));
  }
}

}  // namespace detail

}  // namespace CUGRAPH_EXPORT cugraph

/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "prims/count_if_e.cuh"
#include "prims/detail/extract_transform_if_v_frontier_e.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/key_store.cuh"
#include "prims/kv_store.cuh"
#include "prims/reduce_op.cuh"
#include "prims/transform_reduce_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>

CUCO_DECLARE_BITWISE_COMPARABLE(float)
CUCO_DECLARE_BITWISE_COMPARABLE(double)

namespace cugraph {

namespace {

template <typename vertex_t, typename tag_t, typename key_t>
struct aggregate_vi_t {
  tag_t num_origins{};

  __device__ key_t operator()(thrust::tuple<vertex_t, tag_t> tup) const
  {
    return (static_cast<key_t>(thrust::get<0>(tup)) * static_cast<key_t>(num_origins)) +
           static_cast<key_t>(thrust::get<1>(tup));
  }
};

template <typename vertex_t, typename tag_t, typename key_t>
struct split_vi_t {
  tag_t num_origins{};

  __device__ thrust::tuple<vertex_t, tag_t> operator()(key_t aggregated_vi) const
  {
    return thrust::make_tuple(
      static_cast<vertex_t>(aggregated_vi / static_cast<key_t>(num_origins)),
      static_cast<tag_t>(aggregated_vi % static_cast<key_t>(num_origins)));
  }
};

template <typename vertex_t, typename tag_t, typename key_t>
struct extract_v_t {
  tag_t num_origins{};

  __device__ vertex_t operator()(key_t aggregated_vi) const
  {
    return static_cast<vertex_t>(aggregated_vi / static_cast<key_t>(num_origins));
  }
};

template <typename vertex_t, typename od_idx_t>
struct update_v_to_destination_index_t {
  raft::device_span<vertex_t const> destinations{};
  raft::device_span<od_idx_t> v_to_destination_indices{};

  __device__ void operator()(od_idx_t i) const { v_to_destination_indices[destinations[i]] = i; }
};

template <typename vertex_t, typename tag_t, typename key_t>
struct compute_od_matrix_index_t {
  raft::device_span<tag_t const> v_to_destination_indices{};
  tag_t num_origins{};
  tag_t num_destinations{};

  __device__ size_t operator()(key_t aggregated_vi) const
  {
    auto v   = static_cast<vertex_t>(aggregated_vi / static_cast<key_t>(num_origins));
    auto tag = static_cast<size_t>(aggregated_vi % static_cast<key_t>(num_origins));
    return (tag * static_cast<size_t>(num_destinations)) +
           static_cast<size_t>(v_to_destination_indices[v]);
  }
};

template <typename vertex_t, typename tag_t, typename key_t>
struct check_destination_index_t {
  raft::device_span<tag_t const> v_to_destination_indices{};
  tag_t num_origins{};
  tag_t invalid_od_idx{};

  __device__ bool operator()(key_t aggregated_vi) const
  {
    auto v = static_cast<vertex_t>(aggregated_vi / static_cast<key_t>(num_origins));
    return (v_to_destination_indices[v] != invalid_od_idx);
  }
};

template <typename vertex_t, typename tag_t, typename key_t, typename weight_t>
struct e_op_t {
  detail::kv_cuco_store_find_device_view_t<detail::kv_cuco_store_view_t<key_t, weight_t const*>>
    key_to_dist_map{};
  tag_t num_origins{};

  __device__ thrust::tuple<tag_t, weight_t> operator()(thrust::tuple<vertex_t, tag_t> tagged_src,
                                                       vertex_t dst,
                                                       cuda::std::nullopt_t,
                                                       cuda::std::nullopt_t,
                                                       weight_t w) const
  {
    aggregate_vi_t<vertex_t, tag_t, key_t> aggregator{num_origins};

    auto src_val = key_to_dist_map.find(aggregator(tagged_src));
    assert(src_val != invalid_distance);
    auto origin_idx   = thrust::get<1>(tagged_src);
    auto new_distance = src_val + w;
    return thrust::make_tuple(origin_idx, new_distance);
  }
};

template <typename vertex_t, typename tag_t, typename key_t, typename weight_t>
struct pred_op_t {
  detail::kv_cuco_store_find_device_view_t<detail::kv_cuco_store_view_t<key_t, weight_t const*>>
    key_to_dist_map{};
  tag_t num_origins{};
  weight_t cutoff{};
  weight_t invalid_distance{};

  __device__ bool operator()(thrust::tuple<vertex_t, tag_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             weight_t w) const
  {
    aggregate_vi_t<vertex_t, tag_t, key_t> aggregator{num_origins};

    auto src_val = key_to_dist_map.find(aggregator(tagged_src));
    assert(src_val != invalid_distance);
    auto origin_idx   = thrust::get<1>(tagged_src);
    auto new_distance = src_val + w;
    auto threshold    = cutoff;
    auto dst_val      = key_to_dist_map.find(aggregator(thrust::make_tuple(dst, origin_idx)));
    if (dst_val != invalid_distance) { threshold = dst_val < threshold ? dst_val : threshold; }
    return (new_distance < threshold);
  }
};

template <typename vertex_t, typename edge_t, typename tag_t, typename key_t>
struct insert_nbr_key_t {
  raft::device_span<edge_t const> offsets{};
  raft::device_span<vertex_t const> indices{};
  detail::key_cuco_store_insert_device_view_t<detail::key_cuco_store_view_t<key_t>> key_set{};
  aggregate_vi_t<vertex_t, tag_t, key_t> aggregator{};

  __device__ void operator()(thrust::tuple<vertex_t, tag_t> vi)
  {
    auto v   = thrust::get<0>(vi);
    auto idx = thrust::get<1>(vi);

    for (edge_t nbr_offset = offsets[v]; nbr_offset < offsets[v + 1]; ++nbr_offset) {
      auto nbr = indices[nbr_offset];
      key_set.insert(aggregator(thrust::make_tuple(nbr, idx)));
    }
  }
};

template <typename key_t, typename weight_t>
struct keep_t {
  weight_t old_near_far_threshold{};
  detail::key_cuco_store_contains_device_view_t<detail::key_cuco_store_view_t<key_t>> key_set{};

  __device__ bool operator()(thrust::tuple<key_t, weight_t> pair) const
  {
    return (thrust::get<1>(pair) >= old_near_far_threshold) ||
           (key_set.contains(thrust::get<0>(pair)));
  }
};

template <typename key_t, typename weight_t>
struct is_no_smaller_than_threshold_t {
  weight_t threshold{};
  detail::kv_cuco_store_find_device_view_t<detail::kv_cuco_store_view_t<key_t, weight_t const*>>
    key_to_dist_map{};

  __device__ bool operator()(key_t key) const { return key_to_dist_map.find(key) >= threshold; }
};

size_t compute_kv_store_capacity(size_t new_min_size,
                                 size_t old_capacity,
                                 size_t max_capacity_increment)
{
  if (new_min_size <= old_capacity) {
    return old_capacity;  // shrinking the kv_store has little impact in reducing the peak memory
                          // usage (it may have limited benefit in improving cache hit ratio at the
                          // cost of more collisions)
  } else {
    return old_capacity + raft::round_up_safe(new_min_size - old_capacity, max_capacity_increment);
  }
}

int32_t constexpr multi_partition_copy_block_size = 512;  // tuning parameter

template <int32_t max_num_partitions,
          typename InputIterator,
          typename key_t,
          typename PartitionOp,
          typename KeyOp>
__global__ static void multi_partition_copy(
  InputIterator input_first,
  InputIterator input_last,
  raft::device_span<key_t*> output_buffer_ptrs,
  PartitionOp partition_op,  // returns max_num_partitions to discard
  KeyOp key_op,
  raft::device_span<size_t> partition_counters)
{
  static_assert(max_num_partitions <= static_cast<int32_t>(std::numeric_limits<uint8_t>::max()));
  assert(output_buffer_ptrs.size() == partition_counters.size());
  int32_t num_partitions = output_buffer_ptrs.size();
  assert(num_partitions <= max_num_partitions);

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto idx       = static_cast<size_t>(tid);

  int32_t constexpr tmp_buffer_size =
    max_num_partitions;  // tuning parameter (trade-off between parallelism & memory vs # BlockScan
                         // & atomic operations)
  static_assert(
    static_cast<size_t>(multi_partition_copy_block_size) * static_cast<size_t>(tmp_buffer_size) <=
    static_cast<size_t>(
      std::numeric_limits<int32_t>::max()));  // int32_t is sufficient to store the maximum possible
                                              // number of updates per block

  using BlockScan = cub::BlockScan<int32_t, multi_partition_copy_block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ size_t block_start_offsets[max_num_partitions];

  static_assert(tmp_buffer_size <= static_cast<int32_t>(std::numeric_limits<uint8_t>::max()));
  uint8_t tmp_counts[max_num_partitions];
  int32_t tmp_intra_block_offsets[max_num_partitions];

  uint8_t tmp_partitions[tmp_buffer_size];
  uint8_t tmp_offsets[tmp_buffer_size];

  auto num_elems = static_cast<size_t>(cuda::std::distance(input_first, input_last));
  auto rounded_up_num_elems =
    ((num_elems + static_cast<size_t>(blockDim.x - 1)) / static_cast<size_t>(blockDim.x)) *
    static_cast<size_t>(blockDim.x);
  while (idx < rounded_up_num_elems) {
    thrust::fill(thrust::seq, tmp_counts, tmp_counts + num_partitions, int32_t{0});
    auto tmp_idx = idx;
    for (int32_t i = 0; i < tmp_buffer_size; ++i) {
      if (tmp_idx < num_elems) {
        auto partition    = partition_op(*(input_first + tmp_idx));
        tmp_partitions[i] = partition;
        tmp_offsets[i]    = tmp_counts[partition];
        ++tmp_counts[partition];
      }
      tmp_idx += gridDim.x * blockDim.x;
    }
    for (int32_t i = 0; i < num_partitions; ++i) {
      BlockScan(temp_storage)
        .ExclusiveSum(static_cast<int32_t>(tmp_counts[i]), tmp_intra_block_offsets[i]);
    }
    if (threadIdx.x == (blockDim.x - 1)) {
      for (int32_t i = 0; i < num_partitions; ++i) {
        auto increment = static_cast<size_t>(tmp_intra_block_offsets[i] + tmp_counts[i]);
        cuda::atomic_ref<size_t, cuda::thread_scope_device> atomic_counter(partition_counters[i]);
        block_start_offsets[i] =
          atomic_counter.fetch_add(increment, cuda::std::memory_order_relaxed);
      }
    }
    __syncthreads();
    tmp_idx = idx;
    for (int32_t i = 0; i < tmp_buffer_size; ++i) {
      if (tmp_idx < num_elems) {
        auto partition = tmp_partitions[i];
        if (partition != static_cast<uint8_t>(max_num_partitions)) {
          auto offset = block_start_offsets[partition] +
                        static_cast<size_t>(tmp_intra_block_offsets[partition] + tmp_offsets[i]);
          *(output_buffer_ptrs[partition] + offset) = key_op(*(input_first + tmp_idx));
        }
      }
      tmp_idx += gridDim.x * blockDim.x;
    }

    idx += static_cast<size_t>(gridDim.x * blockDim.x) * tmp_buffer_size;
  }
}

template <typename GraphViewType, typename tag_t, typename key_t, typename weight_t>
kv_store_t<key_t, weight_t, false /* use_binary_search */> filter_key_to_dist_map(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  kv_store_t<key_t, weight_t, false /* use_binary_search */>&& key_to_dist_map,
  key_bucket_t<typename GraphViewType::vertex_type,
               tag_t,
               GraphViewType::is_multi_gpu,
               true /* sorted_unique */> const& near_bucket,
  std::vector<raft::device_span<key_t const>> const& far_buffers,
  weight_t old_near_far_threshold,
  size_t min_extra_capacity,  // ensure at least extra_capacity elements can be inserted
  size_t kv_store_capacity_increment,
  size_t num_origins)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto invalid_key      = key_to_dist_map.invalid_key();
  auto invalid_distance = key_to_dist_map.invalid_value();

  auto old_kv_store_capacity = key_to_dist_map.capacity();

  rmm::device_uvector<key_t> old_key_buffer(0, handle.get_stream());
  rmm::device_uvector<weight_t> old_value_buffer(0, handle.get_stream());
  rmm::device_uvector<bool> keep_flags(0, handle.get_stream());
  size_t keep_count{0};
  {
    std::tie(old_key_buffer, old_value_buffer) = key_to_dist_map.release(handle.get_stream());
    keep_flags.resize(old_key_buffer.size(), handle.get_stream());

    // FIXME: better use a higher-level interface than this.
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(0));
    auto num_edges = edge_partition.compute_number_of_edges(
      near_bucket.vertex_begin(), near_bucket.vertex_end(), handle.get_stream());
    for (size_t i = 0; i < far_buffers.size(); ++i) {
      auto far_vertex_first = thrust::make_transform_iterator(
        far_buffers[i].begin(),
        extract_v_t<vertex_t, tag_t, key_t>{static_cast<tag_t>(num_origins)});
      num_edges += edge_partition.compute_number_of_edges(
        far_vertex_first, far_vertex_first + far_buffers[i].size(), handle.get_stream());
    }

    auto key_set = key_store_t<key_t, false /* use_binary_search */>(
      num_edges, invalid_key, handle.get_stream());

    // FIXME: better implement this using a primitive
    auto offsets = graph_view.local_edge_partition_view().offsets();
    auto indices = graph_view.local_edge_partition_view().indices();

    thrust::for_each(handle.get_thrust_policy(),
                     near_bucket.begin(),
                     near_bucket.end(),
                     insert_nbr_key_t<vertex_t, edge_t, tag_t, key_t>{
                       offsets,
                       indices,
                       detail::key_cuco_store_insert_device_view_t(key_set.view()),
                       aggregate_vi_t<vertex_t, tag_t, key_t>{static_cast<tag_t>(num_origins)}});

    for (size_t i = 0; i < far_buffers.size(); ++i) {
      auto far_vi_first = thrust::make_transform_iterator(
        far_buffers[i].begin(),
        split_vi_t<vertex_t, tag_t, key_t>{static_cast<tag_t>(num_origins)});
      thrust::for_each(handle.get_thrust_policy(),
                       far_vi_first,
                       far_vi_first + far_buffers[i].size(),
                       insert_nbr_key_t<vertex_t, edge_t, tag_t, key_t>{
                         offsets,
                         indices,
                         detail::key_cuco_store_insert_device_view_t(key_set.view()),
                         aggregate_vi_t<vertex_t, tag_t, key_t>{static_cast<tag_t>(num_origins)}});
    }

    auto old_kv_pair_first =
      thrust::make_zip_iterator(old_key_buffer.begin(), old_value_buffer.begin());
    auto old_kv_pair_last = thrust::transform(
      handle.get_thrust_policy(),
      old_kv_pair_first,
      old_kv_pair_first + old_key_buffer.size(),
      keep_flags.begin(),
      keep_t<key_t, weight_t>{old_near_far_threshold,
                              detail::key_cuco_store_contains_device_view_t(key_set.view())});

    keep_count = thrust::count_if(
      handle.get_thrust_policy(), keep_flags.begin(), keep_flags.end(), cuda::std::identity{});
  }

  size_t new_kv_store_capacity = compute_kv_store_capacity(
    keep_count + min_extra_capacity, old_kv_store_capacity, kv_store_capacity_increment);

  key_to_dist_map = kv_store_t<key_t, weight_t, false /* use_binary_search */>(
    new_kv_store_capacity, invalid_key, invalid_distance, handle.get_stream());

  key_to_dist_map.insert_if(old_key_buffer.begin(),
                            old_key_buffer.end(),
                            old_value_buffer.begin(),
                            keep_flags.begin(),
                            cuda::std::identity{},
                            handle.get_stream());

  return std::move(key_to_dist_map);
}

template <typename GraphViewType, typename weight_t>
rmm::device_uvector<weight_t> od_shortest_distances(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  edge_property_view_t<typename GraphViewType::edge_type, weight_t const*> edge_weight_view,
  raft::device_span<typename GraphViewType::vertex_type const> origins,
  raft::device_span<typename GraphViewType::vertex_type const> destinations,
  weight_t cutoff,
  weight_t delta,
  bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = uint64_t;
  using od_idx_t = uint32_t;  // origin/destination idx

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");
  static_assert(!GraphViewType::is_multi_gpu, "We currently do not support multi-GPU.");

  // concurrently runs multiple instances of the Near-Far Pile method in
  // A. Davidson, S. Baxter, M. Garland, and J. D. Owens, "Work-efficient parallel GPU methods for
  // single-source shortest paths," 2014.

  // 1. check input arguments

  auto const num_vertices = graph_view.number_of_vertices();
  auto const num_edges    = graph_view.compute_number_of_edges(handle);

  CUGRAPH_EXPECTS(num_vertices != 0 || (origins.size() == 0 && destinations.size() == 0),
                  "Invalid input argument: the input graph is empty but origins.size() > 0 or "
                  "destinations.size() > 0.");

  CUGRAPH_EXPECTS(
    static_cast<size_t>(num_vertices) * origins.size() <= std::numeric_limits<key_t>::max(),
    "Invalid input arguments: graph_view.number_of_vertices() * origins.size() too large, the "
    "current implementation assumes that a vertex ID and a origin index can be packed in a 64 "
    "bit value.");

  CUGRAPH_EXPECTS(origins.size() <= std::numeric_limits<od_idx_t>::max() &&
                    destinations.size() <= std::numeric_limits<od_idx_t>::max(),
                  "Invalid input arguments: origins.size() or destinations.size() too large, the "
                  "current implementation assumes that the origin/destination index can be "
                  "represented using a 32 bit value.");

  if (do_expensive_check) {
    auto num_negative_edge_weights =
      count_if_e(handle,
                 graph_view,
                 edge_src_dummy_property_t{}.view(),
                 edge_dst_dummy_property_t{}.view(),
                 edge_weight_view,
                 [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w < 0.0; });
    CUGRAPH_EXPECTS(num_negative_edge_weights == 0,
                    "Invalid input argument: input edge weights should have non-negative values.");

    auto num_invalid_origins = thrust::count_if(
      handle.get_thrust_policy(),
      origins.begin(),
      origins.end(),
      [num_vertices] __device__(auto v) { return !is_valid_vertex(num_vertices, v); });
    auto num_invalid_destinations = thrust::count_if(
      handle.get_thrust_policy(),
      destinations.begin(),
      destinations.end(),
      [num_vertices] __device__(auto v) { return !is_valid_vertex(num_vertices, v); });
    CUGRAPH_EXPECTS(num_invalid_origins == 0,
                    "Invalid input arguments: origins contains invalid vertex IDs.");
    CUGRAPH_EXPECTS(num_invalid_destinations == 0,
                    "Invalid input arguments: destinations contains invalid vertex IDs.");

    rmm::device_uvector<vertex_t> tmp_origins(origins.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), origins.begin(), origins.end(), tmp_origins.begin());
    thrust::sort(handle.get_thrust_policy(), tmp_origins.begin(), tmp_origins.end());
    CUGRAPH_EXPECTS(
      thrust::unique(handle.get_thrust_policy(), tmp_origins.begin(), tmp_origins.end()) ==
        tmp_origins.end(),
      "Invalid input arguments: origins should not have duplicates.");

    rmm::device_uvector<vertex_t> tmp_destinations(destinations.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 destinations.begin(),
                 destinations.end(),
                 tmp_destinations.begin());
    thrust::sort(handle.get_thrust_policy(), tmp_destinations.begin(), tmp_destinations.end());
    CUGRAPH_EXPECTS(thrust::unique(handle.get_thrust_policy(),
                                   tmp_destinations.begin(),
                                   tmp_destinations.end()) == tmp_destinations.end(),
                    "Invalid input arguments: destinations should not have duplicates.");
  }

  // 2. set performance tuning parameters

  size_t constexpr num_far_buffers{5};

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;

  size_t key_buffer_capacity_increment = origins.size() * size_t{1024};
  size_t init_far_buffer_size          = origins.size() * size_t{1024};

  size_t kv_store_capacity_increment = raft::round_up_safe(
    std::max(static_cast<size_t>((static_cast<double>(total_global_mem) * 0.01) /
                                 static_cast<double>(sizeof(key_t) + sizeof(weight_t))),
             size_t{1}),
    size_t{1} << 12);
  auto init_kv_store_size =
    std::min(static_cast<size_t>((static_cast<double>(num_vertices) * 0.001) *
                                 static_cast<double>(origins.size())),
             kv_store_capacity_increment * 25);
  init_kv_store_size = std::max(init_kv_store_size, origins.size());

  auto target_near_q_size =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
    size_t{32 * 1024};  // increase the step size if the near queue size is smaller than the target
                        // size (up to num_far_buffers * delta)

  // 3. initialize od_matrix & v_to_destination_indices

  auto constexpr invalid_distance =
    std::numeric_limits<weight_t>::lowest();  // no valid distance can be negative
  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;
  auto constexpr invalid_od_idx = std::numeric_limits<od_idx_t>::max();
  auto constexpr invalid_key    = std::numeric_limits<key_t>::max();

  auto od_matrix_size = origins.size() * destinations.size();
  rmm::device_uvector<weight_t> od_matrix(od_matrix_size, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               od_matrix.begin(),
               od_matrix.end(),
               std::numeric_limits<weight_t>::max());

  if (num_vertices == 0 || num_edges == 0 || od_matrix.size() == 0) { return od_matrix; }

  rmm::device_uvector<od_idx_t> v_to_destination_indices(num_vertices, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               v_to_destination_indices.begin(),
               v_to_destination_indices.end(),
               invalid_od_idx);
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator(od_idx_t{0}),
                   thrust::make_counting_iterator(static_cast<od_idx_t>(destinations.size())),
                   update_v_to_destination_index_t<vertex_t, od_idx_t>{
                     destinations,
                     raft::device_span<od_idx_t>(v_to_destination_indices.data(),
                                                 v_to_destination_indices.size())});

  // 4. initialize SSSP frontier

  constexpr size_t bucket_idx_near = 0;
  constexpr size_t num_buckets     = 1;

  vertex_frontier_t<vertex_t,
                    od_idx_t,
                    GraphViewType::is_multi_gpu,
                    true /* sorted_unique_key_bucket */>
    vertex_frontier(handle, num_buckets);

  std::vector<rmm::device_uvector<key_t>> far_buffers{};
  far_buffers.reserve(num_far_buffers);
  for (size_t i = 0; i < num_far_buffers; ++i) {
    rmm::device_uvector<key_t> buffer(0, handle.get_stream());
    buffer.reserve(init_far_buffer_size, handle.get_stream());
    far_buffers.push_back(std::move(buffer));
  }

  auto init_kv_store_capacity =
    compute_kv_store_capacity(init_kv_store_size, size_t{0}, kv_store_capacity_increment);

  kv_store_t<key_t, weight_t, false /* use_binary_search */> key_to_dist_map(
    init_kv_store_capacity, invalid_key, invalid_distance, handle.get_stream());

  // 5. initialize vertex_frontier & key_to_dist_map, and update od_matrix

  {
    auto tagged_origin_first =
      thrust::make_zip_iterator(origins.begin(), thrust::make_counting_iterator(od_idx_t{0}));
    vertex_frontier.bucket(bucket_idx_near)
      .insert(tagged_origin_first, tagged_origin_first + origins.size());

    auto key_first = thrust::make_transform_iterator(
      tagged_origin_first,
      aggregate_vi_t<vertex_t, od_idx_t, key_t>{static_cast<od_idx_t>(origins.size())});
    key_to_dist_map.insert(key_first,
                           key_first + origins.size(),
                           thrust::make_constant_iterator(weight_t{0.0}),
                           handle.get_stream());

    thrust::transform_if(handle.get_thrust_policy(),
                         thrust::make_constant_iterator(weight_t{0.0}),
                         thrust::make_constant_iterator(weight_t{0.0}) + origins.size(),
                         key_first,
                         thrust::make_permutation_iterator(
                           od_matrix.begin(),
                           thrust::make_transform_iterator(
                             key_first,
                             compute_od_matrix_index_t<vertex_t, od_idx_t, key_t>{
                               raft::device_span<od_idx_t const>(v_to_destination_indices.data(),
                                                                 v_to_destination_indices.size()),
                               static_cast<od_idx_t>(origins.size()),
                               static_cast<od_idx_t>(destinations.size())})),
                         cuda::std::identity{},
                         check_destination_index_t<vertex_t, od_idx_t, key_t>{
                           raft::device_span<od_idx_t const>(v_to_destination_indices.data(),
                                                             v_to_destination_indices.size()),
                           static_cast<od_idx_t>(origins.size()),
                           invalid_od_idx});
  }

  // 6. SSSP iteration

  auto old_near_far_threshold = weight_t{0.0};
  auto near_far_threshold     = delta;
  std::vector<weight_t> next_far_thresholds(num_far_buffers - 1);
  for (size_t i = 0; i < next_far_thresholds.size(); ++i) {
    next_far_thresholds[i] =
      (i == 0) ? (near_far_threshold + delta) : next_far_thresholds[i - 1] + delta;
  }
  size_t prev_near_far_threshold_num_inserts{0};
  size_t prev_num_near_q_insert_buffers{1};
  while (true) {
    // 6-1. enumerate next frontier candidates

    rmm::device_uvector<key_t> new_frontier_keys(0, handle.get_stream());
    rmm::device_uvector<weight_t> distance_buffer(0, handle.get_stream());
    {
      // use detail space functions as sort_by_key with key = key_t is faster than key =
      // thrust::tuple<vertex_t, od_idx_t> and we need to convert thrust::tuple<vertex_t, od_idx_t>
      // to key_t anyways for post processing

      auto e_op = e_op_t<vertex_t, od_idx_t, key_t, weight_t>{
        detail::kv_cuco_store_find_device_view_t(key_to_dist_map.view()),
        static_cast<od_idx_t>(origins.size())};
      detail::transform_reduce_if_v_frontier_call_e_op_t<
        thrust::tuple<vertex_t, od_idx_t>,
        weight_t,
        vertex_t,
        cuda::std::nullopt_t,
        cuda::std::nullopt_t,
        weight_t,
        e_op_t<vertex_t, od_idx_t, key_t, weight_t>>
        e_op_wrapper{e_op};

      auto new_frontier_tagged_vertex_buffer =
        allocate_dataframe_buffer<thrust::tuple<vertex_t, od_idx_t>>(0, handle.get_stream());
      std::tie(new_frontier_tagged_vertex_buffer, distance_buffer) = detail::
        extract_transform_if_v_frontier_e<false, thrust::tuple<vertex_t, od_idx_t>, weight_t>(
          handle,
          graph_view,
          vertex_frontier.bucket(bucket_idx_near),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_weight_view,
          e_op_wrapper,
          pred_op_t<vertex_t, od_idx_t, key_t, weight_t>{
            detail::kv_cuco_store_find_device_view_t(key_to_dist_map.view()),
            static_cast<od_idx_t>(origins.size()),
            cutoff,
            invalid_distance});

      new_frontier_keys.resize(size_dataframe_buffer(new_frontier_tagged_vertex_buffer),
                               handle.get_stream());
      auto key_first = thrust::make_transform_iterator(
        get_dataframe_buffer_begin(new_frontier_tagged_vertex_buffer),
        aggregate_vi_t<vertex_t, od_idx_t, key_t>{static_cast<od_idx_t>(origins.size())});
      thrust::copy(handle.get_thrust_policy(),
                   key_first,
                   key_first + size_dataframe_buffer(new_frontier_tagged_vertex_buffer),
                   new_frontier_keys.begin());
      resize_dataframe_buffer(new_frontier_tagged_vertex_buffer, 0, handle.get_stream());
      shrink_to_fit_dataframe_buffer(new_frontier_tagged_vertex_buffer, handle.get_stream());

      std::tie(new_frontier_keys, distance_buffer) = detail::
        sort_and_reduce_buffer_elements<key_t, key_t, weight_t, reduce_op::minimum<weight_t>>(
          handle,
          std::move(new_frontier_keys),
          std::move(distance_buffer),
          reduce_op::minimum<weight_t>(),
          std::make_tuple(vertex_t{0}, graph_view.number_of_vertices()),
          std::nullopt);
    }
    vertex_frontier.bucket(bucket_idx_near).clear();

    // 6-2. update key_to_dist_map

    {
      auto new_min_capacity = key_to_dist_map.size() + new_frontier_keys.size();
      if (key_to_dist_map.capacity() <
          new_min_capacity) {  // note that this is conservative as some keys may already exist in
                               // key_to_dist_map
        auto new_kv_store_capacity = compute_kv_store_capacity(
          new_min_capacity, key_to_dist_map.capacity(), kv_store_capacity_increment);
        auto [old_key_buffer, old_value_buffer] = key_to_dist_map.release(handle.get_stream());
        key_to_dist_map = kv_store_t<key_t, weight_t, false /* use_binary_search */>(
          new_kv_store_capacity, invalid_key, invalid_distance, handle.get_stream());
        key_to_dist_map.insert(get_dataframe_buffer_begin(old_key_buffer),
                               get_dataframe_buffer_end(old_key_buffer),
                               old_value_buffer.begin(),
                               handle.get_stream());
      }
      key_to_dist_map.insert_and_assign(new_frontier_keys.begin(),
                                        new_frontier_keys.end(),
                                        distance_buffer.begin(),
                                        handle.get_stream());
      prev_near_far_threshold_num_inserts += new_frontier_keys.size();
    }

    // 6-3. update od_matrix

    {
      thrust::transform_if(handle.get_thrust_policy(),
                           distance_buffer.begin(),
                           distance_buffer.end(),
                           new_frontier_keys.begin(),
                           thrust::make_permutation_iterator(
                             od_matrix.begin(),
                             thrust::make_transform_iterator(
                               new_frontier_keys.begin(),
                               compute_od_matrix_index_t<vertex_t, od_idx_t, key_t>{
                                 raft::device_span<od_idx_t const>(v_to_destination_indices.data(),
                                                                   v_to_destination_indices.size()),
                                 static_cast<od_idx_t>(origins.size()),
                                 static_cast<od_idx_t>(destinations.size())})),
                           cuda::std::identity{},
                           check_destination_index_t<vertex_t, od_idx_t, key_t>{
                             raft::device_span<od_idx_t const>(v_to_destination_indices.data(),
                                                               v_to_destination_indices.size()),
                             static_cast<od_idx_t>(origins.size()),
                             invalid_od_idx});
    }

    // 6-4. update the queues

    {
      std::vector<weight_t> h_split_thresholds(
        num_far_buffers);  // total # buffers = 1 (near queue) + num_far_buffers
      h_split_thresholds[0] = near_far_threshold;
      for (size_t i = 1; i < h_split_thresholds.size(); ++i) {
        h_split_thresholds[i] = next_far_thresholds[i - 1];
      }
      rmm::device_uvector<weight_t> d_split_thresholds(h_split_thresholds.size(),
                                                       handle.get_stream());
      raft::update_device(d_split_thresholds.data(),
                          h_split_thresholds.data(),
                          h_split_thresholds.size(),
                          handle.get_stream());

      rmm::device_uvector<size_t> d_counters(d_split_thresholds.size() + 1, handle.get_stream());
      thrust::fill(handle.get_thrust_policy(), d_counters.begin(), d_counters.end(), size_t{0});

      auto num_tagged_vertices = new_frontier_keys.size();
      rmm::device_uvector<key_t> tmp_near_q_keys(0, handle.get_stream());
      tmp_near_q_keys.reserve(num_tagged_vertices, handle.get_stream());

      std::vector<size_t> old_far_buffer_sizes(num_far_buffers);
      for (size_t i = 0; i < num_far_buffers; ++i) {
        old_far_buffer_sizes[i] = far_buffers[i].size();
      }

      auto input_first =
        thrust::make_zip_iterator(new_frontier_keys.begin(), distance_buffer.begin());

      size_t num_copied{0};
      while (num_copied < num_tagged_vertices) {
        auto this_loop_size =
          std::min(key_buffer_capacity_increment, num_tagged_vertices - num_copied);

        std::vector<key_t*> h_buffer_ptrs(d_counters.size());
        tmp_near_q_keys.resize(tmp_near_q_keys.size() + this_loop_size, handle.get_stream());
        h_buffer_ptrs[0] = tmp_near_q_keys.data();
        for (size_t i = 0; i < num_far_buffers; ++i) {
          if (far_buffers[i].size() + this_loop_size > far_buffers[i].capacity()) {
            far_buffers[i].reserve(
              far_buffers[i].capacity() +
                raft::round_up_safe(this_loop_size, key_buffer_capacity_increment),
              handle.get_stream());
          }
          far_buffers[i].resize(far_buffers[i].size() + this_loop_size, handle.get_stream());
          h_buffer_ptrs[i + 1] = far_buffers[i].data() + old_far_buffer_sizes[i];
        }
        rmm::device_uvector<key_t*> d_buffer_ptrs(h_buffer_ptrs.size(), handle.get_stream());
        raft::update_device(
          d_buffer_ptrs.data(), h_buffer_ptrs.data(), h_buffer_ptrs.size(), handle.get_stream());

        raft::grid_1d_thread_t update_grid(this_loop_size,
                                           multi_partition_copy_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        multi_partition_copy<static_cast<int32_t>(1 /* near queue */ + num_far_buffers)>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            input_first + num_copied,
            input_first + num_copied + this_loop_size,
            raft::device_span<key_t*>(d_buffer_ptrs.data(), d_buffer_ptrs.size()),
            [split_thresholds = raft::device_span<weight_t const>(
               d_split_thresholds.data(), d_split_thresholds.size())] __device__(auto pair) {
              return static_cast<uint8_t>(
                cuda::std::distance(split_thresholds.begin(),
                                    thrust::upper_bound(thrust::seq,
                                                        split_thresholds.begin(),
                                                        split_thresholds.end(),
                                                        thrust::get<1>(pair))));
            },
            [] __device__(auto pair) { return thrust::get<0>(pair); },
            raft::device_span<size_t>(d_counters.data(), d_counters.size()));

        std::vector<size_t> h_counters(d_counters.size());
        raft::update_host(
          h_counters.data(), d_counters.data(), d_counters.size(), handle.get_stream());
        handle.sync_stream();

        tmp_near_q_keys.resize(h_counters[0], handle.get_stream());
        for (size_t i = 0; i < num_far_buffers; ++i) {
          far_buffers[i].resize(old_far_buffer_sizes[i] + h_counters[i + 1], handle.get_stream());
        }

        num_copied += this_loop_size;
      }

      thrust::sort(handle.get_thrust_policy(), tmp_near_q_keys.begin(), tmp_near_q_keys.end());
      tmp_near_q_keys.resize(cuda::std::distance(tmp_near_q_keys.begin(),
                                                 thrust::unique(handle.get_thrust_policy(),
                                                                tmp_near_q_keys.begin(),
                                                                tmp_near_q_keys.end())),
                             handle.get_stream());
      auto near_vi_first = thrust::make_transform_iterator(
        tmp_near_q_keys.begin(),
        split_vi_t<vertex_t, od_idx_t, key_t>{static_cast<od_idx_t>(origins.size())});
      vertex_frontier.bucket(bucket_idx_near)
        .insert(near_vi_first, near_vi_first + tmp_near_q_keys.size());
    }

    new_frontier_keys.resize(0, handle.get_stream());
    distance_buffer.resize(0, handle.get_stream());
    new_frontier_keys.shrink_to_fit(handle.get_stream());
    distance_buffer.shrink_to_fit(handle.get_stream());

    if (vertex_frontier.bucket(bucket_idx_near).aggregate_size() > 0) {
      /* nothing to do */
    } else {
      auto num_aggregate_far_keys = far_buffers[0].size();
      for (size_t i = 1; i < num_far_buffers; ++i) {
        num_aggregate_far_keys += far_buffers[i].size();
      }

      if (num_aggregate_far_keys > 0) {  // near queue is empty, split the far queue
        std::vector<weight_t> invalid_thresholds(num_far_buffers);
        for (size_t i = 0; i < invalid_thresholds.size(); ++i) {
          invalid_thresholds[i] = (i == 0) ? near_far_threshold : next_far_thresholds[i - 1];
        }

        size_t num_near_q_insert_buffers{0};
        size_t near_size{0};
        size_t tot_far_size{0};
        do {
          num_near_q_insert_buffers = 0;
          size_t max_near_q_inserts{0};
          old_near_far_threshold = near_far_threshold;
          do {
            near_far_threshold =
              (num_far_buffers > 1) ? next_far_thresholds[0] : (near_far_threshold + delta);
            for (size_t i = 0; i < next_far_thresholds.size(); ++i) {
              next_far_thresholds[i] = (i < next_far_thresholds.size() - 1)
                                         ? next_far_thresholds[i + 1]
                                         : (next_far_thresholds[i] + delta);
            }
            max_near_q_inserts += far_buffers[num_near_q_insert_buffers].size();
            ++num_near_q_insert_buffers;
          } while ((max_near_q_inserts < target_near_q_size) &&
                   (num_near_q_insert_buffers < num_far_buffers));

          rmm::device_uvector<key_t> new_near_q_keys(0, handle.get_stream());
          new_near_q_keys.reserve(max_near_q_inserts, handle.get_stream());

          for (size_t i = 0; i < num_far_buffers; ++i) {
            auto invalid_threshold = invalid_thresholds[i];

            if (i == (num_far_buffers - 1)) {
              std::vector<weight_t> h_split_thresholds(num_near_q_insert_buffers);
              h_split_thresholds[0] =
                (num_far_buffers == num_near_q_insert_buffers)
                  ? near_far_threshold
                  : next_far_thresholds[(num_far_buffers - num_near_q_insert_buffers) - 1];
              for (size_t j = 1; j < h_split_thresholds.size(); ++j) {
                h_split_thresholds[j] =
                  next_far_thresholds[(num_far_buffers - num_near_q_insert_buffers) + (j - 1)];
              }
              rmm::device_uvector<weight_t> d_split_thresholds(h_split_thresholds.size(),
                                                               handle.get_stream());
              raft::update_device(d_split_thresholds.data(),
                                  h_split_thresholds.data(),
                                  h_split_thresholds.size(),
                                  handle.get_stream());

              rmm::device_uvector<key_t> tmp_buffer = std::move(far_buffers.back());
              std::vector<key_t*> h_buffer_ptrs(h_split_thresholds.size() + 1);
              auto old_size = new_near_q_keys.size();
              for (size_t j = 0; j < h_buffer_ptrs.size(); ++j) {
                if (j == 0 && (num_far_buffers == num_near_q_insert_buffers)) {
                  new_near_q_keys.resize(old_size + tmp_buffer.size(), handle.get_stream());
                  h_buffer_ptrs[j] = new_near_q_keys.data() + old_size;
                } else {
                  auto buffer_idx = (num_far_buffers - num_near_q_insert_buffers) + (j - 1);
                  far_buffers[buffer_idx].reserve(
                    raft::round_up_safe(std::max(tmp_buffer.size(), size_t{1}),
                                        key_buffer_capacity_increment),
                    handle.get_stream());
                  far_buffers[buffer_idx].resize(tmp_buffer.size(), handle.get_stream());
                  h_buffer_ptrs[j] = far_buffers[buffer_idx].data();
                }
              }
              rmm::device_uvector<key_t*> d_buffer_ptrs(num_near_q_insert_buffers + 1,
                                                        handle.get_stream());
              raft::update_device(d_buffer_ptrs.data(),
                                  h_buffer_ptrs.data(),
                                  h_buffer_ptrs.size(),
                                  handle.get_stream());
              rmm::device_uvector<size_t> d_counters(num_near_q_insert_buffers + 1,
                                                     handle.get_stream());
              thrust::fill(
                handle.get_thrust_policy(), d_counters.begin(), d_counters.end(), size_t{0});
              if (tmp_buffer.size() > 0) {
                raft::grid_1d_thread_t update_grid(tmp_buffer.size(),
                                                   multi_partition_copy_block_size,
                                                   handle.get_device_properties().maxGridSize[0]);
                auto constexpr max_num_partitions =
                  static_cast<int32_t>(1 /* near queue */ + num_far_buffers);
                multi_partition_copy<max_num_partitions>
                  <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
                    tmp_buffer.begin(),
                    tmp_buffer.end(),
                    raft::device_span<key_t*>(d_buffer_ptrs.data(), d_buffer_ptrs.size()),
                    [key_to_dist_map =
                       detail::kv_cuco_store_find_device_view_t(key_to_dist_map.view()),
                     split_thresholds = raft::device_span<weight_t const>(
                       d_split_thresholds.data(), d_split_thresholds.size()),
                     invalid_threshold] __device__(auto key) {
                      auto dist = key_to_dist_map.find(key);
                      return static_cast<uint8_t>(
                        (dist < invalid_threshold)
                          ? max_num_partitions /* discard */
                          : cuda::std::distance(split_thresholds.begin(),
                                                thrust::upper_bound(thrust::seq,
                                                                    split_thresholds.begin(),
                                                                    split_thresholds.end(),
                                                                    dist)));
                    },
                    cuda::std::identity{},
                    raft::device_span<size_t>(d_counters.data(), d_counters.size()));
              }
              std::vector<size_t> h_counters(d_counters.size());
              raft::update_host(
                h_counters.data(), d_counters.data(), d_counters.size(), handle.get_stream());
              handle.sync_stream();
              for (size_t j = 0; j < (num_near_q_insert_buffers + 1); ++j) {
                if (j == 0 && (num_far_buffers == num_near_q_insert_buffers)) {
                  new_near_q_keys.resize(old_size + h_counters[j], handle.get_stream());
                } else {
                  auto buffer_idx = (num_far_buffers - num_near_q_insert_buffers) + (j - 1);
                  far_buffers[buffer_idx].resize(h_counters[j], handle.get_stream());
                }
              }
            } else if (i < num_near_q_insert_buffers) {
              auto old_size = new_near_q_keys.size();
              new_near_q_keys.resize(old_size + far_buffers[i].size(), handle.get_stream());
              auto last = thrust::copy_if(
                handle.get_thrust_policy(),
                far_buffers[i].begin(),
                far_buffers[i].end(),
                new_near_q_keys.begin() + old_size,
                is_no_smaller_than_threshold_t<key_t, weight_t>{
                  invalid_threshold,
                  detail::kv_cuco_store_find_device_view_t(key_to_dist_map.view())});
              new_near_q_keys.resize(cuda::std::distance(new_near_q_keys.begin(), last),
                                     handle.get_stream());
            } else {
              far_buffers[i - num_near_q_insert_buffers] = std::move(far_buffers[i]);
            }
          }

          thrust::sort(handle.get_thrust_policy(), new_near_q_keys.begin(), new_near_q_keys.end());
          new_near_q_keys.resize(cuda::std::distance(new_near_q_keys.begin(),
                                                     thrust::unique(handle.get_thrust_policy(),
                                                                    new_near_q_keys.begin(),
                                                                    new_near_q_keys.end())),
                                 handle.get_stream());
          auto near_vi_first = thrust::make_transform_iterator(
            new_near_q_keys.begin(),
            split_vi_t<vertex_t, od_idx_t, key_t>{static_cast<od_idx_t>(origins.size())});
          vertex_frontier.bucket(bucket_idx_near)
            .insert(near_vi_first, near_vi_first + new_near_q_keys.size());

          near_size    = vertex_frontier.bucket(bucket_idx_near).size();
          tot_far_size = far_buffers[0].size();
          for (size_t i = 1; i < num_far_buffers; ++i) {
            tot_far_size += far_buffers[i].size();
          }
        } while ((near_size == 0) && (tot_far_size > 0));

        // this assumes that # inserts with the previous near_far_threshold is a good estimate for
        // # inserts with the next near_far_threshold
        auto next_near_far_threshold_num_inserts_estimate =
          static_cast<size_t>(static_cast<double>(prev_near_far_threshold_num_inserts) *
                              std::max(static_cast<double>(num_near_q_insert_buffers) /
                                         static_cast<double>(prev_num_near_q_insert_buffers),
                                       1.0) *
                              1.2);
        prev_near_far_threshold_num_inserts = 0;
        prev_num_near_q_insert_buffers      = num_near_q_insert_buffers;

        if (key_to_dist_map.size() + next_near_far_threshold_num_inserts_estimate >=
            key_to_dist_map.capacity()) {  // if resize is likely to be necessary before reaching
                                           // this check again
          std::vector<raft::device_span<key_t const>> far_buffer_spans(num_far_buffers);
          for (size_t i = 0; i < num_far_buffers; ++i) {
            far_buffer_spans[i] =
              raft::device_span<key_t const>(far_buffers[i].data(), far_buffers[i].size());
          }
          key_to_dist_map = filter_key_to_dist_map(handle,
                                                   graph_view,
                                                   std::move(key_to_dist_map),
                                                   vertex_frontier.bucket(bucket_idx_near),
                                                   far_buffer_spans,
                                                   old_near_far_threshold,
                                                   next_near_far_threshold_num_inserts_estimate,
                                                   kv_store_capacity_increment,
                                                   origins.size());
        }
        if ((near_size == 0) && (tot_far_size == 0)) { break; }
      } else {
        break;
      }
    }
  }

  return od_matrix;
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> od_shortest_distances(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
  raft::device_span<vertex_t const> origins,
  raft::device_span<vertex_t const> destinations,
  weight_t cutoff,
  bool do_expensive_check)
{
  auto const num_vertices = graph_view.number_of_vertices();
  auto const num_edges    = graph_view.compute_number_of_edges(handle);

  weight_t average_vertex_degree =
    static_cast<weight_t>(num_edges) / static_cast<weight_t>(num_vertices);
  auto average_edge_weight = transform_reduce_e(
    handle,
    graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_weight_view,
    [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w; },
    weight_t{0.0});
  {
    // the above transform_reduce_e output can vary in each run due to floating point round-off
    // errro, we reduce the precision of the significand here to reduce the non-determinicity due to
    // the difference in delta in each run (this still does not guarantee that we will get the same
    // delta in every run for every graph)
    assert(average_edge_weight >= 0.0);
    int exponent{};
    auto significand =
      frexp(average_edge_weight,
            &exponent);  // average_edge_weight = significnad * 2^exponent, 0.5 <= significand < 1.0
    significand         = round(significand * 1000.0) / 1000.0;
    average_edge_weight = ldexp(significand, exponent);
  }
  average_edge_weight /= static_cast<weight_t>(num_edges);
  // FIXME: better use min_edge_weight instead of std::numeric_limits<weight_t>::min() * 1e3 for
  // forward progress guarantee transform_reduce_e should better be updated to support min
  // reduction.
  auto delta = std::max(average_edge_weight * 0.5, std::numeric_limits<weight_t>::min() * 1e3);

  return od_shortest_distances<graph_view_t<vertex_t, edge_t, false, multi_gpu>, weight_t>(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<vertex_t const>(origins.data(), origins.size()),
    raft::device_span<vertex_t const>(destinations.data(), destinations.size()),
    cutoff,
    delta,
    do_expensive_check);
}

}  // namespace cugraph

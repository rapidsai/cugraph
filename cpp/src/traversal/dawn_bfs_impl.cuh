/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <type_traits>
#include <thread>
#include <vector>

namespace cugraph {
namespace detail {
namespace dawn {

constexpr int block_size = 256;
constexpr int tile_vertices = 1024;
constexpr int words_per_dirty = tile_vertices / 32;
constexpr int dirty_words_per_super = 32;
constexpr int compact_entries_per_block = 32;
constexpr int compact_dirty_words_per_block = 2;
constexpr int bitset_warp_degree_threshold = 8;
constexpr int bitset_subwarp_width = 4;
constexpr int bitset_subwarp_max_active = 0;
constexpr int bitset_offset_preload_min_active = 33;
constexpr int defer_high_degree_threshold = 512;
constexpr int deferred_edge_chunk_size = 256;
constexpr int deferred_edge_chunk_blocks = 256;

template <typename vertex_t, typename edge_t>
struct raw_csr_view_t {
  edge_t const* row_offsets{};
  vertex_t const* column_indices{};
  vertex_t num_vertices{};
  edge_t num_edges{};
};

template <typename vertex_t, typename edge_t>
struct workspace_t {
  explicit workspace_t(raw_csr_view_t<vertex_t, edge_t> raw, rmm::cuda_stream_view stream)
    : dense_words((static_cast<size_t>(raw.num_vertices) + 31) / 32, stream),
      dense_dirty((dense_words.size() + words_per_dirty - 1) / words_per_dirty, stream),
      dense_dirty_super((dense_dirty.size() + dirty_words_per_super - 1) / dirty_words_per_super,
                        stream),
      active_super_list(dense_dirty_super.size(), stream),
      masks_a(dense_words.size(), stream),
      masks_b(dense_words.size(), stream),
      gamma_a(dense_words.size(), stream),
      gamma_b(dense_words.size(), stream),
      state(8, stream),
      deferred_begins((static_cast<size_t>(raw.num_edges) + deferred_edge_chunk_size - 1) /
                        deferred_edge_chunk_size +
                      dense_words.size(),
                      stream),
      deferred_ends(deferred_begins.size(), stream)
  {
  }

  rmm::device_uvector<uint32_t> dense_words;
  rmm::device_uvector<uint32_t> dense_dirty;
  rmm::device_uvector<uint32_t> dense_dirty_super;
  rmm::device_uvector<vertex_t> active_super_list;
  rmm::device_uvector<uint32_t> masks_a;
  rmm::device_uvector<uint32_t> masks_b;
  rmm::device_uvector<vertex_t> gamma_a;
  rmm::device_uvector<vertex_t> gamma_b;
  rmm::device_uvector<vertex_t> state;
  rmm::device_uvector<edge_t> deferred_begins;
  rmm::device_uvector<edge_t> deferred_ends;
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t graph_exec{nullptr};

  ~workspace_t()
  {
    if (graph_exec != nullptr) { cudaGraphExecDestroy(graph_exec); }
    if (graph != nullptr) { cudaGraphDestroy(graph); }
  }
};

enum state_index_t {
  state_depth = 0,
  state_changed = 1,
  state_frontier_parity = 2,
  state_source_has_frontier = 3,
  state_alpha_word_count = 4,
  state_beta_word_count = 5,
  state_beta_super_count = 6,
  state_edge_chunk_count = 7,
  state_count = 8,
};

template <typename vertex_t, typename edge_t>
static __global__ void init_kernel(raw_csr_view_t<vertex_t, edge_t> graph,
                                   vertex_t source,
                                   vertex_t* distances,
                                   uint32_t* dense_words,
                                   uint32_t* dense_dirty,
                                   uint32_t* dense_dirty_super,
                                   vertex_t* active_super_list,
                                   uint32_t* masks_a,
                                   uint32_t* masks_b,
                                   vertex_t* gamma_a,
                                   vertex_t* gamma_b,
                                   vertex_t* state,
                                   vertex_t word_count,
                                   vertex_t dirty_count,
                                   vertex_t dirty_super_count)
{
  auto tid    = static_cast<vertex_t>(blockIdx.x * blockDim.x + threadIdx.x);
  auto stride = static_cast<vertex_t>(blockDim.x * gridDim.x);
  for (vertex_t v = tid; v < graph.num_vertices; v += stride) {
    distances[v] = std::numeric_limits<vertex_t>::max();
  }
  for (vertex_t w = tid; w < word_count; w += stride) { dense_words[w] = 0; }
  for (vertex_t d = tid; d < dirty_count; d += stride) { dense_dirty[d] = 0; }
  for (vertex_t s = tid; s < dirty_super_count; s += stride) {
    dense_dirty_super[s] = 0;
    active_super_list[s] = 0;
  }
  for (vertex_t i = tid; i < state_count; i += stride) { state[i] = 0; }

  if (tid == 0) {
    distances[source] = 0;
    auto word = static_cast<vertex_t>(source >> 5);
    auto mask = static_cast<uint32_t>(1u << (source & 31));
    auto dirty_word = static_cast<vertex_t>(word / words_per_dirty);
    auto dirty_super = static_cast<vertex_t>(dirty_word / dirty_words_per_super);
    dense_words[word] = mask;
    dense_dirty[dirty_word] = 1u;
    dense_dirty_super[dirty_super] = 1u;
    active_super_list[0] = dirty_super;
    masks_a[0] = mask;
    gamma_a[0] = word;
    masks_b[0] = 0;
    gamma_b[0] = 0;
    state[state_depth] = 0;
    state[state_changed] = 1;
    state[state_frontier_parity] = 0;
    state[state_source_has_frontier] =
      (graph.row_offsets[source + 1] > graph.row_offsets[source]) ? 1 : 0;
    state[state_alpha_word_count] = 1;
    state[state_beta_word_count] = 0;
    state[state_beta_super_count] = 1;
    state[state_edge_chunk_count] = 0;
  }
}

template <typename vertex_t>
static __global__ void init_multi_source_workspace_kernel(vertex_t* distances,
                                                          uint32_t* dense_words,
                                                          uint32_t* dense_dirty,
                                                          uint32_t* dense_dirty_super,
                                                          vertex_t* active_super_list,
                                                          uint32_t* masks_a,
                                                          uint32_t* masks_b,
                                                          vertex_t* gamma_a,
                                                          vertex_t* gamma_b,
                                                          vertex_t* state,
                                                          vertex_t num_vertices,
                                                          vertex_t word_count,
                                                          vertex_t dirty_count,
                                                          vertex_t dirty_super_count)
{
  auto tid    = static_cast<vertex_t>(blockIdx.x * blockDim.x + threadIdx.x);
  auto stride = static_cast<vertex_t>(blockDim.x * gridDim.x);
  for (vertex_t v = tid; v < num_vertices; v += stride) {
    distances[v] = std::numeric_limits<vertex_t>::max();
  }
  for (vertex_t w = tid; w < word_count; w += stride) {
    dense_words[w] = 0;
    masks_a[w]     = 0;
    masks_b[w]     = 0;
    gamma_a[w]     = 0;
    gamma_b[w]     = 0;
  }
  for (vertex_t d = tid; d < dirty_count; d += stride) { dense_dirty[d] = 0; }
  for (vertex_t s = tid; s < dirty_super_count; s += stride) {
    dense_dirty_super[s] = 0;
    active_super_list[s] = 0;
  }
  for (vertex_t i = tid; i < state_count; i += stride) { state[i] = 0; }

  if (tid == 0) {
    state[state_depth]           = 0;
    state[state_changed]         = 0;
    state[state_frontier_parity] = 0;
  }
}

template <typename vertex_t>
__device__ __forceinline__ vertex_t atomic_cas_distance(vertex_t* address,
                                                        vertex_t compare,
                                                        vertex_t value);

template <typename vertex_t>
__device__ __forceinline__ vertex_t atomic_add_value(vertex_t* address, vertex_t value);

template <typename vertex_t>
__device__ __forceinline__ vertex_t atomic_exch_value(vertex_t* address, vertex_t value);

template <typename vertex_t>
__device__ __forceinline__ void mark_candidate(vertex_t dst,
                                               vertex_t depth,
                                               vertex_t* distances,
                                               uint32_t* dense_words,
                                               uint32_t* dense_dirty,
                                               vertex_t* state);

template <typename vertex_t>
static __global__ void seed_source_distances_kernel(vertex_t const* sources,
                                                    vertex_t n_sources,
                                                    vertex_t* distances)
{
  auto tid    = static_cast<vertex_t>(blockIdx.x * blockDim.x + threadIdx.x);
  auto stride = static_cast<vertex_t>(blockDim.x * gridDim.x);
  for (vertex_t i = tid; i < n_sources; i += stride) {
    atomic_cas_distance(
      distances + sources[i], std::numeric_limits<vertex_t>::max(), vertex_t{0});
  }
}

template <typename vertex_t, typename edge_t>
static __global__ void seed_multi_source_neighbors_kernel(raw_csr_view_t<vertex_t, edge_t> graph,
                                                          vertex_t const* sources,
                                                          vertex_t n_sources,
                                                          vertex_t* distances,
                                                          uint32_t* dense_words,
                                                          uint32_t* dense_dirty,
                                                          vertex_t* state)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) { state[state_depth] = 1; }
  for (vertex_t source_idx = static_cast<vertex_t>(blockIdx.x); source_idx < n_sources;
       source_idx += static_cast<vertex_t>(gridDim.x)) {
    auto source = sources[source_idx];
    auto begin = graph.row_offsets[source];
    auto end = graph.row_offsets[source + 1];
    for (auto edge = begin + static_cast<edge_t>(threadIdx.x); edge < end;
         edge += static_cast<edge_t>(blockDim.x)) {
      mark_candidate(
        graph.column_indices[edge], vertex_t{1}, distances, dense_words, dense_dirty, state);
    }
  }
}

template <typename vertex_t>
static __global__ void reset_iteration_kernel(vertex_t* state)
{
  state[state_beta_word_count]  = 0;
  state[state_beta_super_count] = 0;
  state[state_edge_chunk_count] = 0;
}

template <typename vertex_t>
__device__ __forceinline__ vertex_t atomic_cas_distance(vertex_t* address,
                                                        vertex_t compare,
                                                        vertex_t value)
{
  if constexpr (std::is_same_v<vertex_t, int32_t>) {
    return atomicCAS(address, compare, value);
  } else {
    auto old = atomicCAS(reinterpret_cast<unsigned long long*>(address),
                         static_cast<unsigned long long>(compare),
                         static_cast<unsigned long long>(value));
    return static_cast<vertex_t>(old);
  }
}

template <typename vertex_t>
__device__ __forceinline__ vertex_t atomic_add_value(vertex_t* address, vertex_t value)
{
  if constexpr (std::is_same_v<vertex_t, int32_t>) {
    return atomicAdd(address, value);
  } else {
    auto old = atomicAdd(reinterpret_cast<unsigned long long*>(address),
                         static_cast<unsigned long long>(value));
    return static_cast<vertex_t>(old);
  }
}

template <typename vertex_t>
__device__ __forceinline__ vertex_t atomic_exch_value(vertex_t* address, vertex_t value)
{
  if constexpr (std::is_same_v<vertex_t, int32_t>) {
    return atomicExch(address, value);
  } else {
    auto old = atomicExch(reinterpret_cast<unsigned long long*>(address),
                          static_cast<unsigned long long>(value));
    return static_cast<vertex_t>(old);
  }
}

template <typename vertex_t>
__device__ __forceinline__ void mark_candidate(vertex_t dst,
                                               vertex_t depth,
                                               vertex_t* distances,
                                               uint32_t* dense_words,
                                               uint32_t* dense_dirty,
                                               vertex_t* state)
{
  auto old =
    atomic_cas_distance(distances + dst, std::numeric_limits<vertex_t>::max(), depth);
  if (old == std::numeric_limits<vertex_t>::max()) {
    auto word = static_cast<vertex_t>(dst >> 5);
    auto bit = static_cast<uint32_t>(1u << (dst & 31));
    auto old_word = atomicOr(dense_words + word, bit);
    if (old_word == 0u) {
      auto dirty_word = static_cast<vertex_t>(word / words_per_dirty);
      auto dirty_mask = static_cast<uint32_t>(1u << (word & (words_per_dirty - 1)));
      atomicOr(dense_dirty + dirty_word, dirty_mask);
    }
    atomic_exch_value(state + state_changed, vertex_t{1});
  }
}

template <typename vertex_t, typename edge_t>
static __global__ void consume_frontier_kernel(raw_csr_view_t<vertex_t, edge_t> graph,
                                               vertex_t* distances,
                                               uint32_t* dense_words,
                                               uint32_t* dense_dirty,
                                               uint32_t const* masks_a,
                                               uint32_t const* masks_b,
                                               vertex_t const* gamma_a,
                                               vertex_t const* gamma_b,
                                               edge_t* deferred_begins,
                                               edge_t* deferred_ends,
                                               vertex_t deferred_capacity,
                                               vertex_t* state)
{
  __shared__ vertex_t next_entry;
  auto lane = static_cast<vertex_t>(threadIdx.x & (warpSize - 1));
  auto active_lanes = static_cast<vertex_t>(
    bitset_subwarp_width < bitset_warp_degree_threshold ? bitset_subwarp_width
                                                        : bitset_warp_degree_threshold);
  auto frontier_count = state[state_alpha_word_count];
  auto chunk_begin = static_cast<vertex_t>(blockIdx.x) * compact_entries_per_block;
  if (chunk_begin >= frontier_count) { return; }
  auto chunk_end = std::min<vertex_t>(chunk_begin + compact_entries_per_block, frontier_count);
  auto depth = state[state_depth] + vertex_t{1};
  auto read_a = state[state_frontier_parity] == 0;
  auto const* masks = read_a ? masks_a : masks_b;
  auto const* gamma = read_a ? gamma_a : gamma_b;

  if (threadIdx.x == 0) { next_entry = chunk_begin; }
  __syncthreads();

  while (true) {
    vertex_t entry = 0;
    if (lane == 0) { entry = atomic_add_value(&next_entry, vertex_t{1}); }
    entry = __shfl_sync(0xffffffffu, entry, 0);
    if (entry >= chunk_end) { break; }

    auto word = gamma[entry];
    auto mask = masks[entry];
    if (mask == 0u) { continue; }

    auto base_vertex = static_cast<vertex_t>(word << 5);
    auto vertex = base_vertex + lane;
    auto active_vertex = ((mask & (1u << static_cast<uint32_t>(lane))) != 0u) &&
                         vertex < graph.num_vertices;
    auto valid_vertices =
      graph.num_vertices > base_vertex ? graph.num_vertices - base_vertex : vertex_t{0};
    auto valid_mask = valid_vertices >= 32
                        ? 0xffffffffu
                        : (valid_vertices > 0
                             ? ((1u << static_cast<uint32_t>(valid_vertices)) - 1u)
                             : 0u);
    auto active_mask = mask & valid_mask;
    auto active_count = static_cast<vertex_t>(__popc(active_mask));
    auto preload_offsets = active_count >= bitset_offset_preload_min_active;

    edge_t lane_offset = 0;
    if (preload_offsets && base_vertex + lane <= graph.num_vertices) {
      lane_offset = graph.row_offsets[base_vertex + lane];
    }
    auto shuffled_next_offset =
      preload_offsets ? __shfl_down_sync(0xffffffffu, lane_offset, 1) : edge_t{0};

    edge_t edge_begin = 0;
    edge_t edge_end = 0;
    edge_t degree = 0;
    if (active_vertex) {
      edge_begin = preload_offsets ? lane_offset : graph.row_offsets[vertex];
      edge_end = (preload_offsets && lane < 31) ? shuffled_next_offset
                                                : graph.row_offsets[vertex + 1];
      degree = edge_end - edge_begin;
    }

    if (active_count <= bitset_subwarp_max_active) {
      auto low_degree_mask =
        __ballot_sync(0xffffffffu,
                      active_vertex && degree > 0 &&
                        degree < static_cast<edge_t>(bitset_warp_degree_threshold));
      while (low_degree_mask != 0u) {
        auto owner_lane = static_cast<vertex_t>(__ffs(low_degree_mask) - 1);
        auto owner_group = owner_lane / active_lanes;
        auto group_lane = lane - owner_group * active_lanes;
        auto in_group = group_lane >= 0 && group_lane < active_lanes;
        auto low_begin = __shfl_sync(0xffffffffu, edge_begin, owner_lane);
        auto low_end = __shfl_sync(0xffffffffu, edge_end, owner_lane);
        if (in_group) {
          for (auto edge = low_begin + static_cast<edge_t>(group_lane); edge < low_end;
               edge += static_cast<edge_t>(active_lanes)) {
            mark_candidate(
              graph.column_indices[edge], depth, distances, dense_words, dense_dirty, state);
          }
        }
        low_degree_mask &= low_degree_mask - 1u;
      }
    } else if (active_vertex && degree > 0 &&
               degree < static_cast<edge_t>(bitset_warp_degree_threshold)) {
      for (auto edge = edge_begin; edge < edge_end; ++edge) {
        mark_candidate(
          graph.column_indices[edge], depth, distances, dense_words, dense_dirty, state);
      }
    }

    auto high_degree_mask =
      __ballot_sync(0xffffffffu,
                    active_vertex &&
                      degree >= static_cast<edge_t>(bitset_warp_degree_threshold));
    while (high_degree_mask != 0u) {
      auto owner_lane = static_cast<vertex_t>(__ffs(high_degree_mask) - 1);
      auto high_begin = __shfl_sync(0xffffffffu, edge_begin, owner_lane);
      auto high_end = __shfl_sync(0xffffffffu, edge_end, owner_lane);
      auto high_degree = high_end - high_begin;
      if (high_degree >= static_cast<edge_t>(defer_high_degree_threshold)) {
        auto chunk_count = static_cast<vertex_t>(
          (high_degree + deferred_edge_chunk_size - 1) / deferred_edge_chunk_size);
        vertex_t base_slot = 0;
        if (lane == 0) {
          base_slot = atomic_add_value(state + state_edge_chunk_count, chunk_count);
        }
        base_slot = __shfl_sync(0xffffffffu, base_slot, 0);
        for (auto chunk = lane; chunk < chunk_count; chunk += 32) {
          auto slot = base_slot + chunk;
          if (slot < deferred_capacity) {
            auto begin = high_begin + static_cast<edge_t>(chunk) * deferred_edge_chunk_size;
            auto limit = begin + deferred_edge_chunk_size;
            deferred_begins[slot] = begin;
            deferred_ends[slot] = limit < high_end ? limit : high_end;
          }
        }
      } else {
        for (auto edge = high_begin + static_cast<edge_t>(lane); edge < high_end;
             edge += static_cast<edge_t>(warpSize)) {
          mark_candidate(
            graph.column_indices[edge], depth, distances, dense_words, dense_dirty, state);
        }
      }
      high_degree_mask &= high_degree_mask - 1u;
    }
  }
}

template <typename vertex_t, typename edge_t>
static __global__ void consume_deferred_edges_kernel(raw_csr_view_t<vertex_t, edge_t> graph,
                                                     vertex_t* distances,
                                                     uint32_t* dense_words,
                                                     uint32_t* dense_dirty,
                                                     edge_t const* deferred_begins,
                                                     edge_t const* deferred_ends,
                                                     vertex_t deferred_capacity,
                                                     vertex_t* state)
{
  auto deferred_count = state[state_edge_chunk_count];
  auto chunk_limit = deferred_count < deferred_capacity ? deferred_count : deferred_capacity;
  auto depth = state[state_depth] + vertex_t{1};
  for (vertex_t chunk = blockIdx.x; chunk < chunk_limit; chunk += gridDim.x) {
    auto begin = deferred_begins[chunk];
    auto end = deferred_ends[chunk];
    for (auto edge = begin + static_cast<edge_t>(threadIdx.x); edge < end;
         edge += static_cast<edge_t>(blockDim.x)) {
      mark_candidate(
        graph.column_indices[edge], depth, distances, dense_words, dense_dirty, state);
    }
  }
}

template <typename vertex_t>
static __global__ void compact_dense_frontier_kernel(vertex_t word_count,
                                                     uint32_t* dense_words,
                                                     uint32_t* dense_dirty,
                                                     uint32_t* masks_a,
                                                     uint32_t* masks_b,
                                                     vertex_t* gamma_a,
                                                     vertex_t* gamma_b,
                                                     vertex_t* state,
                                                     vertex_t output_count_index)
{
  __shared__ vertex_t warp_counts[compact_dirty_words_per_block];
  __shared__ vertex_t warp_bases[compact_dirty_words_per_block];
  __shared__ vertex_t block_base;

  auto lane = static_cast<vertex_t>(threadIdx.x & (warpSize - 1));
  auto warp_id = static_cast<vertex_t>(threadIdx.x >> 5);
  auto warps_per_block = static_cast<vertex_t>(blockDim.x >> 5);
  auto dirty_count = (word_count + words_per_dirty - 1) / words_per_dirty;
  auto dirty_index = static_cast<vertex_t>(blockIdx.x) * compact_dirty_words_per_block + warp_id;

  uint32_t dirty_word = 0u;
  if (warp_id < compact_dirty_words_per_block && warp_id < warps_per_block &&
      dirty_index < dirty_count) {
    dirty_word = dense_dirty[dirty_index];
  }

  auto read_a = state[state_frontier_parity] == 0;
  auto* output_masks = read_a ? masks_b : masks_a;
  auto* output_gamma = read_a ? gamma_b : gamma_a;
  if (output_count_index == state_alpha_word_count) {
    output_masks = masks_a;
    output_gamma = gamma_a;
  }

  auto active_word = warp_id < compact_dirty_words_per_block && warp_id < warps_per_block &&
                     dirty_word != 0u &&
                     ((dirty_word & (1u << static_cast<uint32_t>(lane))) != 0u);
  auto word_index = dirty_index * words_per_dirty + lane;
  uint32_t mask = 0u;
  if (active_word && word_index < word_count) { mask = dense_words[word_index]; }
  auto has_output = active_word && word_index < word_count && mask != 0u;
  auto active_mask = __ballot_sync(0xffffffffu, has_output);
  auto active_count = static_cast<vertex_t>(__popc(active_mask));

  if (lane == 0 && warp_id < compact_dirty_words_per_block && warp_id < warps_per_block) {
    warp_counts[warp_id] = active_count;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    vertex_t total_count = 0;
    for (vertex_t i = 0; i < compact_dirty_words_per_block; ++i) {
      warp_bases[i] = total_count;
      if (i < warps_per_block) { total_count += warp_counts[i]; }
    }
    block_base =
      total_count > 0 ? atomic_add_value(state + output_count_index, total_count) : vertex_t{0};
  }
  __syncthreads();

  if (has_output) {
    auto lower_mask = active_mask & ((1u << static_cast<uint32_t>(lane)) - 1u);
    auto slot = block_base + warp_bases[warp_id] + static_cast<vertex_t>(__popc(lower_mask));
    output_gamma[slot] = word_index;
    output_masks[slot] = mask;
    dense_words[word_index] = 0u;
  }

  if (lane == 0 && warp_id < compact_dirty_words_per_block && warp_id < warps_per_block &&
      dirty_index < dirty_count) {
    dense_dirty[dirty_index] = 0u;
  }
}

template <typename vertex_t>
static __global__ void start_iteration_kernel(cudaGraphConditionalHandle handle,
                                              vertex_t* state,
                                              vertex_t depth_limit)
{
  auto alpha_count = state[state_alpha_word_count];
  if (alpha_count != 0 && state[state_depth] < depth_limit) {
    cudaGraphSetConditional(handle, 1);
  } else {
    cudaGraphSetConditional(handle, 0);
  }
}

template <typename vertex_t>
static __global__ void finalize_iteration_kernel(cudaGraphConditionalHandle handle,
                                                 vertex_t* state,
                                                 vertex_t depth_limit)
{
  auto beta_count = state[state_beta_word_count];
  if (beta_count != 0 && state[state_depth] + 1 < depth_limit) {
    state[state_depth] += 1;
    state[state_frontier_parity] = 1 - state[state_frontier_parity];
    state[state_alpha_word_count] = beta_count;
    state[state_beta_word_count] = 0;
    state[state_changed] = 1;
    cudaGraphSetConditional(handle, 1);
  } else {
    state[state_alpha_word_count] = 0;
    state[state_beta_word_count] = 0;
    state[state_changed] = 0;
    cudaGraphSetConditional(handle, 0);
  }
}

inline cudaError_t add_kernel_node(cudaGraph_t graph,
                                   cudaGraphNode_t* node,
                                   const cudaGraphNode_t* dependencies,
                                   size_t dependency_count,
                                   void* function,
                                   dim3 grid,
                                   dim3 block,
                                   unsigned int shared_memory,
                                   void** kernel_arguments)
{
  cudaKernelNodeParams params;
  std::memset(&params, 0, sizeof(params));
  params.func = function;
  params.gridDim = grid;
  params.blockDim = block;
  params.sharedMemBytes = shared_memory;
  params.kernelParams = kernel_arguments;
  return cudaGraphAddKernelNode(node, graph, dependencies, dependency_count, &params);
}

template <typename vertex_t, typename edge_t>
inline void create_initialized_frontier_graph(raw_csr_view_t<vertex_t, edge_t> raw,
                                             vertex_t* distances,
                                             uint32_t* dense_words,
                                             uint32_t* dense_dirty,
                                             uint32_t* dense_dirty_super,
                                             vertex_t* active_super_list,
                                             uint32_t* masks_a,
                                             uint32_t* masks_b,
                                             vertex_t* gamma_a,
                                             vertex_t* gamma_b,
                                             vertex_t* state,
                                             edge_t* deferred_begins,
                                             edge_t* deferred_ends,
                                             vertex_t word_count,
                                             vertex_t dirty_count,
                                             vertex_t dirty_super_count,
                                             vertex_t deferred_capacity,
                                             vertex_t depth_limit,
                                             cudaGraph_t* graph,
                                             cudaGraphExec_t* graph_exec)
{
  (void)dense_dirty_super;
  (void)active_super_list;
  (void)dirty_super_count;

  if (*graph_exec != nullptr) { return; }

  RAFT_CUDA_TRY(cudaGraphCreate(graph, 0));

  cudaGraphConditionalHandle conditional_handle = 0;
  RAFT_CUDA_TRY(cudaGraphConditionalHandleCreate(
    &conditional_handle, *graph, 1, cudaGraphCondAssignDefault));

  cudaGraphNode_t while_node = nullptr;
  cudaGraphNodeParams while_params = {};
  std::memset(&while_params, 0, sizeof(while_params));
  while_params.type = cudaGraphNodeTypeConditional;
  while_params.conditional.handle = conditional_handle;
  while_params.conditional.type = cudaGraphCondTypeWhile;
  while_params.conditional.size = 1;

  cudaGraphNode_t start_node = nullptr;
  cudaGraphConditionalHandle start_handle = conditional_handle;
  void* start_args[] = {&start_handle, &state, &depth_limit};
  RAFT_CUDA_TRY(add_kernel_node(*graph,
                                &start_node,
                                nullptr,
                                0,
                                reinterpret_cast<void*>(start_iteration_kernel<vertex_t>),
                                dim3(1),
                                dim3(1),
                                0,
                                start_args));

  RAFT_CUDA_TRY(cudaGraphAddNode(&while_node, *graph, &start_node, nullptr, 1, &while_params));
  cudaGraph_t body_graph = while_params.conditional.phGraph_out[0];

  cudaGraphNode_t reset_node = nullptr;
  void* reset_args[] = {&state};
  RAFT_CUDA_TRY(add_kernel_node(body_graph,
                                &reset_node,
                                nullptr,
                                0,
                                reinterpret_cast<void*>(reset_iteration_kernel<vertex_t>),
                                dim3(1),
                                dim3(1),
                                0,
                                reset_args));

  cudaGraphNode_t consume_node = nullptr;
  auto consume_blocks = std::max(
    1, static_cast<int>((word_count + compact_entries_per_block - 1) / compact_entries_per_block));
  raw_csr_view_t<vertex_t, edge_t> consume_raw = raw;
  void* consume_args[] = {&consume_raw,
                          &distances,
                          &dense_words,
                          &dense_dirty,
                          &masks_a,
                          &masks_b,
                          &gamma_a,
                          &gamma_b,
                          &deferred_begins,
                          &deferred_ends,
                          &deferred_capacity,
                          &state};
  RAFT_CUDA_TRY(add_kernel_node(
    body_graph,
    &consume_node,
    &reset_node,
    1,
    reinterpret_cast<void*>(consume_frontier_kernel<vertex_t, edge_t>),
    dim3(consume_blocks),
    dim3(block_size),
    0,
    consume_args));

  cudaGraphNode_t deferred_node = nullptr;
  raw_csr_view_t<vertex_t, edge_t> deferred_raw = raw;
  void* deferred_args[] = {&deferred_raw,
                           &distances,
                           &dense_words,
                           &dense_dirty,
                           &deferred_begins,
                           &deferred_ends,
                           &deferred_capacity,
                           &state};
  RAFT_CUDA_TRY(add_kernel_node(
    body_graph,
    &deferred_node,
    &consume_node,
    1,
    reinterpret_cast<void*>(consume_deferred_edges_kernel<vertex_t, edge_t>),
    dim3(deferred_edge_chunk_blocks),
    dim3(block_size),
    0,
    deferred_args));

  cudaGraphNode_t compact_node = nullptr;
  auto compact_blocks = std::max(
    1,
    static_cast<int>((dirty_count + compact_dirty_words_per_block - 1) /
                     compact_dirty_words_per_block));
  auto compact_block_size = 32 * compact_dirty_words_per_block;
  vertex_t beta_count_index = static_cast<vertex_t>(state_beta_word_count);
  void* compact_args[] = {&word_count,
                          &dense_words,
                          &dense_dirty,
                          &masks_a,
                          &masks_b,
                          &gamma_a,
                          &gamma_b,
                          &state,
                          &beta_count_index};
  RAFT_CUDA_TRY(add_kernel_node(
    body_graph,
    &compact_node,
    &deferred_node,
    1,
    reinterpret_cast<void*>(compact_dense_frontier_kernel<vertex_t>),
    dim3(compact_blocks),
    dim3(compact_block_size),
    0,
    compact_args));

  cudaGraphNode_t finalize_node = nullptr;
  cudaGraphConditionalHandle finalize_handle = conditional_handle;
  void* finalize_args[] = {&finalize_handle, &state, &depth_limit};
  RAFT_CUDA_TRY(add_kernel_node(
    body_graph,
    &finalize_node,
    &compact_node,
    1,
    reinterpret_cast<void*>(finalize_iteration_kernel<vertex_t>),
    dim3(1),
    dim3(1),
    0,
    finalize_args));

  RAFT_CUDA_TRY(cudaGraphInstantiate(graph_exec, *graph, 0));
}

template <typename vertex_t, typename edge_t>
inline void run_initialized_frontier(raw_csr_view_t<vertex_t, edge_t> raw,
                                     vertex_t* distances,
                                     uint32_t* dense_words,
                                     uint32_t* dense_dirty,
                                     uint32_t* dense_dirty_super,
                                     vertex_t* active_super_list,
                                     uint32_t* masks_a,
                                     uint32_t* masks_b,
                                     vertex_t* gamma_a,
                                     vertex_t* gamma_b,
                                     vertex_t* state,
                                     edge_t* deferred_begins,
                                     edge_t* deferred_ends,
                                     vertex_t word_count,
                                     vertex_t dirty_count,
                                     vertex_t dirty_super_count,
                                     vertex_t deferred_capacity,
                                     vertex_t depth_limit,
                                     rmm::cuda_stream_view stream,
                                     cudaGraph_t* graph,
                                     cudaGraphExec_t* graph_exec)
{
  create_initialized_frontier_graph(raw,
                                    distances,
                                    dense_words,
                                    dense_dirty,
                                    dense_dirty_super,
                                    active_super_list,
                                    masks_a,
                                    masks_b,
                                    gamma_a,
                                    gamma_b,
                                    state,
                                    deferred_begins,
                                    deferred_ends,
                                    word_count,
                                    dirty_count,
                                    dirty_super_count,
                                    deferred_capacity,
                                    depth_limit,
                                    graph,
                                    graph_exec);
  RAFT_CUDA_TRY(cudaGraphLaunch(*graph_exec, stream.value()));
  stream.synchronize();
}

template <typename vertex_t, typename edge_t>
inline void run_single_source(raw_csr_view_t<vertex_t, edge_t> raw,
                              vertex_t* distances,
                              vertex_t source,
                              vertex_t depth_limit,
                              rmm::cuda_stream_view stream)
{
  CUGRAPH_EXPECTS((source >= 0) && (source < raw.num_vertices),
                  "Invalid input argument: source is out of range.");

  rmm::device_uvector<uint32_t> dense_words((static_cast<size_t>(raw.num_vertices) + 31) / 32,
                                            stream);
  auto word_count = static_cast<vertex_t>(dense_words.size());
  auto dirty_count = static_cast<vertex_t>((word_count + words_per_dirty - 1) / words_per_dirty);
  auto dirty_super_count =
    static_cast<vertex_t>((dirty_count + dirty_words_per_super - 1) / dirty_words_per_super);

  rmm::device_uvector<uint32_t> dense_dirty(dirty_count, stream);
  rmm::device_uvector<uint32_t> dense_dirty_super(dirty_super_count, stream);
  rmm::device_uvector<vertex_t> active_super_list(dirty_super_count, stream);
  rmm::device_uvector<uint32_t> masks_a(word_count, stream);
  rmm::device_uvector<uint32_t> masks_b(word_count, stream);
  rmm::device_uvector<vertex_t> gamma_a(word_count, stream);
  rmm::device_uvector<vertex_t> gamma_b(word_count, stream);
  rmm::device_uvector<vertex_t> state(state_count, stream);
  rmm::device_uvector<edge_t> deferred_begins(
    (static_cast<size_t>(raw.num_edges) + deferred_edge_chunk_size - 1) / deferred_edge_chunk_size +
      word_count,
    stream);
  rmm::device_uvector<edge_t> deferred_ends(deferred_begins.size(), stream);

  auto init_blocks = static_cast<int>(
    (std::max(static_cast<size_t>(raw.num_vertices), static_cast<size_t>(word_count)) +
     block_size - 1) /
    block_size);
  init_blocks = std::max(1, std::min(init_blocks, 1024));
  init_kernel<<<init_blocks, block_size, 0, stream.value()>>>(raw,
                                                              source,
                                                              distances,
                                                              dense_words.data(),
                                                              dense_dirty.data(),
                                                              dense_dirty_super.data(),
                                                              active_super_list.data(),
                                                              masks_a.data(),
                                                              masks_b.data(),
                                                              gamma_a.data(),
                                                              gamma_b.data(),
                                                              state.data(),
                                                              word_count,
                                                              dirty_count,
                                                              dirty_super_count);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  run_initialized_frontier(raw,
                           distances,
                           dense_words.data(),
                           dense_dirty.data(),
                           dense_dirty_super.data(),
                           active_super_list.data(),
                           masks_a.data(),
                           masks_b.data(),
                           gamma_a.data(),
                           gamma_b.data(),
                           state.data(),
                           deferred_begins.data(),
                           deferred_ends.data(),
                           word_count,
                           dirty_count,
                           dirty_super_count,
                           static_cast<vertex_t>(deferred_begins.size()),
                           depth_limit,
                           stream,
                           &graph,
                           &graph_exec);
  if (graph_exec != nullptr) { cudaGraphExecDestroy(graph_exec); }
  if (graph != nullptr) { cudaGraphDestroy(graph); }
}

template <typename vertex_t, typename edge_t>
inline void run_multi_source(raw_csr_view_t<vertex_t, edge_t> raw,
                             vertex_t* distances,
                             vertex_t const* sources,
                             size_t n_sources,
                             vertex_t depth_limit,
                             rmm::cuda_stream_view stream,
                             workspace_t<vertex_t, edge_t>& workspace)
{
  CUGRAPH_EXPECTS(n_sources > 0, "Invalid input argument: sources cannot be empty.");

  auto word_count = static_cast<vertex_t>(workspace.dense_words.size());
  auto dirty_count = static_cast<vertex_t>((word_count + words_per_dirty - 1) / words_per_dirty);
  auto dirty_super_count =
    static_cast<vertex_t>((dirty_count + dirty_words_per_super - 1) / dirty_words_per_super);
  CUGRAPH_EXPECTS(static_cast<vertex_t>(workspace.dense_dirty.size()) == dirty_count,
                  "Invalid DAWN workspace: dense_dirty has the wrong size.");
  CUGRAPH_EXPECTS(static_cast<vertex_t>(workspace.dense_dirty_super.size()) == dirty_super_count,
                  "Invalid DAWN workspace: dense_dirty_super has the wrong size.");

  auto init_items = std::max(
    std::max(static_cast<size_t>(raw.num_vertices), static_cast<size_t>(word_count)),
    std::max(static_cast<size_t>(dirty_count), static_cast<size_t>(dirty_super_count)));
  auto init_blocks =
    std::max(1, std::min(static_cast<int>((init_items + block_size - 1) / block_size), 1024));
  init_multi_source_workspace_kernel<<<init_blocks, block_size, 0, stream.value()>>>(
    distances,
    workspace.dense_words.data(),
    workspace.dense_dirty.data(),
    workspace.dense_dirty_super.data(),
    workspace.active_super_list.data(),
    workspace.masks_a.data(),
    workspace.masks_b.data(),
    workspace.gamma_a.data(),
    workspace.gamma_b.data(),
    workspace.state.data(),
    raw.num_vertices,
    word_count,
    dirty_count,
    dirty_super_count);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto seed_blocks =
    std::max(1, std::min(static_cast<int>((n_sources + block_size - 1) / block_size), 1024));
  seed_source_distances_kernel<<<seed_blocks, block_size, 0, stream.value()>>>(
    sources, static_cast<vertex_t>(n_sources), distances);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto compact_block_size = 32 * compact_dirty_words_per_block;
  auto compact_blocks = std::max(
    1,
    static_cast<int>((dirty_count + compact_dirty_words_per_block - 1) /
                     compact_dirty_words_per_block));
  if (depth_limit > 0) {
    auto neighbor_seed_blocks = std::max(1, std::min(static_cast<int>(n_sources), 65535));
    seed_multi_source_neighbors_kernel<<<neighbor_seed_blocks, block_size, 0, stream.value()>>>(
      raw,
      sources,
      static_cast<vertex_t>(n_sources),
      distances,
      workspace.dense_words.data(),
      workspace.dense_dirty.data(),
      workspace.state.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    compact_dense_frontier_kernel<<<compact_blocks, compact_block_size, 0, stream.value()>>>(
      word_count,
      workspace.dense_words.data(),
      workspace.dense_dirty.data(),
      workspace.masks_a.data(),
      workspace.masks_b.data(),
      workspace.gamma_a.data(),
      workspace.gamma_b.data(),
      workspace.state.data(),
      static_cast<vertex_t>(state_alpha_word_count));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  run_initialized_frontier(raw,
                           distances,
                           workspace.dense_words.data(),
                           workspace.dense_dirty.data(),
                           workspace.dense_dirty_super.data(),
                           workspace.active_super_list.data(),
                           workspace.masks_a.data(),
                           workspace.masks_b.data(),
                           workspace.gamma_a.data(),
                           workspace.gamma_b.data(),
                           workspace.state.data(),
                           workspace.deferred_begins.data(),
                           workspace.deferred_ends.data(),
                           word_count,
                           dirty_count,
                           dirty_super_count,
                           static_cast<vertex_t>(workspace.deferred_begins.size()),
                           depth_limit,
                           stream,
                           &workspace.graph,
                           &workspace.graph_exec);
}

template <typename vertex_t, typename edge_t>
inline void run_multi_source(raw_csr_view_t<vertex_t, edge_t> raw,
                             vertex_t* distances,
                             vertex_t const* sources,
                             size_t n_sources,
                             vertex_t depth_limit,
                             rmm::cuda_stream_view stream)
{
  workspace_t<vertex_t, edge_t> workspace(raw, stream);
  run_multi_source(raw, distances, sources, n_sources, depth_limit, stream, workspace);
}

}  // namespace dawn
}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
void dawn_bfs(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
              vertex_t* distances,
              vertex_t const* sources,
              size_t n_sources,
              vertex_t depth_limit,
              bool do_expensive_check)
{
  static_assert(!multi_gpu);

  CUGRAPH_EXPECTS(sources != nullptr, "Invalid input argument: source cannot be null.");
  CUGRAPH_EXPECTS(distances != nullptr, "Invalid input argument: distances cannot be null.");
  auto num_vertices = graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  auto stream = handle.get_stream();
  if (do_expensive_check) {
    std::vector<vertex_t> h_sources(n_sources);
    raft::update_host(h_sources.data(), sources, n_sources, stream);
    stream.synchronize();
    for (auto source : h_sources) {
      CUGRAPH_EXPECTS((source >= 0) && (source < num_vertices),
                      "Invalid input argument: source is out of range.");
    }
  }

  auto edge_partition = graph_view.local_edge_partition_view();
  auto offsets = edge_partition.offsets();
  auto indices = edge_partition.indices();

  detail::dawn::raw_csr_view_t<vertex_t, edge_t> raw{
    offsets.data(),
    indices.data(),
    static_cast<vertex_t>(num_vertices),
    static_cast<edge_t>(indices.size())};

  detail::dawn::run_multi_source(raw, distances, sources, n_sources, depth_limit, stream);

}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void dawn_bfs(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
              vertex_t* distances,
              vertex_t const* sources,
              size_t n_sources,
              vertex_t depth_limit,
              bool do_expensive_check,
              detail::dawn::workspace_t<vertex_t, edge_t>& workspace)
{
  static_assert(!multi_gpu);

  CUGRAPH_EXPECTS(sources != nullptr, "Invalid input argument: source cannot be null.");
  CUGRAPH_EXPECTS(distances != nullptr, "Invalid input argument: distances cannot be null.");
  auto num_vertices = graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  auto stream = handle.get_stream();
  if (do_expensive_check) {
    std::vector<vertex_t> h_sources(n_sources);
    raft::update_host(h_sources.data(), sources, n_sources, stream);
    stream.synchronize();
    for (auto source : h_sources) {
      CUGRAPH_EXPECTS((source >= 0) && (source < num_vertices),
                      "Invalid input argument: source is out of range.");
    }
  }

  auto edge_partition = graph_view.local_edge_partition_view();
  auto offsets = edge_partition.offsets();
  auto indices = edge_partition.indices();

  detail::dawn::raw_csr_view_t<vertex_t, edge_t> raw{
    offsets.data(),
    indices.data(),
    static_cast<vertex_t>(num_vertices),
    static_cast<edge_t>(indices.size())};

  detail::dawn::run_multi_source(
    raw, distances, sources, n_sources, depth_limit, stream, workspace);
}

}  // namespace cugraph

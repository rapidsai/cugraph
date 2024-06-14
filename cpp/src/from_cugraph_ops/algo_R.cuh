/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device.cuh"
#include "format.hpp"

#include <raft/random/rng.cuh>
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>

#include <algorithm>

namespace cugraph::ops::graph {

// single warp-separated field of type IdxT
template <typename IdxT>
using smem_algo_r_t = utils::smem_unit_simple_t<1, IdxT>;

template <typename IdxT, typename GenT, typename RandT>
__device__ __forceinline__ void warp_algo_r_index(IdxT* smem,
                                                  IdxT pop_size,
                                                  IdxT idx_offset,
                                                  int sample_size,
                                                  raft::random::DeviceState<GenT>& rng_state)
{
  auto lane = utils::lane_id();
  // first 'sample_size' are just copied
  CUGRAPH_OPS_UNROLL
  for (int i = lane; i < sample_size; i += utils::WARP_SIZE) {
    smem[i] = idx_offset + i;
  }
  auto sample_size_idxt = IdxT{sample_size};
  if (sample_size_idxt >= pop_size) return;

  // we must synchronize here since we have just written to smem
  utils::warp_sync();
  // TODO(mjoux): when we support more warps per node enable this
  //__syncthreads();

  auto idx_end = idx_offset + pop_size;
  auto n       = idx_offset + sample_size_idxt;
  auto flat_id = uint64_t{threadIdx.x + blockIdx.x * blockDim.x};
  GenT gen(rng_state, flat_id);
  CUGRAPH_OPS_UNROLL
  for (auto nidx = n + IdxT{lane}; nidx < idx_end; nidx += IdxT{utils::WARP_SIZE}) {
    // nidx - idx_offset inclusive (necessary for correctness of algo R)
    auto end = nidx - idx_offset + 1;
    raft::random::UniformIntDistParams<IdxT, RandT> int_params{};
    int_params.start = IdxT{0};
    int_params.end   = IdxT{end};
    int_params.diff  = static_cast<RandT>(end);
    IdxT idx;
    raft::random::custom_next(gen, &idx, int_params, 0, 0 /* idx / stride unused */);
    if (idx < sample_size_idxt) {
      // using atomic max instead of exch here because it leads to the same
      // output as the sequential algorithm (DGL does this, too)
      // Additionally, we use the index instead of the neighbor ID here
      // since this allows copying over other node/edge-related data
      // (useful for heterogeneous graphs for example)
      utils::atomic_max(smem + idx, nidx);
    }
  }
  // must synchronize to make smem valid
  utils::warp_sync();
  // TODO(mjoux): when we support more warps per node enable this
  //__syncthreads();
}

template <typename IdxT, typename GenT, typename RandT>
__device__ __forceinline__ void warp_algo_r(IdxT* smem,
                                            IdxT row_id,
                                            const IdxT* nodes,
                                            const IdxT* fg_offsets,
                                            int sample_size,
                                            IdxT& node_id,
                                            IdxT& node_start,
                                            IdxT& node_end,
                                            raft::random::DeviceState<GenT>& rng_state)
{
  auto lane = utils::lane_id();
  if (nodes == nullptr) {
    node_id = row_id;
    if (lane == 0)
      node_start = fg_offsets[node_id];
    else if (lane == 1)
      node_end = fg_offsets[node_id + 1];
    node_start = utils::shfl(node_start, 0);
    node_end   = utils::shfl(node_end, 1);
  } else {
    if (lane == 0) {
      node_id    = nodes[row_id];
      node_start = fg_offsets[node_id];
      node_end   = fg_offsets[node_id + 1];
    }
    node_id    = utils::shfl(node_id, 0);
    node_start = utils::shfl(node_start, 0);
    node_end   = utils::shfl(node_end, 0);
  }
  auto pop_size = node_end - node_start;
  warp_algo_r_index<IdxT, GenT, RandT>(smem, pop_size, node_start, sample_size, rng_state);
}

// TODO(mjoux): support configuring n_warps_per_node in template
template <typename RandT, int N_WARPS, bool SAMPLE_SELF, bool IS_HG, typename IdxT, typename GenT>
CUGRAPH_OPS_KERNEL void algo_r_kernel(raft::random::DeviceState<GenT> rng_state,
                                      IdxT* neighbors,
                                      IdxT* counts,
                                      // edge_types / node_types should be non-const
                                      // probably detected if `!IS_HG`
                                      // NOLINTNEXTLINE(readability-non-const-parameter)
                                      int32_t* edge_types,
                                      // NOLINTNEXTLINE(readability-non-const-parameter)
                                      int32_t* node_types,
                                      const IdxT* offsets,
                                      const IdxT* indices,
                                      const int32_t* g_edge_types,
                                      const int32_t* g_node_types,
                                      const IdxT* nodes,
                                      IdxT n_dst_nodes,
                                      int sample_size)
{
  auto lane   = utils::lane_id();
  auto warp   = utils::warp_id();  // 1D block with X dim
  auto row_id = warp + static_cast<IdxT>(blockIdx.x) * IdxT{N_WARPS};
  if (row_id >= n_dst_nodes) { return; }
  IdxT* s_idx;
  smem_algo_r_t<IdxT> smem{};
  int32_t smem_sizes[] = {sample_size};
  smem.set_ptrs(warp, N_WARPS, smem_sizes, s_idx);
  IdxT node_id, node_start, node_end;
  warp_algo_r<IdxT, GenT, RandT>(
    s_idx, row_id, nodes, offsets, sample_size, node_id, node_start, node_end, rng_state);

  IdxT count = 0;
  for (int i = lane; i < sample_size; i += utils::WARP_SIZE) {
    auto nidx = s_idx[i];
    // checking for node_end here because sample_size may be larger than
    // the total number of neighbors of the node
    auto val = nidx < node_end ? indices[nidx] : cugraph::legacy::ops::graph::INVALID_ID<IdxT>;
    // TODO(mjoux) it's possible that we break the ELLPACK format here since
    // if we set val to invalid, we should add it to end of list, rather
    // than simply at index "i". This is ignored for now since the case
    // where SAMPLE_SELF := false is rare and unconventional
    if (!SAMPLE_SELF && val == node_id) val = cugraph::legacy::ops::graph::INVALID_ID<IdxT>;
    auto local_id       = row_id * IdxT{sample_size} + i;
    neighbors[local_id] = val;
    if (val != cugraph::legacy::ops::graph::INVALID_ID<IdxT>) {
      ++count;
      if (IS_HG) edge_types[local_id] = g_edge_types[nidx];
    }
  }
  if (IS_HG && lane == 0) node_types[row_id] = g_node_types[node_id];
  if (counts != nullptr) {
    count = utils::warp_reduce(count);
    if (lane == 0) { counts[row_id] = count; }
  }
}

template <typename IdxT, bool SAMPLE_SELF, bool IS_HG>
void algo_r_impl(IdxT* neighbors,
                 IdxT* counts,
                 int32_t* edge_types,
                 int32_t* node_types,
                 raft::random::RngState& rng,
                 const IdxT* offsets,
                 const IdxT* indices,
                 const int32_t* g_edge_types,
                 const int32_t* g_node_types,
                 const IdxT* nodes,
                 IdxT n_dst_nodes,
                 IdxT g_n_dst_nodes,
                 IdxT sample_size,
                 IdxT max_val,
                 cudaStream_t stream)
{
  if (nodes == nullptr) { n_dst_nodes = g_n_dst_nodes; }
  ASSERT(n_dst_nodes <= g_n_dst_nodes,
         "Algo R: expected n_dst_nodes <= graph.n_dst_nodes (%ld > %ld)",
         long(n_dst_nodes),
         long(g_n_dst_nodes));
  ASSERT(
    static_cast<size_t>(sample_size) + 2 < static_cast<size_t>(std::numeric_limits<int>::max()),
    "Expected sample size [+2] to be lower than INT_MAX");
  static constexpr int TPB     = 512;
  static constexpr int N_WARPS = TPB / utils::WARP_SIZE;
  auto n_blks                  = utils::ceil_div<IdxT>(n_dst_nodes, N_WARPS);
  int sample_size_i            = static_cast<int>(sample_size);
  int32_t smem_sizes[]         = {sample_size_i};
  size_t smem_size             = smem_algo_r_t<IdxT>::get_size(N_WARPS, smem_sizes);
  if (static_cast<uint64_t>(max_val) < std::numeric_limits<uint32_t>::max()) {
    // we'll use the 32-bit based method for generating random integers
    // as we most likely do not need less bias
    RAFT_CALL_RNG_FUNC(
      rng,
      (algo_r_kernel<uint32_t, N_WARPS, SAMPLE_SELF, IS_HG><<<n_blks, TPB, smem_size, stream>>>),
      neighbors,
      counts,
      edge_types,
      node_types,
      offsets,
      indices,
      g_edge_types,
      g_node_types,
      nodes,
      n_dst_nodes,
      sample_size_i);
  } else {
    RAFT_CALL_RNG_FUNC(
      rng,
      (algo_r_kernel<uint64_t, N_WARPS, SAMPLE_SELF, IS_HG><<<n_blks, TPB, smem_size, stream>>>),
      neighbors,
      counts,
      edge_types,
      node_types,
      offsets,
      indices,
      g_edge_types,
      g_node_types,
      nodes,
      n_dst_nodes,
      sample_size_i);
  }
  // update the rng state (this is a pessimistic update as it is difficult to
  // compute the number of RNG calls done per thread!)
  auto thread_rs = utils::ceil_div<IdxT>(
    std::max(IdxT{0}, std::min(max_val, g_n_dst_nodes) - sample_size), utils::WARP_SIZE);
  rng.advance(static_cast<uint64_t>(n_blks * TPB), thread_rs);
  RAFT_CUDA_TRY(cudaGetLastError());
}

#if 0
/**
 * @brief Reservoir sampling algorithm R as described in the wiki page here:
 *        https://en.wikipedia.org/wiki/Reservoir_sampling#Simple_algorithm
 */
template <typename IdxT>
void algo_r(IdxT* neighbors,
            IdxT* counts,
            raft::random::RngState& rng,
            const graph::csc<IdxT>& graph,
            const IdxT* nodes,
            IdxT n_dst_nodes,
            IdxT sample_size,
            IdxT max_val,
            bool sample_self,
            cudaStream_t stream)
{
  if (sample_self) {
    algo_r_impl<IdxT, true, false>(neighbors,
                                   counts,
                                   nullptr,
                                   nullptr,
                                   rng,
                                   graph.offsets,
                                   graph.indices,
                                   nullptr,
                                   nullptr,
                                   nodes,
                                   n_dst_nodes,
                                   graph.n_dst_nodes,
                                   sample_size,
                                   max_val,
                                   stream);
  } else {
    algo_r_impl<IdxT, false, false>(neighbors,
                                    counts,
                                    nullptr,
                                    nullptr,
                                    rng,
                                    graph.offsets,
                                    graph.indices,
                                    nullptr,
                                    nullptr,
                                    nodes,
                                    n_dst_nodes,
                                    graph.n_dst_nodes,
                                    sample_size,
                                    max_val,
                                    stream);
  }
}

/**
 * @brief Reservoir sampling algorithm for heterogeneous graphs.
 *
 * @note Simply copies over out node and edge type information
 */
template <typename IdxT>
void algo_r(IdxT* neighbors,
            IdxT* counts,
            int32_t* edge_types,
            int32_t* dst_node_types,
            raft::random::RngState& rng,
            const graph::csc_hg<IdxT>& graph,
            const IdxT* nodes,
            IdxT n_dst_nodes,
            IdxT sample_size,
            IdxT max_val,
            bool sample_self,
            cudaStream_t stream)
{
  if (sample_self) {
    algo_r_impl<IdxT, true, true>(neighbors,
                                  counts,
                                  edge_types,
                                  dst_node_types,
                                  rng,
                                  graph.offsets,
                                  graph.indices,
                                  graph.edge_types,
                                  graph.node_types,
                                  nodes,
                                  n_dst_nodes,
                                  graph.n_dst_nodes,
                                  sample_size,
                                  max_val,
                                  stream);
  } else {
    algo_r_impl<IdxT, false, true>(neighbors,
                                   counts,
                                   edge_types,
                                   dst_node_types,
                                   rng,
                                   graph.offsets,
                                   graph.indices,
                                   graph.edge_types,
                                   graph.node_types,
                                   nodes,
                                   n_dst_nodes,
                                   graph.n_dst_nodes,
                                   sample_size,
                                   max_val,
                                   stream);
  }
}
#endif

}  // namespace cugraph::ops::graph

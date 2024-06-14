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
#include "kernel_params.cuh"

namespace cugraph::ops::utils {

/**
 * @brief Load the CSR offsets given full-graph offsets, and a warp-per-node approach
 *
 * This assumes that all threads of a warp work on the same output node,
 * and the index of that node is given by `node_id`.
 */
template <typename IdxT>
__device__ __forceinline__ void load_csc_offsets_warp(IdxT& off_start,
                                                      IdxT& off_end,
                                                      const IdxT* offsets,
                                                      IdxT node_id)
{
  int lid      = utils::lane_id();
  auto tmp_off = lid <= 1 ? offsets[node_id + lid] : IdxT{0};
  off_start    = utils::shfl(tmp_off, 0);
  off_end      = utils::shfl(tmp_off, 1);
}

/**
 * @brief Load the CSR indexes for a full-graph and a warp-per-node approach
 *
 * This assumes that all threads of a warp work on the same output node,
 * the edge ID to be loaded is given by `edge_id`.
 *
 * You can optionally specify:
 *  - an edge feature index which maps edge IDs to edge feature IDs
 *  - a reverse edge index which maps edge IDs to reverse edge IDs
 *  - in case an edge feature index and a reverse edge index is given,
 *    you can specify whether the feature index should be applied to the
 *    reverse ID immediately, using `USE_MAP_CSC_TO_COO`.
 *  - Both node and edge feature offsets can be added as well.
 */
template <int NEIGH_STRIDE, typename IdxT>
__device__ inline void load_csc_idx_warp(IdxT* s_idx_neigh,
                                         IdxT* s_idx_edge,
                                         IdxT* s_idx_rev,
                                         const IdxT* indices,
                                         const IdxT* rev_edge_ids,
                                         const IdxT* map_csc_to_coo,
                                         const IdxT* map_csc_to_coo_rev,
                                         IdxT edge_id,
                                         IdxT off_end,
                                         IdxT nf_offset = IdxT{0},
                                         IdxT ef_offset = IdxT{0})
{
  if (s_idx_neigh == nullptr && s_idx_edge == nullptr && s_idx_rev == nullptr) return;
  int lid = utils::lane_id();
  IdxT n  = edge_id + IdxT{lid};
  // the NEIGH_STRIDE == utils::WARP_SIZE check is compile-time constant
  // thus, with optimizations, we can avoid the check for `lid < NEIGH_STRIDE`
  // in case `NEIGH_STRIDE == utils::WARP_SIZE` holds.
  // we need to check `n < off_end` in any case
  if ((NEIGH_STRIDE == utils::WARP_SIZE || lid < NEIGH_STRIDE) && (n < off_end)) {
    if (s_idx_neigh != nullptr) s_idx_neigh[lid] = indices[n] + nf_offset;
    // we need both checks here to mimic behavior e.g. in agg_dmpnn where one only
    // loads s_idx_rev based on both map_csc_to_coo and rev_indices
    if (s_idx_edge != nullptr && map_csc_to_coo != nullptr)
      s_idx_edge[lid] = map_csc_to_coo[n] + ef_offset;
    else if (s_idx_edge != nullptr)
      s_idx_edge[lid] = n + ef_offset;

    if (s_idx_rev != nullptr && map_csc_to_coo_rev != nullptr)
      s_idx_rev[lid] = map_csc_to_coo_rev[rev_edge_ids[n]] + ef_offset;
    else if (rev_edge_ids != nullptr)
      s_idx_rev[lid] = rev_edge_ids[n] + ef_offset;
  }
  // make output indices available to all threads in a warp
  utils::warp_sync();
}

/**
 * @brief Load the CSR offsets given full-graph offsets, and a block-per-node approach
 *
 * This assumes that all threads of a block work on the same output node,
 * and the index of that node is given by `node_id`.
 */
template <typename IdxT>
__device__ __forceinline__ void load_csc_offsets_block(
  IdxT& off_start, IdxT& off_end, IdxT* s_offsets, const IdxT* offsets, IdxT node_id)
{
  if (threadIdx.x <= 1) { s_offsets[threadIdx.x] = offsets[node_id + threadIdx.x]; }
  // make offsets available
  __syncthreads();

  off_start = s_offsets[0];
  off_end   = s_offsets[1];
}

/**
 * @brief Load the CSR indexes for a full-graph and a block-per-node approach
 *
 * This assumes that all threads of a block work on the same output node,
 * the edge ID to be loaded is given by `edge_id`.
 *
 * You can optionally specify:
 *  - an edge feature index which maps edge IDs to edge feature IDs
 *  - a reverse edge index which maps edge IDs to reverse edge IDs
 *  - in case an edge feature index and a reverse edge index is given,
 *    you can specify whether the feature index should be applied to the
 *    reverse ID immediately, using `USE_MAP_CSC_TO_COO`.
 *  - Both node and edge feature offsets can be added as well.
 */
template <typename IdxT,
          int NEIGH_STRIDE,
          bool LOAD_NEIGH,
          bool LOAD_MAP_CSC_TO_COO = false,
          bool LOAD_REV_IDX        = false,
          bool USE_MAP_CSC_TO_COO  = LOAD_MAP_CSC_TO_COO>
__device__ inline void load_csc_idx_block(IdxT* s_idx_neigh,
                                          IdxT* s_idx_edge,
                                          IdxT* s_idx_rev,
                                          const IdxT* indices,
                                          const IdxT* map_csc_to_coo,
                                          const IdxT* rev_edge_ids,
                                          IdxT edge_id,
                                          IdxT off_end,
                                          IdxT nf_offset = IdxT{0},
                                          IdxT ef_offset = IdxT{0})
{
  if (!LOAD_NEIGH && !LOAD_MAP_CSC_TO_COO && !LOAD_REV_IDX) return;
  IdxT n = edge_id + static_cast<IdxT>(threadIdx.x);
  // we need to check `n < off_end` in any case
  if (threadIdx.x < NEIGH_STRIDE && n < off_end) {
    if (LOAD_NEIGH) s_idx_neigh[threadIdx.x] = indices[n] + nf_offset;
    if (LOAD_MAP_CSC_TO_COO) s_idx_edge[threadIdx.x] = map_csc_to_coo[n] + ef_offset;
    if (LOAD_REV_IDX && USE_MAP_CSC_TO_COO)
      s_idx_rev[threadIdx.x] = map_csc_to_coo[rev_edge_ids[n]] + ef_offset;
    else if (LOAD_REV_IDX)
      s_idx_rev[threadIdx.x] = rev_edge_ids[n] + ef_offset;
  }
  // make output indices available to all threads in a warp
  __syncthreads();
}

}  // namespace cugraph::ops::utils

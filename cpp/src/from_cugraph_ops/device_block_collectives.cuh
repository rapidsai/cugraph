/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device_atomics.cuh"
#include "device_core.hpp"
#include "device_dim.cuh"
#include "device_warp_collectives.cuh"
#include "loads_stores.cuh"
#include "macros.hpp"

#include <cstdint>

namespace cugraph::ops::utils {

/**
 * Apply a block-wide sync only for `n_warps` warps
 *
 * @note using `__syncthreads()` may often be preferred over this, since
 * `__syncthreads` does not have the limitation that you need to know the exact
 * number of participating warps: in particular, exited threads are disregarded
 * for `__syncthreads` but not here (they must be discounted from `n_warps`)
 *
 * @param[in] n_warps number of participating warps. Must be >= 1 !
 *
 * @note This works on Pascal and earlier archs as well, but all threads of the
 * first `n_warps` warps of the block must participate
 */
__device__ inline void block_sync(int n_warps)
{
  auto n_threads = n_warps * WARP_SIZE;
  asm volatile("bar.sync 0, %0;" : : "r"(n_threads) : "memory");
}

/**
 * @brief block-level sum reduction
 *
 * @tparam DataT data type
 *
 * @param[in]    val  input value
 * @param[inout] s_data shared memory region needed for storing intermediate
 *               results. It must alteast be of size: `n_warps`. If
 *               you want to reuse this block of memory later, then make sure a
 *               `block_sync(n_warps)` is in place to avoid any data hazards.
 *               The pointer must be naturally aligned to `alignof(DataT)`.
 * @param[in]    n_warps number of participating warps. By default, all warps.
 *
 * @return only the thread0 will contain valid reduced result
 *
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block.
 *
 * @note All threads in the block must enter this function together
 *
 * TODO(mjoux) Expand this to support arbitrary reduction ops
 */
template <typename DataT, bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ inline DataT block_reduce(DataT val, DataT* s_data, int n_warps = 0)
{
  if (n_warps <= 0) n_warps = num_warps<USE_X, USE_Y, USE_Z>();
  auto lid = lane_id();
  auto wid = warp_id<USE_X, USE_Y, USE_Z>();
  val      = warp_reduce(val);
  if (lid == 0) { s_data[wid] = val; }
  block_sync(n_warps);
  if (wid == 0) {
    val = lid < n_warps ? s_data[lid] : DataT{0};
    val = warp_reduce(val);
  }
  return val;
}

// see `block_reduce` for details on `s_data`
// TODO(@stadlmax): check max w.r.t. __half2, __nv_bfloat162, we would need max
// for both halves independently
template <typename DataT, bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ inline DataT block_max_reduce(DataT val,
                                         DataT* s_data,
                                         DataT identity = DataT{0},
                                         int n_warps    = 0)
{
  if (n_warps <= 0) n_warps = num_warps<USE_X, USE_Y, USE_Z>();
  auto lid = lane_id();
  auto wid = warp_id<USE_X, USE_Y, USE_Z>();
  val      = warp_max_reduce(val);
  if (lid == 0) { s_data[wid] = val; }
  block_sync(n_warps);
  if (wid == 0) {
    val = lid < n_warps ? s_data[lid] : identity;
    val = warp_max_reduce(val);
  }
  return val;
}

// both shared memory regions must be at least of size `n_warps`
// see `block_reduce` for further details on shared memory regions
template <typename IdxT, typename DataT, bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ inline void block_max_idx_reduce(IdxT& idx,
                                            DataT& val,
                                            IdxT* s_idx,
                                            DataT* s_data,
                                            DataT identity = DataT{0},
                                            int i_identity = -1,
                                            int n_warps    = 0)
{
  if (n_warps <= 0) n_warps = num_warps<USE_X, USE_Y, USE_Z>();
  auto lid = lane_id();
  auto wid = warp_id<USE_X, USE_Y, USE_Z>();
  warp_max_idx_reduce(idx, val);
  if (lid == 0) {
    s_idx[wid]  = idx;
    s_data[wid] = val;
  }
  block_sync(n_warps);
  if (wid == 0) {
    idx = lid < n_warps ? s_idx[lid] : i_identity;
    val = lid < n_warps ? s_data[lid] : identity;
    warp_max_idx_reduce(idx, val);
  }
}

/**
 * @brief block-level lane-wise strided sum reduction
 *
 * All lanes X sum up their respective values.
 *
 * @tparam DataT data type
 *
 * @param[in]    val  input value
 * @param[inout] s_data shared memory region needed for storing intermediate
 *               results.
 *               It must alteast be of size: `WARP_SIZE` and initialized to 0!
 *               If you want to reuse this block of memory later, then make sure a
 *               `block_sync(n_warps)` is in place to avoid any data hazards.
 *               The pointer must be naturally aligned to `alignof(DataT)`.
 * @param[in]    n_warps number of participating warps. By default, all warps.
 *
 * @return only warp 0 (but all lanes of warp 0) will contain valid reduced results
 *
 * @note All threads in the block must enter this function together
 *
 * TODO(mjoux) Expand this to support arbitrary reduction ops
 */
template <typename DataT, bool USE_X = true, bool USE_Y = false, bool USE_Z = false>
__device__ inline DataT block_lane_reduce(DataT val, DataT* s_data, int n_warps = 0)
{
  if (n_warps <= 0) n_warps = num_warps<USE_X, USE_Y, USE_Z>();
  auto lid = lane_id();
  auto wid = warp_id<USE_X, USE_Y, USE_Z>();
  atomic_add(s_data + lid, val);
  block_sync(n_warps);
  DataT out{};
  if (wid == 0) out = s_data[lid];
  return out;
}

/**
 * @brief Block-strided copy
 *
 * @tparam DataT data type
 * @tparam IdxT  indexing type
 *
 * @param[out] out output array
 * @param[in]  in  input array
 * @param[in]  len number of elements to be copied
 */
template <typename DataT, typename IdxT>
__device__ inline void block_strided_copy(DataT* out, DataT* in, IdxT len)
{
  for (IdxT i = threadIdx.x; i < len; i += blockDim.x) {
    out[i] = in[i];
  }
}

/**
 * @brief Load a chunk of data into thread registers
 *
 * @tparam DataT       data type
 * @tparam IdxT        indexing type
 * @tparam NUM_ITEMS   number of items per thread
 * @tparam NUM_THREADS number of threads per block
 *
 * @param[out] vals        output registers
 * @param[out] smem        shared memory used for transpose. It must atleast be
 *                         of length `NUM_ITEMS * NUM_THREADS` in the non-vectorized
 *                         case and else, it must atleast be of length
 *                         `(NUM_ITEMS + VEC_LEN) * NUM_THREADS`. It can be safely
 *                         reused after a `__syncthreads()` call.
 *                         In the vectorized case, it must be aligned to
 *                         `alignof(DataT) * 16 / sizeof(DataT)`
 * @param[in]  in          input data to be loaded [on device] [len = `len`]
 * @param[in]  blk_offset  start offset for the threadblock from where to load
 *                         the data
 * @param[in]  len         length of the input array
 * @param[in]  default_val default value to be stored for OOB locations
 *
 * @note: Currently only tested for int32_t and int64_t (IOW for 4B/8B struct's)!
 */
template <typename DataT, typename IdxT, int NUM_ITEMS, int NUM_THREADS, bool TRY_VECTORIZED = true>
__device__ inline void block_load(DataT (&vals)[NUM_ITEMS],
                                  DataT* smem,
                                  const DataT* in,
                                  IdxT blk_offset,
                                  IdxT len,
                                  DataT default_val)
{
  static constexpr auto VEC_LEN   = 16 / sizeof(DataT);
  static constexpr auto NUM_LOADS = NUM_ITEMS / VEC_LEN;
  if constexpr (TRY_VECTORIZED && ((sizeof(DataT) * NUM_ITEMS) % 16 == 0) && NUM_LOADS > 0) {
    static constexpr auto STRIDE = NUM_ITEMS + (NUM_LOADS == 1 ? 0 : VEC_LEN);
    auto store_offset = (threadIdx.x / NUM_LOADS) * STRIDE + (threadIdx.x % NUM_LOADS) * VEC_LEN;
    auto load_offset  = blk_offset + threadIdx.x * VEC_LEN;
    // ldg + sts
    CUGRAPH_OPS_UNROLL
    for (int i = 0; i < NUM_LOADS; ++i) {
      DataT data[VEC_LEN];
      if (load_offset + VEC_LEN <= len) {
        ldg(data, in + load_offset);
      } else {
        // this will happen only for the last thread in the last block which
        // is supposed to load the last element from the input
        CUGRAPH_OPS_UNROLL
        for (int j = 0; j < VEC_LEN; ++j) {
          data[j] = load_offset + j < len ? in[load_offset + j] : default_val;
        }
      }
      sts(smem + store_offset, data);
      load_offset += NUM_THREADS * VEC_LEN;
      store_offset += (NUM_THREADS / NUM_LOADS) * STRIDE;
    }
    __syncthreads();
    // load to registers
    load_offset = threadIdx.x * STRIDE;
    CUGRAPH_OPS_UNROLL
    for (int i = 0; i < NUM_LOADS; ++i) {
      DataT data[VEC_LEN];
      lds(data, smem + load_offset + i * VEC_LEN);
      CUGRAPH_OPS_UNROLL
      for (int j = 0; j < VEC_LEN; ++j) {
        vals[i * VEC_LEN + j] = data[j];
      }
    }
  } else {
    static constexpr auto STRIDE = NUM_ITEMS + (NUM_ITEMS == 1 ? 0 : 1);
    auto s_offset                = (threadIdx.x / NUM_ITEMS) * STRIDE + (threadIdx.x % NUM_ITEMS);
    auto g_offset                = blk_offset + threadIdx.x;
    CUGRAPH_OPS_UNROLL
    for (int i = 0; i < NUM_ITEMS; ++i) {
      auto data = g_offset < len ? in[g_offset] : default_val;
      g_offset += NUM_THREADS;
      smem[s_offset] = data;
      s_offset += (NUM_THREADS / NUM_ITEMS) * STRIDE;
    }
    __syncthreads();
    CUGRAPH_OPS_UNROLL
    for (int i = 0; i < NUM_ITEMS; ++i) {
      vals[i] = smem[threadIdx.x * STRIDE + i];
    }
  }
}

}  // namespace cugraph::ops::utils

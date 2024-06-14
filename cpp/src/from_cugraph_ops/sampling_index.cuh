/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "algo_R.cuh"
#include "cudart.hpp"
#include "format.hpp"
#include "nvtx.hpp"
#include "sampling.hpp"

#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>

#include <type_traits>

namespace cugraph::legacy::ops::graph {

namespace utils = cugraph::ops::utils;
template <typename IdxT>
using smem_algo_r_t = utils::smem_unit_simple_t<1, IdxT>;

template <typename IdxT, typename GenT>
CUGRAPH_OPS_KERNEL void index_replace_kernel(raft::random::DeviceState<GenT> rng_state,
                                             IdxT* index,
                                             const IdxT* sizes,
                                             IdxT n_sizes,
                                             int sample_size)
{
  using rand_t = std::make_unsigned_t<IdxT>;
  // a warp-wide implementation.
  auto lane    = cugraph::ops::utils::lane_id();
  auto warp    = utils::warp_id();    // 1D block with X dim
  auto n_warps = utils::num_warps();  // 1D block with X dim
  auto row_id  = warp + static_cast<IdxT>(blockIdx.x) * IdxT{n_warps};
  if (row_id >= n_sizes) return;
  // 1. load population size (once per warp)
  IdxT size = IdxT{0};
  if (lane == 0) size = sizes[row_id];

  // 2. shuffle it to all threads in warp
  size = utils::shfl(size, 0);

  // 3. check valid size: possible early-out
  if (size <= 0) {
    CUGRAPH_OPS_UNROLL
    for (auto i = lane; i < sample_size; i += utils::WARP_SIZE) {
      index[row_id * IdxT{sample_size} + IdxT{i}] = graph::INVALID_ID<IdxT>;
    }
    return;
  }

  // 4. every thread generates its indexes
  auto flat_id = static_cast<uint64_t>(threadIdx.x + blockIdx.x * blockDim.x);
  GenT gen(rng_state, flat_id);
  raft::random::UniformIntDistParams<IdxT, rand_t> int_params{};
  int_params.start = IdxT{0};
  int_params.end   = size;
  int_params.diff  = static_cast<rand_t>(size);
  CUGRAPH_OPS_UNROLL
  for (auto i = lane; i < sample_size; i += utils::WARP_SIZE) {
    IdxT idx = IdxT{0};
    raft::random::custom_next(gen, &idx, int_params, 0, 0 /* idx / stride unused */);

    // 5. output index
    index[row_id * IdxT{sample_size} + IdxT{i}] = idx;
  }
}

template <typename IdxT>
void get_sampling_index_replace(IdxT* index,
                                raft::random::RngState& rng,
                                const IdxT* sizes,
                                IdxT n_sizes,
                                int32_t sample_size,
                                cudaStream_t stream)
{
  // keep thread per block fairly low since we can expect sample_size < warp_size
  // thus we want to have as many blocks as possible to increase parallelism
  static constexpr int TPB     = 128;
  static constexpr int N_WARPS = TPB / utils::WARP_SIZE;
  auto n_blks                  = utils::ceil_div<IdxT>(n_sizes, N_WARPS);
  RAFT_CALL_RNG_FUNC(
    rng, (index_replace_kernel<<<n_blks, TPB, 0, stream>>>), index, sizes, n_sizes, sample_size);
  auto thread_rs = utils::ceil_div<IdxT>(IdxT{sample_size}, utils::WARP_SIZE);
  rng.advance(static_cast<uint64_t>(n_blks * TPB), thread_rs * sizeof(IdxT) / sizeof(int32_t));
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <int N_WARPS, typename IdxT, typename GenT>
CUGRAPH_OPS_KERNEL void index_algo_r_kernel(raft::random::DeviceState<GenT> rng_state,
                                            IdxT* index,
                                            const IdxT* sizes,
                                            IdxT n_sizes,
                                            int sample_size)
{
  using rand_t = std::make_unsigned_t<IdxT>;
  // a warp-wide implementation.
  auto lane   = utils::lane_id();
  auto warp   = utils::warp_id();  // 1D block with X dim
  auto row_id = warp + static_cast<IdxT>(blockIdx.x) * IdxT{N_WARPS};
  if (row_id >= n_sizes) return;
  IdxT* s_idx;
  smem_algo_r_t<IdxT> smem{};
  int32_t smem_sizes[] = {sample_size};
  smem.set_ptrs(warp, N_WARPS, smem_sizes, s_idx);
  // 1. load population size (once per warp)
  IdxT size = IdxT{0};
  if (lane == 0) size = sizes[row_id];

  // 2. shuffle it to all threads in warp
  size = utils::shfl(size, 0);

  // 3. Get algo R indexes per warp
  cugraph::ops::graph::warp_algo_r_index<IdxT, GenT, rand_t>(
    s_idx, size, IdxT{0}, sample_size, rng_state);

  CUGRAPH_OPS_UNROLL
  for (auto i = lane; i < sample_size; i += utils::WARP_SIZE) {
    // 4. output index
    // still need to check if the index is actually valid
    auto idx                                    = s_idx[i];
    index[row_id * IdxT{sample_size} + IdxT{i}] = idx >= size ? graph::INVALID_ID<IdxT> : idx;
  }
}

template <typename IdxT>
void get_sampling_index_reservoir(IdxT* index,
                                  raft::random::RngState& rng,
                                  const IdxT* sizes,
                                  IdxT n_sizes,
                                  int32_t sample_size,
                                  cudaStream_t stream)
{
  // same TPB as in algo R: increased SM occupancy is most important here
  static constexpr int TPB     = 512;
  static constexpr int N_WARPS = TPB / utils::WARP_SIZE;
  auto n_blks                  = utils::ceil_div<IdxT>(n_sizes, N_WARPS);
  int32_t smem_sizes[]         = {sample_size};
  size_t smem_size             = smem_algo_r_t<IdxT>::get_size(N_WARPS, smem_sizes);
  RAFT_CALL_RNG_FUNC(rng,
                     (index_algo_r_kernel<N_WARPS><<<n_blks, TPB, smem_size, stream>>>),
                     index,
                     sizes,
                     n_sizes,
                     sample_size);
  auto thread_rs = utils::ceil_div<IdxT>(
    std::max(IdxT{0}, std::min(std::numeric_limits<IdxT>::max(), n_sizes) - IdxT{sample_size}),
    utils::WARP_SIZE);
  rng.advance(static_cast<uint64_t>(n_blks * TPB), thread_rs * sizeof(IdxT) / sizeof(int32_t));
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename IdxT>
void get_sampling_index_impl(IdxT* index,
                             raft::random::RngState& rng,
                             const IdxT* sizes,
                             IdxT n_sizes,
                             int32_t sample_size,
                             bool replace,
                             cudaStream_t stream)
{
  if (replace) {
    get_sampling_index_replace<IdxT>(index, rng, sizes, n_sizes, sample_size, stream);
  } else {
    get_sampling_index_reservoir<IdxT>(index, rng, sizes, n_sizes, sample_size, stream);
  }
}

}  // namespace cugraph::legacy::ops::graph

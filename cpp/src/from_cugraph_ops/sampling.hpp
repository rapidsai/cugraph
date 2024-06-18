/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

// FIXME: This is only here for the prims...
//   Need to look how Seunghwa fixed this in his PR
#include <cugraph/graph.hpp>

#include <raft/random/rng_state.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace cugraph::legacy::ops::graph {

/**
 * @brief Generate indexes given population sizes and a sample size,
 *        with or without replacement
 *
 * @param[out]   index          The (dense) index matrix. [on device]
 *                              [dim = `n_sizes x sample_size`]
 *                              In case `replace` is `false`, this may contain
 *                              `ops::graph::INVALID_ID<IdxT>`
 *                              if no index could be generated.
 * @param[inout] rng            RAFT RngState state object
 * @param[in]    sizes          Input array of population sizes [on device]
 *                              [len = `n_sizes`]
 * @param[in]    n_sizes        number of sizes to sample from.
 * @param[in]    sample_size    max number of indexes to be sampled per element
 *                              in `sizes`. Assumed to be <= 384 at the moment.
 * @param[in]    replace        If `true`, sample with replacement, otherwise
 *                              without replacement.
 * @param[in]    stream         cuda stream
 *
 @{
 */
void get_sampling_index(int32_t* index,
                        raft::random::RngState& rng,
                        const int32_t* sizes,
                        int32_t n_sizes,
                        int32_t sample_size,
                        bool replace,
                        cudaStream_t stream);
void get_sampling_index(int64_t* index,
                        raft::random::RngState& rng,
                        const int64_t* sizes,
                        int64_t n_sizes,
                        int32_t sample_size,
                        bool replace,
                        cudaStream_t stream);

}  // namespace cugraph::legacy::ops::graph

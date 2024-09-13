/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#include "sampling.hpp"
#include "sampling_index.cuh"

namespace cugraph::legacy::ops::graph {

void get_sampling_index(int32_t* index,
                        raft::random::RngState& rng,
                        const int32_t* sizes,
                        int32_t n_sizes,
                        int32_t sample_size,
                        bool replace,
                        cudaStream_t stream)
{
  get_sampling_index_impl(index, rng, sizes, n_sizes, sample_size, replace, stream);
}

void get_sampling_index(int64_t* index,
                        raft::random::RngState& rng,
                        const int64_t* sizes,
                        int64_t n_sizes,
                        int32_t sample_size,
                        bool replace,
                        cudaStream_t stream)
{
  get_sampling_index_impl(index, rng, sizes, n_sizes, sample_size, replace, stream);
}

}  // namespace cugraph::legacy::ops::graph

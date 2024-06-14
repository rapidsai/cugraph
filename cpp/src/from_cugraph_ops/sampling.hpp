/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "format.hpp"

#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

namespace cugraph::legacy::ops::graph {

/**
 * @brief Different kinds of sampling algorithms
 */
enum class SamplingAlgoT : uint8_t {
  /** Reservoir sampling (Algo R) */
  kReservoirAlgoR,
  /** Reservoir sampling (Algo L, single-threaded) */
  kReservoirAlgoLST,
};  // enum SamplingAlgoT

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
/** @} */

/**
 * @brief Uniform graph neighboorhood sampling technique
 *
 * @note explicit self-loops in the input graph are always sampled
 *       (like any other edge)
 *
 * @param[inout] rng            RAFT RngState state object
 * @param[in]    graph          input graph
 * @param[in]    nodes          list of node indices whose neighbors need to be
 *                              sampled [on device] [len = `n_dst_nodes`].
 *                              If this is a `nullptr`, then all nodes
 *                              neighborhood will be sampled.
 * @param[in]    n_dst_nodes        number of nodes to be sampled from.
 * @param[in]    sample_size    max number of nodes to be sampled per output node
 * @param[in]    type           sampling algorithm type
 * @param[in]    max_val        maximum node degree found in the graph. If not
 *                              used by the underlying algo, pass a `0`.
 * @param[in]    stream         cuda stream
 * @return tuple of device vectors representing CSR offsets and indices of
 *         the sub-sampled graph or COO source and destination indices
 *
 @{
 */
std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> uniform_sample_coo(
  raft::random::RngState& rng,
  const csc_s32_t& graph,
  const int32_t* nodes,
  int32_t n_dst_nodes,
  int32_t sample_size,
  SamplingAlgoT type,
  int32_t max_val,
  cudaStream_t stream);
std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> uniform_sample_coo(
  raft::random::RngState& rng,
  const csc_s64_t& graph,
  const int64_t* nodes,
  int64_t n_dst_nodes,
  int64_t sample_size,
  SamplingAlgoT type,
  int64_t max_val,
  cudaStream_t stream);
std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> uniform_sample_csc(
  raft::random::RngState& rng,
  const csc_s32_t& graph,
  const int32_t* nodes,
  int32_t n_dst_nodes,
  int32_t sample_size,
  SamplingAlgoT type,
  int32_t max_val,
  cudaStream_t stream);
std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> uniform_sample_csc(
  raft::random::RngState& rng,
  const csc_s64_t& graph,
  const int64_t* nodes,
  int64_t n_dst_nodes,
  int64_t sample_size,
  SamplingAlgoT type,
  int64_t max_val,
  cudaStream_t stream);
/** @} */

}  // namespace cugraph::legacy::ops::graph

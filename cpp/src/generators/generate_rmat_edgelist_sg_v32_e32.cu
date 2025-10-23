/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_rmat_edgelist.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/tuple>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <tuple>

namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_rmat_edgelist<int32_t>(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                bool clip_and_flip,
                                bool scramble_vertex_ids,
                                std::optional<large_buffer_type_t> large_buffer_type);

template std::vector<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>
generate_rmat_edgelists<int32_t>(raft::handle_t const& handle,
                                 raft::random::RngState& rng_state,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t size_distribution,
                                 generator_distribution_t edge_distribution,
                                 bool clip_and_flip,
                                 bool scramble_vertex_ids,
                                 std::optional<large_buffer_type_t> large_buffer_type);

}  // namespace cugraph

/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "generate_rmat_edgelist.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_rmat_edgelist<int64_t>(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                bool clip_and_flip,
                                bool scramble_vertex_ids);

template std::vector<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
generate_rmat_edgelists<int64_t>(raft::handle_t const& handle,
                                 raft::random::RngState& rng_state,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t size_distribution,
                                 generator_distribution_t edge_distribution,
                                 bool clip_and_flip,
                                 bool scramble_vertex_ids);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_rmat_edgelist<int64_t>(raft::handle_t const& handle,
                                size_t scale,
                                size_t num_edges,
                                double a,
                                double b,
                                double c,
                                uint64_t seed,
                                bool clip_and_flip,
                                bool scramble_vertex_ids);

template std::vector<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>>
generate_rmat_edgelists<int64_t>(raft::handle_t const& handle,
                                 size_t n_edgelists,
                                 size_t min_scale,
                                 size_t max_scale,
                                 size_t edge_factor,
                                 generator_distribution_t size_distribution,
                                 generator_distribution_t edge_distribution,
                                 uint64_t seed,
                                 bool clip_and_flip,
                                 bool scramble_vertex_ids);

}  // namespace cugraph

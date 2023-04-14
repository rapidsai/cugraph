/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#pragma once

#include <cugraph/graph_generators.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace cython {

struct graph_generator_t {
  std::unique_ptr<rmm::device_buffer> d_source;
  std::unique_ptr<rmm::device_buffer> d_destination;
};

// Wrapper for calling graph generator
template <typename vertex_t>
std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist(raft::handle_t const& handle,
                                                               size_t scale,
                                                               size_t num_edges,
                                                               double a,
                                                               double b,
                                                               double c,
                                                               uint64_t seed,
                                                               bool clip_and_flip,
                                                               bool scramble_vertex_ids);
template <typename vertex_t>
std::vector<std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists(raft::handle_t const& handle,
                             size_t n_edgelists,
                             size_t min_scale,
                             size_t max_scale,
                             size_t edge_factor,
                             cugraph::generator_distribution_t size_distribution,
                             cugraph::generator_distribution_t edge_distribution,
                             uint64_t seed,
                             bool clip_and_flip,
                             bool scramble_vertex_ids);

// Helper for setting up subcommunicators, typically called as part of the
// user-initiated comms initialization in Python.
//
// raft::handle_t& handle
//   Raft handle for which the new subcommunicators will be created. The
//   subcommunicators will then be accessible from the handle passed to the
//   parallel processes.
//
// size_t row_comm_size
//   Number of items in a partition row (ie. pcols), needed for creating the
//   appropriate number of subcommunicator instances.
void init_subcomms(raft::handle_t& handle, size_t row_comm_size);

}  // namespace cython
}  // namespace cugraph

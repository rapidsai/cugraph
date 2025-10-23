/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_bipartite_rmat_edgelist.cuh"

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
generate_bipartite_rmat_edgelist<int32_t>(raft::handle_t const& handle,
                                          raft::random::RngState& rng_state,
                                          size_t src_scale,
                                          size_t dst_scale,
                                          size_t num_edges,
                                          double a,
                                          double b,
                                          double c);

}  // namespace cugraph

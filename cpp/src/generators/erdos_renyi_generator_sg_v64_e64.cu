/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generators/erdos_renyi_generator.cuh"

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_erdos_renyi_graph_edgelist_gnp(raft::handle_t const& handle,
                                        int64_t num_vertices,
                                        float p,
                                        int64_t base_vertex_id,
                                        uint64_t seed);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_erdos_renyi_graph_edgelist_gnm(raft::handle_t const& handle,
                                        int64_t num_vertices,
                                        size_t m,
                                        int64_t base_vertex_id,
                                        uint64_t seed);

}  // namespace cugraph

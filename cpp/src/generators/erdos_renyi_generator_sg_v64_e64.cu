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

#include "generators/erdos_renyi_generator.cuh"

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/tuple.h>

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

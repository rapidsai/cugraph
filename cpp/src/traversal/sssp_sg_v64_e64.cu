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
#include "traversal/sssp_impl.cuh"

namespace cugraph {

// SG instantiation

template void sssp(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                   edge_property_view_t<int64_t, float const*> edge_weight_view,
                   float* distances,
                   int64_t* predecessors,
                   int64_t source_vertex,
                   float cutoff,
                   bool do_expensive_check);

template void sssp(raft::handle_t const& handle,
                   graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                   edge_property_view_t<int64_t, double const*> edge_weight_view,
                   double* distances,
                   int64_t* predecessors,
                   int64_t source_vertex,
                   double cutoff,
                   bool do_expensive_check);

}  // namespace cugraph

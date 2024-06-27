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

#include "generators/simple_generators.cuh"

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <numeric>

namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_path_graph_edgelist(raft::handle_t const& handle,
                             std::vector<std::tuple<int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_path_graph_edgelist(raft::handle_t const& handle,
                             std::vector<std::tuple<int64_t, int64_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int32_t, int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int64_t, int64_t, int64_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_complete_graph_edgelist(
  raft::handle_t const& handle, std::vector<std::tuple<int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
generate_complete_graph_edgelist(
  raft::handle_t const& handle, std::vector<std::tuple<int64_t, int64_t>> const& component_parms_v);

}  // namespace cugraph

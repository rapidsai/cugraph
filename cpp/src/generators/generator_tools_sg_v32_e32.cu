/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "generators/generator_tools.cuh"
#include "generators/scramble.cuh"

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <numeric>

namespace cugraph {

template rmm::device_uvector<int32_t> scramble_vertex_ids(raft::handle_t const& handle,
                                                          rmm::device_uvector<int32_t>&& vertices,
                                                          size_t lgN);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> scramble_vertex_ids(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& srcs,
  rmm::device_uvector<int32_t>&& dsts,
  size_t lgN);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<int32_t>>&& sources,
                  std::vector<rmm::device_uvector<int32_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<float>>>&& optional_d_weights,
                  bool remove_multi_edges);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<int32_t>>&& sources,
                  std::vector<rmm::device_uvector<int32_t>>&& dests,
                  std::optional<std::vector<rmm::device_uvector<double>>>&& optional_d_weights,
                  bool remove_multi_edges);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_src_v,
  rmm::device_uvector<int32_t>&& d_dst_v,
  std::optional<rmm::device_uvector<float>>&& optional_d_weights_v,
  bool check_diagonal);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_src_v,
  rmm::device_uvector<int32_t>&& d_dst_v,
  std::optional<rmm::device_uvector<double>>&& optional_d_weights_v,
  bool check_diagonal);

}  // namespace cugraph

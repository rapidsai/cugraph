/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "mtmg/vertex_result.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/mtmg/vertex_result_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <thrust/functional.h>
#include <thrust/gather.h>

namespace cugraph {
namespace mtmg {

template rmm::device_uvector<float> vertex_result_view_t<float>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  std::vector<int64_t> const& vertex_partition_range_lasts,
  vertex_partition_view_t<int64_t, true> vertex_partition_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view,
  float default_value);

template rmm::device_uvector<double> vertex_result_view_t<double>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  std::vector<int64_t> const& vertex_partition_range_lasts,
  vertex_partition_view_t<int64_t, true> vertex_partition_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view,
  double default_value);

template rmm::device_uvector<int64_t> vertex_result_view_t<int64_t>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  std::vector<int64_t> const& vertex_partition_range_lasts,
  vertex_partition_view_t<int64_t, true> vertex_partition_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view,
  int64_t default_value);

}  // namespace mtmg
}  // namespace cugraph

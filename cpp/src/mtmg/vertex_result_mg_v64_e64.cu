/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  vertex_partition_view_t<int64_t, true> vertex_partition_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view,
  float default_value);

template rmm::device_uvector<double> vertex_result_view_t<double>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  vertex_partition_view_t<int64_t, true> vertex_partition_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view,
  double default_value);

template rmm::device_uvector<int64_t> vertex_result_view_t<int64_t>::gather(
  handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  vertex_partition_view_t<int64_t, true> vertex_partition_view,
  std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view,
  int64_t default_value);

}  // namespace mtmg
}  // namespace cugraph

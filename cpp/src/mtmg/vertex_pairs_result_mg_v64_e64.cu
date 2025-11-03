/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/graph_partition_utils.cuh"
#include "mtmg/vertex_pairs_result.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/mtmg/vertex_pair_result_view.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <thrust/functional.h>
#include <thrust/gather.h>

namespace cugraph {
namespace mtmg {

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
  vertex_pair_result_view_t<int64_t, float>::gather(
    handle_t const& handle,
    raft::device_span<int64_t const> vertices,
    raft::host_span<int64_t const> vertex_partition_range_lasts,
    vertex_partition_view_t<int64_t, true> vertex_partition_view,
    std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
  vertex_pair_result_view_t<int64_t, double>::gather(
    handle_t const& handle,
    raft::device_span<int64_t const> vertices,
    raft::host_span<int64_t const> vertex_partition_range_lasts,
    vertex_partition_view_t<int64_t, true> vertex_partition_view,
    std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
  vertex_pair_result_view_t<int64_t, int64_t>::gather(
    handle_t const& handle,
    raft::device_span<int64_t const> vertices,
    raft::host_span<int64_t const> vertex_partition_range_lasts,
    vertex_partition_view_t<int64_t, true> vertex_partition_view,
    std::optional<cugraph::mtmg::renumber_map_view_t<int64_t>>& renumber_map_view);

}  // namespace mtmg
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generators/simple_generators.cuh"

#include <cugraph/graph_generators.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <numeric>

namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_path_graph_edgelist(raft::handle_t const& handle,
                             std::vector<std::tuple<int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int32_t, int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> const& component_parms_v);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
generate_complete_graph_edgelist(
  raft::handle_t const& handle, std::vector<std::tuple<int32_t, int32_t>> const& component_parms_v);

}  // namespace cugraph

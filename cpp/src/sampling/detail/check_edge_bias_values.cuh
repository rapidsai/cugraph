/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#pragma once

#include "prims/count_if_e.cuh"

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/count.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename bias_t, bool multi_gpu>
std::tuple<size_t, size_t> check_edge_bias_values(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view)
{
  auto num_negative_edge_weights =
    count_if_e(handle,
               graph_view,
               edge_src_dummy_property_t{}.view(),
               edge_dst_dummy_property_t{}.view(),
               edge_bias_view,
               [] __device__(vertex_t, vertex_t, auto, auto, bias_t b) { return b < 0.0; });

  size_t num_overflows{0};
  {
    auto bias_sums = compute_out_weight_sums(handle, graph_view, edge_bias_view);
    num_overflows  = thrust::count_if(
      handle.get_thrust_policy(), bias_sums.begin(), bias_sums.end(), [] __device__(auto sum) {
        return sum > std::numeric_limits<bias_t>::max();
      });
  }

  if constexpr (multi_gpu) {
    num_negative_edge_weights = host_scalar_allreduce(
      handle.get_comms(), num_negative_edge_weights, raft::comms::op_t::SUM, handle.get_stream());
    num_overflows = host_scalar_allreduce(
      handle.get_comms(), num_overflows, raft::comms::op_t::SUM, handle.get_stream());
  }

  return std::make_tuple(num_negative_edge_weights, num_overflows);
}

}  // namespace detail
}  // namespace cugraph

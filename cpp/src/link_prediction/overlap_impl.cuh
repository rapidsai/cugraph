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

#include "link_prediction/similarity_impl.cuh"

#include <cugraph/algorithms.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {
namespace detail {

struct overlap_functor_t {
  template <typename weight_t>
  weight_t __device__ compute_score(weight_t weight_a,
                                    weight_t weight_b,
                                    weight_t weight_a_intersect_b,
                                    weight_t weight_a_union_b) const
  {
    return std::min(weight_a, weight_b) <= std::numeric_limits<weight_t>::min()
             ? weight_t{0}
             : weight_a_intersect_b / std::min(weight_a, weight_b);
  }
};

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> overlap_coefficients(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::similarity(handle,
                            graph_view,
                            edge_weight_view,
                            vertex_pairs,
                            detail::overlap_functor_t{},
                            do_expensive_check);
}

}  // namespace cugraph

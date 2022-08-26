/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <link_prediction/similarity_impl.cuh>

#include <raft/handle.hpp>

namespace cugraph {
namespace detail {

struct sorensen_functor_t {
  template <typename weight_t>
  weight_t __device__ compute_score(size_t cardinality_a,
                                    size_t cardinality_b,
                                    size_t cardinality_a_intersect_b,
                                    weight_t,
                                    weight_t,
                                    weight_t)
  {
    return static_cast<weight_t>(2 * cardinality_a_intersect_b) /
           static_cast<weight_t>(cardinality_a + cardinality_b);
  }
};

struct weighted_sorensen_functor_t {
  template <typename weight_t>
  weight_t __device__ compute_score(size_t cardinality_a,
                                    size_t cardinality_b,
                                    size_t cardinality_a_intersect_b,
                                    weight_t weight_a,
                                    weight_t weight_b,
                                    weight_t min_weight_a_intersect_b)
  {
    return (2 * min_weight_a_intersect_b) / (weight_a + weight_b);
  }
};

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  sorensen(raft::handle_t const& handle,
           graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
           std::optional<raft::device_span<vertex_t const>> first,
           std::optional<raft::device_span<vertex_t const>> second,
           bool use_weights)
{
  if (use_weights)
    return detail::similarity(
      handle, graph_view, first, second, use_weights, detail::sorensen_functor_t{});
  else
    return detail::similarity(
      handle, graph_view, first, second, use_weights, detail::weighted_sorensen_functor_t{});
}

}  // namespace cugraph

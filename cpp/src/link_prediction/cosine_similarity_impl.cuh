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

template <typename weight_t>
struct cosine_functor_t {
  weight_t __device__ compute_score(weight_t norm_a,
                                    weight_t norm_b,
                                    weight_t sum_of_product_of_a_and_b,
                                    weight_t reserved_param) const
  {
    return sum_of_product_of_a_and_b / (norm_a * norm_b);
  }
};

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> cosine_similarity_coefficients(
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
                            detail::cosine_functor_t<weight_t>{},
                            detail::coefficient_t::COSINE,
                            do_expensive_check);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>>
  cosine_similarity_all_pairs_coefficients(
    raft::handle_t const& handle,
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::optional<raft::device_span<vertex_t const>> vertices,
    std::optional<size_t> topk,
    bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::all_pairs_similarity(handle,
                                      graph_view,
                                      edge_weight_view,
                                      vertices,
                                      topk,
                                      detail::cosine_functor_t<weight_t>{},
                                      detail::coefficient_t::COSINE,
                                      do_expensive_check);
}

}  // namespace cugraph

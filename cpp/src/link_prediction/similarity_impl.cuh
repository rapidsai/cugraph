/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <prims/per_v_pair_transform_dst_nbr_intersection.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/graph_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu, typename functor_t>
rmm::device_uvector<weight_t> similarity(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
  functor_t functor,
  bool do_expensive_check = false)
{
  using GraphViewType = graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  CUGRAPH_EXPECTS(std::get<0>(vertex_pairs).size() == std::get<1>(vertex_pairs).size(),
                  "vertex pairs have mismatched sizes");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");

  size_t num_vertex_pairs = std::get<0>(vertex_pairs).size();
  auto vertex_pairs_begin =
    thrust::make_zip_iterator(std::get<0>(vertex_pairs).data(), std::get<1>(vertex_pairs).data());

  if (edge_weight_view) {
    // FIXME: need implementation, similar to unweighted
    //    Use compute_out_weight_sums instead of compute_out_degrees
    //    Sum up for each common edge compute (u,a,v): min weight ((u,a), (a,v)) and
    //        max weight((u,a), (a,v)).
    //    Use these to compute weighted score
    //
    CUGRAPH_FAIL("weighted similarity computations are not supported in this release");
  } else {
    rmm::device_uvector<weight_t> similarity_score(num_vertex_pairs, handle.get_stream());

    //
    //  Compute vertex_degree for all vertices, then distribute to each GPU.
    //  Need to use this instead of the dummy properties below
    //
    auto out_degrees = graph_view.compute_out_degrees(handle);

    per_v_pair_transform_dst_nbr_intersection(
      handle,
      graph_view,
      vertex_pairs_begin,
      vertex_pairs_begin + num_vertex_pairs,
      out_degrees.begin(),
      [functor] __device__(auto v1, auto v2, auto v1_degree, auto v2_degree, auto intersection) {
        return functor.compute_score(static_cast<weight_t>(v1_degree),
                                     static_cast<weight_t>(v2_degree),
                                     static_cast<weight_t>(intersection.size()));
      },
      similarity_score.begin(),
      do_expensive_check);

    return similarity_score;
  }
}

}  // namespace detail
}  // namespace cugraph

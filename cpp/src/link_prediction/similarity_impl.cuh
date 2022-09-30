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

#include <prims/per_v_pair_transform_dst_nbr_intersection.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/graph_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu, typename functor_t>
rmm::device_uvector<weight_t> similarity(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
  bool use_weights,
  functor_t functor)
{
  using GraphViewType               = graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu>;
  constexpr bool do_expensive_check = false;

  CUGRAPH_EXPECTS(std::get<0>(vertex_pairs).size() == std::get<1>(vertex_pairs).size(),
                  "vertex pairs have mismatched sizes");

  if (use_weights)
    CUGRAPH_EXPECTS(graph_view.is_weighted(), "attempting to use weights on an unweighted graph");

  size_t num_vertex_pairs = std::get<0>(vertex_pairs).size();
  auto vertex_pairs_begin =
    thrust::make_zip_iterator(std::get<0>(vertex_pairs).data(), std::get<1>(vertex_pairs).data());

  if (use_weights) {
    // FIXME: need implementation, similar to unweighted
    //    Use compute_out_weight_sums instead of compute_out_degrees
    //    Sum up for each common edge compute (u,a,v): min weight ((u,a), (a,v)) and
    //        max weight((u,a), (a,v)).
    //    Use these to compute weighted score
    //
    CUGRAPH_FAIL("weighted similarity computations are not supported in this release");
  } else {
    auto intermediate_scores =
      cugraph::allocate_dataframe_buffer<thrust::tuple<edge_t, edge_t, edge_t>>(
        num_vertex_pairs, handle.get_stream());

    //
    //  Compute vertex_degree for all vertices, then distribute to each GPU.
    //  Need to use this instead of the dummy properties below
    //
    auto in_degrees = graph_view.compute_in_degrees(handle);

    auto src_degrees = edge_src_property_t<GraphViewType, edge_t>(handle, graph_view);
    auto dst_degrees = edge_dst_property_t<GraphViewType, edge_t>(handle, graph_view);
    update_edge_src_property(handle, graph_view, in_degrees.begin(), src_degrees);
    update_edge_dst_property(handle, graph_view, in_degrees.begin(), dst_degrees);

    //
    //  For each vertex pair compute the tuple: (src degree, dst degree, cardinality of
    //  intersection)
    //
    per_v_pair_transform_dst_nbr_intersection(
      handle,
      graph_view,
      vertex_pairs_begin,
      vertex_pairs_begin + num_vertex_pairs,
      in_degrees.begin(),
      // src_degrees.view(),
      // dst_degrees.view(),
      [] __device__(auto src, auto dst, auto src_degree, auto dst_degree, auto intersection) {
        return thrust::make_tuple(src_degree, dst_degree, static_cast<edge_t>(intersection.size()));
      },
      cugraph::get_dataframe_buffer_begin(intermediate_scores),
      do_expensive_check);

    //
    //  Convert to the desired score
    //
    rmm::device_uvector<weight_t> similarity_score(num_vertex_pairs, handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      get_dataframe_buffer_begin(intermediate_scores),
                      get_dataframe_buffer_end(intermediate_scores),
                      similarity_score.begin(),
                      [functor] __device__(auto tuple) {
                        auto src_degree        = static_cast<weight_t>(thrust::get<0>(tuple));
                        auto dst_degree        = static_cast<weight_t>(thrust::get<1>(tuple));
                        auto intersection_size = static_cast<weight_t>(thrust::get<2>(tuple));
                        return functor.compute_score(src_degree, dst_degree, intersection_size);
                      });

    return similarity_score;
  }
}

}  // namespace detail
}  // namespace cugraph

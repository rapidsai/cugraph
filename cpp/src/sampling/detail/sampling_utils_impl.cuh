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

#include <prims/extract_transform_v_frontier_outgoing_e.cuh>
#include <prims/per_v_random_select_transform_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>  // ??
#include <prims/vertex_frontier.cuh>

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/handle.hpp>

#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

// FIXME: Need to move this to a publicly available function
template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<edge_t>>
count_and_remove_duplicates(raft::handle_t const& handle,
                            rmm::device_uvector<vertex_t>&& src,
                            rmm::device_uvector<vertex_t>&& dst,
                            rmm::device_uvector<weight_t>&& wgt)
{
  auto tuple_iter_begin =
    thrust::make_zip_iterator(thrust::make_tuple(src.begin(), dst.begin(), wgt.begin()));

  thrust::sort(handle.get_thrust_policy(), tuple_iter_begin, tuple_iter_begin + src.size());

  auto num_uniques = thrust::count_if(handle.get_thrust_policy(),
                                      thrust::make_counting_iterator(size_t{0}),
                                      thrust::make_counting_iterator(src.size()),
                                      detail::is_first_in_run_t<decltype(tuple_iter_begin)>{tuple_iter_begin});

  rmm::device_uvector<vertex_t> result_src(num_uniques, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dst(num_uniques, handle.get_stream());
  rmm::device_uvector<weight_t> result_wgt(num_uniques, handle.get_stream());
  rmm::device_uvector<edge_t> result_count(num_uniques, handle.get_stream());

  rmm::device_uvector<edge_t> count(src.size(), handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), count.begin(), count.end(), edge_t{1});

  thrust::reduce_by_key(handle.get_thrust_policy(),
                        tuple_iter_begin,
                        tuple_iter_begin + src.size(),
                        count.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                          result_src.begin(), result_dst.begin(), result_wgt.begin())),
                        result_count.begin());

  return std::make_tuple(
    std::move(result_src), std::move(result_dst), std::move(result_wgt), std::move(result_count));
}

template <typename vertex_t, typename weight_t, typename property_t>
struct return_all_edges_e_op {
  using return_type = thrust::optional<thrust::tuple<vertex_t, vertex_t, weight_t>>;

  return_type __device__
  operator()(vertex_t src, vertex_t dst, weight_t wgt, property_t, property_t)
  {
    return thrust::make_optional(thrust::make_tuple(src, dst, wgt));
  }
};

template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  const rmm::device_uvector<typename GraphViewType::vertex_type>& active_majors,
  bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  // FIXME: add as a template parameter
  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(
    handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  // FIXME:  Shouldn't there be a dummy property equivalent here?
  using property_t = int32_t;
  cugraph::edge_src_property_t<GraphViewType, property_t> src_properties(handle, graph_view);
  cugraph::edge_dst_property_t<GraphViewType, property_t> dst_properties(handle, graph_view);

  auto [majors, minors, weights] = cugraph::extract_transform_v_frontier_outgoing_e(
    handle,
    graph_view,
    vertex_frontier.bucket(0),
    // edge_src_dummy_property_t{},
    // edge_dst_dummy_property_t{},
    src_properties.view(),
    dst_properties.view(),
    return_all_edges_e_op<vertex_t, weight_t, property_t>{},
    do_expensive_check);

  return std::make_tuple(
    std::move(majors), std::move(minors), std::make_optional(std::move(weights)));
}

template <typename vertex_t, typename weight_t, typename property_t>
struct sample_edges_op_t {
  using result_t = thrust::tuple<vertex_t, vertex_t, weight_t, property_t, property_t>;

  __device__ result_t operator()(
    vertex_t src, vertex_t dst, weight_t wgt, property_t src_prop, property_t dst_prop) const
  {
    return thrust::make_tuple(src, dst, wgt, src_prop, dst_prop);
  }
};

template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           std::optional<rmm::device_uvector<typename GraphViewType::weight_type>>>
sample_edges(raft::handle_t const& handle,
             GraphViewType const& graph_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<typename GraphViewType::vertex_type> const& active_majors,
             size_t fanout,
             bool with_replacement)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;

  // FIXME: add as a template parameter
  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(
    handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  // FIXME:  Shouldn't there be a dummy property equivalent here?
  using property_t = int32_t;
  cugraph::edge_src_property_t<GraphViewType, property_t> src_properties(handle, graph_view);
  cugraph::edge_dst_property_t<GraphViewType, property_t> dst_properties(handle, graph_view);

  using result_t = thrust::tuple<vertex_t, vertex_t, weight_t, property_t, property_t>;

  auto [sample_offsets, sample_e_op_results] = cugraph::per_v_random_select_transform_outgoing_e(
    handle,
    graph_view,
    vertex_frontier.bucket(0),
    src_properties.view(),
    dst_properties.view(),
    sample_edges_op_t<vertex_t, weight_t, property_t>{},
    rng_state,
    fanout,
    with_replacement,
    std::make_optional(result_t{cugraph::invalid_vertex_id<vertex_t>::value,
                                cugraph::invalid_vertex_id<vertex_t>::value}),
    true);

  //
  // FIXME: Debugging status, EOD 9/18/22
  //   2) Need to consider the case of a directed graph where a vertex is a sink but is selected
  //      as a seed for sampling.  Output degree is 0, so there can be no departing vertices.  ALso
  //      consider case where output degree is 1 and want 2 edges without replacement.
  //   3) Finally... can I switch to using cugraph::invalid_vertex_id<vertex_t> instead of
  //   number_of_vertices()? 4) I'm close, I should do the code cleanup.
  //
  auto end_iter = thrust::copy_if(handle.get_thrust_policy(),
                                  get_dataframe_buffer_begin(sample_e_op_results),
                                  get_dataframe_buffer_end(sample_e_op_results),
                                  get_dataframe_buffer_begin(sample_e_op_results),
                                  [] __device__(auto tuple) {
                                    auto v1 = thrust::get<0>(tuple);
                                    auto v2 = thrust::get<1>(tuple);

                                    return ((v1 != cugraph::invalid_vertex_id<vertex_t>::value) &&
                                            (v1 != cugraph::invalid_vertex_id<vertex_t>::value));
                                  });

  size_t new_size = thrust::distance(get_dataframe_buffer_begin(sample_e_op_results), end_iter);

  if (new_size != size_dataframe_buffer(sample_e_op_results)) {
    resize_dataframe_buffer(sample_e_op_results, new_size, handle.get_stream());
    shrink_to_fit_dataframe_buffer(sample_e_op_results, handle.get_stream());
  }

  auto& [majors, minors, weights, p1, p2] = sample_e_op_results;

  return std::make_tuple(
    std::move(majors), std::move(minors), std::make_optional(std::move(weights)));
}

}  // namespace detail
}  // namespace cugraph

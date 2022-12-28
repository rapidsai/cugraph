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

  auto num_uniques =
    thrust::count_if(handle.get_thrust_policy(),
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

template <typename vertex_t>
struct return_all_edges_e_op {
  template <typename EdgeProperties>
  auto __host__ __device__ operator()(vertex_t src,
                                      vertex_t dst,
                                      thrust::nullopt_t,
                                      thrust::nullopt_t,
                                      EdgeProperties edge_properties)
  {
    if constexpr (std::is_same_v<EdgeProperties, thrust::nullopt_t>) {
      return thrust::make_optional(thrust::make_tuple(src, dst));
    } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
      return thrust::make_optional(thrust::make_tuple(src, dst, edge_properties));
    } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                         (thrust::tuple_size<EdgeProperties>::value == 2)) {
      return thrust::make_optional(thrust::make_tuple(
        src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties)));
    } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                         (thrust::tuple_size<EdgeProperties>::value == 3)) {
      return thrust::make_optional(thrust::make_tuple(src,
                                                      dst,
                                                      thrust::get<0>(edge_properties),
                                                      thrust::get<1>(edge_properties),
                                                      thrust::get<2>(edge_properties)));
    }
  }
};

template <typename vertex_t>
struct sample_edges_op_t {
  template <typename EdgeProperties>
  auto __host__ __device__ operator()(vertex_t src,
                                      vertex_t dst,
                                      thrust::nullopt_t,
                                      thrust::nullopt_t,
                                      EdgeProperties edge_properties) const
  {
    if constexpr (std::is_same_v<EdgeProperties, thrust::nullopt_t>) {
      return thrust::make_tuple(src, dst);
    } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
      return thrust::make_tuple(src, dst, edge_properties);
    } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                         (thrust::tuple_size<EdgeProperties>::value == 2)) {
      return thrust::make_tuple(
        src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties));
    } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                         (thrust::tuple_size<EdgeProperties>::value == 3)) {
      return thrust::make_tuple(src,
                                dst,
                                thrust::get<0>(edge_properties),
                                thrust::get<1>(edge_properties),
                                thrust::get<2>(edge_properties));
    }
  }
};

// FIXME:  Old Implementation, phase out
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  const rmm::device_uvector<vertex_t>& active_majors,
  bool do_expensive_check)
{
  // FIXME: add as a template parameter
  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  if (edge_weight_view) {
    std::tie(majors, minors, weights) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       vertex_frontier.bucket(0),
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       *edge_weight_view,
                                                       return_all_edges_e_op<vertex_t>{},
                                                       do_expensive_check);
  } else {
    std::tie(majors, minors) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       vertex_frontier.bucket(0),
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       edge_dummy_property_t{}.view(),
                                                       return_all_edges_e_op<vertex_t>{},
                                                       do_expensive_check);
  }
  return std::make_tuple(std::move(majors), std::move(minors), std::move(weights));
}

// FIXME:  Old Implementation, phase out
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             raft::random::RngState& rng_state,
             rmm::device_uvector<vertex_t> const& active_majors,
             size_t fanout,
             bool with_replacement)
{
  // FIXME: add as a template parameter
  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  if (edge_weight_view) {
    auto [sample_offsets, sample_e_op_results] = cugraph::per_v_random_select_transform_outgoing_e(
      handle,
      graph_view,
      vertex_frontier.bucket(0),
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      *edge_weight_view,
      sample_edges_op_t<vertex_t>{},
      rng_state,
      fanout,
      with_replacement,
      std::make_optional<thrust::tuple<vertex_t, vertex_t, weight_t>>(
        cugraph::invalid_vertex_id<vertex_t>::value,
        cugraph::invalid_vertex_id<vertex_t>::value,
        weight_t{}),
      true);
    majors  = std::move(std::get<0>(sample_e_op_results));
    minors  = std::move(std::get<1>(sample_e_op_results));
    weights = std::move(std::get<2>(sample_e_op_results));
  } else {
    auto [sample_offsets, sample_e_op_results] = cugraph::per_v_random_select_transform_outgoing_e(
      handle,
      graph_view,
      vertex_frontier.bucket(0),
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_dummy_property_t{}.view(),
      sample_edges_op_t<vertex_t>{},
      rng_state,
      fanout,
      with_replacement,
      std::make_optional<thrust::tuple<vertex_t, vertex_t>>(
        cugraph::invalid_vertex_id<vertex_t>::value, cugraph::invalid_vertex_id<vertex_t>::value),
      true);
    majors = std::move(std::get<0>(sample_e_op_results));
    minors = std::move(std::get<1>(sample_e_op_results));
  }

  //
  // FIXME: Debugging status, EOD 9/18/22
  //   2) Need to consider the case of a directed graph where a vertex is a sink but is selected
  //      as a seed for sampling.  Output degree is 0, so there can be no departing vertices.  ALso
  //      consider case where output degree is 1 and want 2 edges without replacement.
  //   3) Finally... can I switch to using cugraph::invalid_vertex_id<vertex_t> instead of
  //   number_of_vertices()? 4) I'm close, I should do the code cleanup.
  //
  if (weights) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(majors.begin(), minors.begin(), (*weights).begin()));
    auto end_iter =
      thrust::remove_if(handle.get_thrust_policy(),
                        edge_first,
                        edge_first + majors.size(),
                        [] __device__(auto tuple) {
                          auto v1 = thrust::get<0>(tuple);
                          auto v2 = thrust::get<1>(tuple);

                          return ((v1 == cugraph::invalid_vertex_id<vertex_t>::value) ||
                                  (v1 == cugraph::invalid_vertex_id<vertex_t>::value));
                        });

    size_t new_size = thrust::distance(edge_first, end_iter);

    if (new_size != majors.size()) {
      majors.resize(new_size, handle.get_stream());
      majors.shrink_to_fit(handle.get_stream());
      minors.resize(new_size, handle.get_stream());
      minors.shrink_to_fit(handle.get_stream());
      (*weights).resize(new_size, handle.get_stream());
      (*weights).shrink_to_fit(handle.get_stream());
    }
  } else {
    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(majors.begin(), minors.begin()));
    auto end_iter =
      thrust::remove_if(handle.get_thrust_policy(),
                        edge_first,
                        edge_first + majors.size(),
                        [] __device__(auto tuple) {
                          auto v1 = thrust::get<0>(tuple);
                          auto v2 = thrust::get<1>(tuple);

                          return ((v1 == cugraph::invalid_vertex_id<vertex_t>::value) ||
                                  (v1 == cugraph::invalid_vertex_id<vertex_t>::value));
                        });

    size_t new_size = thrust::distance(edge_first, end_iter);

    if (new_size != majors.size()) {
      majors.resize(new_size, handle.get_stream());
      majors.shrink_to_fit(handle.get_stream());
      minors.resize(new_size, handle.get_stream());
      minors.shrink_to_fit(handle.get_stream());
    }
  }

  return std::make_tuple(std::move(majors), std::move(minors), std::move(weights));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<
    edge_property_view_t<edge_t,
                         thrust::zip_iterator<thrust::tuple<edge_t const*, edge_type_t const*>>>>
    edge_id_type_view,
  rmm::device_uvector<vertex_t> const& active_majors,
  std::optional<rmm::device_uvector<int32_t>> const& active_major_labels,
  bool do_expensive_check)
{
  // FIXME: add as a template parameter
  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  rmm::device_uvector<edge_t> edge_ids(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

  if (edge_weight_view) {
    if (edge_id_type_view) {
      auto edge_weight_type_id_view = view_concat(*edge_weight_view, *edge_id_type_view);

      std::tie(majors, minors, weights, edge_ids, edge_types) =
        cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                         graph_view,
                                                         vertex_frontier.bucket(0),
                                                         edge_src_dummy_property_t{}.view(),
                                                         edge_dst_dummy_property_t{}.view(),
                                                         edge_weight_type_id_view,
                                                         return_all_edges_e_op<vertex_t>{},
                                                         do_expensive_check);
    } else {
      std::tie(majors, minors, weights) =
        cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                         graph_view,
                                                         vertex_frontier.bucket(0),
                                                         edge_src_dummy_property_t{}.view(),
                                                         edge_dst_dummy_property_t{}.view(),
                                                         *edge_weight_view,
                                                         return_all_edges_e_op<vertex_t>{},
                                                         do_expensive_check);
    }
  } else {
    if (edge_id_type_view) {
      std::tie(majors, minors, edge_ids, edge_types) =
        cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                         graph_view,
                                                         vertex_frontier.bucket(0),
                                                         edge_src_dummy_property_t{}.view(),
                                                         edge_dst_dummy_property_t{}.view(),
                                                         *edge_id_type_view,
                                                         return_all_edges_e_op<vertex_t>{},
                                                         do_expensive_check);
    } else {
      std::tie(majors, minors) =
        cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                         graph_view,
                                                         vertex_frontier.bucket(0),
                                                         edge_src_dummy_property_t{}.view(),
                                                         edge_dst_dummy_property_t{}.view(),
                                                         edge_dummy_property_t{}.view(),
                                                         return_all_edges_e_op<vertex_t>{},
                                                         do_expensive_check);
    }
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(edge_ids),
                         std::move(weights),
                         std::move(edge_types),
                         std::move(labels));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>>
sample_edges(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<
    edge_property_view_t<edge_t,
                         thrust::zip_iterator<thrust::tuple<edge_t const*, edge_type_t const*>>>>
    edge_id_type_view,
  raft::random::RngState& rng_state,
  rmm::device_uvector<vertex_t> const& active_majors,
  std::optional<rmm::device_uvector<int32_t>> const& active_major_labels,
  size_t fanout,
  bool with_replacement)
{
  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  rmm::device_uvector<edge_t> edge_ids(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

  // NOTE:  sample_offsets breaks down the number of samples for each element in the frontier.
  //        If we convert that offsets array into it's own array we can use that to look up labels
  //   Meaning that I don't have to add the label as a tag - which makes the coding easier.  Fix
  //   that in the above code!
  if (edge_weight_view) {
    if (edge_id_type_view) {
      auto edge_weight_type_id_view = view_concat(*edge_weight_view, *edge_id_type_view);

      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_weight_type_id_view,
          sample_edges_op_t<vertex_t>{},
          rng_state,
          fanout,
          with_replacement,
          std::make_optional(thrust::make_tuple(cugraph::invalid_vertex_id<vertex_t>::value,
                                                cugraph::invalid_vertex_id<vertex_t>::value,
                                                weight_t{},
                                                edge_t{},
                                                edge_type_t{})),
          true);

      std::tie(majors, minors, weights, edge_ids, edge_types) = std::move(sample_e_op_results);
    } else {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          sample_edges_op_t<vertex_t>{},
          rng_state,
          fanout,
          with_replacement,
          std::make_optional<thrust::tuple<vertex_t, vertex_t, weight_t>>(
            cugraph::invalid_vertex_id<vertex_t>::value,
            cugraph::invalid_vertex_id<vertex_t>::value,
            weight_t{}),
          true);

      std::tie(majors, minors, weights) = std::move(sample_e_op_results);
    }
  } else {
    if (edge_id_type_view) {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          *edge_id_type_view,
          sample_edges_op_t<vertex_t>{},
          rng_state,
          fanout,
          with_replacement,
          std::make_optional(thrust::make_tuple(cugraph::invalid_vertex_id<vertex_t>::value,
                                                cugraph::invalid_vertex_id<vertex_t>::value,
                                                edge_t{},
                                                edge_type_t{})),
          true);

      std::tie(majors, minors, edge_ids, edge_types) = std::move(sample_e_op_results);
    } else {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          sample_edges_op_t<vertex_t>{},
          rng_state,
          fanout,
          with_replacement,
          std::make_optional<thrust::tuple<vertex_t, vertex_t>>(
            cugraph::invalid_vertex_id<vertex_t>::value,
            cugraph::invalid_vertex_id<vertex_t>::value),
          true);

      std::tie(majors, minors) = std::move(sample_e_op_results);
    }
  }

  if (weights) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(majors.begin(), minors.begin(), (*weights).begin()));
    auto end_iter =
      thrust::remove_if(handle.get_thrust_policy(),
                        edge_first,
                        edge_first + majors.size(),
                        [] __device__(auto tuple) {
                          auto v1 = thrust::get<0>(tuple);
                          auto v2 = thrust::get<1>(tuple);

                          return ((v1 == cugraph::invalid_vertex_id<vertex_t>::value) ||
                                  (v2 == cugraph::invalid_vertex_id<vertex_t>::value));
                        });

    size_t new_size = thrust::distance(edge_first, end_iter);

    if (new_size != majors.size()) {
      majors.resize(new_size, handle.get_stream());
      majors.shrink_to_fit(handle.get_stream());
      minors.resize(new_size, handle.get_stream());
      minors.shrink_to_fit(handle.get_stream());
      (*weights).resize(new_size, handle.get_stream());
      (*weights).shrink_to_fit(handle.get_stream());
        edge_ids.resize(new_size, handle.get_stream());
        edge_ids.shrink_to_fit(handle.get_stream());
      if(edge_types) {
        (*edge_types).resize(new_size, handle.get_stream());
        (*edge_types).shrink_to_fit(handle.get_stream());
      }
      if(labels) {
        (*labels).resize(new_size, handle.get_stream());
        (*labels).shrink_to_fit(handle.get_stream());
      }
    }
  } else {
    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(majors.begin(), minors.begin()));
    auto end_iter =
      thrust::remove_if(handle.get_thrust_policy(),
                        edge_first,
                        edge_first + majors.size(),
                        [] __device__(auto tuple) {
                          auto v1 = thrust::get<0>(tuple);
                          auto v2 = thrust::get<1>(tuple);

                          return ((v1 == cugraph::invalid_vertex_id<vertex_t>::value) ||
                                  (v2 == cugraph::invalid_vertex_id<vertex_t>::value));
                        });

    size_t new_size = thrust::distance(edge_first, end_iter);

    if (new_size != majors.size()) {
      majors.resize(new_size, handle.get_stream());
      majors.shrink_to_fit(handle.get_stream());
      minors.resize(new_size, handle.get_stream());
      minors.shrink_to_fit(handle.get_stream());
      edge_ids.resize(new_size, handle.get_stream());
      edge_ids.shrink_to_fit(handle.get_stream());
      if(edge_types) {
        (*edge_types).resize(new_size, handle.get_stream());
        (*edge_types).shrink_to_fit(handle.get_stream());
      }
      if(labels) {
        (*labels).resize(new_size, handle.get_stream());
        (*labels).shrink_to_fit(handle.get_stream());
      }
    }
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(edge_ids),
                         std::move(weights),
                         std::move(edge_types),
                         std::move(labels));
}

}  // namespace detail
}  // namespace cugraph

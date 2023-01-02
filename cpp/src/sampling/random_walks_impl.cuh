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

#include <detail/graph_utils.cuh>
#include <prims/per_v_random_select_transform_outgoing_e.cuh>
#include <prims/vertex_frontier.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/optional.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace detail {

inline uint64_t get_current_time_nanoseconds()
{
  auto cur = std::chrono::steady_clock::now();
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(cur.time_since_epoch()).count());
}

template <typename vertex_t, typename weight_t>
struct sample_edges_op_t {
  template <typename W = weight_t>
  __device__ std::enable_if_t<std::is_same_v<W, void>, vertex_t> operator()(
    vertex_t, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return dst;
  }

  template <typename W = weight_t>
  __device__ std::enable_if_t<!std::is_same_v<W, void>, thrust::tuple<vertex_t, W>> operator()(
    vertex_t, vertex_t dst, thrust::nullopt_t, thrust::nullopt_t, W w) const
  {
    return thrust::make_tuple(dst, w);
  }
};

template <typename weight_t>
struct uniform_selector {
  raft::random::RngState rng_state_;

  uniform_selector(uint64_t seed) : rng_state_(seed) {}

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type> const& current_vertices)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    // FIXME: add as a template parameter
    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(
      handle, 1);

    vertex_frontier.bucket(0).insert(current_vertices.begin(), current_vertices.end());

    rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    if (edge_weight_view) {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          sample_edges_op_t<vertex_t, weight_t>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(
            thrust::make_tuple(cugraph::invalid_vertex_id<vertex_t>::value, weight_t{0.0})));

      minors  = std::move(std::get<0>(sample_e_op_results));
      weights = std::move(std::get<1>(sample_e_op_results));
    } else {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          sample_edges_op_t<vertex_t, void>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}));

      minors = std::move(sample_e_op_results);
    }
    return std::make_tuple(std::move(minors), std::move(weights));
  }
};

template <typename weight_t>
struct biased_selector {
  uint64_t seed_{0};

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type> const& current_vertices)
  {
    //  To do biased sampling, I need out_weights instead of out_degrees.
    //  Then I generate a random float between [0, out_weights[v]).  Then
    //  instead of making a decision based on the index I need to find
    //  upper_bound (or is it lower_bound) of the random number and
    //  the cumulative weight.
    CUGRAPH_FAIL("biased sampling not implemented");
  }
};

template <typename weight_t>
struct node2vec_selector {
  weight_t p_;
  weight_t q_;
  uint64_t seed_{0};

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type> const& current_vertices)
  {
    //  To do node2vec, I need the following:
    //    1) transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v to compute the sum of the
    //       node2vec style weights
    //    2) Generate a random number between [0, output_from_trdnioeebv[v])
    //    3) a sampling value that lets me pick the correct edge based on the same computation
    //       (essentially weighted sampling, but with a function that computes the weight rather
    //       than just using the edge weights)
    CUGRAPH_FAIL("node2vec not implemented");
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename random_selector_t>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
random_walk_impl(raft::handle_t const& handle,
                 graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                 std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                 raft::device_span<vertex_t const> start_vertices,
                 size_t max_length,
                 random_selector_t random_selector)
{
  rmm::device_uvector<vertex_t> result_vertices(start_vertices.size() * (max_length + 1),
                                                handle.get_stream());
  auto result_weights = edge_weight_view
                          ? std::make_optional<rmm::device_uvector<weight_t>>(
                              start_vertices.size() * max_length, handle.get_stream())
                          : std::nullopt;

  detail::scalar_fill(handle,
                      result_vertices.data(),
                      result_vertices.size(),
                      cugraph::invalid_vertex_id<vertex_t>::value);
  if (result_weights)
    detail::scalar_fill(handle, result_weights->data(), result_weights->size(), weight_t{0});

  rmm::device_uvector<vertex_t> current_vertices(start_vertices.size(), handle.get_stream());
  rmm::device_uvector<size_t> current_position(0, handle.get_stream());
  rmm::device_uvector<int> current_gpu(0, handle.get_stream());
  auto new_weights = edge_weight_view
                       ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
                       : std::nullopt;

  if constexpr (multi_gpu) {
    current_position.resize(start_vertices.size(), handle.get_stream());
    current_gpu.resize(start_vertices.size(), handle.get_stream());

    raft::copy(
      current_vertices.data(), start_vertices.data(), start_vertices.size(), handle.get_stream());
    detail::scalar_fill(
      handle, current_gpu.data(), current_gpu.size(), handle.get_comms().get_rank());
    detail::sequence_fill(
      handle.get_stream(), current_position.data(), current_position.size(), size_t{0});
  } else {
    raft::copy(
      current_vertices.begin(), start_vertices.begin(), start_vertices.size(), handle.get_stream());
  }

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(current_vertices.size()),
    [current_verts = current_vertices.data(),
     result_verts  = result_vertices.data(),
     max_length] __device__(size_t i) { result_verts[i * (max_length + 1)] = current_verts[i]; });

  rmm::device_uvector<vertex_t> vertex_partition_range_lasts(
    graph_view.vertex_partition_range_lasts().size(), handle.get_stream());
  raft::update_device(vertex_partition_range_lasts.data(),
                      graph_view.vertex_partition_range_lasts().data(),
                      graph_view.vertex_partition_range_lasts().size(),
                      handle.get_stream());

  for (size_t level = 0; level < max_length; ++level) {
    if constexpr (multi_gpu) {
      // Shuffle vertices to correct GPU to compute random indices
      std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position),
                            std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          thrust::make_zip_iterator(
            current_vertices.begin(), current_gpu.begin(), current_position.begin()),
          thrust::make_zip_iterator(
            current_vertices.end(), current_gpu.end(), current_position.end()),
          [key_func =
             cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
               {vertex_partition_range_lasts.begin(),
                vertex_partition_range_lasts.size()}}] __device__(auto val) {
            return key_func(thrust::get<0>(val));
          },
          handle.get_stream());
    }

    std::tie(current_vertices, new_weights) =
      random_selector.follow_random_edge(handle, graph_view, edge_weight_view, current_vertices);

    if constexpr (multi_gpu) {
      //
      //  Now I can iterate over the tuples (current_vertices, new_weights, current_gpu,
      //  current_position) and skip over anything where current_vertices == invalid_vertex_id.
      //  There should, for any vertex, be at most one gpu where the vertex has a new vertex
      //  neighbor.
      //
      if (result_weights) {
        auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                    new_weights->begin(),
                                                    current_gpu.begin(),
                                                    current_position.begin());

        CUGRAPH_EXPECTS(current_vertices.size() < std::numeric_limits<int32_t>::max(),
                        "remove_if will fail, current_vertices.size() is too large");

        // FIXME: remove_if has a 32-bit overflow issue
        // (https://github.com/NVIDIA/thrust/issues/1302) Seems unlikely here (the goal of
        // sampling is to extract small graphs) so not going to work around this for now.
        auto compacted_length = thrust::distance(
          input_iter,
          thrust::remove_if(handle.get_thrust_policy(),
                            input_iter,
                            input_iter + current_vertices.size(),
                            current_vertices.begin(),
                            [] __device__(auto dst) {
                              return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                            }));

        current_vertices.resize(compacted_length, handle.get_stream());
        new_weights->resize(compacted_length, handle.get_stream());
        current_gpu.resize(compacted_length, handle.get_stream());
        current_position.resize(compacted_length, handle.get_stream());

        // Shuffle back to original GPU
        auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                      new_weights->begin(),
                                                      current_gpu.begin(),
                                                      current_position.begin());

        std::forward_as_tuple(
          std::tie(current_vertices, *new_weights, current_gpu, current_position), std::ignore) =
          cugraph::groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            current_iter,
            current_iter + current_vertices.size(),
            [] __device__(auto val) { return thrust::get<2>(val); },
            handle.get_stream());

        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(
            current_vertices.begin(), new_weights->begin(), current_position.begin()),
          thrust::make_zip_iterator(
            current_vertices.end(), new_weights->end(), current_position.end()),
          [result_verts = result_vertices.data(),
           result_wgts  = result_weights->data(),
           level,
           max_length] __device__(auto tuple) {
            vertex_t v                                       = thrust::get<0>(tuple);
            weight_t w                                       = thrust::get<1>(tuple);
            size_t pos                                       = thrust::get<2>(tuple);
            result_verts[pos * (max_length + 1) + level + 1] = v;
            result_wgts[pos * max_length + level]            = w;
          });
      } else {
        auto input_iter = thrust::make_zip_iterator(
          current_vertices.begin(), current_gpu.begin(), current_position.begin());

        CUGRAPH_EXPECTS(current_vertices.size() < std::numeric_limits<int32_t>::max(),
                        "remove_if will fail, current_vertices.size() is too large");

        auto compacted_length = thrust::distance(
          input_iter,
          // FIXME: remove_if has a 32-bit overflow issue
          // (https://github.com/NVIDIA/thrust/issues/1302) Seems unlikely here (the goal of
          // sampling is to extract small graphs) so not going to work around this for now.
          thrust::remove_if(handle.get_thrust_policy(),
                            input_iter,
                            input_iter + current_vertices.size(),
                            current_vertices.begin(),
                            [] __device__(auto dst) {
                              return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                            }));

        current_vertices.resize(compacted_length, handle.get_stream());
        current_gpu.resize(compacted_length, handle.get_stream());
        current_position.resize(compacted_length, handle.get_stream());

        // Shuffle back to original GPU
        auto current_iter = thrust::make_zip_iterator(
          current_vertices.begin(), current_gpu.begin(), current_position.begin());

        std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position),
                              std::ignore) =
          cugraph::groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            current_iter,
            current_iter + current_vertices.size(),
            [] __device__(auto val) { return thrust::get<1>(val); },
            handle.get_stream());

        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(current_vertices.begin(), current_position.begin()),
          thrust::make_zip_iterator(current_vertices.end(), current_position.end()),
          [result_verts = result_vertices.data(), level, max_length] __device__(auto tuple) {
            vertex_t v                                       = thrust::get<0>(tuple);
            size_t pos                                       = thrust::get<1>(tuple);
            result_verts[pos * (max_length + 1) + level + 1] = v;
          });
      }
    } else {
      if (result_weights) {
        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(current_vertices.size()),
                         [current_verts = current_vertices.data(),
                          new_wgts      = new_weights->data(),
                          result_verts  = result_vertices.data(),
                          result_wgts   = result_weights->data(),
                          level,
                          max_length] __device__(size_t i) {
                           result_verts[i * (max_length + 1) + level + 1] = current_verts[i];
                           result_wgts[i * max_length + level]            = new_wgts[i];
                         });
      } else {
        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(current_vertices.size()),
                         [current_verts = current_vertices.data(),
                          result_verts  = result_vertices.data(),
                          level,
                          max_length] __device__(size_t i) {
                           result_verts[i * (max_length + 1) + level + 1] = current_verts[i];
                         });
      }
    }
  }

  return std::make_tuple(std::move(result_vertices), std::move(result_weights));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     raft::device_span<vertex_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed)
{
  return detail::random_walk_impl(handle,
                                  graph_view,
                                  edge_weight_view,
                                  start_vertices,
                                  max_length,
                                  detail::uniform_selector<weight_t>(
                                    (seed == 0 ? detail::get_current_time_nanoseconds() : seed)));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                    raft::device_span<vertex_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed)
{
  return detail::random_walk_impl(
    handle,
    graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>>{edge_weight_view},
    start_vertices,
    max_length,
    detail::biased_selector<weight_t>{(seed == 0 ? detail::get_current_time_nanoseconds() : seed)});
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                      raft::device_span<vertex_t const> start_vertices,
                      size_t max_length,
                      weight_t p,
                      weight_t q,
                      uint64_t seed)
{
  return detail::random_walk_impl(
    handle,
    graph_view,
    edge_weight_view,
    start_vertices,
    max_length,
    detail::node2vec_selector<weight_t>{
      p, q, (seed == 0 ? detail::get_current_time_nanoseconds() : seed)});
}

}  // namespace cugraph

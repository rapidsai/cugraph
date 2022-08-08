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
#include <sampling/detail/graph_functions.hpp>
#include <sampling/detail/sampling_utils_impl.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/handle.hpp>
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

uint64_t get_current_time_nanoseconds()
{
  timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  return current_time.tv_sec * 1000000000 + current_time.tv_nsec;
}

struct uniform_selector {
  raft::random::RngState rng_state_;

  uniform_selector(uint64_t seed) : rng_state_(seed) {}

  template <typename vertex_t, typename edge_t>
  rmm::device_uvector<edge_t> get_random_indices(raft::handle_t const& handle,
                                                 rmm::device_uvector<vertex_t>& current_vertices,
                                                 rmm::device_uvector<edge_t>& out_degrees)
  {
    rmm::device_uvector<edge_t> reply(out_degrees.size(), handle.get_stream());
    rmm::device_uvector<double> random(out_degrees.size(), handle.get_stream());

    // pick a uniform random integer between 0 and out_degrees[i] - 1
    raft::random::uniform<double, size_t>(
      rng_state_, random.data(), random.size(), double{0}, double{1}, handle.get_stream());

    thrust::transform(handle.get_thrust_policy(),
                      thrust::make_zip_iterator(random.begin(), out_degrees.begin()),
                      thrust::make_zip_iterator(random.end(), out_degrees.end()),
                      reply.begin(),
                      [] __device__(auto t) {
                        double rnd        = thrust::get<0>(t);
                        edge_t out_degree = thrust::get<1>(t);

                        return (out_degree > 0) ? static_cast<edge_t>(rnd * out_degree)
                                                : edge_t{-1};
                      });

    return reply;
  }
};

struct biased_selector {
  uint64_t seed_{0};

  template <typename vertex_t, typename edge_t>
  rmm::device_uvector<edge_t> get_random_indices(raft::handle_t const& handle,
                                                 rmm::device_uvector<vertex_t>& current_vertices,
                                                 rmm::device_uvector<edge_t>& out_degrees)
  {
    CUGRAPH_FAIL("biased sampling not implemented");
  }
};

template <typename weight_t>
struct node2vec_selector {
  weight_t p_;
  weight_t q_;
  uint64_t seed_{0};

  template <typename vertex_t, typename edge_t>
  rmm::device_uvector<edge_t> get_random_indices(raft::handle_t const& handle,
                                                 rmm::device_uvector<vertex_t>& current_vertices,
                                                 rmm::device_uvector<edge_t>& out_degrees)
  {
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
                 graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                 raft::device_span<vertex_t const> start_vertices,
                 size_t max_length,
                 random_selector_t random_selector)
{
  // FIXME: This should be the global constant
  vertex_t invalid_vertex_id = graph_view.number_of_vertices();

  // preamble step for out-degree info:
  //
  auto&& [global_degree_offsets, global_out_degrees] =
    detail::get_global_degree_information(handle, graph_view);

  rmm::device_uvector<vertex_t> result_vertices(start_vertices.size() * (max_length + 1),
                                                handle.get_stream());
  auto result_weights = graph_view.is_weighted()
                          ? std::make_optional<rmm::device_uvector<weight_t>>(
                              start_vertices.size() * max_length, handle.get_stream())
                          : std::nullopt;

  thrust::fill(
    handle.get_thrust_policy(), result_vertices.begin(), result_vertices.end(), invalid_vertex_id);
  thrust::fill(
    handle.get_thrust_policy(), result_weights->begin(), result_weights->end(), weight_t{0});

  rmm::device_uvector<vertex_t> current_vertices(start_vertices.size(), handle.get_stream());
  rmm::device_uvector<size_t> current_position(0, handle.get_stream());
  rmm::device_uvector<int> current_gpu(0, handle.get_stream());
  auto new_weights = graph_view.is_weighted()
                       ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
                       : std::nullopt;

  if constexpr (multi_gpu) {
    current_position.resize(start_vertices.size(), handle.get_stream());
    current_gpu.resize(start_vertices.size(), handle.get_stream());
    auto current_iter = thrust::make_zip_iterator(
      current_vertices.begin(), current_gpu.begin(), current_position.begin());

    thrust::tabulate(handle.get_thrust_policy(),
                     current_iter,
                     current_iter + current_vertices.size(),
                     [my_gpu_id      = handle.get_comms().get_rank(),
                      start_vertices = start_vertices.begin()] __device__(auto i) {
                       return thrust::make_tuple(
                         start_vertices[i], my_gpu_id, static_cast<size_t>(i));
                     });
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
      // Shuffle current_vertices tuples
      auto current_iter = thrust::make_zip_iterator(
        current_vertices.begin(), current_gpu.begin(), current_position.begin());

      std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position),
                            std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          current_iter,
          current_iter + current_vertices.size(),
          [key_func =
             cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
               vertex_partition_range_lasts.begin(),
               vertex_partition_range_lasts.size()}] __device__(auto val) {
            return key_func(thrust::get<0>(val));
          },
          handle.get_stream());

      // Need to allgather across the col communicator
      auto const& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      size_t source_count  = current_vertices.size();

      auto external_source_counts =
        cugraph::host_scalar_allgather(col_comm, source_count, handle.get_stream());

      auto total_external_source_count =
        std::accumulate(external_source_counts.begin(), external_source_counts.end(), size_t{0});

      std::vector<size_t> displacements(external_source_counts.size(), size_t{0});
      std::exclusive_scan(external_source_counts.begin(),
                          external_source_counts.end(),
                          displacements.begin(),
                          size_t{0});

      rmm::device_uvector<vertex_t> active_vertices(total_external_source_count,
                                                    handle.get_stream());
      rmm::device_uvector<int> active_gpu(total_external_source_count, handle.get_stream());
      rmm::device_uvector<size_t> active_position(total_external_source_count, handle.get_stream());

      // Get the sources other gpus on the same row are working on
      // FIXME : replace with device_bcast for better scaling
      device_allgatherv(col_comm,
                        current_vertices.data(),
                        active_vertices.data(),
                        external_source_counts,
                        displacements,
                        handle.get_stream());
      device_allgatherv(col_comm,
                        current_gpu.data(),
                        active_gpu.data(),
                        external_source_counts,
                        displacements,
                        handle.get_stream());
      device_allgatherv(col_comm,
                        current_position.data(),
                        active_position.data(),
                        external_source_counts,
                        displacements,
                        handle.get_stream());
      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(
          active_vertices.begin(), active_gpu.begin(), active_position.begin()),
        thrust::make_zip_iterator(active_vertices.end(), active_gpu.end(), active_position.end()));

      current_vertices = std::move(active_vertices);
      current_gpu      = std::move(active_gpu);
      current_position = std::move(active_position);
    }

    auto&& out_degrees =
      get_active_major_global_degrees(handle, graph_view, current_vertices, global_out_degrees);

    // We have replicated the current vertices across the GPU columns.  We rely here upon the
    // fact that the random number generator will use the same seed on each GPU, therefore
    // we will generate the same random number on each of the GPUs for a particular current
    // vertex.
    //
    //  *** NOTE: to support node2vec, the current tuples also should specify a previous src
    //    (invalid_vertex for initialization).  Note, computing node2vec will be more
    //    complicated/expensive
    auto random_indices = random_selector.get_random_indices(handle, current_vertices, out_degrees);

    std::tie(std::ignore, current_vertices, new_weights) =
      detail::gather_local_edges(handle,
                                 graph_view,
                                 current_vertices,
                                 std::move(random_indices),
                                 edge_t{1},
                                 global_degree_offsets,
                                 false);

    if constexpr (multi_gpu) {
      //
      //  Now I can iterate over the tuples (current_vertices, new_weights, current_gpu,
      //  current_position) and skip over anything where current_vertices == invalid_vertex_id.
      //  There should, for any vertex, be at most one gpu where the vertex has a new vertex
      //  neighbor.
      //
      if (new_weights) {
        auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                    new_weights->begin(),
                                                    current_gpu.begin(),
                                                    current_position.begin());

        CUGRAPH_EXPECTS(current_vertices.size() < std::numeric_limits<int32_t>::max(),
                        "remove_if will fail, minors.size() is too large");

        // FIXME: remove_if has a 32-bit overflow issue
        // (https://github.com/NVIDIA/thrust/issues/1302) Seems unlikely here (the goal of sampling
        // is to extract small graphs) so not going to work around this for now.
        auto compacted_length = thrust::distance(
          input_iter,
          thrust::remove_if(
            handle.get_thrust_policy(),
            input_iter,
            input_iter + current_vertices.size(),
            current_vertices.begin(),
            [invalid_vertex_id] __device__(auto dst) { return (dst == invalid_vertex_id); }));

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

        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(current_vertices.size()),
                         [current_verts = current_vertices.data(),
                          new_wgts      = new_weights->data(),
                          current_pos   = current_position.begin(),
                          result_verts  = result_vertices.data(),
                          result_wgts   = result_weights->data(),
                          level,
                          max_length] __device__(size_t i) {
                           result_verts[current_pos[i] * (max_length + 1) + level + 1] =
                             current_verts[i];
                           result_wgts[current_pos[i] * max_length + level] = new_wgts[i];
                         });
      } else {
        auto input_iter = thrust::make_zip_iterator(
          current_vertices.begin(), current_gpu.begin(), current_position.begin());

        CUGRAPH_EXPECTS(current_vertices.size() < std::numeric_limits<int32_t>::max(),
                        "remove_if will fail, minors.size() is too large");

        auto compacted_length = thrust::distance(
          input_iter,
          // FIXME: remove_if has a 32-bit overflow issue
          // (https://github.com/NVIDIA/thrust/issues/1302) Seems unlikely here (the goal of
          // sampling is to extract small graphs) so not going to work around this for now.
          thrust::remove_if(
            handle.get_thrust_policy(),
            input_iter,
            input_iter + current_vertices.size(),
            current_vertices.begin(),
            [invalid_vertex_id] __device__(auto dst) { return (dst == invalid_vertex_id); }));

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

        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(current_vertices.size()),
                         [current_verts = current_vertices.data(),
                          current_pos   = current_position.data(),
                          result_verts  = result_vertices.data(),
                          level,
                          max_length] __device__(size_t i) {
                           result_verts[current_pos[i] * (max_length + 1) + level + 1] =
                             current_verts[i];
                         });
      }
    } else {
      if (new_weights) {
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
                     graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                     raft::device_span<vertex_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed)
{
  return detail::random_walk_impl(
    handle,
    graph_view,
    start_vertices,
    max_length,
    detail::uniform_selector((seed == 0 ? detail::get_current_time_nanoseconds() : seed)));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                    raft::device_span<vertex_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed)
{
  return detail::random_walk_impl(
    handle,
    graph_view,
    start_vertices,
    max_length,
    detail::biased_selector{(seed == 0 ? detail::get_current_time_nanoseconds() : seed)});
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<vertex_t, edge_t, weight_t, false, multi_gpu> const& graph_view,
                      raft::device_span<vertex_t const> start_vertices,
                      size_t max_length,
                      weight_t p,
                      weight_t q,
                      uint64_t seed)
{
  return detail::random_walk_impl(
    handle,
    graph_view,
    start_vertices,
    max_length,
    detail::node2vec_selector<weight_t>{
      p, q, (seed == 0 ? detail::get_current_time_nanoseconds() : seed)});
}

}  // namespace cugraph

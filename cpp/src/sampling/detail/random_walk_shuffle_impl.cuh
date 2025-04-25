/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "sampling_utils.hpp"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<int>,
           rmm::device_uvector<size_t>,
           std::optional<rmm::device_uvector<vertex_t>>>
random_walk_shuffle_input(raft::handle_t const& handle,
                          rmm::device_uvector<vertex_t>&& vertices,
                          rmm::device_uvector<int>&& gpus,
                          rmm::device_uvector<size_t>&& positions,
                          std::optional<rmm::device_uvector<vertex_t>>&& previous_vertices,
                          raft::device_span<vertex_t const> vertex_partition_range_lasts)
{
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t> key_func{
    vertex_partition_range_lasts, major_comm_size, minor_comm_size};

  if (previous_vertices) {
    std::forward_as_tuple(std::tie(vertices, gpus, positions, previous_vertices), std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        handle.get_comms(),
        thrust::make_zip_iterator(
          vertices.begin(), gpus.begin(), positions.begin(), previous_vertices->begin()),
        thrust::make_zip_iterator(
          vertices.end(), gpus.end(), positions.end(), previous_vertices->end()),
        [key_func] __device__(auto val) { return key_func(thrust::get<0>(val)); },
        handle.get_stream());

  } else {
    // Shuffle vertices to correct GPU to compute random indices
    std::forward_as_tuple(std::tie(vertices, gpus, positions), std::ignore) =
      cugraph::groupby_gpu_id_and_shuffle_values(
        handle.get_comms(),
        thrust::make_zip_iterator(vertices.begin(), gpus.begin(), positions.begin()),
        thrust::make_zip_iterator(vertices.end(), gpus.end(), positions.end()),
        [key_func] __device__(auto val) { return key_func(thrust::get<0>(val)); },
        handle.get_stream());
  }

  return std::make_tuple(
    std::move(vertices), std::move(gpus), std::move(positions), std::move(previous_vertices));
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<int>,
           rmm::device_uvector<size_t>,
           std::optional<rmm::device_uvector<vertex_t>>>
random_walk_shuffle_output(raft::handle_t const& handle,
                           rmm::device_uvector<vertex_t>&& vertices,
                           std::optional<rmm::device_uvector<weight_t>>&& weights,
                           rmm::device_uvector<int>&& gpus,
                           rmm::device_uvector<size_t>&& positions,
                           std::optional<rmm::device_uvector<vertex_t>>&& previous_vertices)
{
  if (previous_vertices) {
    if (weights) {
      auto current_iter = thrust::make_zip_iterator(gpus.begin(),
                                                    vertices.begin(),
                                                    positions.begin(),
                                                    weights->begin(),
                                                    previous_vertices->begin());

      std::forward_as_tuple(std::tie(gpus, vertices, positions, weights, previous_vertices),
                            std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          current_iter,
          current_iter + vertices.size(),
          [] __device__(auto val) { return thrust::get<0>(val); },
          handle.get_stream());
    } else {
      auto current_iter = thrust::make_zip_iterator(
        gpus.begin(), vertices.begin(), positions.begin(), previous_vertices->begin());

      std::forward_as_tuple(std::tie(gpus, vertices, positions, previous_vertices), std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          current_iter,
          current_iter + vertices.size(),
          [] __device__(auto val) { return thrust::get<0>(val); },
          handle.get_stream());
    }
  } else {
    if (weights) {
      auto current_iter = thrust::make_zip_iterator(
        gpus.begin(), vertices.begin(), positions.begin(), weights->begin());

      std::forward_as_tuple(std::tie(gpus, vertices, positions, weights), std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          current_iter,
          current_iter + vertices.size(),
          [] __device__(auto val) { return thrust::get<0>(val); },
          handle.get_stream());
    } else {
      auto current_iter =
        thrust::make_zip_iterator(gpus.begin(), vertices.begin(), positions.begin());

      std::forward_as_tuple(std::tie(gpus, vertices, positions), std::ignore) =
        cugraph::groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          current_iter,
          current_iter + vertices.size(),
          [] __device__(auto val) { return thrust::get<0>(val); },
          handle.get_stream());
    }
  }

  return std::make_tuple(std::move(vertices),
                         std::move(weights),
                         std::move(gpus),
                         std::move(positions),
                         std::move(previous_vertices));
}

}  // namespace detail
}  // namespace cugraph

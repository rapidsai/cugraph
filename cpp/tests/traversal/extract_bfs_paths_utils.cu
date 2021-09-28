/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>

#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/transform.h>

template <bool multi_gpu, typename vertex_t>
rmm::device_uvector<vertex_t> randomly_select_destinations(
  raft::handle_t const& handle,
  vertex_t number_of_vertices,
  rmm::device_uvector<vertex_t> const& d_predecessors,
  size_t num_paths_to_check,
  uint64_t seed)
{
  constexpr vertex_t invalid_vertex = cugraph::invalid_vertex_id<vertex_t>::value;

  rmm::device_uvector<vertex_t> d_vertices(number_of_vertices, handle.get_stream());
  rmm::device_uvector<float> d_probabilities(number_of_vertices, handle.get_stream());

  cugraph::detail::uniform_random_fill(
    handle.get_stream(), d_probabilities.data(), d_probabilities.size(), float{0}, float{1}, seed);

  cugraph::detail::sequence_fill(
    handle.get_stream(), d_vertices.begin(), d_vertices.size(), vertex_t{0});

  auto pair_iter =
    thrust::make_zip_iterator(thrust::make_tuple(d_predecessors.begin(), d_probabilities.begin()));

  thrust::transform(handle.get_thrust_policy(),
                    pair_iter,
                    pair_iter + d_predecessors.size(),
                    d_probabilities.begin(),
                    [invalid_vertex] __device__(auto tuple) {
                      vertex_t predecessor = thrust::get<0>(tuple);
                      return (predecessor == invalid_vertex) ? float{1} : thrust::get<1>(tuple);
                    });

  thrust::sort_by_key(
    handle.get_thrust_policy(), d_probabilities.begin(), d_probabilities.end(), d_vertices.data());

  size_t num_good_destinations = thrust::count_if(handle.get_thrust_policy(),
                                                  d_probabilities.begin(),
                                                  d_probabilities.end(),
                                                  [] __device__(auto f) { return f < float{1}; });

  d_vertices.resize(std::min(num_paths_to_check, num_good_destinations), handle.get_stream());
  d_vertices.shrink_to_fit(handle.get_stream());

  return d_vertices;
}

template rmm::device_uvector<int32_t> randomly_select_destinations<false>(
  raft::handle_t const& handle,
  int32_t number_of_vertices,
  rmm::device_uvector<int32_t> const& d_predecessors,
  size_t num_paths_to_check,
  uint64_t seed);

template rmm::device_uvector<int64_t> randomly_select_destinations<false>(
  raft::handle_t const& handle,
  int64_t number_of_vertices,
  rmm::device_uvector<int64_t> const& d_predecessors,
  size_t num_paths_to_check,
  uint64_t seed);

template rmm::device_uvector<int32_t> randomly_select_destinations<true>(
  raft::handle_t const& handle,
  int32_t number_of_vertices,
  rmm::device_uvector<int32_t> const& d_predecessors,
  size_t num_paths_to_check,
  uint64_t seed);

template rmm::device_uvector<int64_t> randomly_select_destinations<true>(
  raft::handle_t const& handle,
  int64_t number_of_vertices,
  rmm::device_uvector<int64_t> const& d_predecessors,
  size_t num_paths_to_check,
  uint64_t seed);

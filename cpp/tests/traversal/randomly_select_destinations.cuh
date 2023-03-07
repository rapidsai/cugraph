/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#pragma once

#include <utilities/thrust_wrapper.hpp>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/distance.h>
#include <thrust/remove.h>

namespace cugraph {
namespace test {

template <bool multi_gpu, typename vertex_t>
rmm::device_uvector<vertex_t> randomly_select_destinations(
  raft::handle_t const& handle,
  vertex_t number_of_vertices,
  vertex_t local_vertex_first,
  rmm::device_uvector<vertex_t> const& d_predecessors,
  size_t num_paths_to_check)
{
  constexpr vertex_t invalid_vertex = cugraph::invalid_vertex_id<vertex_t>::value;

  rmm::device_uvector<vertex_t> d_vertices(number_of_vertices, handle.get_stream());
  cugraph::detail::sequence_fill(
    handle.get_stream(), d_vertices.begin(), d_vertices.size(), local_vertex_first);

  auto end_iter = thrust::remove_if(
    handle.get_thrust_policy(),
    d_vertices.begin(),
    d_vertices.end(),
    [invalid_vertex, predecessors = d_predecessors.data(), local_vertex_first] __device__(auto v) {
      return predecessors[v - local_vertex_first] == invalid_vertex;
    });

  d_vertices.resize(thrust::distance(d_vertices.begin(), end_iter), handle.get_stream());

  return cugraph::test::randomly_select(handle, std::move(d_vertices), num_paths_to_check);
}

}  // namespace test
}  // namespace cugraph

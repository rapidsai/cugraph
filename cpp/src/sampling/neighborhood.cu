/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>

#include <utilities/cugraph_ops_utils.hpp>

#include <cugraph-ops/graph/sampling.hpp>

#include <raft/random/rng_state.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>

#include <type_traits>

namespace cugraph {

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>>
sample_neighbors_adjacency_list(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
                                vertex_t const* ptr_d_start,
                                size_t num_start_vertices,
                                size_t sampling_size,
                                ops::graph::SamplingAlgoT sampling_algo)
{
  using base_vertex_t = std::decay_t<vertex_t>;
  using base_edge_t   = std::decay_t<edge_t>;
  static_assert(std::is_same_v<base_vertex_t, base_edge_t>,
                "cugraph-ops sampling not yet implemented for different node and edge types");

  const auto ops_graph = detail::get_graph(graph_view);
  return ops::graph::uniform_sample_csc(rng_state,
                                        ops_graph,
                                        ptr_d_start,
                                        num_start_vertices,
                                        sampling_size,
                                        sampling_algo,
                                        ops_graph.dst_max_in_degree,
                                        handle.get_stream());
}

template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> sample_neighbors_edgelist(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  vertex_t const* ptr_d_start,
  size_t num_start_vertices,
  size_t sampling_size,
  ops::graph::SamplingAlgoT sampling_algo)
{
  using base_vertex_t = std::decay_t<vertex_t>;
  using base_edge_t   = std::decay_t<edge_t>;
  static_assert(std::is_same_v<base_vertex_t, base_edge_t>,
                "cugraph-ops sampling not yet implemented for different node and edge types");

  const auto ops_graph = detail::get_graph(graph_view);
  return ops::graph::uniform_sample_coo(rng_state,
                                        ops_graph,
                                        ptr_d_start,
                                        num_start_vertices,
                                        sampling_size,
                                        sampling_algo,
                                        ops_graph.dst_max_in_degree,
                                        handle.get_stream());
}

// template explicit instantiation directives (EIDir's):
//
// CSR SG FP32{
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
sample_neighbors_adjacency_list(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                graph_view_t<int32_t, int32_t, false, false> const& gview,
                                int32_t const* ptr_d_start,
                                size_t num_start_vertices,
                                size_t sampling_size,
                                ops::graph::SamplingAlgoT sampling_algo);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
sample_neighbors_adjacency_list(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                graph_view_t<int64_t, int64_t, false, false> const& gview,
                                int64_t const* ptr_d_start,
                                size_t num_start_vertices,
                                size_t sampling_size,
                                ops::graph::SamplingAlgoT sampling_algo);
//}
//
// COO SG FP32{
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
sample_neighbors_edgelist(raft::handle_t const& handle,
                          raft::random::RngState& rng_state,
                          graph_view_t<int32_t, int32_t, false, false> const& gview,
                          int32_t const* ptr_d_start,
                          size_t num_start_vertices,
                          size_t sampling_size,
                          ops::graph::SamplingAlgoT sampling_algo);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
sample_neighbors_edgelist(raft::handle_t const& handle,
                          raft::random::RngState& rng_state,
                          graph_view_t<int64_t, int64_t, false, false> const& gview,
                          int64_t const* ptr_d_start,
                          size_t num_start_vertices,
                          size_t sampling_size,
                          ops::graph::SamplingAlgoT sampling_algo);
//}

rmm::device_uvector<int32_t> get_num_vertices_per_hop_new(raft::handle_t const& handle_,
                                                          rmm::device_uvector<int64_t> srcs,
                                                          rmm::device_uvector<int64_t> dsts,
                                                          rmm::device_uvector<int32_t> hop,
                                                          size_t num_hops)
{
  using vertex_t = int64_t;

  auto hop_0_end_it = thrust::find(handle_.get_thrust_policy(),
                                   thrust::device_pointer_cast<int32_t>(hop.data()),
                                   thrust::device_pointer_cast<int32_t>(hop.data()) + hop.size(),
                                   static_cast<int32_t>(1));

  size_t hop_0_end = hop_0_end_it - thrust::device_pointer_cast<int32_t>(hop.data());
  rmm::device_uvector<vertex_t> vertex_sequence(hop_0_end + dsts.size(), handle_.get_stream());

  raft::copy(vertex_sequence.data(), dsts.data(), dsts.size(), handle_.get_stream());

  // Copy hop 0 vertices
  raft::copy(vertex_sequence.data() + dsts.size(), srcs.data(), hop_0_end, handle_.get_stream());

  hop.resize(hop.size() + hop_0_end, handle_.get_stream());
  auto minus_one = thrust::make_constant_iterator(static_cast<int32_t>(-1));
  thrust::copy(handle_.get_thrust_policy(),
               minus_one,
               minus_one + hop_0_end,
               thrust::device_pointer_cast<int32_t>(hop.data()) + dsts.size());

  size_t n_vertices = dsts.size() + hop_0_end;
  auto zip =
    thrust::make_zip_iterator(thrust::device_pointer_cast<int32_t>(hop.data()),
                              thrust::device_pointer_cast<vertex_t>(vertex_sequence.data()));

  rmm::device_uvector<int32_t> num_vertices_per_hop_result(num_hops + 1, handle_.get_stream());

  thrust::sort(handle_.get_thrust_policy(), zip, zip + n_vertices);

  auto new_end    = thrust::unique(handle_.get_thrust_policy(), zip, zip + n_vertices);
  size_t new_size = new_end - zip;

  hop.resize(new_size, handle_.get_stream());
  vertex_sequence.resize(new_size, handle_.get_stream());

  rmm::device_uvector<vertex_t> seen_vertices(0, handle_.get_stream());

  size_t end = 0;
  for (size_t k = 0; k < num_hops + 1; ++k) {
    size_t start = end;
    auto end_ptr = thrust::find(handle_.get_thrust_policy(),
                                thrust::device_pointer_cast<int32_t>(hop.data()),
                                thrust::device_pointer_cast<int32_t>(hop.data()) + hop.size(),
                                static_cast<int32_t>(k));
    end          = end_ptr - thrust::device_pointer_cast<int32_t>(hop.data());

    rmm::device_uvector<vertex_t> new_seen_vertices(seen_vertices.size() + end - start,
                                                    handle_.get_stream());
    auto output_end = thrust::set_union(
      handle_.get_thrust_policy(),
      thrust::device_pointer_cast<vertex_t>(seen_vertices.data()),
      thrust::device_pointer_cast<vertex_t>(seen_vertices.data()) + seen_vertices.size(),
      thrust::device_pointer_cast<vertex_t>(vertex_sequence.data()) + start,
      thrust::device_pointer_cast<vertex_t>(vertex_sequence.data()) + end,
      thrust::device_pointer_cast<vertex_t>(new_seen_vertices.data()));
    size_t output_size =
      output_end - thrust::device_pointer_cast<vertex_t>(new_seen_vertices.data());
    new_seen_vertices.resize(output_size, handle_.get_stream());
    // shrink to fit?
    seen_vertices = std::move(new_seen_vertices);

    num_vertices_per_hop_result.set_element(k, output_size, handle_.get_stream());
  }

  rmm::device_uvector<int32_t> num_vertices_per_hop_result_copy(num_vertices_per_hop_result,
                                                                handle_.get_stream());
  thrust::transform(handle_.get_thrust_policy(),
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result.data()) + 1,
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result.data()) +
                      num_vertices_per_hop_result.size(),
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result_copy.data()),
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result.data()) + 1,
                    thrust::minus<int32_t>());

  return num_vertices_per_hop_result;
}

rmm::device_uvector<int32_t> get_num_vertices_per_hop(raft::handle_t const& handle_,
                                                      rmm::device_uvector<int64_t> srcs,
                                                      rmm::device_uvector<int64_t> dsts,
                                                      rmm::device_uvector<int32_t> hop,
                                                      size_t num_hops)
{
  using vertex_t = int64_t;

  auto hop_0_end_it = thrust::find(handle_.get_thrust_policy(),
                                   thrust::device_pointer_cast<int32_t>(hop.data()),
                                   thrust::device_pointer_cast<int32_t>(hop.data()) + hop.size(),
                                   static_cast<int32_t>(1));

  size_t hop_0_end = hop_0_end_it - thrust::device_pointer_cast<int32_t>(hop.data());
  rmm::device_uvector<vertex_t> vertex_sequence(hop_0_end + dsts.size(), handle_.get_stream());

  raft::copy(vertex_sequence.data(), dsts.data(), dsts.size(), handle_.get_stream());

  // Copy hop 0 vertices
  raft::copy(vertex_sequence.data() + dsts.size(), srcs.data(), hop_0_end, handle_.get_stream());

  hop.resize(hop.size() + hop_0_end, handle_.get_stream());
  auto minus_one = thrust::make_constant_iterator(static_cast<int32_t>(-1));
  thrust::copy(handle_.get_thrust_policy(),
               minus_one,
               minus_one + hop_0_end,
               thrust::device_pointer_cast<int32_t>(hop.data()) + dsts.size());

  size_t n_vertices = dsts.size() + hop_0_end;
  auto zip =
    thrust::make_zip_iterator(thrust::device_pointer_cast<int32_t>(hop.data()),
                              thrust::device_pointer_cast<vertex_t>(vertex_sequence.data()));

  rmm::device_uvector<int32_t> num_vertices_per_hop_result(num_hops + 1, handle_.get_stream());

  for (size_t k = 0; k < num_hops + 1; ++k) {
    thrust::sort(handle_.get_thrust_policy(), zip, zip + n_vertices);

    auto end_ptr = thrust::find(handle_.get_thrust_policy(),
                                thrust::device_pointer_cast<int32_t>(hop.data()),
                                thrust::device_pointer_cast<int32_t>(hop.data()) + hop.size(),
                                static_cast<int32_t>(0));
    size_t end   = end_ptr - thrust::device_pointer_cast<int32_t>(hop.data());

    vertex_t unique_current =
      thrust::unique_count(handle_.get_thrust_policy(),
                           thrust::device_pointer_cast<vertex_t>(vertex_sequence.data()),
                           thrust::device_pointer_cast<vertex_t>(vertex_sequence.data()) + end);

    num_vertices_per_hop_result.set_element(k, unique_current, handle_.get_stream());

    thrust::transform(handle_.get_thrust_policy(),
                      thrust::device_pointer_cast<int32_t>(hop.data()) + end,
                      thrust::device_pointer_cast<int32_t>(hop.data()) + hop.size(),
                      thrust::make_constant_iterator(static_cast<int32_t>(-1)),
                      thrust::device_pointer_cast<int32_t>(hop.data()) + end,
                      thrust::plus<int32_t>());
  }

  rmm::device_uvector<int32_t> num_vertices_per_hop_result_copy(num_vertices_per_hop_result,
                                                                handle_.get_stream());
  thrust::transform(handle_.get_thrust_policy(),
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result.data()) + 1,
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result.data()) +
                      num_vertices_per_hop_result.size(),
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result_copy.data()),
                    thrust::device_pointer_cast<int32_t>(num_vertices_per_hop_result.data()) + 1,
                    thrust::minus<int32_t>());

  return num_vertices_per_hop_result;
}

}  // namespace cugraph

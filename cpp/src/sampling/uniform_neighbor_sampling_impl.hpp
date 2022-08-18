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

#include <sampling/detail/graph_functions.hpp>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

#include <thrust/optional.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace detail {

template <typename graph_view_t>
std::tuple<rmm::device_uvector<typename graph_view_t::vertex_type>,
           rmm::device_uvector<typename graph_view_t::vertex_type>,
           rmm::device_uvector<typename graph_view_t::weight_type>,
           rmm::device_uvector<typename graph_view_t::edge_type>>
uniform_nbr_sample_impl(
  raft::handle_t const& handle,
  graph_view_t const& graph_view,
  rmm::device_uvector<typename graph_view_t::vertex_type>& d_in,
  raft::host_span<const int> h_fan_out,
  rmm::device_uvector<typename graph_view_t::edge_type> const& global_out_degrees,
  rmm::device_uvector<typename graph_view_t::edge_type> const& global_degree_offsets,
  bool with_replacement,
  uint64_t seed)
{
  using vertex_t = typename graph_view_t::vertex_type;
  using edge_t   = typename graph_view_t::edge_type;
  using weight_t = typename graph_view_t::weight_type;

#ifdef NO_CUGRAPH_OPS
  CUGRAPH_FAIL(
    "uniform_nbr_sampl_impl not supported in this configuration, built with NO_CUGRAPH_OPS");
#else

  namespace cugraph_ops = cugraph::ops::gnn::graph;

  CUGRAPH_EXPECTS(h_fan_out.size() > 0,
                  "Invalid input argument: number of levels must be non-zero.");

  rmm::device_uvector<vertex_t> d_result_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_result_dst(0, handle.get_stream());
  auto d_result_indices =
    thrust::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()));

  size_t level{0};
  size_t row_comm_size{1};

  if constexpr (graph_view_t::is_multi_gpu) {
    auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    seed += row_comm.get_rank();
    row_comm_size = row_comm.get_size();
  }

  for (auto&& k_level : h_fan_out) {
    // prep step for extracting out-degs(sources):
    if constexpr (graph_view_t::is_multi_gpu) {
      d_in = shuffle_int_vertices_by_gpu_id(
        handle, std::move(d_in), graph_view.vertex_partition_range_lasts());
      d_in = allgather_active_majors(handle, std::move(d_in));
    }

    rmm::device_uvector<vertex_t> d_out_src(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_out_dst(0, handle.get_stream());
    auto d_out_indices = std::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()));

    if (k_level > 0) {
      // extract out-degs(sources):
      auto&& d_out_degs =
        get_active_major_global_degrees(handle, graph_view, d_in, global_out_degrees);

      // eliminate 0 degree vertices
      std::tie(d_in, d_out_degs) =
        cugraph::detail::filter_degree_0_vertices(handle, std::move(d_in), std::move(d_out_degs));

      // segmented-random-generation of indices:
      rmm::device_uvector<edge_t> d_rnd_indices(d_in.size() * k_level, handle.get_stream());

      raft::random::RngState rng_state(seed);
      seed += d_rnd_indices.size() * row_comm_size;

      if (d_rnd_indices.size() > 0) {
        // FIXME: This cugraph_ops function does not handle 0 inputs properly
        cugraph_ops::get_sampling_index(d_rnd_indices.data(),
                                        rng_state,
                                        d_out_degs.data(),
                                        static_cast<edge_t>(d_out_degs.size()),
                                        static_cast<int32_t>(k_level),
                                        with_replacement,
                                        handle.get_stream());
      }

      std::tie(d_out_src, d_out_dst, d_out_indices) =
        gather_local_edges(handle,
                           graph_view,
                           d_in,
                           std::move(d_rnd_indices),
                           static_cast<edge_t>(k_level),
                           global_degree_offsets);
    } else {
      std::tie(d_out_src, d_out_dst, d_out_indices) =
        gather_one_hop_edgelist(handle, graph_view, d_in);
    }

    // resize accumulators:
    auto old_sz = d_result_dst.size();
    auto add_sz = d_out_dst.size();
    auto new_sz = old_sz + add_sz;

    d_result_src.resize(new_sz, handle.get_stream());
    d_result_dst.resize(new_sz, handle.get_stream());
    d_result_indices->resize(new_sz, handle.get_stream());

    raft::copy(
      d_result_src.begin() + old_sz, d_out_src.begin(), d_out_src.size(), handle.get_stream());
    raft::copy(
      d_result_dst.begin() + old_sz, d_out_dst.begin(), d_out_dst.size(), handle.get_stream());
    raft::copy(d_result_indices->begin() + old_sz,
               d_out_indices->begin(),
               d_out_indices->size(),
               handle.get_stream());

    d_in = std::move(d_out_dst);

    ++level;
  }

  return count_and_remove_duplicates<vertex_t, edge_t, weight_t>(
    handle, std::move(d_result_src), std::move(d_result_dst), std::move(*d_result_indices));
#endif
}
}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<edge_t>>
uniform_nbr_sample(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  raft::device_span<vertex_t> starting_vertices,
  raft::host_span<const int> fan_out,
  bool with_replacement,
  uint64_t seed)
{
  rmm::device_uvector<vertex_t> d_start_vs(starting_vertices.size(), handle.get_stream());
  raft::copy(
    d_start_vs.data(), starting_vertices.data(), starting_vertices.size(), handle.get_stream());

  // preamble step for out-degree info:
  //
  auto&& [global_degree_offsets, global_out_degrees] =
    detail::get_global_degree_information(handle, graph_view);

  return detail::uniform_nbr_sample_impl(handle,
                                         graph_view,
                                         d_start_vs,
                                         fan_out,
                                         global_out_degrees,
                                         global_degree_offsets,
                                         with_replacement,
                                         seed);
}

}  // namespace cugraph

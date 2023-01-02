/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace cugraph {
namespace test {

struct test_jaccard_t {
  std::string testname{"Jaccard"};

  template <typename weight_t>
  weight_t compute_score(size_t u_size, size_t v_size, weight_t intersection_count) const
  {
    return static_cast<weight_t>(intersection_count) /
           static_cast<weight_t>(u_size + v_size - intersection_count);
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(
    raft::handle_t const& handle,
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
    bool use_weights) const
  {
    return cugraph::jaccard_coefficients(handle, graph_view, edge_weight_view, vertex_pairs, true);
  }
};

struct test_sorensen_t {
  std::string testname{"Sorensen"};

  template <typename weight_t>
  weight_t compute_score(size_t u_size, size_t v_size, weight_t intersection_count) const
  {
    return static_cast<weight_t>(2 * intersection_count) / static_cast<weight_t>(u_size + v_size);
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(
    raft::handle_t const& handle,
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
    bool use_weights) const
  {
    return cugraph::sorensen_coefficients(handle, graph_view, edge_weight_view, vertex_pairs, true);
  }
};

struct test_overlap_t {
  std::string testname{"Overlap"};

  template <typename weight_t>
  weight_t compute_score(size_t u_size, size_t v_size, weight_t intersection_count) const
  {
    return static_cast<weight_t>(intersection_count) /
           static_cast<weight_t>(std::min(u_size, v_size));
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(
    raft::handle_t const& handle,
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
    bool use_weights) const
  {
    return cugraph::overlap_coefficients(handle, graph_view, edge_weight_view, vertex_pairs, true);
  }
};

template <typename vertex_t, typename weight_t, typename test_t>
void similarity_compare(
  vertex_t num_vertices,
  std::tuple<std::vector<vertex_t>&, std::vector<vertex_t>&, std::optional<std::vector<weight_t>>&>
    edge_list,
  std::tuple<std::vector<vertex_t>&, std::vector<vertex_t>&> vertex_pairs,
  std::vector<weight_t>& similarity_score,
  test_t const& test_functor);

}  // namespace test
}  // namespace cugraph

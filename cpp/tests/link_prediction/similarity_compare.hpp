/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
  bool is_jaccard_or_sorensen_or_overlap{true};

  template <typename weight_t>
  weight_t compute_score(weight_t weight_a,
                         weight_t weight_b,
                         weight_t weight_a_intersect_b,
                         weight_t weight_a_union_b) const
  {
    if (std::abs(static_cast<double>(weight_a_union_b) - double{0}) <
        double{2} / std::numeric_limits<double>::max()) {
      return weight_t{0};
    } else {
      return weight_a_intersect_b / weight_a_union_b;
    }
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

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(raft::handle_t const& handle,
           graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
           std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
           std::optional<raft::device_span<vertex_t const>> vertices,
           bool use_weights,
           std::optional<size_t> topk) const
  {
    return cugraph::jaccard_all_pairs_coefficients(
      handle, graph_view, edge_weight_view, vertices, topk);
  }
};

struct test_sorensen_t {
  std::string testname{"Sorensen"};
  bool is_jaccard_or_sorensen_or_overlap{true};

  template <typename weight_t>
  weight_t compute_score(weight_t weight_a,
                         weight_t weight_b,
                         weight_t weight_a_intersect_b,
                         weight_t weight_a_union_b) const
  {
    if (std::abs(static_cast<double>(weight_a_union_b) - double{0}) <
        double{2} / std::numeric_limits<double>::max()) {
      return weight_t{0};
    } else {
      return (2 * weight_a_intersect_b) / (weight_a + weight_b);
    }
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

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(raft::handle_t const& handle,
           graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
           std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
           std::optional<raft::device_span<vertex_t const>> vertices,
           bool use_weights,
           std::optional<size_t> topk) const
  {
    return cugraph::sorensen_all_pairs_coefficients(
      handle, graph_view, edge_weight_view, vertices, topk);
  }
};

struct test_overlap_t {
  std::string testname{"Overlap"};
  bool is_jaccard_or_sorensen_or_overlap{true};

  template <typename weight_t>
  weight_t compute_score(weight_t weight_a,
                         weight_t weight_b,
                         weight_t weight_a_intersect_b,
                         weight_t weight_a_union_b) const
  {
    if (std::abs(static_cast<double>(weight_a_union_b) - double{0}) <
        double{2} / std::numeric_limits<double>::max()) {
      return weight_t{0};
    } else {
      return weight_a_intersect_b / std::min(weight_a, weight_b);
    }
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

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(raft::handle_t const& handle,
           graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
           std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
           std::optional<raft::device_span<vertex_t const>> vertices,
           bool use_weights,
           std::optional<size_t> topk) const
  {
    return cugraph::overlap_all_pairs_coefficients(
      handle, graph_view, edge_weight_view, vertices, topk);
  }
};

struct test_cosine_t {
  std::string testname{"Cosine"};
  bool is_jaccard_or_sorensen_or_overlap{false};

  template <typename weight_t>
  weight_t compute_score(weight_t norm_a,
                         weight_t norm_b,
                         weight_t sum_of_product_of_a_and_b,
                         weight_t reserved_param) const
  {
    if (std::abs(static_cast<double>(norm_a * norm_b)) <
        double{2} / std::numeric_limits<double>::max()) {
      return weight_t{0};
    } else {
      return sum_of_product_of_a_and_b / (norm_a * norm_b);
    }
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(
    raft::handle_t const& handle,
    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
    bool use_weights) const
  {
    return cugraph::cosine_similarity_coefficients(
      handle, graph_view, edge_weight_view, vertex_pairs, true);
  }

  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  auto run(raft::handle_t const& handle,
           graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
           std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
           std::optional<raft::device_span<vertex_t const>> vertices,
           bool use_weights,
           std::optional<size_t> topk) const
  {
    return cugraph::cosine_similarity_all_pairs_coefficients(
      handle, graph_view, edge_weight_view, vertices, topk);
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

template <typename vertex_t, typename weight_t, typename test_t>
void weighted_similarity_compare(
  vertex_t num_vertices,
  std::tuple<std::vector<vertex_t>&, std::vector<vertex_t>&, std::optional<std::vector<weight_t>>&>
    edge_list,
  std::tuple<std::vector<vertex_t>&, std::vector<vertex_t>&> vertex_pairs,
  std::vector<weight_t>& similarity_score,
  test_t const& test_functor);
}  // namespace test
}  // namespace cugraph

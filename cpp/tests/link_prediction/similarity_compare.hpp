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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#pragma once

#include <cugraph/algorithms.hpp>

#include <raft/handle.hpp>
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

  template <typename graph_view_t>
  auto run(raft::handle_t const& handle,
           graph_view_t const& graph_view,
           std::optional<raft::device_span<typename graph_view_t::vertex_type const>> first,
           std::optional<raft::device_span<typename graph_view_t::vertex_type const>> second,
           bool use_weights) const
  {
    return cugraph::jaccard(handle, graph_view, first, second, use_weights);
  }
};

struct test_sorensen_t {
  std::string testname{"Sorensen"};

  template <typename weight_t>
  weight_t compute_score(size_t u_size, size_t v_size, weight_t intersection_count) const
  {
    return static_cast<weight_t>(2 * intersection_count) / static_cast<weight_t>(u_size + v_size);
  }

  template <typename graph_view_t>
  auto run(raft::handle_t const& handle,
           graph_view_t const& graph_view,
           std::optional<raft::device_span<typename graph_view_t::vertex_type const>> first,
           std::optional<raft::device_span<typename graph_view_t::vertex_type const>> second,
           bool use_weights) const
  {
    return cugraph::sorensen(handle, graph_view, first, second, use_weights);
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

  template <typename graph_view_t>
  auto run(raft::handle_t const& handle,
           graph_view_t const& graph_view,
           std::optional<raft::device_span<typename graph_view_t::vertex_type const>> first,
           std::optional<raft::device_span<typename graph_view_t::vertex_type const>> second,
           bool use_weights) const
  {
    return cugraph::overlap(handle, graph_view, first, second, use_weights);
  }
};

template <typename vertex_t, typename weight_t, typename test_t>
void similarity_compare(vertex_t num_vertices,
                        std::vector<vertex_t>&& src,
                        std::vector<vertex_t>&& dst,
                        std::optional<std::vector<weight_t>>&& wgt,
                        std::vector<vertex_t>&& result_src,
                        std::vector<vertex_t>&& result_dst,
                        std::vector<weight_t>&& result_score,
                        test_t const& test_functor);

}  // namespace test
}  // namespace cugraph

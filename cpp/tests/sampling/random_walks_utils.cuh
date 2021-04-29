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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/thrust_rmm_allocator.h>
#include <graph.hpp>
#include <sampling/random_walks.cuh>

#include <raft/handle.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

// utilities for testing / verification of Random Walks functionality:
//
namespace cugraph {
namespace test {

template <typename value_t>
using vector_test_t = cugraph::experimental::detail::device_vec_t<value_t>;  // for debug purposes

// host side utility to check a if a sequence of vertices is connected:
//
template <typename vertex_t, typename edge_t, typename weight_t>
bool host_check_path(std::vector<edge_t> const& row_offsets,
                     std::vector<vertex_t> const& col_inds,
                     std::vector<weight_t> const& values,
                     typename std::vector<vertex_t>::const_iterator v_path_begin,
                     typename std::vector<vertex_t>::const_iterator v_path_end,
                     typename std::vector<weight_t>::const_iterator w_path_begin)
{
  bool assert1 = (row_offsets.size() > 0);
  bool assert2 = (col_inds.size() == values.size());

  vertex_t num_rows = row_offsets.size() - 1;
  edge_t nnz        = row_offsets.back();

  bool assert3 = (nnz == static_cast<edge_t>(col_inds.size()));
  if (assert1 == false || assert2 == false || assert3 == false) {
    std::cout << "CSR inconsistency\n";
    return false;
  }

  auto it_w = w_path_begin;
  for (auto it_v = v_path_begin; it_v != v_path_end - 1; ++it_v, ++it_w) {
    auto crt_vertex  = *it_v;
    auto next_vertex = *(it_v + 1);

    auto begin      = col_inds.begin() + row_offsets[crt_vertex];
    auto end        = col_inds.begin() + row_offsets[crt_vertex + 1];
    auto found_next = std::find_if(
      begin, end, [next_vertex](auto dst_vertex) { return dst_vertex == next_vertex; });
    if (found_next == end) {
      std::cout << "vertex not found: " << next_vertex << " as neighbor of " << crt_vertex << '\n';
      return false;
    }

    auto delta = row_offsets[crt_vertex] + std::distance(begin, found_next);

    // std::cout << "delta in ci: " << delta << '\n';
    auto found_edge = values.begin() + delta;
    if (*found_edge != *it_w) {
      std::cout << "weight not found: " << *found_edge << " between " << crt_vertex << " and "
                << next_vertex << '\n';
      return false;
    }
  }
  return true;
}

template <typename vertex_t, typename edge_t, typename weight_t, typename index_t = edge_t>
bool host_check_rw_paths(
  raft::handle_t const& handle,
  cugraph::experimental::graph_view_t<vertex_t, edge_t, weight_t, false, false> const& graph_view,
  vector_test_t<vertex_t> const& d_coalesced_v,
  vector_test_t<weight_t> const& d_coalesced_w,
  vector_test_t<index_t> const& d_sizes)
{
  edge_t num_edges      = graph_view.get_number_of_edges();
  vertex_t num_vertices = graph_view.get_number_of_vertices();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vals(
    num_edges, 1);  // account for unweighted graph, for which RW provides default weights{1}

  raft::update_host(v_ro.data(), offsets, v_ro.size(), handle.get_stream());
  raft::update_host(v_ci.data(), indices, v_ci.size(), handle.get_stream());

  if (graph_view.is_weighted()) {
    raft::update_host(v_vals.data(), values, v_vals.size(), handle.get_stream());
  }

  std::vector<vertex_t> v_coalesced(d_coalesced_v.size());
  std::vector<weight_t> w_coalesced(d_coalesced_w.size());
  std::vector<index_t> v_sizes(d_sizes.size());

  raft::update_host(v_coalesced.data(),
                    cugraph::experimental::detail::raw_const_ptr(d_coalesced_v),
                    d_coalesced_v.size(),
                    handle.get_stream());
  raft::update_host(w_coalesced.data(),
                    cugraph::experimental::detail::raw_const_ptr(d_coalesced_w),
                    d_coalesced_w.size(),
                    handle.get_stream());
  raft::update_host(v_sizes.data(),
                    cugraph::experimental::detail::raw_const_ptr(d_sizes),
                    d_sizes.size(),
                    handle.get_stream());

  auto it_v_begin = v_coalesced.begin();
  auto it_w_begin = w_coalesced.begin();
  for (auto&& crt_sz : v_sizes) {
    auto it_v_end = it_v_begin + crt_sz;

    bool test_path = host_check_path(v_ro, v_ci, v_vals, it_v_begin, it_v_end, it_w_begin);

    it_v_begin = it_v_end;
    it_w_begin += crt_sz - 1;

    if (!test_path) {  // something went wrong; print to debug (since it's random)
      raft::print_host_vector("sizes", v_sizes.data(), v_sizes.size(), std::cout);

      raft::print_host_vector("coalesced v", v_coalesced.data(), v_coalesced.size(), std::cout);

      raft::print_host_vector("coalesced w", w_coalesced.data(), w_coalesced.size(), std::cout);

      return false;
    }
  }
  return true;
}

}  // namespace test
}  // namespace cugraph

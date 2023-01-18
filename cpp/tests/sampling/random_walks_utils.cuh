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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/exec_policy.hpp>
#include <sampling/random_walks.cuh>

#include <raft/core/handle.hpp>

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
using vector_test_t = cugraph::detail::original::device_vec_t<value_t>;  // for debug purposes

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
    std::cerr << "CSR inconsistency\n";
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
      std::cerr << "vertex not found: " << next_vertex << " as neighbor of " << crt_vertex << '\n';
      return false;
    }

    auto delta = row_offsets[crt_vertex] + std::distance(begin, found_next);

    // std::cerr << "delta in ci: " << delta << '\n';
    auto found_edge = values.begin() + delta;
    if (*found_edge != *it_w) {
      std::cerr << "weight not found: " << *found_edge << " between " << crt_vertex << " and "
                << next_vertex << '\n';
      return false;
    }
  }
  return true;
}

template <typename vertex_t, typename edge_t, typename weight_t, typename index_t = edge_t>
bool host_check_rw_paths(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, false> const& graph_view,
  vertex_t const* ptr_d_coalesced_v,
  size_t num_path_vertices,
  weight_t const* ptr_d_coalesced_w,
  size_t num_path_edges,
  index_t const* ptr_d_sizes,
  size_t path_sizes,
  index_t num_paths)
{
  vertex_t num_vertices = graph_view.number_of_vertices();
  edge_t num_edges      = graph_view.number_of_edges();

  auto v_ro = cugraph::test::to_host(handle, graph_view.local_edge_partition_view().offsets());
  auto v_ci = cugraph::test::to_host(handle, graph_view.local_edge_partition_view().indices());
  std::vector<weight_t> v_vals{};
  if (graph_view.is_weighted()) {
    v_vals = cugraph::test::to_host(handle, *(graph_view.local_edge_partition_view().weights()));
  } else {
    v_vals = std::vector<weight_t>(
      num_edges,
      weight_t{1.0});  // account for unweighted graph, for which RW provides default weights{1}
  }

  auto v_coalesced = cugraph::test::to_host(
    handle,
    raft::device_span<vertex_t const>(ptr_d_coalesced_v, ptr_d_coalesced_v + num_path_vertices));
  auto w_coalesced = cugraph::test::to_host(
    handle,
    raft::device_span<weight_t const>(ptr_d_coalesced_w, ptr_d_coalesced_w + num_path_edges));
  std::vector<index_t> v_sizes{};

  if (path_sizes > 0) {  // coalesced case
    v_sizes = cugraph::test::to_host(
      handle, raft::device_span<index_t const>(ptr_d_sizes, ptr_d_sizes + path_sizes));
  } else {  // padded case
    if (num_paths == 0) {
      std::cerr << "ERROR: padded case requires `num_paths` info.\n";
      return false;
    }

    // extract sizes from v_coalesced (which now contains padded info)
    //
    auto max_depth     = v_coalesced.size() / num_paths;
    auto it_start_path = v_coalesced.begin();
    for (index_t row_index = 0; row_index < num_paths; ++row_index) {
      auto it_end_path      = it_start_path + max_depth;
      auto it_padding_found = std::find(it_start_path, it_end_path, num_vertices);

      v_sizes.push_back(std::distance(it_start_path, it_padding_found));

      it_start_path = it_end_path;
    }

    // truncate padded vectors v_coalesced, w_coalesced:
    //
    v_coalesced.erase(std::remove(v_coalesced.begin(), v_coalesced.end(), num_vertices),
                      v_coalesced.end());

    w_coalesced.erase(std::remove(w_coalesced.begin(), w_coalesced.end(), weight_t{0}),
                      w_coalesced.end());
  }

  auto it_v_begin = v_coalesced.begin();
  auto it_w_begin = w_coalesced.begin();
  for (auto&& crt_sz : v_sizes) {
    auto it_v_end = it_v_begin + crt_sz;

    bool test_path = host_check_path(v_ro, v_ci, v_vals, it_v_begin, it_v_end, it_w_begin);

    it_v_begin = it_v_end;
    it_w_begin += crt_sz - 1;

    if (!test_path) {  // something went wrong; print to debug (since it's random)
      raft::print_host_vector("sizes", v_sizes.data(), v_sizes.size(), std::cerr);

      raft::print_host_vector("coalesced v", v_coalesced.data(), v_coalesced.size(), std::cerr);

      raft::print_host_vector("coalesced w", w_coalesced.data(), w_coalesced.size(), std::cerr);

      return false;
    }
  }
  return true;
}

// convenience trampoline function for when `device_uvector`'s are readily available;
// (e.g., returned by an algorithm);
//
template <typename vertex_t, typename edge_t, typename weight_t, typename index_t = edge_t>
bool host_check_rw_paths(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, weight_t, false, false> const& graph_view,
  vector_test_t<vertex_t> const& d_coalesced_v,
  vector_test_t<weight_t> const& d_coalesced_w,
  vector_test_t<index_t> const& d_sizes,
  index_t num_paths = 0)  // only relevant for the padded case (in which case it must be non-zero)
{
  return host_check_rw_paths(handle,
                             graph_view,
                             d_coalesced_v.data(),
                             d_coalesced_v.size(),
                             d_coalesced_w.data(),
                             d_coalesced_w.size(),
                             d_sizes.data(),
                             d_sizes.size(),
                             num_paths);
}

template <typename index_t>
bool host_check_query_rw(raft::handle_t const& handle,
                         vector_test_t<index_t> const& d_v_sizes,
                         vector_test_t<index_t> const& d_v_offsets,
                         vector_test_t<index_t> const& d_w_sizes,
                         vector_test_t<index_t> const& d_w_offsets)
{
  index_t num_paths = d_v_sizes.size();

  if (num_paths == 0) return false;

  auto v_sizes =
    cugraph::test::to_host(handle,
                           raft::device_span<index_t const>(
                             cugraph::detail::original::raw_const_ptr(d_v_sizes),
                             cugraph::detail::original::raw_const_ptr(d_v_sizes) + num_paths));
  auto v_offsets =
    cugraph::test::to_host(handle,
                           raft::device_span<index_t const>(
                             cugraph::detail::original::raw_const_ptr(d_v_offsets),
                             cugraph::detail::original::raw_const_ptr(d_v_offsets) + num_paths));
  auto w_sizes =
    cugraph::test::to_host(handle,
                           raft::device_span<index_t const>(
                             cugraph::detail::original::raw_const_ptr(d_w_sizes),
                             cugraph::detail::original::raw_const_ptr(d_w_sizes) + num_paths));
  auto w_offsets =
    cugraph::test::to_host(handle,
                           raft::device_span<index_t const>(
                             cugraph::detail::original::raw_const_ptr(d_w_offsets),
                             cugraph::detail::original::raw_const_ptr(d_w_offsets) + num_paths));

  index_t crt_v_offset = 0;
  index_t crt_w_offset = 0;
  auto it_v_sz         = v_sizes.begin();
  auto it_w_sz         = w_sizes.begin();
  auto it_v_offset     = v_offsets.begin();
  auto it_w_offset     = w_offsets.begin();

  bool flag_passed{true};

  for (; it_v_sz != v_sizes.end(); ++it_v_sz, ++it_w_sz, ++it_v_offset, ++it_w_offset) {
    if (*it_w_sz != (*it_v_sz) - 1) {
      std::cerr << "ERROR: Incorrect weight path size: " << *it_w_sz << ", " << *it_v_sz << '\n';
      flag_passed = false;
      break;
    }

    if (*it_v_offset != crt_v_offset) {
      std::cerr << "ERROR: Incorrect vertex path offset: " << *it_v_offset << ", " << crt_v_offset
                << '\n';
      flag_passed = false;
      break;
    }

    if (*it_w_offset != crt_w_offset) {
      std::cerr << "ERROR: Incorrect weight path offset: " << *it_w_offset << ", " << crt_w_offset
                << '\n';
      flag_passed = false;
      break;
    }

    crt_v_offset += *it_v_sz;
    crt_w_offset += *it_w_sz;
  }

  if (!flag_passed) {
    std::cerr << "v sizes:";
    std::copy(v_sizes.begin(), v_sizes.end(), std::ostream_iterator<index_t>(std::cerr, ", "));
    std::cerr << '\n';

    std::cerr << "v offsets:";
    std::copy(v_offsets.begin(), v_offsets.end(), std::ostream_iterator<index_t>(std::cerr, ", "));
    std::cerr << '\n';

    std::cerr << "w sizes:";
    std::copy(w_sizes.begin(), w_sizes.end(), std::ostream_iterator<index_t>(std::cerr, ", "));
    std::cerr << '\n';

    std::cerr << "w offsets:";
    std::copy(w_offsets.begin(), w_offsets.end(), std::ostream_iterator<index_t>(std::cerr, ", "));
    std::cerr << '\n';
  }

  return flag_passed;
}

}  // namespace test
}  // namespace cugraph

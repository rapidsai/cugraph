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

#include "cuda_profiler_api.h"
#include "gtest/gtest.h"

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/random.h>

#include <algorithms.hpp>
#include <experimental/random_walks.cuh>
#include <graph.hpp>

#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <utilities/high_res_timer.hpp>
#include <vector>

using namespace cugraph::experimental;

template <typename value_t>
using vector_test_t = detail::device_vec_t<value_t>;  // for debug purposes

namespace {  // anonym.

template <typename value_t>
void copy_n(raft::handle_t const& handle,
            rmm::device_uvector<value_t>& d_dst,
            std::vector<value_t> const& v_src)
{
  auto stream = handle.get_stream();

  // Copy starting set on device:
  //
  CUDA_TRY(cudaMemcpyAsync(
    d_dst.data(), v_src.data(), v_src.size() * sizeof(value_t), cudaMemcpyHostToDevice, stream));

  cudaStreamSynchronize(stream);
}

template <typename value_t>
void copy_n(raft::handle_t const& handle,
            std::vector<value_t>& v_dst,
            value_t const* d_src,
            size_t nelems)
{
  auto stream = handle.get_stream();

  // Copy starting set on device:
  //
  CUDA_TRY(
    cudaMemcpyAsync(v_dst.data(), d_src, nelems * sizeof(value_t), cudaMemcpyDeviceToHost, stream));

  cudaStreamSynchronize(stream);
}

template <typename vertex_t, typename edge_t, typename weight_t>
graph_t<vertex_t, edge_t, weight_t, false, false> make_graph(raft::handle_t const& handle,
                                                             std::vector<vertex_t> const& v_src,
                                                             std::vector<vertex_t> const& v_dst,
                                                             std::vector<weight_t> const& v_w,
                                                             vertex_t num_vertices,
                                                             edge_t num_edges)
{
  vector_test_t<vertex_t> d_src(num_edges, handle.get_stream());
  vector_test_t<vertex_t> d_dst(num_edges, handle.get_stream());
  vector_test_t<weight_t> d_weights(num_edges, handle.get_stream());

  copy_n(handle, d_src, v_src);
  copy_n(handle, d_dst, v_dst);
  copy_n(handle, d_weights, v_w);

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    d_src.data(), d_dst.data(), d_weights.data(), num_edges};

  graph_t<vertex_t, edge_t, weight_t, false, false> graph(
    handle, edgelist, num_vertices, graph_properties_t{}, false);

  return graph;
}

template <typename vertex_t, typename edge_t, typename index_t>
bool check_col_indices(raft::handle_t const& handle,
                       vector_test_t<edge_t> const& d_crt_out_degs,
                       vector_test_t<vertex_t> const& d_col_indx,
                       index_t num_paths)
{
  bool all_indices_within_degs = thrust::all_of(
    rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
    thrust::make_counting_iterator<index_t>(0),
    thrust::make_counting_iterator<index_t>(num_paths),
    [p_d_col_indx     = detail::raw_const_ptr(d_col_indx),
     p_d_crt_out_degs = detail::raw_const_ptr(d_crt_out_degs)] __device__(auto indx) {
      if (p_d_crt_out_degs[indx] > 0)
        return ((p_d_col_indx[indx] >= 0) && (p_d_col_indx[indx] < p_d_crt_out_degs[indx]));
      else
        return true;
    });
  return all_indices_within_degs;
}

// std::vector printer:
//
template <typename value_t>
void print_vec(std::vector<value_t> const& vec, std::ostream& os)
{
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<value_t>(os, ", "));
  std::cout << '\n';
}

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
bool host_check_rw_paths(raft::handle_t const& handle,
                         graph_view_t<vertex_t, edge_t, weight_t, false, false> const& graph_view,
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
  std::vector<weight_t> v_vals(num_edges);

  copy_n(handle, v_ro, offsets, v_ro.size());
  copy_n(handle, v_ci, indices, v_ci.size());
  copy_n(handle, v_vals, values, v_vals.size());

  std::vector<vertex_t> v_coalesced(d_coalesced_v.size());
  std::vector<weight_t> w_coalesced(d_coalesced_w.size());
  std::vector<index_t> v_sizes(d_sizes.size());

  copy_n(handle, v_coalesced, detail::raw_const_ptr(d_coalesced_v), d_coalesced_v.size());
  copy_n(handle, w_coalesced, detail::raw_const_ptr(d_coalesced_w), d_coalesced_w.size());
  copy_n(handle, v_sizes, detail::raw_const_ptr(d_sizes), d_sizes.size());

  auto it_v_begin = v_coalesced.begin();
  auto it_w_begin = w_coalesced.begin();
  for (auto&& crt_sz : v_sizes) {
    auto it_v_end = it_v_begin + crt_sz;

    bool test_path = host_check_path(v_ro, v_ci, v_vals, it_v_begin, it_v_end, it_w_begin);

    it_v_begin = it_v_end;
    it_w_begin += crt_sz - 1;

    if (!test_path) {  // something went wrong; print to debug (since it's random)
      std::cout << "sizes:\n";
      print_vec(v_sizes, std::cout);

      std::cout << "coalesced v:\n";
      print_vec(v_coalesced, std::cout);

      std::cout << "coalesced w:\n";
      print_vec(w_coalesced, std::cout);

      return false;
    }
  }
  return true;
}
}  // namespace

struct RandomWalksPrimsTest : public ::testing::Test {
};

TEST_F(RandomWalksPrimsTest, SimpleGraphRWStart)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vs(num_edges);

  copy_n(handle, v_ro, offsets, num_vertices + 1);
  copy_n(handle, v_ci, indices, num_edges);
  copy_n(handle, v_vs, values, num_edges);

  std::vector<edge_t> v_ro_expected{0, 1, 3, 6, 7, 8, 8};
  std::vector<vertex_t> v_ci_expected{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_vs_expected{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  EXPECT_EQ(v_ro, v_ro_expected);
  EXPECT_EQ(v_ci, v_ci_expected);
  EXPECT_EQ(v_vs, v_vs_expected);

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<vertex_t> v_coalesced(total_sz, -1);
  std::vector<weight_t> w_coalesced(total_sz - num_paths, -1);

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<weight_t> d_coalesced_w(total_sz - num_paths, handle.get_stream());

  copy_n(handle, d_coalesced_v, v_coalesced);
  copy_n(handle, d_coalesced_w, w_coalesced);

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  copy_n(handle, d_start, v_start);

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  std::vector<vertex_t> v_coalesced_exp{1, -1, -1, 0, -1, -1, 4, -1, -1, 2, -1, -1};
  copy_n(handle, v_coalesced, raw_const_ptr(d_coalesced_v), total_sz);
  EXPECT_EQ(v_coalesced, v_coalesced_exp);

  std::vector<index_t> v_sizes{1, 1, 1, 1};
  std::vector<index_t> v_sz_exp(num_paths);
  copy_n(handle, v_sz_exp, raw_const_ptr(d_sizes), num_paths);

  EXPECT_EQ(v_sizes, v_sz_exp);
}

TEST_F(RandomWalksPrimsTest, SimpleGraphCoalesceExperiments)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<vertex_t> v_coalesced(total_sz, -1);
  std::vector<weight_t> w_coalesced(total_sz - num_paths, -1);

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<weight_t> d_coalesced_w(total_sz - num_paths, handle.get_stream());

  copy_n(handle, d_coalesced_v, v_coalesced);
  copy_n(handle, d_coalesced_w, w_coalesced);

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  copy_n(handle, d_start, v_start);

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  auto const& d_out_degs = rand_walker.get_out_degs();
  EXPECT_EQ(static_cast<size_t>(num_vertices), d_out_degs.size());

  std::vector<edge_t> v_out_degs(num_vertices);
  copy_n(handle, v_out_degs, raw_const_ptr(d_out_degs), num_vertices);

  std::vector<edge_t> v_out_degs_exp{1, 2, 3, 1, 1, 0};
  EXPECT_EQ(v_out_degs, v_out_degs_exp);

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // update crt_out_degs:
  //
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  rand_walker.gather_from_coalesced(
    d_coalesced_v, d_out_degs, d_sizes, d_crt_out_degs, max_depth, num_paths);

  std::vector<edge_t> v_crt_out_degs(num_paths);
  copy_n(handle, v_crt_out_degs, raw_const_ptr(d_crt_out_degs), num_paths);

  std::vector<edge_t> v_crt_out_degs_exp{2, 1, 1, 3};
  EXPECT_EQ(v_crt_out_degs, v_crt_out_degs_exp);
}

TEST_F(RandomWalksPrimsTest, SimpleGraphColExtraction)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<vertex_t> v_coalesced(total_sz, -1);
  std::vector<weight_t> w_coalesced(total_sz - num_paths, -1);

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<weight_t> d_coalesced_w(total_sz - num_paths, handle.get_stream());

  copy_n(handle, d_coalesced_v, v_coalesced);
  copy_n(handle, d_coalesced_w, w_coalesced);

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  copy_n(handle, d_start, v_start);

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  auto const& d_out_degs = rand_walker.get_out_degs();

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // update crt_out_degs:
  //
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  rand_walker.gather_from_coalesced(
    d_coalesced_v, d_out_degs, d_sizes, d_crt_out_degs, max_depth, num_paths);

  col_indx_extract_t<decltype(graph_view), index_t> col_extractor{handle,
                                                                  graph_view,
                                                                  raw_const_ptr(d_crt_out_degs),
                                                                  raw_const_ptr(d_sizes),
                                                                  num_paths,
                                                                  max_depth};

  // typically given by random engine:
  //
  std::vector<vertex_t> v_col_indx{1, 0, 0, 2};
  vector_test_t<vertex_t> d_col_indx(num_paths, handle.get_stream());

  copy_n(handle, d_col_indx, v_col_indx);

  vector_test_t<vertex_t> d_next_v(num_paths, handle.get_stream());
  vector_test_t<weight_t> d_next_w(num_paths, handle.get_stream());

  col_extractor(d_coalesced_v, d_col_indx, d_next_v, d_next_w);

  std::vector<vertex_t> v_next_v(num_paths);
  std::vector<weight_t> v_next_w(num_paths);

  copy_n(handle, v_next_v, raw_const_ptr(d_next_v), num_paths);
  copy_n(handle, v_next_w, raw_const_ptr(d_next_w), num_paths);

  std::vector<vertex_t> v_next_v_exp{4, 1, 5, 3};
  std::vector<weight_t> v_next_w_exp{2.1f, 0.1f, 7.1f, 5.1f};

  EXPECT_EQ(v_next_v, v_next_v_exp);
  EXPECT_EQ(v_next_w, v_next_w_exp);
}

TEST_F(RandomWalksPrimsTest, SimpleGraphRndGenColIndx)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;
  using real_t   = float;
  using seed_t   = long;

  using random_engine_t = rrandom_gen_t<vertex_t, edge_t>;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<vertex_t> v_coalesced(total_sz, -1);
  std::vector<weight_t> w_coalesced(total_sz - num_paths, -1);

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<weight_t> d_coalesced_w(total_sz - num_paths, handle.get_stream());

  copy_n(handle, d_coalesced_v, v_coalesced);
  copy_n(handle, d_coalesced_w, w_coalesced);

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  copy_n(handle, d_start, v_start);

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  auto const& d_out_degs = rand_walker.get_out_degs();

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // update crt_out_degs:
  //
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  rand_walker.gather_from_coalesced(
    d_coalesced_v, d_out_degs, d_sizes, d_crt_out_degs, max_depth, num_paths);

  // random engine generated:
  //
  vector_test_t<vertex_t> d_col_indx(num_paths, handle.get_stream());
  vector_test_t<real_t> d_random(num_paths, handle.get_stream());

  seed_t seed = static_cast<seed_t>(std::time(nullptr));
  random_engine_t rgen(handle, num_paths, d_random, d_crt_out_degs, seed);
  rgen.generate_col_indices(d_col_indx);

#ifdef PRINT_RANDOM
  std::vector<vertex_t> v_col_indx(num_paths);
  std::vector<edge_t> v_crt_out_degs(num_paths);
  std::vector<real_t> v_random(num_paths);

  copy_n(handle, v_col_indx, raw_const_ptr(d_col_indx), num_paths);
  copy_n(handle, v_crt_out_degs, raw_const_ptr(d_crt_out_degs), num_paths);
  copy_n(handle, v_random, raw_const_ptr(d_random), num_paths);

  std::cout << "v_random:\n";
  std::copy(v_random.begin(), v_random.end(), std::ostream_iterator<real_t>(std::cout, ", "));
  std::cout << '\n';

  std::cout << "crt_out_degs:\n";
  std::copy(
    v_crt_out_degs.begin(), v_crt_out_degs.end(), std::ostream_iterator<edge_t>(std::cout, ", "));
  std::cout << '\n';

  std::cout << "col_indx:\n";
  std::copy(v_col_indx.begin(), v_col_indx.end(), std::ostream_iterator<vertex_t>(std::cout, ", "));
  std::cout << '\n';
#endif

  bool all_indices_within_degs = check_col_indices(handle, d_crt_out_degs, d_col_indx, num_paths);

  ASSERT_TRUE(all_indices_within_degs);
}

TEST_F(RandomWalksPrimsTest, SimpleGraphUpdatePathSizes)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;
  using real_t   = float;
  using seed_t   = long;

  using random_engine_t = rrandom_gen_t<vertex_t, edge_t>;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<vertex_t> v_coalesced(total_sz, -1);
  std::vector<weight_t> w_coalesced(total_sz - num_paths, -1);

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<weight_t> d_coalesced_w(total_sz - num_paths, handle.get_stream());

  copy_n(handle, d_coalesced_v, v_coalesced);
  copy_n(handle, d_coalesced_w, w_coalesced);

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  copy_n(handle, d_start, v_start);

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  auto const& d_out_degs = rand_walker.get_out_degs();

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // say, if d_crt_out_degs were this:
  //
  std::vector<edge_t> v_crt_out_degs{2, 0, 1, 0};
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  copy_n(handle, d_crt_out_degs, v_crt_out_degs);

  rand_walker.update_path_sizes(d_crt_out_degs, d_sizes);

  std::vector<index_t> v_sizes(num_paths);
  copy_n(handle, v_sizes, raw_const_ptr(d_sizes), num_paths);
  std::vector<index_t> v_sizes_exp{2, 1, 2, 1};
  // i.e., corresponding 0-entries in crt-out-degs, don't get updated;

  EXPECT_EQ(v_sizes, v_sizes_exp);
}

TEST_F(RandomWalksPrimsTest, SimpleGraphScatterUpdate)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<vertex_t> v_coalesced(total_sz, -1);
  std::vector<weight_t> w_coalesced(total_sz - num_paths, -1);

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<weight_t> d_coalesced_w(total_sz - num_paths, handle.get_stream());

  copy_n(handle, d_coalesced_v, v_coalesced);
  copy_n(handle, d_coalesced_w, w_coalesced);

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  copy_n(handle, d_start, v_start);

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  auto const& d_out_degs = rand_walker.get_out_degs();

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // update crt_out_degs:
  //
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  rand_walker.gather_from_coalesced(
    d_coalesced_v, d_out_degs, d_sizes, d_crt_out_degs, max_depth, num_paths);

  col_indx_extract_t<decltype(graph_view), index_t> col_extractor{handle,
                                                                  graph_view,
                                                                  raw_const_ptr(d_crt_out_degs),
                                                                  raw_const_ptr(d_sizes),
                                                                  num_paths,
                                                                  max_depth};

  // typically given by random engine:
  //
  std::vector<vertex_t> v_col_indx{1, 0, 0, 2};
  vector_test_t<vertex_t> d_col_indx(num_paths, handle.get_stream());

  copy_n(handle, d_col_indx, v_col_indx);

  vector_test_t<vertex_t> d_next_v(num_paths, handle.get_stream());
  vector_test_t<weight_t> d_next_w(num_paths, handle.get_stream());

  col_extractor(d_coalesced_v, d_col_indx, d_next_v, d_next_w);

  rand_walker.update_path_sizes(d_crt_out_degs, d_sizes);

  // check start():
  //
  {
    std::vector<vertex_t> v_coalesced_exp{1, -1, -1, 0, -1, -1, 4, -1, -1, 2, -1, -1};
    copy_n(handle, v_coalesced, raw_const_ptr(d_coalesced_v), total_sz);
    EXPECT_EQ(v_coalesced, v_coalesced_exp);
  }

  // check crt_out_degs:
  //
  {
    std::vector<edge_t> v_crt_out_degs(num_paths);
    copy_n(handle, v_crt_out_degs, raw_const_ptr(d_crt_out_degs), num_paths);
    std::vector<edge_t> v_crt_out_degs_exp{2, 1, 1, 3};
    EXPECT_EQ(v_crt_out_degs, v_crt_out_degs_exp);
  }

  // check paths sizes update:
  //
  {
    std::vector<index_t> v_sizes(num_paths);
    copy_n(handle, v_sizes, raw_const_ptr(d_sizes), num_paths);
    std::vector<index_t> v_sizes_exp{2, 2, 2, 2};
    // i.e., corresponding 0-entries in crt-out-degs, don't get updated;
    EXPECT_EQ(v_sizes, v_sizes_exp);
  }

  // check next step:
  //
  {
    std::vector<vertex_t> v_next_v(num_paths);
    std::vector<weight_t> v_next_w(num_paths);

    copy_n(handle, v_next_v, raw_const_ptr(d_next_v), num_paths);
    copy_n(handle, v_next_w, raw_const_ptr(d_next_w), num_paths);

    std::vector<vertex_t> v_next_v_exp{4, 1, 5, 3};
    std::vector<weight_t> v_next_w_exp{2.1f, 0.1f, 7.1f, 5.1f};

    EXPECT_EQ(v_next_v, v_next_v_exp);
    EXPECT_EQ(v_next_w, v_next_w_exp);
  }

  rand_walker.scatter_vertices(d_next_v, d_coalesced_v, d_crt_out_degs, d_sizes);
  rand_walker.scatter_weights(d_next_w, d_coalesced_w, d_crt_out_degs, d_sizes);

  // check vertex/weight scatter:
  //
  {
    copy_n(handle, v_coalesced, raw_const_ptr(d_coalesced_v), total_sz);
    copy_n(handle, w_coalesced, raw_const_ptr(d_coalesced_w), total_sz - num_paths);

    std::vector<vertex_t> v_coalesced_exp{1, 4, -1, 0, 1, -1, 4, 5, -1, 2, 3, -1};
    std::vector<weight_t> w_coalesced_exp{2.1, -1, 0.1, -1, 7.1, -1, 5.1, -1};

    EXPECT_EQ(v_coalesced, v_coalesced_exp);
    EXPECT_EQ(w_coalesced, w_coalesced_exp);
  }
}

TEST_F(RandomWalksPrimsTest, SimpleGraphCoalesceDefragment)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<index_t> v_sizes{1, 2, 2, 1};
  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());
  copy_n(handle, d_sizes, v_sizes);

  std::vector<vertex_t> v_coalesced(total_sz, -1);
  v_coalesced[0]                 = 3;
  v_coalesced[max_depth]         = 5;
  v_coalesced[max_depth + 1]     = 2;
  v_coalesced[2 * max_depth]     = 4;
  v_coalesced[2 * max_depth + 1] = 0;
  v_coalesced[3 * max_depth]     = 1;

  std::vector<weight_t> w_coalesced(total_sz - num_paths, -1);
  w_coalesced[max_depth - 1]     = 10.1;
  w_coalesced[2 * max_depth - 2] = 11.2;

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<weight_t> d_coalesced_w(total_sz - num_paths, handle.get_stream());

  copy_n(handle, d_coalesced_v, v_coalesced);
  copy_n(handle, d_coalesced_w, w_coalesced);

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  rand_walker.stop(d_coalesced_v, d_coalesced_w, d_sizes);

  // check vertex/weight defragment:
  //
  {
    v_coalesced.resize(d_coalesced_v.size());
    w_coalesced.resize(d_coalesced_w.size());

    copy_n(handle, v_coalesced, raw_const_ptr(d_coalesced_v), d_coalesced_v.size());
    copy_n(handle, w_coalesced, raw_const_ptr(d_coalesced_w), d_coalesced_w.size());

    std::vector<vertex_t> v_coalesced_exp{3, 5, 2, 4, 0, 1};
    std::vector<weight_t> w_coalesced_exp{10.1, 11.2};

    EXPECT_EQ(v_coalesced, v_coalesced_exp);
    EXPECT_EQ(w_coalesced, w_coalesced_exp);
  }
}

TEST_F(RandomWalksPrimsTest, SimpleGraphRandomWalk)
{
  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vals(num_edges);

  copy_n(handle, v_ro, offsets, v_ro.size());
  copy_n(handle, v_ci, indices, v_ci.size());
  copy_n(handle, v_vals, values, v_vals.size());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_v_start(v_start.size(), handle.get_stream());
  copy_n(handle, d_v_start, v_start);

  index_t num_paths = v_start.size();
  index_t max_depth = 5;
  auto triplet      = random_walks(handle, graph_view, d_v_start.data(), num_paths, max_depth);

  auto& d_coalesced_v = std::get<0>(triplet);
  auto& d_coalesced_w = std::get<1>(triplet);
  auto& d_sizes       = std::get<2>(triplet);

  bool test_all_paths =
    host_check_rw_paths(handle, graph_view, d_coalesced_v, d_coalesced_w, d_sizes);

  ASSERT_TRUE(test_all_paths);
}

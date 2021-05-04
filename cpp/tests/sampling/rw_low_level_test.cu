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

#include <gtest/gtest.h>
#include "cuda_profiler_api.h"

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/random.h>

#include <algorithms.hpp>
#include <graph.hpp>
#include <sampling/random_walks.cuh>

#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include "random_walks_utils.cuh"

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

template <typename vertex_t, typename edge_t, typename weight_t>
graph_t<vertex_t, edge_t, weight_t, false, false> make_graph(raft::handle_t const& handle,
                                                             std::vector<vertex_t> const& v_src,
                                                             std::vector<vertex_t> const& v_dst,
                                                             std::vector<weight_t> const& v_w,
                                                             vertex_t num_vertices,
                                                             edge_t num_edges,
                                                             bool is_weighted)
{
  vector_test_t<vertex_t> d_src(num_edges, handle.get_stream());
  vector_test_t<vertex_t> d_dst(num_edges, handle.get_stream());
  vector_test_t<weight_t> d_weights(num_edges, handle.get_stream());

  raft::update_device(d_src.data(), v_src.data(), d_src.size(), handle.get_stream());
  raft::update_device(d_dst.data(), v_dst.data(), d_dst.size(), handle.get_stream());

  weight_t* ptr_d_weights{nullptr};
  if (is_weighted) {
    raft::update_device(d_weights.data(), v_w.data(), d_weights.size(), handle.get_stream());

    ptr_d_weights = d_weights.data();
  }

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    d_src.data(), d_dst.data(), ptr_d_weights, num_edges};

  graph_t<vertex_t, edge_t, weight_t, false, false> graph(
    handle, edgelist, num_vertices, graph_properties_t{false, false, is_weighted}, false);

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

}  // namespace

// FIXME (per rlratzel request):
// This test may be considered an e2e test
// which could be moved to a different test suite:
//
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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vs(num_edges);

  raft::update_host(v_ro.data(), offsets, num_vertices + 1, handle.get_stream());
  raft::update_host(v_ci.data(), indices, num_edges, handle.get_stream());
  raft::update_host(v_vs.data(), values, num_edges, handle.get_stream());

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

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(
    d_coalesced_w.data(), w_coalesced.data(), d_coalesced_w.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  raft::update_device(d_start.data(), v_start.data(), d_start.size(), handle.get_stream());

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  std::vector<vertex_t> v_coalesced_exp{1, -1, -1, 0, -1, -1, 4, -1, -1, 2, -1, -1};
  raft::update_host(
    v_coalesced.data(), raw_const_ptr(d_coalesced_v), total_sz, handle.get_stream());
  EXPECT_EQ(v_coalesced, v_coalesced_exp);

  std::vector<index_t> v_sizes{1, 1, 1, 1};
  std::vector<index_t> v_sz_exp(num_paths);
  raft::update_host(v_sz_exp.data(), raw_const_ptr(d_sizes), num_paths, handle.get_stream());

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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

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

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(
    d_coalesced_w.data(), w_coalesced.data(), d_coalesced_w.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  raft::update_device(d_start.data(), v_start.data(), d_start.size(), handle.get_stream());

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  auto const& d_out_degs = rand_walker.get_out_degs();
  EXPECT_EQ(static_cast<size_t>(num_vertices), d_out_degs.size());

  std::vector<edge_t> v_out_degs(num_vertices);
  raft::update_host(
    v_out_degs.data(), raw_const_ptr(d_out_degs), num_vertices, handle.get_stream());

  std::vector<edge_t> v_out_degs_exp{1, 2, 3, 1, 1, 0};
  EXPECT_EQ(v_out_degs, v_out_degs_exp);

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // update crt_out_degs:
  //
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  rand_walker.gather_from_coalesced(
    d_coalesced_v, d_out_degs, d_sizes, d_crt_out_degs, max_depth, num_paths);

  std::vector<edge_t> v_crt_out_degs(num_paths);
  raft::update_host(
    v_crt_out_degs.data(), raw_const_ptr(d_crt_out_degs), num_paths, handle.get_stream());

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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

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

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(
    d_coalesced_w.data(), w_coalesced.data(), d_coalesced_w.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  raft::update_device(d_start.data(), v_start.data(), d_start.size(), handle.get_stream());

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

  raft::update_device(d_col_indx.data(), v_col_indx.data(), d_col_indx.size(), handle.get_stream());

  vector_test_t<vertex_t> d_next_v(num_paths, handle.get_stream());
  vector_test_t<weight_t> d_next_w(num_paths, handle.get_stream());

  col_extractor(d_coalesced_v, d_col_indx, d_next_v, d_next_w);

  std::vector<vertex_t> v_next_v(num_paths);
  std::vector<weight_t> v_next_w(num_paths);

  raft::update_host(v_next_v.data(), raw_const_ptr(d_next_v), num_paths, handle.get_stream());
  raft::update_host(v_next_w.data(), raw_const_ptr(d_next_w), num_paths, handle.get_stream());

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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

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

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(
    d_coalesced_w.data(), w_coalesced.data(), d_coalesced_w.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  raft::update_device(d_start.data(), v_start.data(), d_start.size(), handle.get_stream());

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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

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

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(
    d_coalesced_w.data(), w_coalesced.data(), d_coalesced_w.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  raft::update_device(d_start.data(), v_start.data(), d_start.size(), handle.get_stream());

  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  auto const& d_out_degs = rand_walker.get_out_degs();

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // Fixed  set of out-degs, as opposed to have them generated by the algorithm.
  // That's because I want to test a certain functionality in isolation
  //
  std::vector<edge_t> v_crt_out_degs{2, 0, 1, 0};
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  raft::update_device(
    d_crt_out_degs.data(), v_crt_out_degs.data(), d_crt_out_degs.size(), handle.get_stream());

  rand_walker.update_path_sizes(d_crt_out_degs, d_sizes);

  std::vector<index_t> v_sizes(num_paths);
  raft::update_host(v_sizes.data(), raw_const_ptr(d_sizes), num_paths, handle.get_stream());
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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

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

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(
    d_coalesced_w.data(), w_coalesced.data(), d_coalesced_w.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_start(num_paths, handle.get_stream());

  raft::update_device(d_start.data(), v_start.data(), d_start.size(), handle.get_stream());

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

  raft::update_device(d_col_indx.data(), v_col_indx.data(), d_col_indx.size(), handle.get_stream());

  vector_test_t<vertex_t> d_next_v(num_paths, handle.get_stream());
  vector_test_t<weight_t> d_next_w(num_paths, handle.get_stream());

  col_extractor(d_coalesced_v, d_col_indx, d_next_v, d_next_w);

  rand_walker.update_path_sizes(d_crt_out_degs, d_sizes);

  // check start():
  //
  {
    std::vector<vertex_t> v_coalesced_exp{1, -1, -1, 0, -1, -1, 4, -1, -1, 2, -1, -1};
    raft::update_host(
      v_coalesced.data(), raw_const_ptr(d_coalesced_v), total_sz, handle.get_stream());
    EXPECT_EQ(v_coalesced, v_coalesced_exp);
  }

  // check crt_out_degs:
  //
  {
    std::vector<edge_t> v_crt_out_degs(num_paths);
    raft::update_host(
      v_crt_out_degs.data(), raw_const_ptr(d_crt_out_degs), num_paths, handle.get_stream());
    std::vector<edge_t> v_crt_out_degs_exp{2, 1, 1, 3};
    EXPECT_EQ(v_crt_out_degs, v_crt_out_degs_exp);
  }

  // check paths sizes update:
  //
  {
    std::vector<index_t> v_sizes(num_paths);
    raft::update_host(v_sizes.data(), raw_const_ptr(d_sizes), num_paths, handle.get_stream());
    std::vector<index_t> v_sizes_exp{2, 2, 2, 2};
    // i.e., corresponding 0-entries in crt-out-degs, don't get updated;
    EXPECT_EQ(v_sizes, v_sizes_exp);
  }

  // check next step:
  //
  {
    std::vector<vertex_t> v_next_v(num_paths);
    std::vector<weight_t> v_next_w(num_paths);

    raft::update_host(v_next_v.data(), raw_const_ptr(d_next_v), num_paths, handle.get_stream());
    raft::update_host(v_next_w.data(), raw_const_ptr(d_next_w), num_paths, handle.get_stream());

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
    raft::update_host(
      v_coalesced.data(), raw_const_ptr(d_coalesced_v), total_sz, handle.get_stream());
    raft::update_host(
      w_coalesced.data(), raw_const_ptr(d_coalesced_w), total_sz - num_paths, handle.get_stream());

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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  index_t num_paths = 4;
  index_t max_depth = 3;
  index_t total_sz  = num_paths * max_depth;

  std::vector<index_t> v_sizes{1, 2, 2, 1};
  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());
  raft::update_device(d_sizes.data(), v_sizes.data(), d_sizes.size(), handle.get_stream());

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

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(
    d_coalesced_w.data(), w_coalesced.data(), d_coalesced_w.size(), handle.get_stream());

  random_walker_t<decltype(graph_view)> rand_walker{handle, graph_view, num_paths, max_depth};

  rand_walker.stop(d_coalesced_v, d_coalesced_w, d_sizes);

  // check vertex/weight defragment:
  //
  {
    v_coalesced.resize(d_coalesced_v.size());
    w_coalesced.resize(d_coalesced_w.size());

    raft::update_host(
      v_coalesced.data(), raw_const_ptr(d_coalesced_v), d_coalesced_v.size(), handle.get_stream());
    raft::update_host(
      w_coalesced.data(), raw_const_ptr(d_coalesced_w), d_coalesced_w.size(), handle.get_stream());

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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vals(num_edges);

  raft::update_host(v_ro.data(), offsets, v_ro.size(), handle.get_stream());
  raft::update_host(v_ci.data(), indices, v_ci.size(), handle.get_stream());
  raft::update_host(v_vals.data(), values, v_vals.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_v_start(v_start.size(), handle.get_stream());
  raft::update_device(d_v_start.data(), v_start.data(), d_v_start.size(), handle.get_stream());

  index_t num_paths = v_start.size();
  index_t max_depth = 5;

  // 0-copy const device view:
  //
  detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(), num_paths};
  auto quad = detail::random_walks_impl(handle, graph_view, d_start_view, max_depth);

  auto& d_coalesced_v = std::get<0>(quad);
  auto& d_coalesced_w = std::get<1>(quad);
  auto& d_sizes       = std::get<2>(quad);
  auto seed0          = std::get<3>(quad);

  bool test_all_paths =
    cugraph::test::host_check_rw_paths(handle, graph_view, d_coalesced_v, d_coalesced_w, d_sizes);

  if (!test_all_paths) std::cout << "starting seed on failure: " << seed0 << '\n';

  ASSERT_TRUE(test_all_paths);
}

TEST(RandomWalksQuery, GraphRWQueryOffsets)
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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vals(num_edges);

  raft::update_host(v_ro.data(), offsets, v_ro.size(), handle.get_stream());
  raft::update_host(v_ci.data(), indices, v_ci.size(), handle.get_stream());
  raft::update_host(v_vals.data(), values, v_vals.size(), handle.get_stream());

  std::vector<vertex_t> v_start{1, 0, 4, 2};
  vector_test_t<vertex_t> d_v_start(v_start.size(), handle.get_stream());
  raft::update_device(d_v_start.data(), v_start.data(), d_v_start.size(), handle.get_stream());

  index_t num_paths = v_start.size();
  index_t max_depth = 5;

  // 0-copy const device view:
  //
  detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(), num_paths};
  auto quad = detail::random_walks_impl(handle, graph_view, d_start_view, max_depth);

  auto& d_v_sizes = std::get<2>(quad);
  auto seed0      = std::get<3>(quad);

  auto triplet = query_rw_sizes_offsets(handle, num_paths, detail::raw_const_ptr(d_v_sizes));

  auto& d_v_offsets = std::get<0>(triplet);
  auto& d_w_sizes   = std::get<1>(triplet);
  auto& d_w_offsets = std::get<2>(triplet);

  bool test_paths_sz =
    cugraph::test::host_check_query_rw(handle, d_v_sizes, d_v_offsets, d_w_sizes, d_w_offsets);

  if (!test_paths_sz) std::cout << "starting seed on failure: " << seed0 << '\n';

  ASSERT_TRUE(test_paths_sz);
}

TEST(RandomWalksSpecialCase, SingleRandomWalk)
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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vals(num_edges);

  raft::update_host(v_ro.data(), offsets, v_ro.size(), handle.get_stream());
  raft::update_host(v_ci.data(), indices, v_ci.size(), handle.get_stream());
  raft::update_host(v_vals.data(), values, v_vals.size(), handle.get_stream());

  std::vector<vertex_t> v_start{2};
  vector_test_t<vertex_t> d_v_start(v_start.size(), handle.get_stream());
  raft::update_device(d_v_start.data(), v_start.data(), d_v_start.size(), handle.get_stream());

  index_t num_paths = v_start.size();
  index_t max_depth = 5;

  // 0-copy const device view:
  //
  detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(), num_paths};
  auto quad = detail::random_walks_impl(handle, graph_view, d_start_view, max_depth);

  auto& d_coalesced_v = std::get<0>(quad);
  auto& d_coalesced_w = std::get<1>(quad);
  auto& d_sizes       = std::get<2>(quad);
  auto seed0          = std::get<3>(quad);

  bool test_all_paths =
    cugraph::test::host_check_rw_paths(handle, graph_view, d_coalesced_v, d_coalesced_w, d_sizes);

  if (!test_all_paths) std::cout << "starting seed on failure: " << seed0 << '\n';

  ASSERT_TRUE(test_all_paths);
}

TEST(RandomWalksSpecialCase, UnweightedGraph)
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
  std::vector<weight_t> v_w;

  auto graph =
    make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, false);  // un-weighted

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  ASSERT_TRUE(values == nullptr);

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);

  raft::update_host(v_ro.data(), offsets, v_ro.size(), handle.get_stream());
  raft::update_host(v_ci.data(), indices, v_ci.size(), handle.get_stream());

  std::vector<vertex_t> v_start{2};
  vector_test_t<vertex_t> d_v_start(v_start.size(), handle.get_stream());
  raft::update_device(d_v_start.data(), v_start.data(), d_v_start.size(), handle.get_stream());

  index_t num_paths = v_start.size();
  index_t max_depth = 5;

  // 0-copy const device view:
  //
  detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(), num_paths};
  auto quad = detail::random_walks_impl(handle, graph_view, d_start_view, max_depth);

  auto& d_coalesced_v = std::get<0>(quad);
  auto& d_coalesced_w = std::get<1>(quad);
  auto& d_sizes       = std::get<2>(quad);
  auto seed0          = std::get<3>(quad);

  bool test_all_paths =
    cugraph::test::host_check_rw_paths(handle, graph_view, d_coalesced_v, d_coalesced_w, d_sizes);

  if (!test_all_paths) std::cout << "starting seed on failure: " << seed0 << '\n';

  ASSERT_TRUE(test_all_paths);
}

TEST(RandomWalksPadded, SimpleGraph)
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

  auto graph = make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.offsets();
  vertex_t const* indices = graph_view.indices();
  weight_t const* values  = graph_view.weights();

  std::vector<edge_t> v_ro(num_vertices + 1);
  std::vector<vertex_t> v_ci(num_edges);
  std::vector<weight_t> v_vals(num_edges);

  raft::update_host(v_ro.data(), offsets, v_ro.size(), handle.get_stream());
  raft::update_host(v_ci.data(), indices, v_ci.size(), handle.get_stream());
  raft::update_host(v_vals.data(), values, v_vals.size(), handle.get_stream());

  std::vector<vertex_t> v_start{2};
  vector_test_t<vertex_t> d_v_start(v_start.size(), handle.get_stream());
  raft::update_device(d_v_start.data(), v_start.data(), d_v_start.size(), handle.get_stream());

  index_t num_paths = v_start.size();
  index_t max_depth = 5;

  // 0-copy const device view:
  //
  detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(), num_paths};
  bool use_padding{true};
  auto quad = detail::random_walks_impl(handle, graph_view, d_start_view, max_depth, use_padding);

  auto& d_coalesced_v = std::get<0>(quad);
  auto& d_coalesced_w = std::get<1>(quad);
  auto& d_sizes       = std::get<2>(quad);
  auto seed0          = std::get<3>(quad);

  ASSERT_TRUE(d_sizes.size() == 0);

  bool test_all_paths = cugraph::test::host_check_rw_paths(
    handle, graph_view, d_coalesced_v, d_coalesced_w, d_sizes, num_paths);

  if (!test_all_paths) std::cout << "starting seed on failure: " << seed0 << '\n';

  ASSERT_TRUE(test_all_paths);
}

TEST(RandomWalksUtility, PathsToCOO)
{
  using namespace cugraph::experimental::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  std::vector<index_t> v_sizes{2, 1, 3, 5, 1};
  std::vector<vertex_t> v_coalesced{5, 3, 4, 9, 0, 1, 6, 2, 7, 3, 2, 5};
  std::vector<weight_t> w_coalesced{0.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto num_paths = v_sizes.size();
  auto total_sz  = v_coalesced.size();
  auto num_edges = w_coalesced.size();

  ASSERT_TRUE(num_edges == total_sz - num_paths);

  vector_test_t<vertex_t> d_coalesced_v(total_sz, handle.get_stream());
  vector_test_t<index_t> d_sizes(num_paths, handle.get_stream());

  raft::update_device(
    d_coalesced_v.data(), v_coalesced.data(), d_coalesced_v.size(), handle.get_stream());
  raft::update_device(d_sizes.data(), v_sizes.data(), d_sizes.size(), handle.get_stream());

  index_t coalesced_v_sz = d_coalesced_v.size();

  auto tpl_coo_offsets = convert_paths_to_coo<vertex_t>(handle,
                                                        coalesced_v_sz,
                                                        static_cast<index_t>(num_paths),
                                                        d_coalesced_v.release(),
                                                        d_sizes.release());

  auto&& d_src     = std::move(std::get<0>(tpl_coo_offsets));
  auto&& d_dst     = std::move(std::get<1>(tpl_coo_offsets));
  auto&& d_offsets = std::move(std::get<2>(tpl_coo_offsets));

  ASSERT_TRUE(d_src.size() == num_edges);
  ASSERT_TRUE(d_dst.size() == num_edges);

  std::vector<vertex_t> v_src(num_edges, 0);
  std::vector<vertex_t> v_dst(num_edges, 0);
  std::vector<index_t> v_offsets(d_offsets.size(), 0);

  raft::update_host(v_src.data(), raw_const_ptr(d_src), d_src.size(), handle.get_stream());
  raft::update_host(v_dst.data(), raw_const_ptr(d_dst), d_dst.size(), handle.get_stream());
  raft::update_host(
    v_offsets.data(), raw_const_ptr(d_offsets), d_offsets.size(), handle.get_stream());

  std::vector<vertex_t> v_src_exp{5, 9, 0, 6, 2, 7, 3};
  std::vector<vertex_t> v_dst_exp{3, 0, 1, 2, 7, 3, 2};
  std::vector<index_t> v_offsets_exp{0, 1, 3};

  EXPECT_EQ(v_src, v_src_exp);
  EXPECT_EQ(v_dst, v_dst_exp);
  EXPECT_EQ(v_offsets, v_offsets_exp);
}

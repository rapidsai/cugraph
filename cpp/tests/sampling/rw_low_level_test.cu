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

#include "cuda_profiler_api.h"
#include <gtest/gtest.h>

#include <topology/topology.cuh>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <rmm/exec_policy.hpp>
#include <thrust/random.h>

#include <cugraph/algorithms.hpp>
#include <sampling/random_walks.cuh>

#include <raft/handle.hpp>
#include <raft/random/rng.cuh>

#include "random_walks_utils.cuh"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <utilities/high_res_timer.hpp>
#include <vector>

template <typename value_t>
using vector_test_t = cugraph::detail::device_vec_t<value_t>;  // for debug purposes

namespace {  // anonym.

template <typename vertex_t, typename edge_t, typename index_t>
bool check_col_indices(raft::handle_t const& handle,
                       vector_test_t<edge_t> const& d_crt_out_degs,
                       vector_test_t<vertex_t> const& d_col_indx,
                       index_t num_paths)
{
  bool all_indices_within_degs = thrust::all_of(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<index_t>(0),
    thrust::make_counting_iterator<index_t>(num_paths),
    [p_d_col_indx     = cugraph::detail::raw_const_ptr(d_col_indx),
     p_d_crt_out_degs = cugraph::detail::raw_const_ptr(d_crt_out_degs)] __device__(auto indx) {
      if (p_d_crt_out_degs[indx] > 0)
        return ((p_d_col_indx[indx] >= 0) && (p_d_col_indx[indx] < p_d_crt_out_degs[indx]));
      else
        return true;
    });
  return all_indices_within_degs;
}

template <typename vertex_t, typename real_t, typename selector_t>
void next_biased(raft::handle_t const& handle,
                 vector_test_t<vertex_t> const& d_src_v,
                 vector_test_t<real_t> const& d_rnd,
                 vector_test_t<vertex_t>& d_next_v,
                 selector_t const& selector)
{
  thrust::transform(handle.get_thrust_policy(),
                    d_src_v.begin(),
                    d_src_v.end(),
                    d_rnd.begin(),
                    d_next_v.begin(),
                    [sampler = selector.get_strategy()] __device__(auto src_v_indx, auto rnd_val) {
                      auto next_vw = sampler(src_v_indx, rnd_val);
                      return (next_vw.has_value() ? thrust::get<0>(*next_vw) : src_v_indx);
                    });
}

// simulates max_depth==1 traversal of multiple paths,
// where num_paths = distance(begin, end), below:
//
template <typename vertex_t, typename real_t, typename selector_t>
void next_node2vec(raft::handle_t const& handle,
                   vector_test_t<vertex_t> const& d_src_v,
                   vector_test_t<thrust::optional<vertex_t>> const& d_prev_v,
                   vector_test_t<real_t> const& d_rnd,
                   vector_test_t<vertex_t>& d_next_v,
                   selector_t const& selector)
{
  size_t num_paths{d_src_v.size()};
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(
    d_src_v.begin(), d_prev_v.begin(), thrust::make_counting_iterator<size_t>(0)));
  auto end   = thrust::make_zip_iterator(thrust::make_tuple(
    d_src_v.end(), d_prev_v.end(), thrust::make_counting_iterator<size_t>(num_paths)));

  thrust::transform(handle.get_thrust_policy(),
                    begin,
                    end,
                    d_rnd.begin(),
                    d_next_v.begin(),
                    [sampler = selector.get_strategy()] __device__(auto tpl, auto rnd_val) {
                      vertex_t src_v = thrust::get<0>(tpl);

                      size_t path_index = thrust::get<2>(tpl);

                      if (thrust::get<1>(tpl) != thrust::nullopt) {
                        vertex_t prev_v = *thrust::get<1>(tpl);

                        auto next_vw = sampler(src_v, rnd_val, prev_v, path_index, false);
                        return (next_vw.has_value() ? thrust::get<0>(*next_vw) : src_v);
                      } else {
                        return src_v;
                      }
                    });
}

template <typename vertex_t, typename edge_t, typename weight_t>
void alpha_node2vec(std::vector<edge_t> const& row_offsets,
                    std::vector<vertex_t> const& col_indices,
                    std::vector<weight_t>& weights,  // to be scaled!
                    std::vector<thrust::optional<vertex_t>> const& v_pred,
                    std::vector<vertex_t> const& v_crt,
                    weight_t p,
                    weight_t q)
{
  auto num_vs = v_crt.size();
  for (size_t indx = 0; indx < num_vs; ++indx) {
    auto src_v = v_crt[indx];

    size_t num_neighbors = row_offsets[src_v + 1] - row_offsets[src_v];

    if (num_neighbors == 0) { continue; }

    if (v_pred[indx].has_value()) {
      auto pred_v = *(v_pred[indx]);

      for (auto offset_indx = row_offsets[src_v]; offset_indx < row_offsets[src_v + 1];
           ++offset_indx) {
        auto next_v = col_indices[offset_indx];

        weight_t alpha{0};

        if (next_v == pred_v) {
          alpha = 1.0 / p;
        } else {
          auto begin    = col_indices.begin() + row_offsets[pred_v];
          auto end      = col_indices.begin() + row_offsets[pred_v + 1];
          auto it_found = std::find(begin, end, next_v);

          if (it_found != end) {
            alpha = 1.0;
          } else {
            alpha = 1.0 / q;
          }
        }

        weights[offset_indx] *= alpha;  // scale weights
      }
    }
  }
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
  using namespace cugraph::detail;

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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

  random_walker_t<decltype(graph_view)> rand_walker{handle, num_vertices, num_paths, max_depth};

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
  using namespace cugraph::detail;

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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

  random_walker_t<decltype(graph_view)> rand_walker{handle, num_vertices, num_paths, max_depth};

  auto d_out_degs = rand_walker.get_out_degs(graph_view);
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
  using namespace cugraph::detail;

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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

  random_walker_t<decltype(graph_view)> rand_walker{handle, num_vertices, num_paths, max_depth};

  auto d_out_degs = rand_walker.get_out_degs(graph_view);

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // update crt_out_degs:
  //
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  rand_walker.gather_from_coalesced(
    d_coalesced_v, d_out_degs, d_sizes, d_crt_out_degs, max_depth, num_paths);

  col_indx_extract_t<decltype(graph_view), index_t> col_extractor{
    handle, graph_view, raw_ptr(d_crt_out_degs), raw_ptr(d_sizes), num_paths, max_depth};

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
  using namespace cugraph::detail;

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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

  random_walker_t<decltype(graph_view)> rand_walker{handle, num_vertices, num_paths, max_depth};

  auto d_out_degs = rand_walker.get_out_degs(graph_view);

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
  random_engine_t rgen(handle, num_paths, d_random, seed);
  rgen.generate_col_indices(d_crt_out_degs, d_col_indx);

  bool all_indices_within_degs = check_col_indices(handle, d_crt_out_degs, d_col_indx, num_paths);

  ASSERT_TRUE(all_indices_within_degs);
}

TEST_F(RandomWalksPrimsTest, SimpleGraphUpdatePathSizes)
{
  using namespace cugraph::detail;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;
  using real_t   = float;
  using seed_t   = long;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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

  random_walker_t<decltype(graph_view)> rand_walker{handle, num_vertices, num_paths, max_depth};

  auto d_out_degs = rand_walker.get_out_degs(graph_view);

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
  using namespace cugraph::detail;

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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

  random_walker_t<decltype(graph_view)> rand_walker{handle, num_vertices, num_paths, max_depth};

  auto d_out_degs = rand_walker.get_out_degs(graph_view);

  rand_walker.start(d_start, d_coalesced_v, d_sizes);

  // update crt_out_degs:
  //
  vector_test_t<edge_t> d_crt_out_degs(num_paths, handle.get_stream());
  rand_walker.gather_from_coalesced(
    d_coalesced_v, d_out_degs, d_sizes, d_crt_out_degs, max_depth, num_paths);

  col_indx_extract_t<decltype(graph_view), index_t> col_extractor{
    handle, graph_view, raw_ptr(d_crt_out_degs), raw_ptr(d_sizes), num_paths, max_depth};

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
  using namespace cugraph::detail;

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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

  random_walker_t<decltype(graph_view)> rand_walker{handle, num_vertices, num_paths, max_depth};

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
  using real_t   = float;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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
  cugraph::detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(),
                                                                            num_paths};
  using graph_t = decltype(graph_view);

  cugraph::detail::uniform_selector_t<graph_t, real_t> selector{handle, graph_view, real_t{0}};

  auto quad =
    cugraph::detail::random_walks_impl(handle, graph_view, d_start_view, max_depth, selector);

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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
  cugraph::detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(),
                                                                            num_paths};
  using graph_t = decltype(graph_view);
  using real_t  = float;
  cugraph::detail::uniform_selector_t<graph_t, real_t> selector{handle, graph_view, real_t{0}};

  auto quad =
    cugraph::detail::random_walks_impl(handle, graph_view, d_start_view, max_depth, selector);

  auto& d_v_sizes = std::get<2>(quad);
  auto seed0      = std::get<3>(quad);

  auto triplet =
    cugraph::query_rw_sizes_offsets(handle, num_paths, cugraph::detail::raw_const_ptr(d_v_sizes));

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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
  cugraph::detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(),
                                                                            num_paths};
  using graph_t = decltype(graph_view);
  using real_t  = float;
  cugraph::detail::uniform_selector_t<graph_t, real_t> selector{handle, graph_view, real_t{0}};

  auto quad =
    cugraph::detail::random_walks_impl(handle, graph_view, d_start_view, max_depth, selector);

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

  auto graph = cugraph::test::make_graph<vertex_t, edge_t, weight_t>(
    handle, v_src, v_dst, std::nullopt, num_vertices, num_edges);  // un-weighted

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  ASSERT_TRUE(graph_view.local_edge_partition_view().weights().has_value() == false);

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
  cugraph::detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(),
                                                                            num_paths};
  using graph_t = decltype(graph_view);
  using real_t  = float;
  cugraph::detail::uniform_selector_t<graph_t, real_t> selector{handle, graph_view, real_t{0}};

  auto quad =
    cugraph::detail::random_walks_impl(handle, graph_view, d_start_view, max_depth, selector);

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto graph_view = graph.view();

  edge_t const* offsets   = graph_view.local_edge_partition_view().offsets();
  vertex_t const* indices = graph_view.local_edge_partition_view().indices();
  weight_t const* values  = *(graph_view.local_edge_partition_view().weights());

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
  cugraph::detail::device_const_vector_view<vertex_t, index_t> d_start_view{d_v_start.data(),
                                                                            num_paths};
  bool use_padding{true};

  using graph_t = decltype(graph_view);
  using real_t  = float;
  cugraph::detail::uniform_selector_t<graph_t, real_t> selector{handle, graph_view, real_t{0}};

  auto quad = cugraph::detail::random_walks_impl(
    handle, graph_view, d_start_view, max_depth, selector, use_padding);

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
  using namespace cugraph::detail;

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

  auto tpl_coo_offsets = cugraph::convert_paths_to_coo<vertex_t>(handle,
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

TEST(BiasedRandomWalks, SelectorSmallGraph)
{
  namespace topo = cugraph::topology;

  raft::handle_t handle{};

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;
  using real_t   = weight_t;

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  /*
    0 --(.1)--> 1 --(1.1)--> 4
   /|\       /\ |            |
    |       /   |            |
   (5.1) (3.1)(2.1)        (3.2)
    |   /       |            |
    | /        \|/          \|/
    2 --(4.1)-->3 --(7.2)--> 5
   */
  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  std::vector<real_t> v_rnd{0.0,
                            1.0,
                            0.2,  // 0
                            0.0,
                            1.0,
                            0.8,  // 1
                            0.0,
                            1.0,
                            0.5,  // 2
                            0.0,
                            1.0,  // 3
                            0.0,
                            1.0,  // 4
                            0.0,
                            1.0};  // 5

  std::vector<vertex_t> v_src_v{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5};

  vector_test_t<real_t> d_rnd(v_rnd.size(), handle.get_stream());
  vector_test_t<vertex_t> d_src_v(v_src_v.size(), handle.get_stream());

  EXPECT_EQ(d_rnd.size(), d_src_v.size());

  raft::update_device(d_rnd.data(), v_rnd.data(), d_rnd.size(), handle.get_stream());
  raft::update_device(d_src_v.data(), v_src_v.data(), d_src_v.size(), handle.get_stream());

  auto graph_view = graph.view();

  edge_t const* offsets = graph_view.local_edge_partition_view().offsets();

  vertex_t const* indices = graph_view.local_edge_partition_view().indices();

  weight_t const* values = *(graph_view.local_edge_partition_view().weights());

  cugraph::detail::biased_selector_t selector{handle, graph_view, 0.0f};

  std::vector<weight_t> h_correct_sum_w{0.1f, 3.2f, 12.3f, 7.2f, 3.2f, 0.0f};
  std::vector<weight_t> v_sum_weights(num_vertices);

  auto const& d_sum_weights = selector.get_sum_weights();

  raft::update_host(
    v_sum_weights.data(), d_sum_weights.data(), d_sum_weights.size(), handle.get_stream());

  // test floating point equality between vectors:
  //
  weight_t eps = 1.0e-6f;

  auto end =
    thrust::make_zip_iterator(thrust::make_tuple(v_sum_weights.end(), h_correct_sum_w.end()));
  auto it = thrust::find_if(
    thrust::host,
    thrust::make_zip_iterator(thrust::make_tuple(v_sum_weights.begin(), h_correct_sum_w.begin())),
    end,
    [eps](auto const& tpl) { return std::abs(thrust::get<0>(tpl) - thrust::get<1>(tpl)) > eps; });

  EXPECT_EQ(it, end);

  vector_test_t<vertex_t> d_next_v(v_src_v.size(), handle.get_stream());

  next_biased(handle, d_src_v, d_rnd, d_next_v, selector);

  std::vector<edge_t> v_next_v(v_src_v.size());

  raft::update_host(v_next_v.data(), d_next_v.data(), v_src_v.size(), handle.get_stream());

  std::vector h_next_v{1,
                       1,
                       1, /*<-0*/
                       3,
                       4,
                       4, /*<-1*/
                       0,
                       3,
                       1, /*<-2*/
                       5,
                       5, /*<-3*/
                       5,
                       5, /*<-4*/
                       5,
                       5}; /*<-5*/

  EXPECT_EQ(v_next_v, h_next_v);
}

TEST(Node2VecRandomWalks, Node2VecSmallGraph)
{
  namespace topo = cugraph::topology;

  raft::handle_t handle{};

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;
  using real_t   = weight_t;

  weight_t p = 2.0;
  weight_t q = 4.0;

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  // Step 1: graph construction:
  //
  /*
    0 --(.1)--> 1 --(1.1)--> 4
   /|\       /\ |            |
    |       /   |            |
   (5.1) (3.1)(2.1)        (3.2)
    |   /       |            |
    | /        \|/          \|/
    2 --(4.1)-->3 --(7.2)--> 5
   */
  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w(num_edges, 1.0);  //{0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  std::vector<real_t> v_rnd{0.2, 0.5, 1.0, 0.1, 0.8};
  std::vector<vertex_t> v_src_v{0, 1, 3, 4, 5};
  std::vector<thrust::optional<vertex_t>> v_pred_v{2, 0, 1, 1, 4};

  vector_test_t<real_t> d_rnd(v_rnd.size(), handle.get_stream());
  vector_test_t<vertex_t> d_src_v(v_src_v.size(), handle.get_stream());

  EXPECT_EQ(d_rnd.size(), d_src_v.size());

  raft::update_device(d_rnd.data(), v_rnd.data(), d_rnd.size(), handle.get_stream());
  raft::update_device(d_src_v.data(), v_src_v.data(), d_src_v.size(), handle.get_stream());

  auto graph_view = graph.view();

  edge_t const* offsets = graph_view.local_edge_partition_view().offsets();

  vertex_t const* indices = graph_view.local_edge_partition_view().indices();

  weight_t const* values = *(graph_view.local_edge_partition_view().weights());

  // Step 2: `node2vec` selection on original graph:
  //
  cugraph::detail::node2vec_selector_t n2v_selector{handle, graph_view, 0.0f, p, q};

  vector_test_t<thrust::optional<vertex_t>> d_pred_v(v_pred_v.size(), handle.get_stream());

  raft::update_device(d_pred_v.data(), v_pred_v.data(), v_pred_v.size(), handle.get_stream());

  vector_test_t<vertex_t> d_next_v(v_src_v.size(), handle.get_stream());

  // `node2vec` stepping:
  //
  next_node2vec(handle, d_src_v, d_pred_v, d_rnd, d_next_v, n2v_selector);

  std::vector<vertex_t> n2v_next_v(v_src_v.size());
  raft::update_host(n2v_next_v.data(), d_next_v.data(), v_src_v.size(), handle.get_stream());

  EXPECT_EQ(n2v_next_v.size(), d_src_v.size());

  // Step 3: construct similar graph, just with
  //         alpha scaled weights;
  //
  std::vector<weight_t> scaled_weights(v_w);
  std::vector<edge_t> row_offsets(num_vertices + 1);
  std::vector<vertex_t> col_indices(num_edges);

  raft::update_host(
    row_offsets.data(), offsets, static_cast<size_t>(num_vertices + 1), handle.get_stream());

  raft::update_host(
    col_indices.data(), indices, static_cast<size_t>(num_edges), handle.get_stream());

  std::vector<edge_t> v_ro{0, 1, 3, 6, 7, 8, 8};
  std::vector<vertex_t> v_ci{1, 3, 4, 0, 1, 3, 5, 5};

  EXPECT_EQ(row_offsets, v_ro);
  EXPECT_EQ(col_indices, v_ci);
  EXPECT_EQ(scaled_weights.size(), static_cast<size_t>(num_edges));

  alpha_node2vec(row_offsets, col_indices, scaled_weights, v_pred_v, v_src_v, p, q);

  auto scaled_graph =
    cugraph::test::make_graph(handle,
                              v_src,
                              v_dst,
                              std::optional<std::vector<weight_t>>{scaled_weights},
                              num_vertices,
                              num_edges);

  auto scaled_graph_view = scaled_graph.view();

  // Step 4: biased selection on alpha scaled graph:
  //
  cugraph::detail::biased_selector_t selector{handle, scaled_graph_view, 0.0f};

  next_biased(handle, d_src_v, d_rnd, d_next_v, selector);

  std::vector<vertex_t> biased_next_v(v_src_v.size());
  raft::update_host(biased_next_v.data(), d_next_v.data(), v_src_v.size(), handle.get_stream());

  // Step 5: compare `node2vec` on original graph
  //         with biased on graph with alpha scaled weights:
  //
  EXPECT_EQ(biased_next_v, n2v_next_v);
}

TEST(Node2VecRandomWalks, CachedNode2VecSmallGraph)
{
  namespace topo = cugraph::topology;

  raft::handle_t handle{};

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;
  using index_t  = vertex_t;
  using real_t   = weight_t;

  weight_t p = 2.0;
  weight_t q = 4.0;

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  // Step 1: graph construction:
  //
  /*
    0 --(.1)--> 1 --(1.1)--> 4
   /|\       /\ |            |
    |       /   |            |
   (5.1) (3.1)(2.1)        (3.2)
    |   /       |            |
    | /        \|/          \|/
    2 --(4.1)-->3 --(7.2)--> 5
   */
  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w(num_edges, 1.0);  //{0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  std::vector<real_t> v_rnd{0.2, 0.5, 1.0, 0.1, 0.8};
  std::vector<vertex_t> v_src_v{0, 1, 3, 4, 5};
  std::vector<thrust::optional<vertex_t>> v_pred_v{2, 0, 1, 1, 4};

  vector_test_t<real_t> d_rnd(v_rnd.size(), handle.get_stream());
  vector_test_t<vertex_t> d_src_v(v_src_v.size(), handle.get_stream());

  EXPECT_EQ(d_rnd.size(), d_src_v.size());

  raft::update_device(d_rnd.data(), v_rnd.data(), d_rnd.size(), handle.get_stream());
  raft::update_device(d_src_v.data(), v_src_v.data(), d_src_v.size(), handle.get_stream());

  auto graph_view = graph.view();

  edge_t const* offsets = graph_view.local_edge_partition_view().offsets();

  vertex_t const* indices = graph_view.local_edge_partition_view().indices();

  weight_t const* values = *(graph_view.local_edge_partition_view().weights());

  // Step 2: `node2vec` selection on original graph:
  //
  // CAVEAT: next_node2vec(), steps in parallel, so it simulates
  //         traversing multiple paths (of size max_depth == 1);
  //         if ignored, this creates a data race on the cached
  //         alpha buffer!
  //
  edge_t num_paths(d_src_v.size());
  cugraph::detail::node2vec_selector_t n2v_selector{
    handle, graph_view, 0.0f, p, q, num_paths};  // use cached approach

  auto const& d_cached_alpha = n2v_selector.get_alpha_cache();

  size_t expected_max_degree{3};
  EXPECT_EQ(d_cached_alpha.size(), expected_max_degree * num_paths);

  auto&& coalesced_alpha = n2v_selector.get_strategy().get_alpha_buffer();

  ASSERT_TRUE(coalesced_alpha != thrust::nullopt);

  EXPECT_EQ(static_cast<size_t>(thrust::get<0>(*coalesced_alpha)), expected_max_degree);
  EXPECT_EQ(thrust::get<1>(*coalesced_alpha), num_paths);
  EXPECT_EQ(thrust::get<2>(*coalesced_alpha), d_cached_alpha.data());

  vector_test_t<thrust::optional<vertex_t>> d_pred_v(v_pred_v.size(), handle.get_stream());

  raft::update_device(d_pred_v.data(), v_pred_v.data(), v_pred_v.size(), handle.get_stream());

  vector_test_t<vertex_t> d_next_v(v_src_v.size(), handle.get_stream());

  // `node2vec` stepping:
  //
  // CAVEAT: next_node2vec(), steps in parallel, so it simulates
  //         traversing multiple paths (of size max_depth == 1);
  //         if ignored, this creates a data race on the cached
  //         alpha buffer!
  //
  next_node2vec(handle, d_src_v, d_pred_v, d_rnd, d_next_v, n2v_selector);

  std::vector<vertex_t> n2v_next_v(v_src_v.size());
  raft::update_host(n2v_next_v.data(), d_next_v.data(), v_src_v.size(), handle.get_stream());

  EXPECT_EQ(n2v_next_v.size(), d_src_v.size());

  // Step 3: construct similar graph, just with
  //         alpha scaled weights;
  //
  std::vector<weight_t> scaled_weights(v_w);
  std::vector<edge_t> row_offsets(num_vertices + 1);
  std::vector<vertex_t> col_indices(num_edges);

  raft::update_host(
    row_offsets.data(), offsets, static_cast<size_t>(num_vertices + 1), handle.get_stream());

  raft::update_host(
    col_indices.data(), indices, static_cast<size_t>(num_edges), handle.get_stream());

  std::vector<edge_t> v_ro{0, 1, 3, 6, 7, 8, 8};
  std::vector<vertex_t> v_ci{1, 3, 4, 0, 1, 3, 5, 5};

  EXPECT_EQ(row_offsets, v_ro);
  EXPECT_EQ(col_indices, v_ci);
  EXPECT_EQ(scaled_weights.size(), static_cast<size_t>(num_edges));

  alpha_node2vec(row_offsets, col_indices, scaled_weights, v_pred_v, v_src_v, p, q);

  auto scaled_graph =
    cugraph::test::make_graph(handle,
                              v_src,
                              v_dst,
                              std::optional<std::vector<weight_t>>{scaled_weights},
                              num_vertices,
                              num_edges);

  auto scaled_graph_view = scaled_graph.view();

  // Step 4: biased selection on alpha scaled graph:
  //
  cugraph::detail::biased_selector_t selector{handle, scaled_graph_view, 0.0f};

  next_biased(handle, d_src_v, d_rnd, d_next_v, selector);

  std::vector<vertex_t> biased_next_v(v_src_v.size());
  raft::update_host(biased_next_v.data(), d_next_v.data(), v_src_v.size(), handle.get_stream());

  // Step 5: compare `node2vec` on original graph
  //         with biased on graph with alpha scaled weights:
  //
  EXPECT_EQ(biased_next_v, n2v_next_v);
}

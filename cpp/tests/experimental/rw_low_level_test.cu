/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
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

  // ASSERT_TRUE(true);
}

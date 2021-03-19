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
}  // namespace

struct RandomWalksPrimsTest : public ::testing::Test {
};

using namespace cugraph::experimental;

TEST_F(RandomWalksPrimsTest, SimpleGraphColExtraction)
{
  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = float;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  rmm::device_uvector<vertex_t> d_src(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst(num_edges, handle.get_stream());
  rmm::device_uvector<weight_t> d_weights(num_edges, handle.get_stream());

  copy_n(handle, d_src, v_src);
  copy_n(handle, d_dst, v_dst);
  copy_n(handle, d_weights, v_w);

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    d_src.data(), d_dst.data(), d_weights.data(), num_edges};

  graph_t<vertex_t, edge_t, weight_t, false, false> graph(
    handle, edgelist, num_vertices, graph_properties_t{}, false);

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

  // ASSERT_TRUE(true);
}

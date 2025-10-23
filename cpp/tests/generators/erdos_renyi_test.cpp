/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"

#include <cugraph/graph.hpp>
#include <cugraph/graph_generators.hpp>

#include <cuda/std/tuple>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <gtest/gtest.h>

struct GenerateErdosRenyiTest : public ::testing::Test {};

template <typename vertex_t>
void test_symmetric(std::vector<vertex_t>& h_src_v, std::vector<vertex_t>& h_dst_v)
{
  std::vector<vertex_t> reverse_src_v(h_src_v.size());
  std::vector<vertex_t> reverse_dst_v(h_dst_v.size());

  std::copy(h_src_v.begin(), h_src_v.end(), reverse_dst_v.begin());
  std::copy(h_dst_v.begin(), h_dst_v.end(), reverse_src_v.begin());

  thrust::sort(thrust::host,
               thrust::make_zip_iterator(h_src_v.begin(), h_dst_v.begin()),
               thrust::make_zip_iterator(h_src_v.end(), h_dst_v.end()));

  thrust::sort(thrust::host,
               thrust::make_zip_iterator(reverse_src_v.begin(), reverse_dst_v.begin()),
               thrust::make_zip_iterator(reverse_src_v.end(), reverse_dst_v.end()));

  EXPECT_EQ(reverse_src_v, h_src_v);
  EXPECT_EQ(reverse_dst_v, h_dst_v);
}

template <typename vertex_t>
void er_test(size_t num_vertices, float p)
{
  raft::handle_t handle;
  rmm::device_uvector<vertex_t> d_src_v(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst_v(0, handle.get_stream());

  std::tie(d_src_v, d_dst_v) =
    cugraph::generate_erdos_renyi_graph_edgelist_gnp<vertex_t>(handle, num_vertices, p, 0);

  handle.sync_stream();

  auto h_src_v = cugraph::test::to_host(handle, d_src_v);
  auto h_dst_v = cugraph::test::to_host(handle, d_dst_v);

  float expected_edge_count = p * num_vertices * num_vertices;

  ASSERT_GE(h_src_v.size(), static_cast<size_t>(expected_edge_count * 0.8));
  ASSERT_LE(h_src_v.size(), static_cast<size_t>(expected_edge_count * 1.2));
  ASSERT_EQ(std::count_if(h_src_v.begin(),
                          h_src_v.end(),
                          [n = static_cast<vertex_t>(num_vertices)](auto v) {
                            return !cugraph::is_valid_vertex(n, v);
                          }),
            0);
  ASSERT_EQ(std::count_if(h_dst_v.begin(),
                          h_dst_v.end(),
                          [n = static_cast<vertex_t>(num_vertices)](auto v) {
                            return !cugraph::is_valid_vertex(n, v);
                          }),
            0);
}

TEST_F(GenerateErdosRenyiTest, ERTest)
{
  er_test<int32_t>(size_t{10}, float{0.1});
  er_test<int32_t>(size_t{10}, float{0.5});
  er_test<int32_t>(size_t{20}, float{0.1});
  er_test<int32_t>(size_t{50}, float{0.1});
  er_test<int32_t>(size_t{10000}, float{0.1});
}

CUGRAPH_TEST_PROGRAM_MAIN()

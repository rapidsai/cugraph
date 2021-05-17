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

#include <cugraph/serialization/serializer.hpp>

TEST(SerializationTest, GraphSerUnser)
{
  using namespace cugraph::serializer;

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

  auto graph = cugraph::test::make_graph(handle, v_src, v_dst, v_w, num_vertices, num_edges, true);

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

  size_t total_ser_sz = (num_vertices + 1) * sizeof(edge_t) + num_edges * sizeof(vertex_t) +
                        num_edges * sizeof(weight_t);

  auto evaluated_sz = serializer_t::get_device_graph_sz_bytes(graph);
  EXPECT_EQ(total_ser_sz, evaluated_sz);

  serializer_t ser(handle, total_ser_sz);
  auto graph_meta = ser.serialize(graph);

  total_ser_sz = serializer_t::get_device_graph_sz_bytes(graph_meta);
  EXPECT_EQ(total_ser_sz, evaluated_sz);

  auto graph_copy = ser.unserialize(graph_meta);

  auto graph_copy_view   = graph_copy.view();
  auto num_vertices_copy = graph_copy_view.get_number_of_vertices();
  auto num_edges_copy    = graph_copy_view.get_number_of_edges();

  EXPECT_EQ(num_vertices, num_vertices_copy);
  EXPECT_EQ(num_edges, num_edges_copy);
  EXPECT_EQ(graph.is_symmetric(), graph_copy.is_symmetric());
  EXPECT_EQ(graph.is_multigraph(), graph_copy.is_multigraph());
  EXPECT_EQ(graph.is_weighted(), graph_copy.is_weighted());

  std::vector<edge_t> v_ro_copy(num_vertices + 1);
  std::vector<vertex_t> v_ci_copy(num_edges);
  std::vector<weight_t> v_vs_copy(num_edges);

  raft::update_host(
    v_ro_copy.data(), graph_copy_view.offsets(), num_vertices + 1, handle.get_stream());
  raft::update_host(v_ci_copy.data(), graph_copy_view.indices(), num_edges, handle.get_stream());
  raft::update_host(v_vs_copy.data(), graph_copy_view.weights(), num_edges, handle.get_stream());

  EXPECT_EQ(v_ro, v_ro_copy);
  EXPECT_EQ(v_ci, v_ci_copy);
  EXPECT_EQ(v_vs, v_vs_copy);
  EXPECT_EQ(graph_view.get_local_adj_matrix_partition_segment_offsets(0),
            graph_copy_view.get_local_adj_matrix_partition_segment_offsets(0));
}

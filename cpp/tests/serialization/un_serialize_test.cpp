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

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

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

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto pair_sz      = serializer_t::get_device_graph_sz_bytes(graph);
  auto total_ser_sz = pair_sz.first + pair_sz.second;

  serializer_t ser(handle, total_ser_sz);
  serializer_t::graph_meta_t<decltype(graph)> graph_meta{};
  ser.serialize(graph, graph_meta);

  pair_sz          = serializer_t::get_device_graph_sz_bytes(graph_meta);
  auto post_ser_sz = pair_sz.first + pair_sz.second;

  EXPECT_EQ(total_ser_sz, post_ser_sz);

  auto graph_copy = ser.unserialize<decltype(graph)>(pair_sz.first, pair_sz.second);

  auto pair = cugraph::test::compare_graphs(handle, graph, graph_copy);
  if (pair.first == false) std::cerr << "Test failed with " << pair.second << ".\n";

  ASSERT_TRUE(pair.first);
}

TEST(SerializationTest, GraphDecoupledSerUnser)
{
  using namespace cugraph::serializer;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = double;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<weight_t> v_w{0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};

  auto graph = cugraph::test::make_graph(
    handle, v_src, v_dst, std::optional<std::vector<weight_t>>{v_w}, num_vertices, num_edges);

  auto pair_sz      = serializer_t::get_device_graph_sz_bytes(graph);
  auto total_ser_sz = pair_sz.first + pair_sz.second;

  // use the following buffer to simulate communication between
  // sender and reciever of the serialization:
  //
  rmm::device_uvector<serializer_t::byte_t> d_storage_comm(0, handle.get_stream());

  {
    serializer_t ser(handle, total_ser_sz);
    serializer_t::graph_meta_t<decltype(graph)> graph_meta{};
    ser.serialize(graph, graph_meta);

    pair_sz          = serializer_t::get_device_graph_sz_bytes(graph_meta);
    auto post_ser_sz = pair_sz.first + pair_sz.second;

    EXPECT_EQ(total_ser_sz, post_ser_sz);

    d_storage_comm.resize(total_ser_sz, handle.get_stream());
    raft::copy(d_storage_comm.data(), ser.get_storage(), total_ser_sz, handle.get_stream());
  }

  {
    serializer_t ser(handle, d_storage_comm.data());

    auto graph_copy = ser.unserialize<decltype(graph)>(pair_sz.first, pair_sz.second);

    auto pair = cugraph::test::compare_graphs(handle, graph, graph_copy);
    if (pair.first == false) std::cerr << "Test failed with " << pair.second << ".\n";

    ASSERT_TRUE(pair.first);
  }
}

TEST(SerializationTest, UnweightedGraphDecoupledSerUnser)
{
  using namespace cugraph::serializer;

  using vertex_t = int32_t;
  using edge_t   = vertex_t;
  using weight_t = double;
  using index_t  = vertex_t;

  raft::handle_t handle{};

  edge_t num_edges      = 8;
  vertex_t num_vertices = 6;

  std::vector<vertex_t> v_src{0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<vertex_t> v_dst{1, 3, 4, 0, 1, 3, 5, 5};

  auto graph = cugraph::test::make_graph<vertex_t, edge_t, weight_t>(
    handle, v_src, v_dst, std::nullopt, num_vertices, num_edges);

  ASSERT_TRUE(graph.view().local_edge_partition_view().weights().has_value() == false);

  auto pair_sz      = serializer_t::get_device_graph_sz_bytes(graph);
  auto total_ser_sz = pair_sz.first + pair_sz.second;

  // use the following buffer to simulate communication between
  // sender and reciever of the serialization:
  //
  rmm::device_uvector<serializer_t::byte_t> d_storage_comm(0, handle.get_stream());

  {
    serializer_t ser(handle, total_ser_sz);
    serializer_t::graph_meta_t<decltype(graph)> graph_meta{};
    ser.serialize(graph, graph_meta);

    pair_sz          = serializer_t::get_device_graph_sz_bytes(graph_meta);
    auto post_ser_sz = pair_sz.first + pair_sz.second;

    EXPECT_EQ(total_ser_sz, post_ser_sz);

    d_storage_comm.resize(total_ser_sz, handle.get_stream());
    raft::copy(d_storage_comm.data(), ser.get_storage(), total_ser_sz, handle.get_stream());
  }

  {
    serializer_t ser(handle, d_storage_comm.data());

    auto graph_copy = ser.unserialize<decltype(graph)>(pair_sz.first, pair_sz.second);

    ASSERT_TRUE(graph_copy.view().local_edge_partition_view().weights().has_value() == false);

    auto pair = cugraph::test::compare_graphs(handle, graph, graph_copy);
    if (pair.first == false) std::cerr << "Test failed with " << pair.second << ".\n";

    ASSERT_TRUE(pair.first);
  }
}

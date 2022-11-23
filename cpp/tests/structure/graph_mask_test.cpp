/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_mask.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

TEST(Test_GraphMask, BasicGraphMaskTestInt64)
{
  raft::handle_t handle;

  int number_of_vertices = 500;
  int number_of_edges    = 1000;

  cugraph::graph_mask_t<std::int64_t, std::int64_t> mask(
    handle, number_of_vertices, number_of_edges);

  auto mask_view = mask.view();

  ASSERT_EQ(false, mask.has_vertex_mask());
  ASSERT_EQ(false, mask.has_edge_mask());
  ASSERT_EQ(false, mask_view.has_vertex_mask());
  ASSERT_EQ(false, mask_view.has_edge_mask());

  mask.initialize_vertex_mask();
  mask.initialize_edge_mask();

  auto mask_view2 = mask.view();

  ASSERT_EQ(true, mask.has_vertex_mask());
  ASSERT_EQ(true, mask.has_edge_mask());
  ASSERT_EQ(true, mask_view2.has_vertex_mask());
  ASSERT_EQ(true, mask_view2.has_edge_mask());
}
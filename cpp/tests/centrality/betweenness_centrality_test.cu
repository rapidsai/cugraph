/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <thrust/device_vector.h>

#include <graph.hpp>
#include <algorithms.hpp>

struct BetweennessCentralityTest : public ::testing::Test
{
};

TEST_F(BetweennessCentralityTest, SimpleGraph)
{
  std::vector<int>  graph_offsets{ { 0, 1, 2, 5, 7, 10, 12, 14 } };
  std::vector<int>  graph_indices{ { 2, 2, 0, 1, 3, 2, 4, 3, 5, 6, 4, 6, 4, 5 } };

  std::vector<float> expected{ {0.0, 0.0, 0.6, 0.6, 0.5333333, 0.0, 0.0 } };

  int num_verts = graph_offsets.size() - 1;
  int num_edges = graph_indices.size();

  thrust::device_vector<int>    d_graph_offsets(graph_offsets);
  thrust::device_vector<int>    d_graph_indices(graph_indices);
  thrust::device_vector<float>  d_result(num_verts);

  std::vector<float>            result(num_verts);

  cugraph::experimental::GraphCSRView<int,int,float> G(d_graph_offsets.data().get(),
                                                   d_graph_indices.data().get(),
                                                   nullptr,
                                                   num_verts,
                                                   num_edges);

  cugraph::betweenness_centrality(G, d_result.data().get());

  cudaMemcpy(result.data(), d_result.data().get(), sizeof(float) * num_verts, cudaMemcpyDeviceToHost);

  for (int i = 0 ; i < num_verts ; ++i)
    EXPECT_FLOAT_EQ(result[i], expected[i]);
}

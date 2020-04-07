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
#include "test_utils.h"
#include <utility>

#include <graph.hpp>
#include <algorithms.hpp>

#include <queue>
#include <stack>

// =============================================================================
// C++ Reference Implementation
// =============================================================================
template<typename VT, typename ET>
void populate_neighbors(VT *indices, ET *offsets,
                        VT w, std::vector<VT> &neighbors) {
  ET edge_start = offsets[w];
  ET edge_end = offsets[w + 1];
  ET edge_count = edge_end - edge_start;

  neighbors.clear(); // Reset neighbors vector's size
  for (ET edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
    VT dst = indices[edge_start + edge_idx];
    neighbors.push_back(dst);
  }
}
// Algorithm 1: Shortest-path vertex betweenness, (Brandes, 2001)
template <typename VT, typename ET, typename WT, typename result_t>
void reference_betweenness_centrality_impl(VT *indices, ET *offsets,
                                           VT const number_of_vertices,
                                           result_t *result) {
  std::queue<VT> Q;
  std::stack<VT> S;
  // NOTE: dist is of type VT not WT
  std::vector<VT> dist(number_of_vertices);
  std::vector<std::vector<VT>> pred(number_of_vertices);
  std::vector<result_t> sigmas(number_of_vertices);
  std::vector<result_t> deltas(number_of_vertices);

  std::vector<VT> neighbors;

  for (VT s = 0; s < number_of_vertices; ++s) { 
    // Step 1: Single-source shortest-paths problem
    //   a. Initialization
    for (VT w = 0 ; w < number_of_vertices; ++w) {
      pred[w].clear();
      dist[w] = std::numeric_limits<VT>::max();
      sigmas[w] = 0;
    }
    dist[s] = 0;
    sigmas[s] = 1;
    Q.push(s);
    //   b. Traversal
    while (!Q.empty()) {
      VT v = Q.front();
      Q.pop();
      S.push(v);
      populate_neighbors<VT, ET>(indices, offsets, v, neighbors);
      for (VT w : neighbors) {
        // Path Discovery:
        // Found for the first time?
        if (dist[w] == std::numeric_limits<VT>::max()) {
          dist[w] = dist[v] + 1;
          Q.push(w);
        }
        // Path counting
        // Edge(v, w) on  a shortest path?
        if (dist[w] == dist[v] + 1) {
          sigmas[w] +=  sigmas[v];
          pred[w].push_back(v);
        }
      }
    }
    //  Step 2: Accumulation
    //          Back propagation of dependencies
    for (VT v = 0; v < number_of_vertices; ++v) {
      deltas[v] = 0;
    }
    while (!S.empty()) {
      VT w = S.top();
      S.pop();
      for (VT v : pred[w]) {
        deltas[v] +=  (sigmas[v] / sigmas[w]) * (1 + deltas[w]);
      }
      if (w != s) {
        result[w] += deltas[w];
      }
    }
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void reference_betweenness_centrality(cugraph::experimental::GraphCSR<VT, ET, WT> &graph,
                                      result_t *result, bool normalize) {

  VT number_of_vertices = graph.number_of_vertices;
  ET number_of_edges = graph.number_of_edges;
  std::vector<VT> indices(number_of_edges);
  std::vector<ET> offsets(number_of_vertices + 1);

  cudaMemcpy(indices.data(), graph.indices,
             sizeof(VT) * indices.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(offsets.data(), graph.offsets,
             sizeof(ET) * offsets.size(), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  reference_betweenness_centrality_impl<VT, ET, WT, result_t>(indices.data(), offsets.data(),
                                        number_of_vertices, result);
  if (normalize && number_of_vertices > 2) {
    result_t factor = static_cast<result_t>(number_of_vertices - 1) * static_cast<result_t>(number_of_vertices - 2);
    for (VT v = 0; v < number_of_vertices; ++v) {
      result[v] /= factor;
    }
  }
}
// Explicit declaration
template void reference_betweenness_centrality<int, int, float, float>(cugraph::experimental::GraphCSR<int, int, float> &,
                                                                  float *, bool);

// =============================================================================
// Test Suite
// =============================================================================
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

  cugraph::experimental::GraphCSR<int,int,float> G(d_graph_offsets.data().get(),
                                                   d_graph_indices.data().get(),
                                                   nullptr,
                                                   num_verts,
                                                   num_edges);

  cugraph::betweenness_centrality(G, d_result.data().get());

  cudaMemcpy(result.data(), d_result.data().get(), sizeof(float) * num_verts, cudaMemcpyDeviceToHost);

  for (int i = 0 ; i < num_verts ; ++i)
    EXPECT_FLOAT_EQ(result[i], expected[i]);

  // TODO(xcadet) Remove this part, it is for testing the reference
  std::vector<float> ref_result(num_verts);
  reference_betweenness_centrality(G, ref_result.data(), true);
  for (int i = 0 ; i < num_verts ; ++i)
    EXPECT_FLOAT_EQ(ref_result[i], expected[i]);
}

int main( int argc, char** argv )
{
    rmmInitialize(nullptr);
    testing::InitGoogleTest(&argc,argv);
    int rc = RUN_ALL_TESTS();
    rmmFinalize();
    return rc;
}
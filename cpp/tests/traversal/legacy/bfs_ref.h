/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#pragma once

#include <limits>
#include <queue>
#include <stack>
#include <vector>

template <typename VT, typename ET>
void populate_neighbors(VT* indices, ET* offsets, VT w, std::vector<VT>& neighbors)
{
  ET edge_start = offsets[w];
  ET edge_end   = offsets[w + 1];

  neighbors.assign(indices + edge_start, indices + edge_end);
}

// This implements the BFS based on (Brandes, 2001) for shortest path counting
template <typename VT, typename ET>
void ref_bfs(VT* indices,
             ET* offsets,
             VT const number_of_vertices,
             std::queue<VT>& Q,
             std::stack<VT>& S,
             std::vector<VT>& dist,
             std::vector<std::vector<VT>>& pred,
             std::vector<double>& sigmas,
             VT source)
{
  std::vector<VT> neighbors;
  pred.clear();
  pred.resize(number_of_vertices);
  dist.assign(number_of_vertices, std::numeric_limits<VT>::max());
  sigmas.assign(number_of_vertices, 0);
  dist[source]   = 0;
  sigmas[source] = 1;
  Q.push(source);
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
        sigmas[w] += sigmas[v];
        pred[w].push_back(v);
      }
    }
  }
}

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

// Author: Xavier Cadet xcadet@nvidia.com
#pragma once

namespace cugraph {
namespace detail {
template <typename VT, typename WT>
class BC {
   private:
      // --- Information concerning the graph ---
      Graph *graph = nullptr;
      // --- These information are extracted on setup ---
      VT number_vertices;        // Number of vertices in the graph
      VT number_edges;           // Number of edges in the graph
      VT *offsets_ptr;           // Pointer of the offsets
      VT *indices_ptr;           // Pointers to the indices
      WT *edge_weights_ptr;      // Pointer to the weights

      bool configured = false;   // Falg to ensure configuration was called

      // --- Output ----
      // betweenness is set/read by users - using Vectors
      WT *betweenness = nullptr;

      // --- Data required to perform computation ----
      /*
      VT *predecessors = nullptr; // Predecessors required by sssp
      VT *sp_counters = nullptr;  // Shortest-Path counter required by sssp
      WT *sigmas = nullptr;       // Floating point version of sp_counters
      WT *deltas = nullptr;       // Dependencies counter
      */

      cudaStream_t stream;
      void setup();
      void clean();
      void accumulate(thrust::host_vector<WT> &h_betweenness,
                      thrust::host_vector<VT> &predecessors,
                      thrust::host_vector<VT> &h_sp_counters,
                      VT source);

   public:
      virtual ~BC(void) { clean(); }
      BC(Graph *_graph, cudaStream_t _stream = 0) :graph(_graph), stream(_stream) { setup(); }
      void configure(WT *betweenness);
      void compute();
};
} // namespace detail

} // namespace cugraph
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

#pragma once

#include <raft/handle.hpp>

namespace cugraph {
namespace detail {

  template<typename vertex_t, typename edge_t, typename weight_t>
  class TSP {
    public:
      TSP(const raft::handle_t &handle,
          const GraphCOOView<vertex_t, edge_t, weight_t> &graph);
      ~TSP();

    private:
      cudaStream_t stream;
      int blocks;
      int threads;
      int sm_count;
  }

} // namespace detail;
} // namespace cugraph;

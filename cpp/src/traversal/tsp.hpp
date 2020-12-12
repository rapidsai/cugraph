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
#include <rmm/device_uvector.hpp>

namespace cugraph {

  template<typename vertex_t, typename edge_t, typename weight_t>
  class TSP {
    public:
      TSP(const raft::handle_t &handle,
          GraphCOOView<vertex_t, edge_t, weight_t> &graph,
          const float *x_pos,
          const float *y_pos,
          const int restarts);

      float compute();

      ~TSP() {};

    private:
      const raft::handle_t &handle_;
      cudaStream_t stream_

      // COO
      const vertex_t *src_;
      const vertex_t *dst_;
      const weight_t *weight_;
      const vertext_t v_;
      const edge_t e_;

      const float *x_pos_;
      const float *y_pos_;
      const int restarts_;

      int max_blocks_;
      int max_threads_;
      int sm_count_;
  }

} // namespace cugraph;

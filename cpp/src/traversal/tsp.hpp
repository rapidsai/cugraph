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

#include <algorithms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/thrust_rmm_allocator.h>

namespace cugraph {
  namespace detail {
  template <typename vertex_t, typename edge_t, typename weight_t>
  class TSP {
    public:
      TSP(const raft::handle_t &handle,
          GraphCOOView<vertex_t, edge_t, weight_t> &graph_,
          const float *x_pos,
          const float *y_pos,
          const int restarts);

      float compute();
      void knn();
      ~TSP() {};

    private:
      const raft::handle_t &handle_;
      cudaStream_t stream_;
      int max_blocks_;
      int max_threads_;
      int sm_count_;

      // COO
      vertex_t *srcs_;
      vertex_t *dsts_;
      weight_t *weights_;
      vertex_t nodes_;
      edge_t edges_;

      // TSP
      const int restarts_;
      const float *x_pos_;
      const float *y_pos_;
      int *neighbors_;
  };
  } // namespace detail
} // namespace cugraph

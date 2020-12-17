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
  class TSP {
    public:
      TSP(const raft::handle_t &handle,
          const float *x_pos,
          const float *y_pos,
          const int nodes,
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

      // TSP
      const float *x_pos_;
      const float *y_pos_;
      const int restarts_;
      const int nodes_;

      rmm::device_vector<int> neighbors_vec_;
      int *neighbors_;
  };
  } // namespace detail
} // namespace cugraph

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <algorithms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

struct TSPResults{
  float **best_x_pos;
  float **best_y_pos;
  int **best_route;
  int *best_cost;
};

class TSP {
 public:
  TSP(raft::handle_t const &handle,
      int const *vtx_ptr,
      float const *x_pos,
      float const *y_pos,
      int nodes,
      int restarts,
      bool beam_search,
      int k,
      int nstart,
      bool verbose,
      int *route);

  void setup();
  void reset_batch();
  void get_initial_solution(int const batch);
  float compute();
  void knn();
  ~TSP(){};

 private:
  // Config
  raft::handle_t const &handle_;
  cudaStream_t stream_;
  int max_blocks_;
  int max_threads_;
  int warp_size_;
  int sm_count_;
  // how large a grid we want to run, this is fixed
  int restart_batch_;
  int best_thread_num_;

  // TSP
  int const *vtx_ptr_;
  int *route_;
  float const *x_pos_;
  float const *y_pos_;
  int nodes_;
  int restarts_;
  bool beam_search_;
  int k_;
  int nstart_;
  bool verbose_;

  // Scalars
  rmm::device_scalar<int> mylock_scalar_;
  rmm::device_scalar<int> best_cost_scalar_;
  rmm::device_scalar<int> climbs_scalar_;

  int *mylock_;
  int *best_cost_;

  // Vectors
  rmm::device_vector<int64_t> neighbors_vec_;
  rmm::device_vector<int> work_vec_;
  rmm::device_vector<float*> best_x_pos_vec_;
  rmm::device_vector<float*> best_y_pos_vec_;
  rmm::device_vector<int*> best_route_vec_;

  int64_t *neighbors_;
  int *work_;
  int *work_route_;
  TSPResults results_;
};
}  // namespace detail
}  // namespace cugraph

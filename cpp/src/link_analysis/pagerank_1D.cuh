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

// snmg pagerank
// Author: Alex Fender afender@nvidia.com

#pragma once

#include <rmm/thrust_rmm_allocator.h>
#include <numeric>
#include <raft/handle.hpp>

#include "utilities/error_utils.h"
#include "utilities/spmv_1D.cuh"

namespace cugraph {
namespace opg {

template <typename VT, typename ET, typename WT>
class Pagerank {
 private:
  size_t v_glob;                     // global number of vertices
  size_t v_loc;                      // local number of vertices
  size_t e_loc;                      // local number of edges
  WT alpha;                          // damping factor
  const raft::comms::comms_t &comm;  // info about the opg comm setup
  cudaStream_t stream;
  int sm_count;

  // Vertex offsets for each partition.
  // This information should be available on all threads/devices
  // part_offsets[device_id] contains the global ID
  // of the first vertex of the partion owned by device_id.
  // part_offsets[num_devices] contains the global number of vertices
  size_t *part_off;

  // Google matrix
  ET *off;
  VT *ind;
  rmm::device_vector<WT> val;       // values of the substochastic matrix
  rmm::device_vector<WT> bookmark;  // constant vector with dangling node info

  bool is_setup;

 public:
  Pagerank(const raft::handle_t &handle, const experimental::GraphCSCView<VT, ET, WT> &G);
  ~Pagerank();

  void transition_vals(const VT *degree);

  void flag_leafs(const VT *degree);

  // Artificially create the google matrix by setting val and bookmark
  void setup(WT _alpha, VT *degree);

  // run the power iteration on the google matrix
  void solve(int max_iter, WT *pagerank);
};

template <typename VT, typename ET, typename WT>
void pagerank(raft::handle_t const &handle,
              const experimental::GraphCSCView<VT, ET, WT> &G,
              WT *pagerank_result,
              const float damping_factor = 0.85,
              const int n_iter           = 40)
{
  // null pointers check
  CUGRAPH_EXPECTS(G.offsets != nullptr, "Invalid API parameter - offsets is null");
  CUGRAPH_EXPECTS(G.indices != nullptr, "Invalid API parameter - indidices is null");
  CUGRAPH_EXPECTS(pagerank_result != nullptr,
                  "Invalid API parameter - pagerank output memory must be allocated");

  // parameter values
  CUGRAPH_EXPECTS(damping_factor > 0.0,
                  "Invalid API parameter - invalid damping factor value (alpha<0)");
  CUGRAPH_EXPECTS(damping_factor < 1.0,
                  "Invalid API parameter - invalid damping factor value (alpha>1)");
  CUGRAPH_EXPECTS(n_iter > 0, "Invalid API parameter - n_iter must be > 0");

  rmm::device_vector<VT> degree(G.number_of_vertices);

  // in-degree of CSC (equivalent to out-degree of original edge list)
  G.degree(degree.data().get(), experimental::DegreeDirection::IN);

  // Allocate and intialize Pagerank class
  Pagerank<VT, ET, WT> pr_solver(handle, G);

  // Set all constants info
  pr_solver.setup(damping_factor, degree.data().get());

  // Run n_iter pagerank MG SPMVs.
  pr_solver.solve(n_iter, pagerank_result);
}

}  // namespace opg
}  // namespace cugraph

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

// Author: Alex Fender afender@nvidia.com

#pragma once

#include <rmm/thrust_rmm_allocator.h>
#include <numeric>
#include <raft/handle.hpp>

#include "utilities/error.hpp"
#include "utilities/spmv_1D.cuh"

namespace cugraph {
namespace mg {

template <typename VT, typename ET, typename WT>
class Pagerank {
 private:
  VT v_glob{};  // global number of vertices
  VT v_loc{};   // local number of vertices
  ET e_loc{};   // local number of edges
  WT alpha{};   // damping factor
  bool has_personalization;
  // CUDA
  const raft::comms::comms_t &comm;  // info about the mg comm setup
  cudaStream_t stream;
  int blocks;
  int threads;
  int sm_count;

  // Vertex offsets for each partition.
  VT *part_off;
  VT *local_vertices;

  // Google matrix
  ET *off;
  VT *ind;

  rmm::device_vector<WT> val;                     // values of the substochastic matrix
  rmm::device_vector<WT> bookmark;                // constant vector with dangling node info
  rmm::device_vector<WT> prev_pr;                 // record the last pagerank for convergence check
  rmm::device_vector<WT> personalization_vector;  // personalization vector after reconstruction

  bool is_setup;
  raft::handle_t const &handle;  // raft handle propagation for SpMV, etc.

 public:
  Pagerank(const raft::handle_t &handle, const GraphCSCView<VT, ET, WT> &G);
  ~Pagerank();

  void transition_vals(const VT *degree);

  void flag_leafs(const VT *degree);

  // Artificially create the google matrix by setting val and bookmark
  void setup(WT _alpha,
             VT *degree,
             VT personalization_subset_size,
             VT *personalization_subset,
             WT *personalization_values);

  // run the power iteration on the google matrix, return the number of iterations
  int solve(int max_iter, float tolerance, WT *pagerank);
};

template <typename VT, typename ET, typename WT>
int pagerank(raft::handle_t const &handle,
             const GraphCSCView<VT, ET, WT> &G,
             WT *pagerank_result,
             VT personalization_subset_size,
             VT *personalization_subset,
             WT *personalization_values,
             const double damping_factor = 0.85,
             const int64_t n_iter        = 100,
             const double tolerance      = 1e-5)
{
  // null pointers check
  CUGRAPH_EXPECTS(G.offsets != nullptr, "Invalid input argument - offsets is null");
  CUGRAPH_EXPECTS(G.indices != nullptr, "Invalid input argument - indidices is null");
  CUGRAPH_EXPECTS(pagerank_result != nullptr,
                  "Invalid input argument - pagerank output memory must be allocated");

  // parameter values
  CUGRAPH_EXPECTS(damping_factor > 0.0,
                  "Invalid input argument - invalid damping factor value (alpha<0)");
  CUGRAPH_EXPECTS(damping_factor < 1.0,
                  "Invalid input argument - invalid damping factor value (alpha>1)");
  CUGRAPH_EXPECTS(n_iter > 0, "Invalid input argument - n_iter must be > 0");

  rmm::device_vector<VT> degree(G.number_of_vertices);

  // in-degree of CSC (equivalent to out-degree of original edge list)
  G.degree(degree.data().get(), DegreeDirection::IN);

  // Allocate and intialize Pagerank class
  Pagerank<VT, ET, WT> pr_solver(handle, G);

  // Set all constants info
  pr_solver.setup(damping_factor,
                  degree.data().get(),
                  personalization_subset_size,
                  personalization_subset,
                  personalization_values);

  // Run pagerank
  return pr_solver.solve(n_iter, tolerance, pagerank_result);
}

}  // namespace mg
}  // namespace cugraph

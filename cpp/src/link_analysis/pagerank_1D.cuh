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

// snmg pagerank
// Author: Alex Fender afender@nvidia.com

#pragma once

#include <rmm/thrust_rmm_allocator.h>
#include <numeric>
#include <raft/handle.hpp>

#include "utilities/error_utils.h"

namespace cugraph {
namespace opg {

template <typename VT, typename ET, typename WT>
class Pagerank {
 private:
  size_t v_glob;               // global number of vertices
  size_t v_loc;                // local number of vertices
  size_t e_loc;                // local number of edges
  int id;                      // thread id
  int nt;                      // number of threads
  WT alpha;                    // damping factor
  const comms::comms_t &comm;  // info about the opg comm setup
  cudaStream_t stream;

  // Vertex offsets for each partition.
  // This information should be available on all threads/devices
  // part_offsets[device_id] contains the global ID
  // of the first vertex of the partion owned by device_id.
  // part_offsets[num_devices] contains the global number of vertices
  size_t *part_off;

  // local CSR matrix
  ET *off;
  VT *ind;
  rmm::device_vector<WT> val;

  // vectors of size v_glob
  rmm::device_vector<WT> bookmark;  // constant vector with dangling node info

  bool is_setup;

 public:
  Pagerank(
    const raft::handle_t &handle, size_t *part_off_, ET *off_, VT *ind_, cudaStream_t stream = 0);
  ~Pagerank();

  void transition_vals(const VT *degree);

  void flag_leafs(const VT *degree);

  // Artificially create the google matrix by setting val and bookmark
  void setup(WT _alpha, VT *degree);

  // run the power iteration on the google matrix
  void solve(int max_iter, WT *pagerank);
};

template <typename VT, typename ET, typename WT>
void pagerank(const raft::handle_t &handle,
              size_t v_loc,
              ET *csr_off,
              VT *csr_ind,
              VT *degree,
              WT *pagerank_result,
              const float damping_factor = 0.85,
              const int n_iter           = 40,
              cudaStream_t stream        = 0)
{
  // null pointers check
  CUGRAPH_EXPECTS(csr_off != nullptr, "Invalid API parameter - csr_off is null");
  CUGRAPH_EXPECTS(csr_ind != nullptr, "Invalid API parameter - csr_ind is null");
  CUGRAPH_EXPECTS(pagerank_result != nullptr,
                  "Invalid API parameter - pagerank output memory must be allocated");

  // parameter values
  CUGRAPH_EXPECTS(damping_factor > 0.0,
                  "Invalid API parameter - invalid damping factor value (alpha<0)");
  CUGRAPH_EXPECTS(damping_factor < 1.0,
                  "Invalid API parameter - invalid damping factor value (alpha>1)");
  CUGRAPH_EXPECTS(n_iter > 0, "Invalid API parameter - n_iter must be > 0");

  CUGRAPH_EXPECTS(v_loc > 0, "Invalid API parameter - v_loc must be > 0");

  // Must be shared
  std::vector<size_t> part_offset(comm.get_size() + 1);

  // MPICHECK(MPI_Allgather(&v_loc, 1, MPI_SIZE_T, &part_offset[1], 1, MPI_SIZE_T, MPI_COMM_WORLD));
  std::partial_sum(part_offset.begin(), part_offset.end(), part_offset.begin());
  if (comm.is_master())
    for (auto i = part_offset.begin(); i != part_offset.end(); ++i) std::cout << *i << ' ';
  std::cout << std::endl;
  sync_all();

  // Allocate and intialize Pagerank class
  Pagerank<VT, ET, WT> pr_solver(&comm, &part_offset[0], csr_off, csr_ind, stream);

  // Set all constants info
  pr_solver.setup(damping_factor, degree);

  // Run n_iter pagerank MG SPMVs.
  pr_solver.solve(n_iter, pagerank_result);
}

}  // namespace opg
}  // namespace cugraph

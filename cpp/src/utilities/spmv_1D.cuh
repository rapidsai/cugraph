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
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include "utilities/error.hpp"

namespace cugraph {
namespace opg {

template <typename vertex_t, typename edge_t, typename weight_t>
class MGcsrmv {
 private:
  size_t v_glob;
  size_t v_loc;
  size_t e_loc;
  raft::comms::comms_t const& comm;
  vertex_t* part_off;
  vertex_t* local_vertices;
  int i;
  int p;
  edge_t* off;
  vertex_t* ind;
  weight_t* val;
  rmm::device_vector<weight_t> y_loc;
  std::vector<size_t> v_locs_h;
  std::vector<vertex_t> displs_h;

  raft::handle_t const& handle;  // raft handle propagation for SpMV, etc.

 public:
  MGcsrmv(raft::handle_t const& handle_,
          raft::comms::comms_t const& comm,
          vertex_t* local_vertices,
          vertex_t* part_off,
          edge_t* off_,
          vertex_t* ind_,
          weight_t* val_,
          weight_t* x);

  ~MGcsrmv();

  void run(weight_t* x);
};

}  // namespace opg
}  // namespace cugraph

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

#pragma once
#include <cugraph/utilities/error.hpp>
#include <raft/handle.hpp>
#include <rmm/device_vector.hpp>

namespace cugraph {
namespace mg {

template <typename vertex_t, typename edge_t, typename weight_t>
class MGcsrmv {
 private:
  size_t v_glob_;
  size_t v_loc_;
  size_t e_loc_;

  raft::handle_t const& handle_;  // raft handle propagation for SpMV, etc.

  vertex_t* part_off_;
  vertex_t* local_vertices_;
  int i_;
  int p_;
  edge_t* off_;
  vertex_t* ind_;
  weight_t* val_;
  rmm::device_vector<weight_t> y_loc_;
  std::vector<size_t> v_locs_h_;
  std::vector<vertex_t> displs_h_;

 public:
  MGcsrmv(raft::handle_t const& r_handle,
          vertex_t* local_vertices,
          vertex_t* part_off,
          edge_t* row_off,
          vertex_t* col_ind,
          weight_t* vals,
          weight_t* x);

  ~MGcsrmv();

  void run(weight_t* x);
};

}  // namespace mg
}  // namespace cugraph

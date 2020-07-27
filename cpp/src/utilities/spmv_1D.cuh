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

template <typename VT, typename ET, typename WT>
class OPGcsrmv {
 private:
  size_t v_glob;
  size_t v_loc;
  size_t e_loc;
  raft::comms::comms_t const& comm;
  VT* part_off;
  VT* local_vertices;
  int i;
  int p;
  ET* off;
  VT* ind;
  WT* val;
  rmm::device_vector<WT> y_loc;
  std::vector<size_t> v_locs_h;
  std::vector<VT> displs_h;

  /// cudaStream_t stream;
  raft::handle_t const& handle;  // raft handle propagation for SpMV, etc.

 public:
  OPGcsrmv(raft::handle_t const& handle_,
           raft::comms::comms_t const& comm,
           VT* local_vertices,
           VT* part_off,
           ET* off_,
           VT* ind_,
           WT* val_,
           WT* x);

  ~OPGcsrmv();

  void run(WT* x);
};

}  // namespace opg
}  // namespace cugraph

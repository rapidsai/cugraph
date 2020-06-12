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

// snmg spmv
// Author: Alex Fender afender@nvidia.com

#pragma once
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include "cusparse_helper.h"
// FIX ME #include <raft/sparse/cusparse_wrappers.h>
#include "error_utils.cuh"

namespace cugraph {
namespace opg {

template <typename VT, typename ET, typename WT>
class OPGcsrmv {
 private:
  size_t v_glob;
  size_t v_loc;
  size_t e_loc;
  const comms::comms_t& comm;
  size_t* part_off;
  int i;
  int p;
  ET* off;
  VT* ind;
  WT* val;
  rmm::device_vector<WT> y_loc;
  rmm::device_vector<size_t> displs_d;
  std::vector<size_t> displs_h;

  WT* y_loc;
  cudaStream_t stream;
  // FIX ME - access csrmv through RAFT
  cugraph::detail::CusparseCsrMV<WT> spmv;

 public:
  OPGcsrmv(const raft::handle_t& handle, size_t* part_off_, ET* off_, VT* ind_, WT* val_, WT* x);

  ~OPGcsrmv();

  void run(WT* x);
};

}  // namespace opg
}  // namespace cugraph

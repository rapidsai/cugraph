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

#include "spmv_1D.cuh"

namespace cugraph {
namespace opg {
template <typename VT, typename ET, typename WT>
OPGcsrmv<VT, ET, WT>::OPGcsrmv(
  const raft::handle_t &handle, size_t *part_off_, ET *off_, VT *ind_, WT *val_, WT *x)
  : comm(handle.get_comms()), part_off(part_off_), off(off_), ind(ind_), val(val_)
{
  sync_all();
  stream = nullptr;
  i      = comm.get_rank();
  p      = comm.get_size();
  v_glob = part_off[p];
  v_loc  = part_off[i + 1] - part_off[i];
  VT tmp;
  cudaMemcpy(&tmp, &off[v_loc], sizeof(VT), cudaMemcpyDeviceToHost);
  CUDA_CHECK_LAST();
  e_loc = tmp;
  y_loc.resize(v_loc);
  WT h_one  = 1.0;
  WT h_zero = 0.0;

  // displs_d.resize(p);
  // displs_h.resize(p);
  // comm.allgather(v_loc, displs_d, 1, stream);
  // memcpy displs_h displs_d

  spmv.setup(v_loc, v_glob, e_loc, &h_one, val, off, ind, x, &h_zero, y_loc);
}

template <typename VT, typename ET, typename WT>
OPGcsrmv<VT, ET, WT>::~OPGcsrmv()
{
}

template <typename VT, typename ET, typename WT>
void OPGcsrmv<VT, ET, WT>::run(WT *x)
{
  sync_all();
  WT h_one  = 1.0;
  WT h_zero = 0.0;
  spmv.run(v_loc, v_glob, e_loc, &h_one, val, off, ind, x, &h_zero, y_loc);
  comm.allgatherv(y_loc, x, v_loc, displs_h, stream);
}

template class OPGcsrmv<int, double>;
template class OPGcsrmv<int, float>;

template <typename VT, typename ET, typename WT>
void snmg_csrmv_impl(size_t *part_offsets, ET *off, VT *ind, WT *val, WT *x)
{
  CUGRAPH_EXPECTS(part_offsets != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(off != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(ind != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(val != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(x != nullptr, "Invalid API parameter");

  cugraph::detail::Cusparse::get_handle();

  OPGcsrmv<VT, ET, WT> spmv_solver(snmg_env, part_offsets, off, ind, val, x);
  spmv_solver.run(x);
  cugraph::detail::Cusparse::destroy_handle();
}

}  // namespace opg
}  // namespace cugraph
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

// snmg spmv
// Author: Alex Fender afender@nvidia.com

#include "spmv_1D.cuh"

namespace cugraph {
namespace opg {
template <typename VT, typename ET, typename WT>
OPGcsrmv<VT, ET, WT>::OPGcsrmv(const raft::comms::comms_t &comm_,
                               VT *local_vertices_,
                               VT *part_off_,
                               ET *off_,
                               VT *ind_,
                               WT *val_,
                               WT *x)
  : comm(comm_),
    local_vertices(local_vertices_),
    part_off(part_off_),
    off(off_),
    ind(ind_),
    val(val_)
{
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

  spmv.setup(v_loc, v_glob, e_loc, &h_one, val, off, ind, x, &h_zero, y_loc.data().get());
}

template <typename VT, typename ET, typename WT>
OPGcsrmv<VT, ET, WT>::~OPGcsrmv()
{
}

template <typename VT, typename ET, typename WT>
void OPGcsrmv<VT, ET, WT>::run(WT *x)
{
  WT h_one  = 1.0;
  WT h_zero = 0.0;
  spmv.run(v_loc, v_glob, e_loc, &h_one, val, off, ind, x, &h_zero, y_loc.data().get());
  // FIXME https://github.com/rapidsai/raft/issues/21
  size_t recvbuf[comm.get_size()];
  for (int i = 0; i < comm.get_size(); i++) recvbuf[i] = local_vertices[i];
  comm.allgatherv(y_loc.data().get(), x, recvbuf, part_off, stream);
}

template class OPGcsrmv<int, int, double>;
template class OPGcsrmv<int, int, float>;

}  // namespace opg
}  // namespace cugraph

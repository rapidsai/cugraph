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
#include <raft/spectral/matrix_wrappers.hpp>
#include "spmv_1D.cuh"

namespace cugraph {
namespace opg {
template <typename vertex_t, typename edge_t, typename weight_t>
MGcsrmv<vertex_t, edge_t, weight_t>::MGcsrmv(raft::handle_t const &handle_,
                                             const raft::comms::comms_t &comm_,
                                             vertex_t *local_vertices_,
                                             vertex_t *part_off_,
                                             edge_t *off_,
                                             vertex_t *ind_,
                                             weight_t *val_,
                                             weight_t *x)
  : comm(comm_),
    local_vertices(local_vertices_),
    part_off(part_off_),
    off(off_),
    ind(ind_),
    val(val_),
    handle(handle_)
{
  i      = comm.get_rank();
  p      = comm.get_size();
  v_glob = part_off[p - 1] + local_vertices[p - 1];
  v_loc  = local_vertices[i];
  vertex_t tmp;
  CUDA_TRY(cudaMemcpy(&tmp, &off[v_loc], sizeof(vertex_t), cudaMemcpyDeviceToHost));
  e_loc = tmp;
  y_loc.resize(v_loc);
}

template <typename vertex_t, typename edge_t, typename weight_t>
MGcsrmv<vertex_t, edge_t, weight_t>::~MGcsrmv()
{
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MGcsrmv<vertex_t, edge_t, weight_t>::run(weight_t *x)
{
  using namespace raft::matrix;

  weight_t h_one  = 1.0;
  weight_t h_zero = 0.0;

  sparse_matrix_t<vertex_t, weight_t> mat{handle,
                                          off,                            // CSR row_offsets
                                          ind,                            // CSR col_indices
                                          val,                            // CSR values
                                          static_cast<vertex_t>(v_loc),   // n_rows
                                          static_cast<vertex_t>(v_glob),  // n_cols
                                          static_cast<vertex_t>(e_loc)};  // nnz

  mat.mv(h_one,                             // alpha
         x,                                 // x
         h_zero,                            // beta
         y_loc.data().get(),                // y
         sparse_mv_alg_t::SPARSE_MV_ALG2);  // SpMV algorithm

  auto stream = handle.get_stream();

  std::vector<size_t> recvbuf(comm.get_size());
  std::copy(local_vertices, local_vertices + comm.get_size(), recvbuf.begin());
  comm.allgatherv(y_loc.data().get(), x, recvbuf.data(), part_off, stream);
}

template class MGcsrmv<int, int, double>;
template class MGcsrmv<int, int, float>;

}  // namespace opg
}  // namespace cugraph

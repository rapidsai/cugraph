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
#include <raft/spectral/matrix_wrappers.hpp>
#include "spmv_1D.cuh"

namespace cugraph {
namespace mg {
template <typename vertex_t, typename edge_t, typename weight_t>
MGcsrmv<vertex_t, edge_t, weight_t>::MGcsrmv(raft::handle_t const& handle,
                                             vertex_t* local_vertices,
                                             vertex_t* part_off,
                                             edge_t* off,
                                             vertex_t* ind,
                                             weight_t* val,
                                             weight_t* x)
  : handle_(handle),
    local_vertices_(local_vertices),
    part_off_(part_off),
    off_(off),
    ind_(ind),
    val_(val)
{
  i_      = handle_.get_comms().get_rank();
  p_      = handle_.get_comms().get_size();
  v_glob_ = part_off_[p_ - 1] + local_vertices_[p_ - 1];
  v_loc_  = local_vertices_[i_];
  vertex_t tmp;
  CUDA_TRY(cudaMemcpy(&tmp, &off_[v_loc_], sizeof(vertex_t), cudaMemcpyDeviceToHost));
  e_loc_ = tmp;
  y_loc_.resize(v_loc_);
}

template <typename vertex_t, typename edge_t, typename weight_t>
MGcsrmv<vertex_t, edge_t, weight_t>::~MGcsrmv()
{
}

template <typename vertex_t, typename edge_t, typename weight_t>
void MGcsrmv<vertex_t, edge_t, weight_t>::run(weight_t* x)
{
  using namespace raft::matrix;

  weight_t h_one  = 1.0;
  weight_t h_zero = 0.0;

  sparse_matrix_t<vertex_t, weight_t> mat{handle_,                         // raft handle
                                          off_,                            // CSR row_offsets
                                          ind_,                            // CSR col_indices
                                          val_,                            // CSR values
                                          static_cast<vertex_t>(v_loc_),   // n_rows
                                          static_cast<vertex_t>(v_glob_),  // n_cols
                                          static_cast<vertex_t>(e_loc_)};  // nnz

  mat.mv(h_one,                             // alpha
         x,                                 // x
         h_zero,                            // beta
         y_loc_.data().get(),               // y
         sparse_mv_alg_t::SPARSE_MV_ALG2);  // SpMV algorithm

  auto stream = handle_.get_stream();

  auto const& comm{handle_.get_comms()};  // local

  std::vector<size_t> recvbuf(comm.get_size());
  std::vector<size_t> displs(comm.get_size());
  std::copy(local_vertices_, local_vertices_ + comm.get_size(), recvbuf.begin());
  std::copy(part_off_, part_off_ + comm.get_size(), displs.begin());
  comm.allgatherv(y_loc_.data().get(), x, recvbuf.data(), displs.data(), stream);
}

template class MGcsrmv<int32_t, int32_t, double>;
template class MGcsrmv<int32_t, int32_t, float>;

}  // namespace mg
}  // namespace cugraph

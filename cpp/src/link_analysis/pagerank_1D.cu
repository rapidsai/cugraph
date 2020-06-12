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

// opg 1D pagerank
// Author: Alex Fender afender@nvidia.com

#include <graph.hpp>
#include "pagerank_1D.cuh"
#include "utilities/graph_utils.cuh"

namespace cugraph {
namespace opg {

#define CUDA_MAX_KERNEL_THREADS 256
#define CUDA_MAX_BLOCKS 65535

template <typename VT, typename ET, typename WT>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  transition_kernel(const size_t e, const VT *ind, const VT *degree, WT *val)
{
  for (auto i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
    val[i] = 1.0 / degree[ind[i]];
}

template <typename VT, typename ET, typename WT>
Pagerank<VT, ET, WT>::Pagerank(const raft::handle_t &handle_,
                               experimental::GraphCSCView<VT, ET, WT> const &G)
  : comm(handle_.get_comms())
{
  v_glob = G.number_of_vertices;

  // FIXME needs PR #944
  // v_loc  = G.number_of_local_vertices;
  // e_loc  = G.number_of_local_edges;
  // part_off = G.local_offset;

  off      = G.offsets;
  ind      = G.indices;
  sm_count = handle_.get_device_properties().multiProcessorCount;

  is_setup = false;
  bookmark.resize(v_glob);
  val.resize(e_loc);

  // intialize cusparse. This can take some time.
  cugraph::detail::Cusparse::get_handle();
}

template <typename VT, typename ET, typename WT>
Pagerank<VT, ET, WT>::~Pagerank()
{
  // cugraph::detail::Cusparse::destroy_handle();
}

template <typename VT, typename ET, typename WT>
void Pagerank<VT, ET, WT>::transition_vals(const VT *degree)
{
  int threads = min(static_cast<VT>(e_loc), 256);
  int blocks  = min(static_cast<VT>(32 * sm_count), CUDA_MAX_BLOCKS);
  transition_kernel<VT, ET, WT><<<blocks, threads>>>(e_loc, ind, degree, val.data().get());
  CUDA_CHECK_LAST();
}

template <typename VT, typename ET, typename WT>
void Pagerank<VT, ET, WT>::flag_leafs(const VT *degree)
{
  int threads = min(static_cast<VT>(v_glob), 256);
  int blocks  = min(static_cast<VT>(32 * sm_count), CUDA_MAX_BLOCKS);
  cugraph::detail::flag_leafs_kernel<VT, WT>
    <<<blocks, threads>>>(v_glob, degree, bookmark.data().get());
  CUDA_CHECK_LAST();
}

// Artificially create the google matrix by setting val and bookmark
template <typename VT, typename ET, typename WT>
void Pagerank<VT, ET, WT>::setup(WT _alpha, VT *degree)
{
  if (!is_setup) {
    alpha   = _alpha;
    WT zero = 0.0;

    // Update dangling node vector
    cugraph::detail::fill(v_glob, bookmark.data().get(), zero);
    flag_leafs(degree);
    cugraph::detail::update_dangling_nodes(v_glob, bookmark.data().get(), alpha);

    // Transition matrix
    transition_vals(degree);

    is_setup = true;
  } else
    CUGRAPH_FAIL("OPG PageRank : Setup can be called only once");
}

// run the power iteration on the google matrix
template <typename VT, typename ET, typename WT>
void Pagerank<VT, ET, WT>::solve(int max_iter, WT *pagerank)
{
  if (is_setup) {
    WT dot_res;
    WT one = 1.0;
    WT *pr = pagerank;
    cugraph::detail::fill(v_glob, pagerank, one / v_glob);
    // This cuda sync was added to fix #426
    // This should not be requiered in theory
    // This is not needed on one GPU at this time
    cudaDeviceSynchronize();
    dot_res = cugraph::detail::dot(v_glob, bookmark.data().get(), pr);
    OPGcsrmv<VT, ET, WT> spmv_solver(comm, part_off, off, ind, val.data().get(), pagerank);
    for (auto i = 0; i < max_iter; ++i) {
      spmv_solver.run(pagerank);
      cugraph::detail::scal(v_glob, alpha, pr);
      cugraph::detail::addv(v_glob, dot_res * (one / v_glob), pr);
      dot_res = cugraph::detail::dot(v_glob, bookmark.data().get(), pr);
      cugraph::detail::scal(v_glob, one / cugraph::detail::nrm2(v_glob, pr), pr);
    }
    cugraph::detail::scal(v_glob, one / cugraph::detail::nrm1(v_glob, pr), pr);
  } else {
    CUGRAPH_FAIL("OPG PageRank : Solve was called before setup");
  }
}

template class Pagerank<int, int, double>;
template class Pagerank<int, int, float>;

}  // namespace opg
}  // namespace cugraph

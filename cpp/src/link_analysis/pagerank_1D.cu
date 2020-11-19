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

// Author: Alex Fender afender@nvidia.com

#include <algorithm>
#include <graph.hpp>
#include "pagerank_1D.cuh"
#include "utilities/graph_utils.cuh"

namespace cugraph {
namespace mg {

template <typename VT, typename WT>
__global__ void transition_kernel(const size_t e, const VT *ind, const VT *degree, WT *val)
{
  for (auto i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
    val[i] = 1.0 / degree[ind[i]];  // Degree contains IN degree. So all degree[ind[i]] were
                                    // incremented by definition (no div by 0).
}

template <typename VT, typename ET, typename WT>
Pagerank<VT, ET, WT>::Pagerank(const raft::handle_t &handle_, GraphCSCView<VT, ET, WT> const &G)
  : comm(handle_.get_comms()),
    bookmark(G.number_of_vertices),
    prev_pr(G.number_of_vertices),
    val(G.local_edges[comm.get_rank()]),
    handle(handle_),
    has_personalization(false)
{
  v_glob         = G.number_of_vertices;
  v_loc          = G.local_vertices[comm.get_rank()];
  e_loc          = G.local_edges[comm.get_rank()];
  part_off       = G.local_offsets;
  local_vertices = G.local_vertices;
  off            = G.offsets;
  ind            = G.indices;
  blocks         = handle_.get_device_properties().maxGridSize[0];
  threads        = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count       = handle_.get_device_properties().multiProcessorCount;

  is_setup = false;
}

template <typename VT, typename ET, typename WT>
Pagerank<VT, ET, WT>::~Pagerank()
{
}

template <typename VT, typename ET, typename WT>
void Pagerank<VT, ET, WT>::transition_vals(const VT *degree)
{
  if (e_loc > 0) {
    int threads = std::min(e_loc, this->threads);
    int blocks  = std::min(32 * sm_count, this->blocks);
    transition_kernel<VT, WT><<<blocks, threads>>>(e_loc, ind, degree, val.data().get());
    CHECK_CUDA(nullptr);
  }
}

template <typename VT, typename ET, typename WT>
void Pagerank<VT, ET, WT>::flag_leafs(const VT *degree)
{
  if (v_glob > 0) {
    int threads = std::min(v_glob, this->threads);
    int blocks  = std::min(32 * sm_count, this->blocks);
    cugraph::detail::flag_leafs_kernel<VT, WT>
      <<<blocks, threads>>>(v_glob, degree, bookmark.data().get());
    CHECK_CUDA(nullptr);
  }
}

// Artificially create the google matrix by setting val and bookmark
template <typename VT, typename ET, typename WT>
void Pagerank<VT, ET, WT>::setup(WT _alpha,
                                 VT *degree,
                                 VT personalization_subset_size,
                                 VT *personalization_subset,
                                 WT *personalization_values)
{
  if (!is_setup) {
    alpha   = _alpha;
    WT zero = 0.0;
    WT one  = 1.0;
    // Update dangling node vector
    cugraph::detail::fill(v_glob, bookmark.data().get(), zero);
    flag_leafs(degree);
    cugraph::detail::update_dangling_nodes(v_glob, bookmark.data().get(), alpha);

    // Transition matrix
    transition_vals(degree);

    // personalize
    if (personalization_subset_size != 0) {
      CUGRAPH_EXPECTS(personalization_subset != nullptr,
                      "Invalid API parameter: personalization_subset array should be of size "
                      "personalization_subset_size");
      CUGRAPH_EXPECTS(personalization_values != nullptr,
                      "Invalid API parameter: personalization_values array should be of size "
                      "personalization_subset_size");
      CUGRAPH_EXPECTS(personalization_subset_size <= v_glob,
                      "Personalization size should be smaller than V");

      WT sum = cugraph::detail::nrm1(personalization_subset_size, personalization_values);
      if (sum != zero) {
        has_personalization = true;
        personalization_vector.resize(v_glob);
        cugraph::detail::fill(v_glob, personalization_vector.data().get(), zero);
        cugraph::detail::scal(v_glob, one / sum, personalization_values);
        cugraph::detail::scatter(personalization_subset_size,
                                 personalization_values,
                                 personalization_vector.data().get(),
                                 personalization_subset);
      }
    }
    is_setup = true;
  } else
    CUGRAPH_FAIL("MG PageRank : Setup can be called only once");
}

// run the power iteration on the google matrix
template <typename VT, typename ET, typename WT>
int Pagerank<VT, ET, WT>::solve(int max_iter, float tolerance, WT *pagerank)
{
  if (is_setup) {
    WT dot_res;
    WT one = 1.0;
    WT *pr = pagerank;
    cugraph::detail::fill(v_glob, pagerank, one / v_glob);
    cugraph::detail::fill(v_glob, prev_pr.data().get(), one / v_glob);
    // This cuda sync was added to fix #426
    // This should not be requiered in theory
    // This is not needed on one GPU at this time
    cudaDeviceSynchronize();
    dot_res = cugraph::detail::dot(v_glob, bookmark.data().get(), pr);
    MGcsrmv<VT, ET, WT> spmv_solver(
      handle, local_vertices, part_off, off, ind, val.data().get(), pagerank);

    WT residual;
    int i;
    for (i = 0; i < max_iter; ++i) {
      spmv_solver.run(pagerank);
      cugraph::detail::scal(v_glob, alpha, pr);

      // personalization
      if (has_personalization)
        cugraph::detail::axpy(v_glob, dot_res, personalization_vector.data().get(), pr);
      else
        cugraph::detail::addv(v_glob, dot_res * (one / v_glob), pr);

      dot_res = cugraph::detail::dot(v_glob, bookmark.data().get(), pr);
      cugraph::detail::scal(v_glob, one / cugraph::detail::nrm2(v_glob, pr), pr);

      // convergence check
      cugraph::detail::axpy(v_glob, (WT)-1.0, pr, prev_pr.data().get());
      residual = cugraph::detail::nrm2(v_glob, prev_pr.data().get());
      if (residual < tolerance)
        break;
      else
        cugraph::detail::copy(v_glob, pr, prev_pr.data().get());
    }
    cugraph::detail::scal(v_glob, one / cugraph::detail::nrm1(v_glob, pr), pr);
    return i;
  } else {
    CUGRAPH_FAIL("MG PageRank : Solve was called before setup");
  }
}

template class Pagerank<int, int, double>;
template class Pagerank<int, int, float>;

}  // namespace mg
}  // namespace cugraph

#include "utilities/eidir_graph_utils.hpp"

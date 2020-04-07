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

#include <cugraph.h>                                                            
#include <graph.hpp>                                                            
#include <rmm/device_buffer.hpp>                                                
#include <rmm/thrust_rmm_allocator.h>                                           
#include <rmm_utils.h>                                                          
#include <stdio.h>                                                              
                                                                                
#include "utilities/error_utils.h"                                              
#include "utilities/graph_utils.cuh"     

#include "bh_kernels.h"
#include "utils.h"

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void barnes_hut(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const edge_t e, const vertex_t n, float *x_pos,
        float *y_pos, const float theta,
        const float scaling_ratio) {

  const int blocks = getMultiProcessorCount();
  const int epssq = 0.0025;
  int nnodes = n * 2;
  if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
  while ((nnodes & (32 - 1)) != 0) nnodes++;
  nnodes--;
  printf("N_nodes = %d blocks = %d\n", nnodes, blocks);

  // Allocate more space
  //---------------------------------------------------
  rmm::device_vector<unsigned>d_limiter(1);
  rmm::device_vector<int>d_maxdepthd(1);
  rmm::device_vector<int>d_bottomd(1);
  rmm::device_vector<float>d_radiusd(1);

  unsigned *limiter = d_limiter.data().get();
  int *maxdepthd = d_maxdepthd.data().get();
  int *bottomd = d_bottomd.data().get();
  float *radiusd = d_radiusd.data().get();

 InitializationKernel<<<1, 1, 0>>>(/*errl,*/ limiter, maxdepthd,
                                                  radiusd);
  CUDA_CHECK_LAST();

  const int FOUR_NNODES = 4 * nnodes;
  const int FOUR_N = 4 * n;
  const float theta_squared = theta * theta;
  const int NNODES = nnodes;

  rmm::device_vector<int>d_startl(nnodes + 1, 0);
  rmm::device_vector<int>d_childl((nnodes + 1) * 4, 0);
  rmm::device_vector<float>d_massl(nnodes + 1, 0);

  rmm::device_vector<float>d_maxxl(blocks * FACTOR1, 0);
  rmm::device_vector<float>d_maxyl(blocks * FACTOR1, 0);
  rmm::device_vector<float>d_minxl(blocks * FACTOR1, 0);
  rmm::device_vector<float>d_minyl(blocks * FACTOR1, 0);


  // Actual mallocs
  int *startl = d_startl.data().get();
  int *childl = d_childl.data().get();
  float *massl = d_massl.data().get();
  fill(nnodes + 1, massl, 1.0f);

  float *maxxl = d_maxxl.data().get();
  float *maxyl = d_maxyl.data().get();
  float *minxl = d_minxl.data().get();
  float *minyl = d_minyl.data().get();

  // SummarizationKernel
  rmm::device_vector<int>d_countl(nnodes + 1, 0);
  int *countl = d_countl.data().get();

  // SortKernel
  rmm::device_vector<int>d_sortl(nnodes + 1, 0);
  int *sortl = d_sortl.data().get();

  // RepulsionKernel
  rmm::device_vector<float>d_rep_forces((nnodes + 1) * 2, 0);
  float *rep_forces = d_rep_forces.data().get();

  rmm::device_vector<float>d_Z_norm(1, 0);
  float *Z_norm = d_Z_norm.data().get();

  rmm::device_vector<float>d_radius_squared(1, 0);
  float *radiusd_squared = d_radius_squared.data().get();

  int random_state = 0;
  float *YY = random_vector((nnodes + 1) * 2, random_state);

  // Set cache levels for faster algorithm execution
  //---------------------------------------------------
  cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(RepulsionKernel, cudaFuncCachePreferL1);

    BoundingBoxKernel<<<blocks * FACTOR1, THREADS1, 0>>>(
      startl, childl, massl, YY, YY + nnodes + 1, maxxl, maxyl, minxl, minyl,
      FOUR_NNODES, NNODES, n, limiter, radiusd);
    CUDA_CHECK_LAST();

    //printv((nnodes + 1) * 2,YY, 0);
    ClearKernel1<<<blocks, 1024, 0>>>(childl, FOUR_NNODES,
                                                    FOUR_N);
    CUDA_CHECK_LAST();

    TreeBuildingKernel<<<blocks * FACTOR2, THREADS2, 0>>>(
      childl, YY, YY + nnodes + 1, NNODES, n, maxdepthd, bottomd,
      radiusd);
    CUDA_CHECK_LAST();

    ClearKernel2<<<blocks * 1, 1024, 0>>>(startl, massl, NNODES,
                                                        bottomd);
    CUDA_CHECK_LAST();

    SummarizationKernel<<<blocks * FACTOR3, THREADS3, 0>>>(
      countl, childl, massl, YY, YY + nnodes + 1, NNODES, n, bottomd);
    CUDA_CHECK_LAST();

    SortKernel<<<blocks * FACTOR4, THREADS4, 0>>>(
      sortl, countl, startl, childl, NNODES, n, bottomd);
    CUDA_CHECK_LAST();

    RepulsionKernel<<<blocks * FACTOR5, THREADS5, 0>>>(
      theta, epssq, sortl, childl, massl, YY, YY + nnodes + 1,
      rep_forces, rep_forces + nnodes + 1, Z_norm, theta_squared, NNODES,
      FOUR_NNODES, n, radiusd_squared, maxdepthd);
    CUDA_CHECK_LAST();

    ALLOC_FREE_TRY(YY, nullptr);
}

} // namespace detail
} // namespace cugraph

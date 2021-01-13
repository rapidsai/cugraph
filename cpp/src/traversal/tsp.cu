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

#include <converters/COOtoCSR.cuh>
#include "utilities/graph_utils.cuh"

#include "tsp.hpp"
#include "tsp_kernels.hpp"
#include "tsp_knn.hpp"
#include "tsp_utils.hpp"

namespace cugraph {
namespace detail {

TSP::TSP(const raft::handle_t &handle,
         int *route,
         const float *x_pos,
         const float *y_pos,
         const int nodes,
         const int restarts,
         const int k,
         const bool verbose)
  : handle_(handle), route_(route), x_pos_(x_pos), y_pos_(y_pos), nodes_(nodes),
  restarts_(restarts), k_(k), verbose_(verbose)
{
  stream_      = handle_.get_stream();
  max_blocks_  = handle_.get_device_properties().maxGridSize[0];
  max_threads_ = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count_    = handle_.get_device_properties().multiProcessorCount;
  // how large a grid we want to run, this is fixed
  restart_batch_ = 4096;
}

void TSP::allocate()
{
  // Scalars
  mylock_scalar_.set_value(1, stream_);
  n_climbs_scalar_.set_value(1, stream_);
  best_tour_scalar_.set_value(1, stream_);

  mylock_    = mylock_scalar_.data();
  n_climbs_  = n_climbs_scalar_.data();
  best_tour_ = best_tour_scalar_.data();

  // Vectors
  neighbors_vec_.resize(k_ * nodes_);
  work_vec_.resize(4 * restart_batch_ * ((3 * nodes_ + 2 + 31) / 32 * 32));

  neighbors_ = neighbors_vec_.data().get();
  work_      = work_vec_.data().get();
}

float TSP::compute()
{
  int num_graphs       = 1;
  float valid_coo_dist = 0.f;

  int num_restart_batch_es = (restarts_ + restart_batch_ - 1) / restart_batch_;
  int restart_resid        = restarts_ - (num_restart_batch_es - 1) * restart_batch_;
  if (verbose_) {
    printf(" doing %d batches of size %d, with %d tail \n",
        num_restart_batch_es - 1,
        restart_batch_,
        restart_resid);
    printf("configuration: %d nodes, %d restart\n", nodes_, restarts_);
  }
  // Tell the cache how we want it to behave
  cudaFuncSetCacheConfig(simulOpt, cudaFuncCachePreferEqual);

  int threads = best_thread_count(nodes_);
  if (verbose_)
    printf(" calculated best thread number = %d\n", threads);
  // pre-allocate workspace for climbs, each block needs a separate permutation space and search
  // buffer

  float *pos = (float *)malloc(sizeof(float) * (nodes_ + 1) * 2);
  if (pos == NULL) {
    fprintf(stderr, "cannot allocate pos\n");
    exit(-1);
  }

  int *offsets = (int *)malloc((sizeof(int) * (2)));
  offsets[0]   = 0;
  offsets[1]   = nodes_;

  for (int g = 0; g < num_graphs; g++) {
    int global_best = INT_MAX;
    float *soln     = NULL;
    int best        = 0;

    if (verbose_)
      printf("optimizing graph %d kswap = %d \n", g, kswaps);
    for (int b = 0; b < num_restart_batch_es; b++) {
      Init<<<1, 1, 0, stream_>>>(mylock_, n_climbs_, best_tour_);
      CHECK_CUDA(stream_);

      if (b == num_restart_batch_es - 1) restart_batch_ = restart_resid;

      simulOpt<<<restart_batch_, threads, sizeof(int) * threads, stream_>>>(mylock_,
                                                                            n_climbs_,
                                                                            best_tour_,
                                                                            k_,
                                                                            nodes_,
                                                                            neighbors_,
                                                                            x_pos_ + offsets[g],
                                                                            y_pos_ + offsets[g],
                                                                            work_);
      CHECK_CUDA(stream_);
      cudaDeviceSynchronize();

      CUDA_TRY(cudaMemcpy(&best, best_tour_, sizeof(int), cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
      if (verbose_)
        printf("best reported by kernel = %d\n", best);

      if (best < global_best) {
        global_best = best;
        CUDA_TRY(cudaMemcpyFromSymbol(&soln, best_soln, sizeof(void *)));
        cudaDeviceSynchronize();

        CUDA_TRY(cudaMemcpy(pos, soln, sizeof(float) * (nodes_ + 1) * 2, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
      }
    }

    if (verbose_)
      printf("Optimized tour length = %d\n", global_best);

      for (int i = 0; i < nodes_; i++) {
        if (verbose_)
          printf("%.1f %.1f\n", pos[i], pos[i + nodes_ + 1]);
        valid_coo_dist += cpudist(i, i + 1);
      }
  }
  return valid_coo_dist;
}

void TSP::knn()
{
  int numpackages  = nodes_;
  int *neighbors_h = (int *)malloc(k_ * nodes_ * sizeof(int));
  float *input_x_h = (float *)malloc(nodes_ * sizeof(float));
  float *input_y_h = (float *)malloc(nodes_ * sizeof(float));
  CUDA_TRY(cudaMemcpy(input_x_h, x_pos_, sizeof(float) * nodes_, cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpy(input_y_h, y_pos_, sizeof(float) * nodes_, cudaMemcpyDeviceToHost));

  // re-scale arbitrary inputs to fit inside (0,1024)x(0,1024) box
  float xmin = 1e6;
  float xmax = -1e6;
  float ymin = 1e6;
  float ymax = -1e6;
  for (int np = 0; np < numpackages; np++) {
    float xc = input_x_h[np];
    if (xc < xmin) xmin = xc;
    if (xc > xmax) xmax = xc;
    float yc = input_y_h[np];
    if (yc < ymin) ymin = yc;
    if (yc > ymax) ymax = yc;
  }

  // Calculate affine transform A*x + b so that all (x,y) pairs lie in (0,1024)x(0,1024)
  // also calculate inverse so we can recover the original coords
  // We need to use the same scaling for x and y so the Euclidean distance is just scaled
  // otherwise we can get bad neighbors as a result of the scaling
  float forward_b = max(-xmin, -ymin);
  float forward_A = 1024. / max((xmax + forward_b), (ymax + forward_b));
  float back_A    = 1. / forward_A;
  float back_b    = -forward_b;
  affineTrans(numpackages, 1, input_x_h, forward_A, forward_b);
  affineTrans(numpackages, 1, input_y_h, forward_A, forward_b);

  findKneighbors(numpackages, k_, &input_x_h, &input_y_h, &neighbors_h, 0);

  // Reverse the transform
  affineTrans(numpackages, 0, input_x_h, back_A, back_b);
  affineTrans(numpackages, 0, input_y_h, back_A, back_b);

  for (int np = 0; np < numpackages; np++) {
    float xc = input_x_h[np];
    if (xc < xmin) xmin = xc;
    if (xc > xmax) xmax = xc;
    float yc = input_y_h[np];
    if (yc < ymin) ymin = yc;
    if (yc > ymax) ymax = yc;
  }
  CUDA_TRY(cudaMemcpy(neighbors_, neighbors_h, sizeof(int) * k_ * nodes_, cudaMemcpyHostToDevice));
}

}  // namespace detail

float traveling_salesman(const raft::handle_t &handle,
                         int *route,
                         const float *x_pos,
                         const float *y_pos,
                         const int nodes,
                         const int restarts,
                         const int k,
                         const bool verbose)
{
  RAFT_EXPECTS(route != nullptr, "route should be of size V");
  RAFT_EXPECTS(nodes > 0, "0 vertices");
  RAFT_EXPECTS(restarts > 0, "0 restarts");
  RAFT_EXPECTS(k > 0, "0 neighbors");

  cugraph::detail::TSP tsp(handle, route, x_pos, y_pos,
                           nodes, restarts, k, verbose);
  tsp.allocate();
  tsp.knn();
  return tsp.compute();
}

}  // namespace cugraph

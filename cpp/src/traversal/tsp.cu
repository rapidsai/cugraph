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

#include "tsp.hpp"
#include "tsp_kernels.hpp"
#include "tsp_utils.hpp"

namespace cugraph {
namespace detail {
template <typename vertex_t, typename edge_t, typename weight_t>
TSP<vertex_t, edge_t, weight_t>::TSP(const raft::handle_t &handle,
    GraphCOOView<vertex_t, edge_t, weight_t> &graph,
    const float *x_pos,
    const float *y_pos,
    const int restarts)
: handle_(handle),
  x_pos_(x_pos),
  y_pos_(y_pos_),
  restarts_(restarts) {
	stream_ = handle_.get_stream();
  sort(graph, stream_);
  srcs_ = graph.src_indices;
  dsts_ = graph.dst_indices;
  weights_ = graph.edge_data;
  nodes_ = graph.number_of_vertices;
  edges_ = graph.number_of_edges;
  max_blocks_ = handle_.get_device_properties().maxGridSize[0];
  max_threads_ = handle_.get_device_properties().maxThreadsPerBlock;
  sm_count_ = handle_.get_device_properties().multiProcessorCount;
}

template <typename vertex_t, typename edge_t, typename weight_t>
float TSP<vertex_t, edge_t, weight_t>::compute() {


  int restart_batch = 4096; // how large a grid we want to run, this is fixed
  int total_climbs = 0;
  int num_graphs = 1;

  int num_restart_batches = (restarts_ + restart_batch -1) / restart_batch;
  int restart_resid = restarts_ - (num_restart_batches - 1) * restart_batch;

  //Tell the cache how we want it to behave
  cudaFuncSetCacheConfig(simulOpt, cudaFuncCachePreferEqual);

  int threads = best_thread_count(nodes_);
  printf(" calculated best thread number = %d\n", threads);
  //pre-allocate workspace for climbs, each block needs a separate permutation space and search buffer
  rmm::device_vector<int> work(4 * restart_batch * ((3 * nodes_ + 2 + 31) / 32 * 32));
  int *work_d = work.data().get();
  int *offsets = (int *)malloc((sizeof(int) * 2));
  offsets[0] = 0;
  offsets[1] = nodes_;

  float *pos = (float *)malloc(sizeof(float) * (nodes_ + 1) * 2);
  if (pos == NULL) {
    fprintf(stderr, "cannot allocate pos\n");
    exit(-1);
  }

  for (int g = 0; g < num_graphs; g++) {

    int global_best  = INT_MAX;
    int total_climbs = 0;
    int soln = NULL;
    int best;
    int num_climbs;

    printf("optimizing graph %d kswap = %d \n",g, kswaps);
    for (int b = 0; b < num_restart_batches; b++) {
      Init<<<1, 1>>>();

      if (b == num_restart_batches-1)
        restart_batch = restart_resid;

      simulOpt<<<restart_batch, threads, sizeof(int) * threads>>>(
          nodes_, neighbors_, x_pos_ + offsets[g], y_pos_ + offsets[g], work_d);
      cudaDeviceSynchronize();

      copyFromGPUSymbol(&best, best_tour, sizeof(int));
      cudaDeviceSynchronize();
      printf("best reported by kernel = %d\n", best);

      if (best < global_best) {
        global_best = best;
        copyFromGPUSymbol(&soln, best_soln, sizeof(void *));
        cudaDeviceSynchronize();

        copyFromGPU(pos, soln, sizeof(float) * (nodes_ + 1) * 2);
        cudaDeviceSynchronize();

        float valid_dist = 0.0;
        for (int i = 0; i < nodes_; i++) {
          valid_dist += cpudist(i, i + 1) ;
        }
        printf(" validating route gpudist= %d cpudist = %f\n",global_best, valid_dist);

      }
      copyFromGPUSymbol(&num_climbs, n_climbs, sizeof(int));
      total_climbs += num_climbs;
    }

    printf("Optimized tour length = %d\n", global_best);
    float valid_coo_dist = 0.0;
    for (int i = 0; i < nodes; i++) {
      printf("%.1f %.1f\n",  pos[i], pos[i+nodes+1]);
      valid_coo_dist += cpudist(i , i + 1) ;
    }
    printf(" validating route dist = %f\n", valid_coo_dist);
  }
  return valid_coo_dist;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void TSP<vertex_t, edge_t, weight_t>::knn() {
     int numpackages = nodes_;
     int *neighbors_h = (int *)malloc(bw * nodes_ * sizeof(int));
     CUDA_TRY(cudaMemcpy(input_x_h_, x_pos_, sizeof(float) * nodes_, cudaMemcpyDeviceToHost));
     CUDA_TRY(cudaMemcpy(input_y_h_, y_pos_, sizeof(float) * nodes_, cudaMemcpyDeviceToHost));

     //re-scale arbitrary inputs to fit inside (0,1024)x(0,1024) box
     float xmin = 1e6;
     float xmax = -1e6;
     float ymin = 1e6;
     float ymax = -1e6;
     for (int np = 0; np < numpackages; np++) {
         float xc = input_x_h_[np];
         if (xc < xmin) xmin = xc;
         if (xc > xmax) xmax = xc;
         float yc = input_y_h_[np];
         if (yc < ymin) ymin = yc;
         if (yc > ymax) ymax = yc;
     }

     // Calculate affine transform A*x + b so that all (x,y) pairs lie in (0,1024)x(0,1024)
     // also calculate inverse so we can recover the original coords
     // We need to use the same scaling for x and y so the Euclidean distance is just scaled
     // otherwise we can get bad neighbors as a result of the scaling
     float forward_b = max(-xmin, -ymin);
     float forward_A = 1024. / max((xmax + forward_b), (ymax + forward_b));
     float back_A = 1. / forward_A;
     float back_b = -forward_b;
     affineTrans(numpackages, 1, input_x_h_, forward_A, forward_b);
     affineTrans(numpackages, 1, input_y_h_, forward_A, forward_b);

     findKneighbors(numpackages, bw, &input_x_h_, &input_y_h_, &neighbors_h, 0);

     // Reverse the transform
     affineTrans(numpackages, 0, input.x, back_A, back_b);
     affineTrans(numpackages, 0, input.y, back_A, back_b);

     for (int np = 0; np < numpackages; np++) {
         float xc = input_x_h_[np];
         if (xc < xmin) xmin = xc;
         if (xc > xmax) xmax = xc;
         float yc = input_y_h_[np];
         if (yc < ymin) ymin = yc;
         if (yc > ymax) ymax = yc;
     }
     CUDA_TRY(cudaMemcpy(neighbors_, neighbors_h_, sizeof(int) * bw * nodes_, cudaMemcpyHostToDevice));
}

} // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t>
float traveling_salesman(const raft::handle_t &handle,
          GraphCOOView<vertex_t, edge_t, weight_t> &graph,
          const float *x_pos,
          const float *y_pos,
          const int restarts)
{
  RAFT_EXPECTS(graph.number_of_vertices > 0, "0 vertices");
  RAFT_EXPECTS(graph.number_of_edges > 0, "0 edges");
  RAFT_EXPECTS(graph.src_indices != nullptr, "Null sources");
  RAFT_EXPECTS(graph.dst_indices != nullptr, "Null destinations");
  RAFT_EXPECTS(graph.edge_data != nullptr, "Null weights");
  RAFT_EXPECTS(restarts > 0, "0 restarts");

  cugraph::detail::TSP<vertex_t, edge_t, weight_t> tsp(
      handle,
      graph,
      x_pos,
      y_pos,
      restarts);
  return tsp.compute();
}

template float traveling_salesman<int, int, float>(
    const raft::handle_t &handle,
    GraphCOOView<int, int, float> &graph,
    const float *x_pos,
    const float *y_pos,
    const int restarts);

template float traveling_salesman<int, int, double>(
    const raft::handle_t &handle,
    GraphCOOView<int, int, double> &graph,
    const float *x_pos,
    const float *y_pos,
    const int restarts);
} // namespace cugraph


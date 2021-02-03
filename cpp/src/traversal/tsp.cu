/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/spatial/knn/knn.hpp>

#include "tsp.hpp"
#include "tsp_solver.hpp"
#include "tsp_utils.hpp"

namespace cugraph {
namespace detail {

TSP::TSP(raft::handle_t &handle,
         const int *vtx_ptr,
         int *route,
         const float *x_pos,
         const float *y_pos,
         const int nodes,
         const int restarts,
         const bool beam_search,
         const int k,
         const int nstart,
         const bool verbose)
  : handle_(handle),
    vtx_ptr_(vtx_ptr),
    route_(route),
    x_pos_(x_pos),
    y_pos_(y_pos),
    nodes_(nodes),
    restarts_(restarts),
    beam_search_(beam_search),
    k_(k),
    nstart_(nstart),
    verbose_(verbose)
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
  neighbors_vec_.resize((k_ + 1) * nodes_);
  work_vec_.resize(4 * restart_batch_ * ((3 * nodes_ + 2 + 31) / 32 * 32));
  work_route_vec_.resize(4 * restart_batch_ * ((3 * nodes_ + 2 + 31) / 32 * 32));

  // Pointers
  neighbors_  = neighbors_vec_.data().get();
  work_       = work_vec_.data().get();
  work_route_ = work_route_vec_.data().get();
}

float TSP::compute()
{
  float valid_coo_dist    = 0.f;
  int num_restart_batches = (restarts_ + restart_batch_ - 1) / restart_batch_;
  int restart_resid       = restarts_ - (num_restart_batches - 1) * restart_batch_;
  int global_best         = INT_MAX;
  float *soln             = nullptr;
  int *route_sol          = nullptr;
  int best                = 0;
  std::vector<float> pos;
  pos.reserve((nodes_ + 1) * 2);

  // Allocate GPU buffers
  allocate();

  // KNN call
  knn();

  if (verbose_) {
    printf("Doing %d batches of size %d, with %d tail \n",
           num_restart_batches - 1,
           restart_batch_,
           restart_resid);
    printf("configuration: %d nodes, %d restart\n", nodes_, restarts_);
    printf("optimizing graph with kswap = %d \n", kswaps);
  }

  // Tell the cache how we want it to behave
  cudaFuncSetCacheConfig(two_opt_search, cudaFuncCachePreferEqual);

  int threads = best_thread_count(nodes_);
  if (verbose_) printf("Calculated best thread number = %d\n", threads);

  for (int b = 0; b < num_restart_batches; b++) {
    init<<<1, 1, 0, stream_>>>(mylock_, n_climbs_, best_tour_);
    CHECK_CUDA(stream_);

    if (b == num_restart_batches - 1) restart_batch_ = restart_resid;

    two_opt_search<<<restart_batch_, threads, sizeof(int) * threads, stream_>>>(mylock_,
                                                                                n_climbs_,
                                                                                best_tour_,
                                                                                vtx_ptr_,
                                                                                work_route_,
                                                                                beam_search_,
                                                                                k_,
                                                                                nodes_,
                                                                                neighbors_,
                                                                                x_pos_,
                                                                                y_pos_,
                                                                                work_,
                                                                                nstart_);

    CHECK_CUDA(stream_);
    cudaDeviceSynchronize();

    CUDA_TRY(cudaMemcpy(&best, best_tour_, sizeof(int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    if (verbose_) printf("Best reported by kernel = %d\n", best);

    if (best < global_best) {
      global_best = best;
      CUDA_TRY(cudaMemcpyFromSymbol(&soln, best_soln, sizeof(void *)));
      cudaDeviceSynchronize();
      CUDA_TRY(cudaMemcpyFromSymbol(&route_sol, best_route, sizeof(void *)));
      cudaDeviceSynchronize();

      CUDA_TRY(
        cudaMemcpy(pos.data(), soln, sizeof(float) * (nodes_ + 1) * 2, cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
    }
  }

  if (verbose_) printf("Optimized tour length = %d\n", global_best);

  for (int i = 0; i < nodes_; i++) {
    if (verbose_) { printf("%.1f %.1f\n", pos[i], pos[i + nodes_ + 1]); }
    valid_coo_dist += cpudist(i, i + 1);
  }
  raft::copy(route_, route_sol, nodes_, stream_);
  return valid_coo_dist;
}

void TSP::knn()
{
  if (verbose_) printf("Looking at %i nearest neighbors\n", k_);

  int dim              = 2;
  bool row_major_order = false;

  rmm::device_vector<float> input(nodes_ * dim);
  float *input_ptr = input.data().get();
  raft::copy(input_ptr, x_pos_, nodes_, stream_);
  raft::copy(input_ptr + nodes_, y_pos_, nodes_, stream_);

  rmm::device_vector<float> search_data(nodes_ * dim);
  float *search_data_ptr = search_data.data().get();
  raft::copy(search_data_ptr, input_ptr, nodes_ * dim, stream_);

  rmm::device_vector<float> distances(nodes_ * (k_ + 1));
  float *distances_ptr = distances.data().get();

  std::vector<float *> input_vec;
  std::vector<int> sizes_vec;
  input_vec.push_back(input_ptr);
  sizes_vec.push_back(nodes_);

  // k neighbors + 1 is needed because the nearest neighbor of each point is
  // the point itself that we don't want to take into account.

  raft::spatial::knn::brute_force_knn(handle_,
                                      input_vec,
                                      sizes_vec,
                                      dim,
                                      search_data_ptr,
                                      nodes_,
                                      neighbors_,
                                      distances_ptr,
                                      k_ + 1,
                                      row_major_order,
                                      row_major_order);
}
}  // namespace detail

float traveling_salesman(raft::handle_t &handle,
                         const int *vtx_ptr,
                         int *route,
                         const float *x_pos,
                         const float *y_pos,
                         const int nodes,
                         const int restarts,
                         const bool beam_search,
                         const int k,
                         const int nstart,
                         const bool verbose)
{
  RAFT_EXPECTS(route != nullptr, "route should equal the number of nodes");
  RAFT_EXPECTS(nodes > 0, "nodes should be strictly positive");
  RAFT_EXPECTS(restarts > 0, "restarts should be strictly positive");
  RAFT_EXPECTS(k > 0, "k should be strictly positive");

  cugraph::detail::TSP tsp(
    handle, vtx_ptr, route, x_pos, y_pos, nodes, restarts, beam_search, k, nstart, verbose);
  return tsp.compute();
}

}  // namespace cugraph

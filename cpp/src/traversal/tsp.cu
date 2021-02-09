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

#include <numeric>
#include <raft/spatial/knn/knn.hpp>

#include "tsp.hpp"
#include "tsp_solver.hpp"

namespace cugraph {
namespace detail {

TSP::TSP(raft::handle_t &handle,
         int const *vtx_ptr,
         float const *x_pos,
         float const *y_pos,
         int nodes,
         int restarts,
         bool beam_search,
         int k,
         int nstart,
         bool verbose,
         int *route)
  : handle_(handle),
    vtx_ptr_(vtx_ptr),
    x_pos_(x_pos),
    y_pos_(y_pos),
    nodes_(nodes),
    restarts_(restarts),
    beam_search_(beam_search),
    k_(k),
    nstart_(nstart),
    verbose_(verbose),
    route_(route),
    stream_(handle_.get_stream()),
    max_blocks_(handle_.get_device_properties().maxGridSize[0]),
    max_threads_(handle_.get_device_properties().maxThreadsPerBlock),
    sm_count_(handle_.get_device_properties().multiProcessorCount),
    restart_batch_(4096)
{
  allocate();
}

void TSP::allocate()
{
  // Scalars
  mylock_scalar_.set_value(1, stream_);
  best_tour_scalar_.set_value(1, stream_);

  mylock_    = mylock_scalar_.data();
  best_tour_ = best_tour_scalar_.data();

  // Vectors
  neighbors_vec_.resize((k_ + 1) * nodes_);
  // We allocate a work buffer that will store the computed distances, px, py
  // and the route. We align it on the warp size since each block will work on
  // his own scope of the work array (see solver).
  work_vec_.resize(sizeof(int) * restart_batch_ * ((4 * nodes_ + 3 + 31) / 32 * 32));

  // Pointers
  neighbors_  = neighbors_vec_.data().get();
  work_       = work_vec_.data().get();
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
  clock_t start, end;
  std::vector<long> h_times;
  h_times.reserve(4);
  rmm::device_vector<long> times(4);
  std::vector<float> h_x_pos;
  std::vector<float> h_y_pos;
  h_x_pos.reserve(nodes_ + 1);
  h_y_pos.reserve(nodes_ + 1);

  // KNN call
  start = clock();
  knn();
  end = clock();
  h_times[0] = end - start;

  if (verbose_) {
    std::cout << "Doing " << num_restart_batches - 1 << " batches of size " << restart_batch_
              << ", with " << restart_resid << " tail\n";
    std::cout << "configuration: " << nodes_ << " nodes, " << restarts_ << " restart\n";
    std::cout << "optimizing graph with kswap = " << kswaps << "\n";
  }

  // Tell the cache how we want it to behave
  cudaFuncSetCacheConfig(search_solution, cudaFuncCachePreferEqual);

  int threads = best_thread_count(nodes_, max_threads_, sm_count_);
  if (verbose_) std::cout << "Calculated best thread number = " << threads << "\n";

  for (int b = 0; b < num_restart_batches; ++b) {
    init<<<1, 1, 0, stream_>>>(mylock_, best_tour_);
    CHECK_CUDA(stream_);

    if (b == num_restart_batches - 1) restart_batch_ = restart_resid;

    search_solution<<<restart_batch_, threads, sizeof(int) * threads, stream_>>>(mylock_,
                                                                                 best_tour_,
                                                                                 vtx_ptr_,
                                                                                 beam_search_,
                                                                                 k_,
                                                                                 nodes_,
                                                                                 neighbors_,
                                                                                 x_pos_,
                                                                                 y_pos_,
                                                                                 work_,
                                                                                 nstart_,
                                                                                 times.data().get());

    CHECK_CUDA(stream_);
    cudaDeviceSynchronize();

    CUDA_TRY(cudaMemcpy(&best, best_tour_, sizeof(int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    if (verbose_) std::cout << "Best reported by kernel = " << best << "\n";

    if (best < global_best) {
      global_best = best;
      CUDA_TRY(cudaMemcpyFromSymbol(&soln, best_soln, sizeof(void *)));
      cudaDeviceSynchronize();
      CUDA_TRY(cudaMemcpyFromSymbol(&route_sol, best_route, sizeof(void *)));
      cudaDeviceSynchronize();
    }
  }

  if (verbose_) std::cout << "Optimized tour length = " << global_best << "\n";

  raft::copy(route_, route_sol, nodes_, stream_);

  CUDA_TRY(cudaMemcpy(h_x_pos.data(), soln, sizeof(float) * (nodes_ + 1), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  CUDA_TRY(cudaMemcpy(
    h_y_pos.data(), soln + nodes_ + 1, sizeof(float) * (nodes_ + 1), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  CUDA_TRY(cudaMemcpy(
    h_y_pos.data(), soln + nodes_ + 1, sizeof(float) * (nodes_ + 1), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  for (int i = 0; i < nodes_; ++i) {
    if (verbose_) { std::cout << h_x_pos[i] << " " << h_y_pos[i] << "\n"; }
    valid_coo_dist += euclidean_dist(h_x_pos.data(), h_y_pos.data(), i, i + 1);
  }

  CUDA_TRY(cudaMemcpy(
    h_times.data() + 1, times.data().get(), sizeof(long) * 4, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  if (verbose_) {
    double total = 0;
    for (int i = 0; i < 4; ++i)
      total += h_times[i];

    std::cout << "Knn time: " << h_times[0] << " " << (h_times[0] / total) * 100.0 << "%\n";
    std::cout << "Init time: " << h_times[1] << " " << (h_times[1] / total) * 100.0 << "%\n";
    std::cout << "Hill Climbing time: " << h_times[2] << " " << (h_times[2] / total) * 100.0 << "%\n";
    std::cout << "Retrieve best solution time: " << h_times[3] << " " << (h_times[3] / total) * 100.0 << "%\n";
    std::cout << "Total time: " << total << "\n";
  }
  return valid_coo_dist;
}

void TSP::knn()
{
  if (verbose_) std::cout << "Looking at " << k_ << " nearest neighbors\n";

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
                         int const *vtx_ptr,
                         float const *x_pos,
                         float const *y_pos,
                         int nodes,
                         int restarts,
                         bool beam_search,
                         int k,
                         int nstart,
                         bool verbose,
                         int *route)
{
  RAFT_EXPECTS(route != nullptr, "route should equal the number of nodes");
  RAFT_EXPECTS(nodes > 0, "nodes should be strictly positive");
  RAFT_EXPECTS(restarts > 0, "restarts should be strictly positive");
  RAFT_EXPECTS(nstart >= 0 && nstart < nodes, "nstart should be between 0 and nodes - 1");
  RAFT_EXPECTS(k > 0, "k should be strictly positive");

  cugraph::detail::TSP tsp(
    handle, vtx_ptr, x_pos, y_pos, nodes, restarts, beam_search, k, nstart, verbose, route);
  return tsp.compute();
}

}  // namespace cugraph

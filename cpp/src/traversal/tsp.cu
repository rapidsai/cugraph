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

#include <utilities/high_res_timer.hpp>

#include <rmm/cuda_stream_view.hpp>

#include "tsp.hpp"
#include "tsp_solver.hpp"

namespace cugraph {
namespace detail {

TSP::TSP(raft::handle_t const &handle,
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
    warp_size_(handle_.get_device_properties().warpSize),
    sm_count_(handle_.get_device_properties().multiProcessorCount),
    restart_batch_(8192),
    mylock_scalar_(stream_),
    best_cost_scalar_(stream_),
    neighbors_vec_((k_ + 1) * nodes_, stream_),
    work_vec_(restart_batch_ * ((4 * nodes_ + 3 + warp_size_ - 1) / warp_size_ * warp_size_),
              stream_),
    best_x_pos_vec_(1, stream_),
    best_y_pos_vec_(1, stream_),
    best_route_vec_(1, stream_)
{
  setup();
}

void TSP::setup()
{
  mylock_ = mylock_scalar_.data();

  neighbors_ = neighbors_vec_.data();
  // pre-allocate workspace for climbs, each block needs a separate permutation space and search
  // buffer. We allocate a work buffer that will store the computed distances, px, py and the route.
  // We align it on the warp size.
  work_ = work_vec_.data();

  results_.best_x_pos = best_x_pos_vec_.data();
  results_.best_y_pos = best_y_pos_vec_.data();
  results_.best_route = best_route_vec_.data();
  results_.best_cost  = best_cost_scalar_.data();
}

void TSP::reset_batch()
{
  mylock_scalar_.set_value_to_zero_async(stream_);
  auto const max{std::numeric_limits<int>::max()};
  best_cost_scalar_.set_value_async(max, stream_);
}

void TSP::get_initial_solution(int const batch)
{
  if (!beam_search_) {
    random_init<<<restart_batch_, best_thread_num_>>>(
      work_, x_pos_, y_pos_, vtx_ptr_, nstart_, nodes_, batch, restart_batch_);
    CHECK_CUDA(stream_);
  } else {
    knn_init<<<restart_batch_, best_thread_num_>>>(
      work_, x_pos_, y_pos_, vtx_ptr_, neighbors_, nstart_, nodes_, k_, batch, restart_batch_);
    CHECK_CUDA(stream_);
  }
}

float TSP::compute()
{
  float final_cost        = 0.f;
  int num_restart_batches = (restarts_ + restart_batch_ - 1) / restart_batch_;
  int restart_resid       = restarts_ - (num_restart_batches - 1) * restart_batch_;
  int global_best         = std::numeric_limits<int>::max();
  int best                = 0;

  std::vector<float> h_x_pos;
  std::vector<float> h_y_pos;
  std::vector<int> h_route;
  h_x_pos.reserve(nodes_ + 1);
  h_y_pos.reserve(nodes_ + 1);
  h_route.reserve(nodes_);
  std::vector<float *> addr_best_x_pos(1);
  std::vector<float *> addr_best_y_pos(1);
  std::vector<int *> addr_best_route(1);
  HighResTimer hr_timer;
  auto create_timer = [&hr_timer, this](char const *name) {
    return VerboseTimer(name, hr_timer, verbose_);
  };

  if (verbose_) {
    std::cout << "Doing " << num_restart_batches << " batches of size " << restart_batch_
              << ", with " << restart_resid << " tail\n";
    std::cout << "configuration: " << nodes_ << " nodes, " << restarts_ << " restart\n";
    std::cout << "optimizing graph with kswap = " << kswaps << "\n";
  }

  // Tell the cache how we want it to behave
  cudaFuncSetCacheConfig(search_solution, cudaFuncCachePreferEqual);
  best_thread_num_ = best_thread_count(nodes_, max_threads_, sm_count_, warp_size_);

  if (verbose_) std::cout << "Calculated best thread number = " << best_thread_num_ << "\n";

  if (beam_search_) {
    auto timer = create_timer("knn");
    knn();
  }

  for (auto batch = 0; batch < num_restart_batches; ++batch) {
    reset_batch();
    if (batch == num_restart_batches - 1) restart_batch_ = restart_resid;

    {
      auto timer = create_timer("initial_sol");
      get_initial_solution(batch);
    }

    {
      auto timer = create_timer("search_sol");
      search_solution<<<restart_batch_,
                        best_thread_num_,
                        sizeof(int) * best_thread_num_,
                        stream_>>>(
        results_, mylock_, vtx_ptr_, beam_search_, k_, nodes_, x_pos_, y_pos_, work_, nstart_);
      CHECK_CUDA(stream_);
    }

    {
      auto timer = create_timer("optimal_tour");
      get_optimal_tour<<<restart_batch_,
                         best_thread_num_,
                         sizeof(int) * best_thread_num_,
                         stream_>>>(results_, mylock_, work_, nodes_);
      CHECK_CUDA(stream_);
    }

    cudaDeviceSynchronize();
    best = best_cost_scalar_.value(stream_);

    if (verbose_) std::cout << "Best reported by kernel = " << best << "\n";

    if (best < global_best) {
      global_best = best;

      raft::update_host(addr_best_x_pos.data(), results_.best_x_pos, 1, stream_);
      raft::update_host(addr_best_y_pos.data(), results_.best_y_pos, 1, stream_);
      raft::update_host(addr_best_route.data(), results_.best_route, 1, stream_);
      CUDA_TRY(cudaStreamSynchronize(stream_));

      raft::copy(h_x_pos.data(), addr_best_x_pos[0], nodes_ + 1, stream_);
      raft::copy(h_y_pos.data(), addr_best_y_pos[0], nodes_ + 1, stream_);
      raft::copy(h_route.data(), addr_best_route[0], nodes_, stream_);
      raft::copy(route_, addr_best_route[0], nodes_, stream_);
      CHECK_CUDA(stream_);
    }
  }

  for (auto i = 0; i < nodes_; ++i) {
    if (verbose_) { std::cout << h_route[i] << ": " << h_x_pos[i] << " " << h_y_pos[i] << "\n"; }
    final_cost += euclidean_dist(h_x_pos.data(), h_y_pos.data(), i, i + 1);
  }

  if (verbose_) {
    hr_timer.display(std::cout);
    std::cout << "Optimized tour length = " << global_best << "\n";
  }

  return final_cost;
}

void TSP::knn()
{
  if (verbose_) std::cout << "Looking at " << k_ << " nearest neighbors\n";

  int dim              = 2;
  bool row_major_order = false;

  rmm::device_uvector<float> input(nodes_ * dim, stream_);
  float *input_ptr = input.data();
  raft::copy(input_ptr, x_pos_, nodes_, stream_);
  raft::copy(input_ptr + nodes_, y_pos_, nodes_, stream_);

  rmm::device_uvector<float> search_data(nodes_ * dim, stream_);
  float *search_data_ptr = search_data.data();
  raft::copy(search_data_ptr, input_ptr, nodes_ * dim, stream_);

  rmm::device_uvector<float> distances(nodes_ * (k_ + 1), stream_);
  float *distances_ptr = distances.data();

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

float traveling_salesperson(raft::handle_t const &handle,
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

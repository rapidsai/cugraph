/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cugraph/mtmg/handle.hpp>

#include <raft/comms/std_comms.hpp>

#include <vector>

namespace cugraph {
namespace mtmg {

/**
 * @brief Manages a subset of the cluster for a set of graph computations
 */
class instance_manager_t {
 public:
  /**
   * @brief Constructor
   *
   * @param handles   Vector of RAFT handles, one for each device on this node
   */
  instance_manager_t(std::vector<std::unique_ptr<raft::handle_t>>&& handles,
                     std::vector<std::unique_ptr<ncclComm_t>>&& nccl_comms,
                     std::vector<rmm::cuda_device_id>&& device_ids)
    : thread_counter_{0},
      raft_handle_{std::move(handles)},
      nccl_comms_{std::move(nccl_comms)},
      device_ids_{std::move(device_ids)}
  {
  }

  ~instance_manager_t()
  {
    for (size_t i = 0; i < nccl_comms_.size(); ++i) {
      rmm::cuda_set_device_raii local_set_device(device_ids_[i]);
      RAFT_NCCL_TRY(ncclCommDestroy(*nccl_comms_[i]));
    }
  }

  /**
   * @brief Get handle
   *
   * The instance manager will construct a handle appropriate for the thread making
   * the request.  Threads will be assigned to GPUs in a round-robin fashion to
   * spread requesting threads around the GPU resources.
   *
   * This function is CPU thread-safe.
   *
   * @return a handle for this thread.
   */
  handle_t get_handle()
  {
    int local_id  = thread_counter_++;
    int gpu_id    = local_id % raft_handle_.size();
    int thread_id = local_id / raft_handle_.size();

    return handle_t(*raft_handle_[gpu_id], thread_id, device_ids_[gpu_id]);
  }

  /**
   * @brief Get handle for particular GPU
   *
   * Return a handle for a particular GPU.  In a context-free environment
   * this lets the caller reconstitute the handle for the right host thread.
   * It does assume that the caller will not allow multiple threads to
   * concurrently use a gpu_id/thread_id pair.
   *
   * @return a handle for this thread.
   */
  handle_t get_handle(int gpu_id, int thread_id = 0)
  {
    return handle_t(*raft_handle_[gpu_id], thread_id, device_ids_[gpu_id]);
  }

  /**
   * @brief Reset the thread counter
   *
   * After a parallel activity is completed, we need to reset the thread counter so that
   * future threads will round robin around the GPUs properly.
   */
  void reset_threads() { thread_counter_.store(0); }

  /**
   * @brief Number of local GPUs in the instance
   */
  int get_local_gpu_count() { return static_cast<int>(raft_handle_.size()); }

 private:
  // FIXME: Should this be an std::map<> where the key is the rank?
  //        On a multi-node system we might have nodes with fewer
  //        (or no) GPUs, so mapping rank to a handle might be a challenge
  //
  std::vector<std::unique_ptr<raft::handle_t>> raft_handle_{};

  // FIXME: Explore what RAFT changes might be desired to allow the ncclComm_t
  //        to be managed by RAFT instead of cugraph::mtmg
  std::vector<std::unique_ptr<ncclComm_t>> nccl_comms_{};
  std::vector<rmm::cuda_device_id> device_ids_{};

  std::atomic<int> thread_counter_{0};
};

}  // namespace mtmg
}  // namespace cugraph

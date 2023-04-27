/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
  instance_manager_t(std::vector<std::shared_ptr<raft::handle_t>>&& handles)
    : thread_counter_{0}, raft_handle_{handles}
  {
  }

  /**
   * @brief Get handle
   *
   * The instance manager will construct a handle appropriate for the thread making
   * the request.  Threads will be assigned to GPUs in a round-robin fashion to
   * spread requesting threads around the GPU resources.
   *
   * This function will be CPU thread-safe.
   *
   * @return a handle for this thread.
   */
  handle_t get_handle()
  {
    int local_id = ++thread_counter_;

    return handle_t(raft_handle_[local_id % raft_handle_.size()], local_id / raft_handle_.size());
  }

  /**
   * @brief Reset the thread counter
   *
   * After a parallel activity is completed, we need to reset the thread counter so that
   * future threads will round robin around the GPUs properly.
   */
  void reset_threads() { thread_counter_.store(0); }

 private:
  std::atomic<int> thread_counter_{0};
  std::vector<std::shared_ptr<raft::handle_t>> raft_handle_{};
};

}  // namespace mtmg
}  // namespace cugraph

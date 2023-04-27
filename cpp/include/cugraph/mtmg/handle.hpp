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

#include <raft/core/handle.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Resource handler
 *
 * Multi-threaded resource handler.  Every GPU gets a raft::handle object that provides access to
 * the GPU resources.  In a multi-threaded environment multiple threads will share a particular GPU.
 * Following the MPI model, each thread will be assigned to a thread rank.
 *
 */
class handle_t {
 public:
  /**
   * @brief Constructor
   *
   * @param raft_handle   Raft handle for the resources
   * @param thread_rank   Rank for this thread
   */
  handle_t(std::shared_ptr<raft::handle_t>& raft_handle, int thread_rank)
    : raft_handle_(raft_handle), thread_rank_(thread_rank)
  {
  }

  /**
   * @brief Get the raft handle
   *
   * @return const reference to a raft handle
   */
  raft::handle_t const& raft_handle() const { return *raft_handle_; }

  /**
   * @brief Get thread rank
   *
   * @return thread rank
   */
  int get_thread_rank() const { return thread_rank_; }

  /**
   * @brief Get number of gpus
   *
   * @return number of gpus
   */
  int get_size() const { return raft_handle_->get_comms().get_size(); }

  /**
   * @brief Get gpu rank
   *
   * @return gpu rank
   */
  int get_rank() const { return raft_handle_->get_comms().get_rank(); }

 private:
  std::shared_ptr<raft::handle_t> raft_handle_;
  int thread_rank_;
};

}  // namespace mtmg
}  // namespace cugraph

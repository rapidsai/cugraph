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
#include <cugraph/mtmg/instance_manager.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/std_comms.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <execution>

namespace cugraph {
namespace mtmg {

/**
 * @brief Class for managing local and remote GPU resources for use in
 *   multi-threaded multi-GPU interface.
 *
 * Each process in a multi-GPU configuration should have an instance of this
 * class.  The resource manager object should be configured by calling
 * register_local_gpu (or register_remote_gpu once we support a multi-node
 * configuration) to allocate resources that can be used in the mtmg space.
 *
 * When we want to execute some graph computations, we need to create an instance for execution.
 * Based on how big a subset of the desired compute resources is desired, we can allocate some
 * number of GPUs to the problem (up to the total set of managed resources).
 *
 * The returned instance can be used to create a graph, execute one or more algorithms, etc.  Once
 * we are done the caller can delete the instance.
 *
 * At the moment, the caller is assumed to be responsible for scheduling use of the resources.
 *
 * For our first release, we will only consider a single node multi-GPU configuration, so the remote
 * GPU methods are currently disabled via ifdef.
 */
class resource_manager_t {
 public:
  /**
   * @brief Default constructor
   */
  resource_manager_t() {}

  /**
   * @brief add a local GPU to the resource manager.
   *
   * @param rank       The rank to assign to the local GPU
   * @param device_id  The device_id corresponding to this rank
   */
  void register_local_gpu(int rank, rmm::cuda_device_id device_id)
  {
    std::lock_guard<std::mutex> lock(lock_);

    CUGRAPH_EXPECTS(local_rank_map_.find(rank) == local_rank_map_.end(),
                    "cannot register same rank multiple times");

    int num_gpus_this_node;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_this_node));

    CUGRAPH_EXPECTS((device_id.value() >= 0) && (device_id.value() < num_gpus_this_node),
                    "device id out of range");

    local_rank_map_.insert(std::pair(rank, device_id));

    RAFT_CUDA_TRY(cudaSetDevice(device_id.value()));

    // FIXME: There is a bug in the cuda_memory_resource that results in a Hang.
    //   using the pool resource as a work-around.
    //
    // There is a deprecated environment variable: NCCL_LAUNCH_MODE=GROUP
    // which should temporarily work around this problem.
    //
    // Ultimately there should be some RMM parameters passed into this function
    // (or the constructor of the object) to configure this behavior
#if 0
    auto per_device_it = per_device_rmm_resources_.insert(
      std::pair{rank, std::make_shared<rmm::mr::cuda_memory_resource>()});
#else
    auto const [free, total] = rmm::detail::available_device_memory();
    auto const min_alloc =
      rmm::detail::align_down(std::min(free, total / 6), rmm::detail::CUDA_ALLOCATION_ALIGNMENT);

    auto per_device_it = per_device_rmm_resources_.insert(
      std::pair{rank,
                rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
                  std::make_shared<rmm::mr::cuda_memory_resource>(), min_alloc)});
#endif

    rmm::mr::set_per_device_resource(device_id, per_device_it.first->second.get());
  }

  /**
   * @brief Create an instance using a subset of the registered resources
   *
   * The selected set of resources will be configured as an instance manager.
   * If @ranks_to_include is a proper subset of the registered resources,
   * ranks will be renumbered into the range [0, @p ranks_to_use.size()), making
   * it a proper configuration.
   *
   * @param ranks_to_use        a vector containing the ranks to include in the instance.
   *   Must be a subset of the entire set of available ranks.
   * @param instance_manager_id a ncclUniqueId that is shared by all processes participating
   *   in this instance.  All processes must use the same ID in this call, it is up
   *   to the calling code to share this ID properly before the call.
   *
   * @return unique pointer to instance manager
   */
  std::unique_ptr<instance_manager_t> create_instance_manager(
    std::vector<int> ranks_to_include, ncclUniqueId instance_manager_id) const
  {
    std::for_each(
      ranks_to_include.begin(), ranks_to_include.end(), [local_ranks = local_rank_map_](int rank) {
        CUGRAPH_EXPECTS(local_ranks.find(rank) != local_ranks.end(),
                        "requesting inclusion of an invalid rank");
      });

    std::vector<std::unique_ptr<ncclComm_t>> nccl_comms{};
    std::vector<std::unique_ptr<raft::handle_t>> handles{};
    std::vector<rmm::cuda_device_id> device_ids{};

    nccl_comms.reserve(ranks_to_include.size());
    handles.reserve(ranks_to_include.size());
    device_ids.reserve(ranks_to_include.size());

    // FIXME: not quite right for multi-node
    auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(ranks_to_include.size())));
    while (ranks_to_include.size() % gpu_row_comm_size != 0) {
      --gpu_row_comm_size;
    }

    // FIXME: not quite right for multi-node
    for (size_t i = 0; i < ranks_to_include.size(); ++i) {
      int rank = ranks_to_include[i];
      auto pos = local_rank_map_.find(rank);
      RAFT_CUDA_TRY(cudaSetDevice(pos->second.value()));

      raft::handle_t tmp_handle;

      nccl_comms.push_back(std::make_unique<ncclComm_t>());
      handles.push_back(
        std::make_unique<raft::handle_t>(tmp_handle, per_device_rmm_resources_.find(rank)->second));
      device_ids.push_back(pos->second);
    }

    std::vector<std::thread> running_threads;

    for (size_t i = 0; i < ranks_to_include.size(); ++i) {
      running_threads.emplace_back([instance_manager_id,
                                    idx = i,
                                    gpu_row_comm_size,
                                    comm_size = ranks_to_include.size(),
                                    &ranks_to_include,
                                    &local_rank_map = local_rank_map_,
                                    &nccl_comms,
                                    &handles]() {
        int rank = ranks_to_include[idx];
        auto pos = local_rank_map.find(rank);
        RAFT_CUDA_TRY(cudaSetDevice(pos->second.value()));

        NCCL_TRY(ncclCommInitRank(nccl_comms[idx].get(), comm_size, instance_manager_id, rank));

        raft::comms::build_comms_nccl_only(handles[idx].get(), *nccl_comms[idx], comm_size, rank);

        cugraph::partition_manager::init_subcomm(*handles[idx], gpu_row_comm_size);
      });
    }

    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });

    // FIXME: Update for multi-node
    return std::make_unique<instance_manager_t>(
      std::move(handles), std::move(nccl_comms), std::move(device_ids), ranks_to_include.size());
  }

  /**
   * @brief Get a list of all of the currently registered ranks
   *
   * @return A copy of the list of ranks.
   */
  std::vector<int> registered_ranks() const
  {
    std::lock_guard<std::mutex> lock(lock_);

    //
    // C++20 mechanism:
    // return std::vector<int>{ std::views::keys(local_rank_map_).begin(),
    //                          std::views::keys(local_rank_map_).end() };
    //  Would need a bit more complicated to handle remote_rank_map_ also
    //
    std::vector<int> registered_ranks(local_rank_map_.size());
    std::transform(
      local_rank_map_.begin(), local_rank_map_.end(), registered_ranks.begin(), [](auto pair) {
        return pair.first;
      });

    return registered_ranks;
  }

 private:
  mutable std::mutex lock_{};
  std::map<int, rmm::cuda_device_id> local_rank_map_{};
  std::map<int, std::shared_ptr<rmm::mr::device_memory_resource>> per_device_rmm_resources_{};
};

}  // namespace mtmg
}  // namespace cugraph

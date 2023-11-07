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
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
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
 * Each GPU in the cluster should be given a unique global rank, an integer
 * that will be used to reference the GPU within the resource manager.  It
 * is recommended that the GPUs be numbered sequentially from 0, although this
 * is not required.
 *
 * When we want to execute some graph computations, we need to create an instance for execution.
 * Based on how big a subset of the desired compute resources is desired, we can allocate some
 * number of GPUs to the problem (up to the total set of managed resources).
 *
 * The returned instance can be used to create a graph, execute one or more algorithms, etc.  Once
 * we are done the caller can delete the instance.
 *
 * The caller is assumed to be responsible for scheduling use of the resources.
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
   * @param global_rank       The global rank to assign to the local GPU
   * @param local_device_id  The local device_id corresponding to this rank
   */
  void register_local_gpu(int global_rank, rmm::cuda_device_id local_device_id)
  {
    std::lock_guard<std::mutex> lock(lock_);

    CUGRAPH_EXPECTS(remote_rank_set_.find(global_rank) == remote_rank_set_.end(),
                    "cannot register same global_rank as local and remote");
    CUGRAPH_EXPECTS(local_rank_map_.find(global_rank) == local_rank_map_.end(),
                    "cannot register same global_rank multiple times");

    int num_gpus_this_node;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_this_node));

    CUGRAPH_EXPECTS(
      (local_device_id.value() >= 0) && (local_device_id.value() < num_gpus_this_node),
      "local device id out of range");

    local_rank_map_.insert(std::pair(global_rank, local_device_id));

    RAFT_CUDA_TRY(cudaSetDevice(local_device_id.value()));

    // FIXME: There is a bug in the cuda_memory_resource that results in a Hang.
    //   using the pool resource as a work-around.
    //
    // There is a deprecated environment variable: NCCL_LAUNCH_MODE=GROUP
    // which should temporarily work around this problem.
    //
    // Further NOTE: multi-node requires the NCCL_LAUNCH_MODE=GROUP feature
    // to be enabled even with the pool memory resource.
    //
    // Ultimately there should be some RMM parameters passed into this function
    // (or the constructor of the object) to configure this behavior
#if 0
    auto per_device_it = per_device_rmm_resources_.insert(
      std::pair{global_rank, std::make_shared<rmm::mr::cuda_memory_resource>()});
#else
    auto const [free, total] = rmm::detail::available_device_memory();
    auto const min_alloc =
      rmm::detail::align_down(std::min(free, total / 6), rmm::detail::CUDA_ALLOCATION_ALIGNMENT);

    auto per_device_it = per_device_rmm_resources_.insert(
      std::pair{global_rank,
                rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
                  std::make_shared<rmm::mr::cuda_memory_resource>(), min_alloc)});
#endif

    rmm::mr::set_per_device_resource(local_device_id, per_device_it.first->second.get());
  }

  /**
   * @brief add a remote GPU to the resource manager.
   *
   * @param global_rank             The global rank to assign to the remote GPU
   */
  void register_remote_gpu(int global_rank)
  {
    std::lock_guard<std::mutex> lock(lock_);

    CUGRAPH_EXPECTS(local_rank_map_.find(global_rank) == local_rank_map_.end(),
                    "cannot register same global_rank as local and remote");
    CUGRAPH_EXPECTS(remote_rank_set_.find(global_rank) == remote_rank_set_.end(),
                    "cannot register same global_rank multiple times");

    remote_rank_set_.insert(global_rank);
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
   * @param n_streams           The number of streams to create in a stream pool for
   *   each GPU.  Defaults to 16.
   *
   * @return unique pointer to instance manager
   */
  std::unique_ptr<instance_manager_t> create_instance_manager(std::vector<int> ranks_to_include,
                                                              ncclUniqueId instance_manager_id,
                                                              size_t n_streams = 16) const
  {
    std::vector<int> local_ranks_to_include;

    std::copy_if(ranks_to_include.begin(),
                 ranks_to_include.end(),
                 std::back_inserter(local_ranks_to_include),
                 [&local_ranks = local_rank_map_](int rank) {
                   return (local_ranks.find(rank) != local_ranks.end());
                 });

    // FIXME: Explore what RAFT changes might be desired to allow the ncclComm_t
    //        to be managed by RAFT instead of cugraph::mtmg
    std::vector<std::unique_ptr<ncclComm_t>> nccl_comms{};
    std::vector<std::unique_ptr<raft::handle_t>> handles{};
    std::vector<rmm::cuda_device_id> device_ids{};

    nccl_comms.reserve(local_ranks_to_include.size());
    handles.reserve(local_ranks_to_include.size());
    device_ids.reserve(local_ranks_to_include.size());

    auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(ranks_to_include.size())));
    while (ranks_to_include.size() % gpu_row_comm_size != 0) {
      --gpu_row_comm_size;
    }

    int current_device{};
    RAFT_CUDA_TRY(cudaGetDevice(&current_device));
    RAFT_NCCL_TRY(ncclGroupStart());

    for (size_t i = 0; i < local_ranks_to_include.size(); ++i) {
      int rank = local_ranks_to_include[i];
      auto pos = local_rank_map_.find(rank);
      RAFT_CUDA_TRY(cudaSetDevice(pos->second.value()));

      nccl_comms.push_back(std::make_unique<ncclComm_t>());
      handles.push_back(
        std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread,
                                         std::make_shared<rmm::cuda_stream_pool>(n_streams),
                                         per_device_rmm_resources_.find(rank)->second));
      device_ids.push_back(pos->second);

      RAFT_NCCL_TRY(
        ncclCommInitRank(nccl_comms[i].get(), ranks_to_include.size(), instance_manager_id, rank));
      raft::comms::build_comms_nccl_only(
        handles[i].get(), *nccl_comms[i], ranks_to_include.size(), rank);
    }
    RAFT_NCCL_TRY(ncclGroupEnd());
    RAFT_CUDA_TRY(cudaSetDevice(current_device));

    std::vector<std::thread> running_threads;

    for (size_t i = 0; i < local_ranks_to_include.size(); ++i) {
      running_threads.emplace_back([instance_manager_id,
                                    idx = i,
                                    gpu_row_comm_size,
                                    comm_size = ranks_to_include.size(),
                                    &local_ranks_to_include,
                                    &device_ids,
                                    &nccl_comms,
                                    &handles]() {
        int rank = local_ranks_to_include[idx];
        RAFT_CUDA_TRY(cudaSetDevice(device_ids[idx].value()));

        cugraph::partition_manager::init_subcomm(*handles[idx], gpu_row_comm_size);
      });
    }

    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });

    return std::make_unique<instance_manager_t>(
      std::move(handles), std::move(nccl_comms), std::move(device_ids));
  }

  /**
   * @brief Get a list of all of the currently registered ranks
   *
   * @return A copy of the list of ranks.
   */
  std::vector<int> registered_ranks() const
  {
    std::lock_guard<std::mutex> lock(lock_);

    std::vector<int> registered_ranks(local_rank_map_.size() + remote_rank_set_.size());
    std::transform(
      local_rank_map_.begin(), local_rank_map_.end(), registered_ranks.begin(), [](auto pair) {
        return pair.first;
      });

    std::copy(remote_rank_set_.begin(),
              remote_rank_set_.end(),
              registered_ranks.begin() + local_rank_map_.size());

    std::sort(registered_ranks.begin(), registered_ranks.end());
    return registered_ranks;
  }

 private:
  mutable std::mutex lock_{};
  std::map<int, rmm::cuda_device_id> local_rank_map_{};
  std::set<int> remote_rank_set_{};
  std::map<int, std::shared_ptr<rmm::mr::device_memory_resource>> per_device_rmm_resources_{};
};

}  // namespace mtmg
}  // namespace cugraph

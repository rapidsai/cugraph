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

#include <raft/comms/std_comms.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

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
  resource_manager_t() {}

  /**
   * @brief add a local GPU to the resource manager.
   *
   * @param rank       The rank to assign to the local GPU
   * @param device_id  The device_id corresponding to this rank
   */
  void register_local_gpu(int rank, int device_id)
  {
    std::lock_guard<std::mutex> lock(lock_);

    CUGRAPH_EXPECTS(local_rank_map_.find(rank) == local_rank_map_.end(),
                    "cannot register same rank multiple times");

    int num_gpus_this_node;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_this_node));

    CUGRAPH_EXPECTS((device_id >= 0) && (device_id < num_gpus_this_node), "device id out of range");

    local_rank_map_[rank] = device_id;
  }

#if 0
  /**
   * @brief add a remote GPU to the resource manager.
   *
   * FIXME: Need some sort of comms information for the remote GPU here...
   *
   * @param rank    The rank to assign to the remote GPU
   */
  void register_remote_gpu(int rank, TBD const& remote_info);
#endif

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

    ncclGroupStart();

    std::vector<std::shared_ptr<ncclComm_t>> nccl_comms(ranks_to_include.size());
    std::vector<std::shared_ptr<raft::handle_t>> handles(ranks_to_include.size());
    std::vector<int> device_ids(ranks_to_include.size());

    std::vector<std::thread> running_threads;

    for (size_t i = 0; i < ranks_to_include.size(); ++i) {
      running_threads.emplace_back([instance_manager_id,
                                    idx = i,
                                    this,
                                    &ranks_to_include,
                                    &local_ranks = local_rank_map_,
                                    &nccl_comms,
                                    &handles,
                                    &device_ids]() {
        int rank = ranks_to_include[idx];

        // FIXME: not quite right for multi-node
        auto nccl_comm = std::make_shared<ncclComm_t>();
        auto pos       = local_ranks.find(rank);
        cudaSetDevice(pos->second);
        ncclCommInitRank(nccl_comm.get(), local_ranks.size(), instance_manager_id, rank);

        // FIXME: I should pass in RMM parameters here?
        auto handle = std::make_shared<raft::handle_t>();

        raft::comms::build_comms_nccl_only(handle.get(), *nccl_comm, local_ranks.size(), rank);

        auto& comm           = handle->get_comms();
        auto const comm_size = comm.get_size();

        auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
        while (comm_size % gpu_row_comm_size != 0) {
          --gpu_row_comm_size;
        }

        cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

        nccl_comms[idx] = nccl_comm;
        handles[idx]    = handle;
        device_ids[idx] = pos->second;
      });
    }

    std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });

    ncclGroupEnd();

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
  std::map<int, int> local_rank_map_{};

#if 0
  //
  // TBD: Probably a map with rank used as the key and
  //    some sort of comms class as the value.  Needs
  //    to be something we can make a raft handle and
  //    initialize comms
  //
  std::map<int, TBD> remote_rank_map_{};
#endif
};

}  // namespace mtmg
}  // namespace cugraph

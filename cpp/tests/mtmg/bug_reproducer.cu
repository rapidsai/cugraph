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
#include <raft/comms/std_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <nccl.h>

#include <vector>

int main(int argc, char** argv)
{
  int comm_size{2};
  int gpu_row_comm_size{1};

  ncclUniqueId instance_manager_id;
  ncclGetUniqueId(&instance_manager_id);

  std::vector<std::thread> running_threads;

  std::vector<std::shared_ptr<ncclComm_t>> nccl_comms{};
  std::vector<std::shared_ptr<raft::handle_t>> handles{};

  nccl_comms.reserve(comm_size);
  handles.reserve(comm_size);

  cudaSetDevice(0);
  auto resource_0 = std::make_shared<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_per_device_resource(rmm::cuda_device_id{0}, resource_0.get());

  raft::handle_t tmp_handle_0;

  nccl_comms.push_back(std::make_shared<ncclComm_t>());
  handles.push_back(std::make_shared<raft::handle_t>(tmp_handle_0, resource_0));

  cudaSetDevice(1);
  auto resource_1 = std::make_shared<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_per_device_resource(rmm::cuda_device_id{1}, resource_1.get());

  raft::handle_t tmp_handle_1;

  nccl_comms.push_back(std::make_shared<ncclComm_t>());
  handles.push_back(std::make_shared<raft::handle_t>(tmp_handle_1, resource_1));

  for (int i = 0; i < comm_size; ++i) {
    running_threads.emplace_back(
      [&instance_manager_id, rank = i, gpu_row_comm_size, comm_size, &nccl_comms, &handles]() {
        cudaSetDevice(rank);

        std::cout << "call nccl_comms_init_rank, rank = " << rank << std::endl;

        ncclCommInitRank(nccl_comms[rank].get(), comm_size, instance_manager_id, rank);

        std::cout << "call build_comms_nccl_only" << std::endl;
        raft::comms::build_comms_nccl_only(handles[rank].get(), *nccl_comms[rank], comm_size, rank);
        std::cout << "back from build_comms_nccl_only" << std::endl;

        handles[rank]->sync_stream();

        auto& comm = handles[rank]->get_comms();

        int row_idx = rank / gpu_row_comm_size;
        int col_idx = rank % gpu_row_comm_size;

        handles[rank]->set_subcomm(
          "gpu_row_comm",
          std::make_shared<raft::comms::comms_t>(comm.comm_split(row_idx, col_idx)));
        handles[rank]->set_subcomm(
          "gpu_col_comm",
          std::make_shared<raft::comms::comms_t>(comm.comm_split(col_idx, row_idx)));

        handles[rank]->sync_stream();

        std::cout << "back from init_subcomm" << std::endl;
      });
  }

  std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
}

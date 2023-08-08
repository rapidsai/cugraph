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
#include <cugraph/mtmg/resource_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <nccl.h>

#include <vector>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

int main(int argc, char** argv)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  int num_gpus{2};

  ncclUniqueId instance_manager_id;
  ncclGetUniqueId(&instance_manager_id);

  cugraph::mtmg::resource_manager_t resource_manager;

  std::vector<std::thread> running_threads;

  resource_manager.register_local_gpu(0, rmm::cuda_device_id{0});
  resource_manager.register_local_gpu(1, rmm::cuda_device_id{1});

  std::cout << "create instance_manager" << std::endl;

  auto instance_manager = resource_manager.create_instance_manager(
    resource_manager.registered_ranks(), instance_manager_id);

  for (int i = 0; i < num_gpus; ++i) {
    running_threads.emplace_back([&instance_manager]() {
      auto thread_handle = instance_manager->get_handle();

      if (thread_handle.get_thread_rank() > 0) return;

      auto& comm = thread_handle.raft_handle().get_comms();

      std::vector<vertex_t> h_sorted_local_vertices;
      if (thread_handle.get_rank() == 0) {
        h_sorted_local_vertices = {{32}};
      } else {
        h_sorted_local_vertices = {{33}};
      }

      rmm::device_uvector<vertex_t> sorted_local_vertices(1, thread_handle.get_stream());
      raft::update_device(sorted_local_vertices.data(),
                          h_sorted_local_vertices.data(),
                          h_sorted_local_vertices.size(),
                          thread_handle.get_stream());
      thread_handle.raft_handle().sync_stream();

      std::cout << "in compute_renumber_map... step 3 shuffle, rank = " << thread_handle.get_rank()
                << std::endl;
      raft::print_device_vector("  sorted_local_vertices",
                                sorted_local_vertices.data(),
                                sorted_local_vertices.size(),
                                std::cout);
      sorted_local_vertices =
        cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
          thread_handle.raft_handle(), std::move(sorted_local_vertices));

      std::cout << "computing local min, rank = " << comm.get_rank() << std::endl;

      vertex_t min = 32 + comm.get_rank();

      std::cout << "rank = " << comm.get_rank() << ", local min = " << min << std::endl;
      min = cugraph::host_scalar_allreduce(
        comm, min, raft::comms::op_t::MIN, thread_handle.get_stream());

      std::cout << "after detail::host_scalar_allreduce call, rank = " << thread_handle.get_rank()
                << ", min = " << min << std::endl;
    });
  }

  // Wait for CPU threads to complete
  std::for_each(running_threads.begin(), running_threads.end(), [](auto& t) { t.join(); });
  running_threads.resize(0);
  instance_manager->reset_threads();
}

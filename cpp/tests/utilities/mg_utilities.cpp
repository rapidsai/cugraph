/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include "utilities/mg_utilities.hpp"

#include <cugraph/partition_manager.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <vector>

namespace cugraph {
namespace test {

void initialize_mpi(int argc, char** argv) { RAFT_MPI_TRY(MPI_Init(&argc, &argv)); }

void finalize_mpi() { RAFT_MPI_TRY(MPI_Finalize()); }

int query_mpi_comm_world_rank()
{
  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
  return comm_rank;
}

int query_mpi_comm_world_size()
{
  int comm_size{};
  RAFT_MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
  return comm_size;
}

std::unique_ptr<raft::handle_t> initialize_mg_handle(size_t pool_size)
{
  std::unique_ptr<raft::handle_t> handle{nullptr};

  handle = std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread,
                                            std::make_shared<rmm::cuda_stream_pool>(pool_size));

  auto comm_size = query_mpi_comm_world_size();

  raft::comms::initialize_mpi_comms(handle.get(), MPI_COMM_WORLD);

  auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
  while (comm_size % gpu_row_comm_size != 0) {
    --gpu_row_comm_size;
  }
  cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

  return std::move(handle);
}

void enforce_p2p_initialization(raft::comms::comms_t const& comm, rmm::cuda_stream_view stream)
{
  auto const comm_size = comm.get_size();

  constexpr size_t p2p_count = (128 * 1024) / sizeof(int32_t);  // 128 MB

  rmm::device_uvector<int32_t> tx_ints(comm_size * p2p_count, stream);
  rmm::device_uvector<int32_t> rx_ints(comm_size * p2p_count, stream);

  comm.device_alltoall(tx_ints.data(), rx_ints.data(), p2p_count, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

}  // namespace test
}  // namespace cugraph

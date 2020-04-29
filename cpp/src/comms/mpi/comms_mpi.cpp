/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "comms/mpi/comms_mpi.hpp"

#include <iostream>
#include <nccl.h>

namespace cugraph { 
namespace experimental {

Comm::Comm(int p) : _p{p} {
#if USE_NCCL
  // MPI
  int flag{};

  MPI_TRY(MPI_Initialized(&flag));

  if (flag == false) {
    int provided{};
    MPI_TRY(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
    if (provided != MPI_THREAD_MULTIPLE) {
      MPI_TRY(MPI_ERR_OTHER);
    }
    _finalize_mpi = true;
  }

  MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &_mpi_world_rank));
  MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &_mpi_world_size));
  CUGRAPH_EXPECTS( (_p == _mpi_world_size), 
                   "Invalid input arguments: p should match the number of MPI processes.");

  _mpi_comm = MPI_COMM_WORLD;

  // CUDA

  CUDA_TRY(cudaGetDeviceCount(&_device_count));
  _device_id = _mpi_world_rank % _device_count;
  CUDA_TRY(cudaSetDevice(_device_id));

  CUDA_TRY(
    cudaDeviceGetAttribute(&_sm_count_per_device, cudaDevAttrMultiProcessorCount, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_max_grid_dim_1D, cudaDevAttrMaxGridDimX, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_max_block_dim_1D, cudaDevAttrMaxBlockDimX, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_l2_cache_size, cudaDevAttrL2CacheSize, _device_id));
  CUDA_TRY(
    cudaDeviceGetAttribute(
      &_shared_memory_size_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, _device_id));

  // NCCL

  ncclUniqueId nccl_unique_id_p{};
  if (get_rank() == 0) {
    NCCL_TRY(ncclGetUniqueId(&nccl_unique_id_p));
  }
  MPI_TRY(MPI_Bcast(&nccl_unique_id_p, sizeof(ncclUniqueId), MPI_BYTE, 0, _mpi_comm));
  NCCL_TRY(ncclCommInitRank(&_nccl_comm, get_p(), nccl_unique_id_p, get_rank()));
  _finalize_nccl = true;
#endif

}

Comm::~Comm() {
#if USE_NCCL
  // NCCL
  if (_finalize_nccl)
    ncclCommDestroy(_nccl_comm);

  if (_finalize_mpi) {
    MPI_Finalize();
  }
#endif
}

void Comm::barrier() {
#if USE_NCCL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}
} }//namespace

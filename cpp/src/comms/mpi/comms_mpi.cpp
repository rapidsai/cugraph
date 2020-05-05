/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <comms_mpi.hpp>
#include <iostream>
#include <vector>
#include "utilities/error_utils.h"

namespace cugraph {
namespace experimental {
#if ENABLE_OPG

/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a NCCL error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct nccl_error : public std::runtime_error {
  nccl_error(std::string const &message) : std::runtime_error(message) {}
};

inline void throw_nccl_error(ncclResult_t error, const char *file, unsigned int line)
{
  throw nccl_error(std::string{"NCCL error encountered at: " + std::string{file} + ":" +
                               std::to_string(line) + ": " + ncclGetErrorString(error)});
}

#define NCCL_TRY(call)                                                                     \
  {                                                                                        \
    ncclResult_t nccl_status = (call);                                                     \
    if (nccl_status != ncclSuccess) { throw_nccl_error(nccl_status, __FILE__, __LINE__); } \
  }
// MPI errors are expected to be fatal before reaching this.
// FIXME : improve when adding raft comms
#define MPI_TRY(cmd)                                             \
  {                                                              \
    int e = cmd;                                                 \
    if (e != MPI_SUCCESS) { CUGRAPH_FAIL("Failed: MPI error"); } \
  }

template <typename value_t>
constexpr MPI_Datatype get_mpi_type()
{
  if (std::is_integral<value_t>::value) {
    if (std::is_signed<value_t>::value) {
      if (sizeof(value_t) == 1) {
        return MPI_INT8_T;
      } else if (sizeof(value_t) == 2) {
        return MPI_INT16_T;
      } else if (sizeof(value_t) == 4) {
        return MPI_INT32_T;
      } else if (sizeof(value_t) == 8) {
        return MPI_INT64_T;
      } else {
        CUGRAPH_FAIL("unsupported type");
      }
    } else {
      if (sizeof(value_t) == 1) {
        return MPI_UINT8_T;
      } else if (sizeof(value_t) == 2) {
        return MPI_UINT16_T;
      } else if (sizeof(value_t) == 4) {
        return MPI_UINT32_T;
      } else if (sizeof(value_t) == 8) {
        return MPI_UINT64_T;
      } else {
        CUGRAPH_FAIL("unsupported type");
      }
    }
  } else if (std::is_same<value_t, float>::value) {
    return MPI_FLOAT;
  } else if (std::is_same<value_t, double>::value) {
    return MPI_DOUBLE;
  } else {
    CUGRAPH_FAIL("unsupported type");
  }
}

template <typename value_t>
constexpr ncclDataType_t get_nccl_type()
{
  if (std::is_integral<value_t>::value) {
    if (std::is_signed<value_t>::value) {
      if (sizeof(value_t) == 1) {
        return ncclInt8;
      } else if (sizeof(value_t) == 4) {
        return ncclInt32;
      } else if (sizeof(value_t) == 8) {
        return ncclInt64;
      } else {
        CUGRAPH_FAIL("unsupported type");
      }
    } else {
      if (sizeof(value_t) == 1) {
        return ncclUint8;
      } else if (sizeof(value_t) == 4) {
        return ncclUint32;
      } else if (sizeof(value_t) == 8) {
        return ncclUint64;
      } else {
        CUGRAPH_FAIL("unsupported type");
      }
    }
  } else if (std::is_same<value_t, float>::value) {
    return ncclFloat32;
  } else if (std::is_same<value_t, double>::value) {
    return ncclFloat64;
  } else {
    CUGRAPH_FAIL("unsupported type");
  }
}

constexpr MPI_Op get_mpi_reduce_op(ReduceOp reduce_op)
{
  if (reduce_op == ReduceOp::SUM) {
    return MPI_SUM;
  } else if (reduce_op == ReduceOp::MAX) {
    return MPI_MAX;
  } else if (reduce_op == ReduceOp::MIN) {
    return MPI_MIN;
  } else {
    CUGRAPH_FAIL("unsupported type");
  }
}

constexpr ncclRedOp_t get_nccl_reduce_op(ReduceOp reduce_op)
{
  if (reduce_op == ReduceOp::SUM) {
    return ncclSum;
  } else if (reduce_op == ReduceOp::MAX) {
    return ncclMax;
  } else if (reduce_op == ReduceOp::MIN) {
    return ncclMin;
  } else {
    CUGRAPH_FAIL("unsupported type");
  }
}
#endif

Comm::Comm(int p) : _p{p}
{
#if ENABLE_OPG
  // MPI
  int flag{}, mpi_world_size;

  MPI_TRY(MPI_Initialized(&flag));

  if (flag == false) {
    int provided{};
    MPI_TRY(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
    if (provided != MPI_THREAD_MULTIPLE) { MPI_TRY(MPI_ERR_OTHER); }
    _finalize_mpi = true;
  }

  MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
  MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));
  CUGRAPH_EXPECTS((_p == mpi_world_size),
                  "Invalid input arguments: p should match the number of MPI processes.");

  _mpi_comm = MPI_COMM_WORLD;

  // CUDA

  CUDA_TRY(cudaGetDeviceCount(&_device_count));
  _device_id = _rank % _device_count;  // FIXME : assumes each node has the same number of GPUs
  CUDA_TRY(cudaSetDevice(_device_id));

  CUDA_TRY(
    cudaDeviceGetAttribute(&_sm_count_per_device, cudaDevAttrMultiProcessorCount, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_max_grid_dim_1D, cudaDevAttrMaxGridDimX, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_max_block_dim_1D, cudaDevAttrMaxBlockDimX, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_l2_cache_size, cudaDevAttrL2CacheSize, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(
    &_shared_memory_size_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, _device_id));

  // NCCL

  ncclUniqueId nccl_unique_id_p{};
  if (get_rank() == 0) { NCCL_TRY(ncclGetUniqueId(&nccl_unique_id_p)); }
  MPI_TRY(MPI_Bcast(&nccl_unique_id_p, sizeof(ncclUniqueId), MPI_BYTE, 0, _mpi_comm));
  NCCL_TRY(ncclCommInitRank(&_nccl_comm, get_p(), nccl_unique_id_p, get_rank()));
  _finalize_nccl = true;
#endif
}

#if ENABLE_OPG
Comm::Comm(ncclComm_t comm, int size, int rank) : _nccl_comm(comm), _p(size), _rank(rank)
{
  // CUDA
  CUDA_TRY(cudaGetDeviceCount(&_device_count));
  _device_id = _rank % _device_count;   // FIXME : assumes each node has the same number of GPUs
  CUDA_TRY(cudaSetDevice(_device_id));  // FIXME : check if this is needed or if
                                        // python takes care of this

  CUDA_TRY(
    cudaDeviceGetAttribute(&_sm_count_per_device, cudaDevAttrMultiProcessorCount, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_max_grid_dim_1D, cudaDevAttrMaxGridDimX, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_max_block_dim_1D, cudaDevAttrMaxBlockDimX, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(&_l2_cache_size, cudaDevAttrL2CacheSize, _device_id));
  CUDA_TRY(cudaDeviceGetAttribute(
    &_shared_memory_size_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, _device_id));
}
#endif

Comm::~Comm()
{
#if ENABLE_OPG
  // NCCL
  if (_finalize_nccl) ncclCommDestroy(_nccl_comm);

  if (_finalize_mpi) { MPI_Finalize(); }
#endif
}

void Comm::barrier()
{
#if ENABLE_OPG
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

template <typename value_t>
void Comm::allgather(size_t size, value_t *sendbuff, value_t *recvbuff) const
{
#if ENABLE_OPG
  NCCL_TRY(ncclAllGather((const void *)sendbuff,
                         (void *)recvbuff,
                         size,
                         get_nccl_type<value_t>(),
                         _nccl_comm,
                         cudaStreamDefault));
#endif
}

template <typename value_t>
void Comm::allreduce(size_t size, value_t *sendbuff, value_t *recvbuff, ReduceOp reduce_op) const
{
#if ENABLE_OPG
  NCCL_TRY(ncclAllReduce((const void *)sendbuff,
                         (void *)recvbuff,
                         size,
                         get_nccl_type<value_t>(),
                         get_nccl_reduce_op(reduce_op),
                         _nccl_comm,
                         cudaStreamDefault));
#endif
}

// explicit
template void Comm::allgather<int>(size_t size, int *sendbuff, int *recvbuff) const;
template void Comm::allgather<float>(size_t size, float *sendbuff, float *recvbuff) const;
template void Comm::allgather<double>(size_t size, double *sendbuff, double *recvbuff) const;
template void Comm::allreduce<int>(size_t size,
                                   int *sendbuff,
                                   int *recvbuff,
                                   ReduceOp reduce_op) const;
template void Comm::allreduce<float>(size_t size,
                                     float *sendbuff,
                                     float *recvbuff,
                                     ReduceOp reduce_op) const;
template void Comm::allreduce<double>(size_t size,
                                      double *sendbuff,
                                      double *recvbuff,
                                      ReduceOp reduce_op) const;

}  // namespace experimental
}  // namespace cugraph

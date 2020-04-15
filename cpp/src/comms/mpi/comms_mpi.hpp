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

// snmg utils
// Author: Alex Fender afender@nvidia.com
 
#pragma once
#include <mpi.h>
#include <nccl.h>
#include <omp.h>
#include <unistd.h>
#include <vector>
#include "mem_utils.h"
#include "basic_kernels.cuh"

#define USE_NCCL 1

namespace cugraph { 
namespace opg {

template <typename value_t>
constexpr MPI_Datatype get_mpi_type() {
  if (std::is_integral<value_t>::value) {
    if (std::is_signed<value_t>::value) {
      if (sizeof(value_t) == 1) {
        return MPI_INT8_T;
      }
      else if (sizeof(value_t) == 2) {
        return MPI_INT16_T;
      }
      else if (sizeof(value_t) == 4) {
        return MPI_INT32_T;
      }
      else if (sizeof(value_t) == 8) {
        return MPI_INT64_T;
      }
      else {
        CUGRAPH_FAIL("unsupported type");
      }
    }
    else {
      if (sizeof(value_t) == 1) {
        return MPI_UINT8_T;
      }
      else if (sizeof(value_t) == 2) {
        return MPI_UINT16_T;
      }
      else if (sizeof(value_t) == 4) {
        return MPI_UINT32_T;
      }
      else if (sizeof(value_t) == 8) {
        return MPI_UINT64_T;
      }
      else {
        CUGRAPH_FAIL("unsupported type");
      }
    }
  }
  else if(std::is_same<value_t, float>::value) {
    return MPI_FLOAT;
  }
  else if(std::is_same<value_t, double>::value) {
    return MPI_DOUBLE;
  }
  else {
    CUGRAPH_FAIL("unsupported type");
  }
}
#if USE_NCCL
template <typename value_t>
constexpr ncclDataType_t get_nccl_type() {
  if (std::is_integral<value_t>::value) {
    if (std::is_signed<value_t>::value) {
      if (sizeof(value_t) == 1) {
        return ncclInt8;
      }
      else if (sizeof(value_t) == 4) {
        return ncclInt32;
      }
      else if (sizeof(value_t) == 8) {
        return ncclInt64;
      }
      else {
        CUGRAPH_FAIL("unsupported type");
      }
    }
    else {
      if (sizeof(value_t) == 1) {
        return ncclUint8;
      }
      else if (sizeof(value_t) == 4) {
        return ncclUint32;
      }
      else if (sizeof(value_t) == 8) {
        return ncclUint64;
      }
      else {
        CUGRAPH_FAIL("unsupported type");
      }
    }
  }
  else if(std::is_same<value_t, float>::value) {
    return ncclFloat32;
  }
  else if(std::is_same<value_t, double>::value) {
    return ncclFloat64;
  }
  else {
    CUGRAPH_FAIL("unsupported type");
  }
}
#endif
enum class ReduceOp { SUM, MAX, MIN };

constexpr MPI_Op get_mpi_reduce_op(ReduceOp reduce_op) {
  if (reduce_op == ReduceOp::SUM) {
    return MPI_SUM;
  }
  else if (reduce_op == ReduceOp::MAX) {
    return MPI_MAX;
  }
  else if (reduce_op == ReduceOp::MIN) {
    return MPI_MIN;
  }
  else {
    CUGRAPH_FAIL("unsupported type");
  }
}

#if USE_NCCL
constexpr ncclRedOp_t get_nccl_reduce_op(ReduceOp reduce_op) {
  if (reduce_op == ReduceOp::SUM) {
    return ncclSum;
  }
  else if (reduce_op == ReduceOp::MAX) {
    return ncclMax;
  }
  else if (reduce_op == ReduceOp::MIN) {
    return ncclMin;
  }
  else {
    CUGRAPH_FAIL("unsupported type");
  }
}
#endif

// basic info about the snmg env setup
class Comm 
{ 
  private:
  int _p_x{0};
  int _p_y{0};

  int _mpi_world_rank{0};
  int _mpi_world_size{0};
  bool _finalize_mpi{false};

  int _device_id{0};
  int _device_count{0};

  std::vector<void*> _p_ipc_mems{};
  std::vector<size_t> _local_ipc_mem_offsets{};

  int _sm_count_per_device{0};
  int _max_grid_dim_1D{0};
  int _max_block_dim_1D{0};
  int _l2_cache_size{0};
  int _shared_memory_size_per_sm{0};
  int _cuda_stream_least_priority{0};
  int _cuda_stream_greatest_priority{0};

  MPI_Comm _mpi_comm_p_x{};
  MPI_Comm _mpi_comm_p_y{};
  MPI_Comm _mpi_comm_p{};

  cudaStream_t _default_stream{};
  std::vector<cudaStream_t> _extra_streams{};

  ncclComm_t _nccl_comm{};
  
  public: 
    Comm();
    ~Comm();
    int get_rank() const { return _mpi_world_rank; }
    int get_p() const { return _mpi_world_size; }
    int get_dev() const { return _device_id; }
    int get_dev_count() const { return _device_count; }
    int get_sm_count() const { return _sm_count_per_device; }
    bool is_master() const return { return (_mpi_world_rank == 0)? true : false; }
    void init();

    template <typename val_t>
    void allgather (size_t size, val_t* sendbuff, val_t* recvbuff);

    template <typename val_t>
    void allreduce (size_t size, val_t* sendbuff, val_t* recvbuff, ReduceOp reduce_op);

};

// Wait for all host threads 
void sync_all() {
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
}

template <typename val_t>
void Comm::allgather (size_t size, val_t* sendbuff, val_t* recvbuff) {
#if USE_NCCL
  if(typeid(val_t) == typeid(float))
    NCCL_TRY(ncclAllGather((const void*)sendbuff, (void*)recvbuff, size, get_nccl_type<value_t>(), _nccl_comm, cudaStreamDefault));
  else 
    CUGRAPH_FAIL("allgather needs floats");
#endif
}

template <typename val_t>
void Comm::allreduce (size_t size, val_t* sendbuff, val_t* recvbuff, ReduceOp reduce_op) {
#if USE_NCCL
    NCCL_TRY(ncclAllReduce(const void*)sendbuff, (void*)recvbuff, size, get_nccl_type<value_t>(), get_nccl_reduce_op(reduce_op), _nccl_comm, cudaStreamDefault)););
#endif
}

} } //namespace

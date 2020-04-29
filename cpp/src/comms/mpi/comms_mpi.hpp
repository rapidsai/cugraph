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


#pragma once

#if USE_NCCL
#include <mpi.h>
#include <nccl.h>
#endif

#include <omp.h>
#include <unistd.h>
#include <vector>
#include "utilities/error_utils.h"

namespace cugraph { 
namespace experimental {

enum class ReduceOp { SUM, MAX, MIN };

#if USE_NCCL
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a NCCL error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct nccl_error : public std::runtime_error {
  nccl_error(std::string const& message) : std::runtime_error(message) {}
};

inline void throw_nccl_error(ncclResult_t error, const char* file,
                             unsigned int line) {
  throw nccl_error(
      std::string{"NCCL error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + ncclGetErrorString(error)});
}

#define NCCL_TRY(call) {                                           \
  ncclResult_t nccl_status = (call);                               \
  if (nccl_status!= ncclSuccess) {                                 \
    throw_nccl_error(nccl_status, __FILE__, __LINE__); \
  }                                                                \
} 

// MPI errors are expected to be fatal before reaching this.
// Fix me : improve when adding raft comms
#define MPI_TRY(cmd) {                           \
  int e = cmd;                                   \
  if ( e != MPI_SUCCESS ) {                      \
    CUGRAPH_FAIL("Failed: MPI error");           \
  }                                              \
}                                               

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
    int _p{0};

    int _mpi_world_rank{0};
    int _mpi_world_size{0};
    bool _finalize_mpi{false};
    bool _finalize_nccl{false};


    int _device_id{0};
    int _device_count{0};

    std::vector<void*> _p_ipc_mems{};
    std::vector<size_t> _local_ipc_mem_offsets{};

    int _sm_count_per_device{0};
    int _max_grid_dim_1D{0};
    int _max_block_dim_1D{0};
    int _l2_cache_size{0};
    int _shared_memory_size_per_sm{0};

#if USE_NCCL
    MPI_Comm _mpi_comm{};
    ncclComm_t _nccl_comm{};
 #endif
   
  public: 
    Comm(){};
    Comm(int p);
    ~Comm();
    int get_rank() const { return _mpi_world_rank; }
    int get_p() const { return _mpi_world_size; }
    int get_dev() const { return _device_id; }
    int get_dev_count() const { return _device_count; }
    int get_sm_count() const { return _sm_count_per_device; }
    bool is_master() const { return (_mpi_world_rank == 0)? true : false; }

    void barrier();

    template <typename value_t>
    void allgather (size_t size, value_t* sendbuff, value_t* recvbuff) const;

    template <typename value_t>
    void allreduce (size_t size, value_t* sendbuff, value_t* recvbuff, ReduceOp reduce_op) const;

};

template <typename value_t>
void Comm::allgather (size_t size, value_t* sendbuff, value_t* recvbuff) const {
#if USE_NCCL
    NCCL_TRY(ncclAllGather((const void*)sendbuff, (void*)recvbuff, size, get_nccl_type<value_t>(), _nccl_comm, cudaStreamDefault));
#endif
}

template <typename value_t>
void Comm::allreduce (size_t size, value_t* sendbuff, value_t* recvbuff, ReduceOp reduce_op) const {
#if USE_NCCL
    NCCL_TRY(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, get_nccl_type<value_t>(), get_nccl_reduce_op(reduce_op), _nccl_comm, cudaStreamDefault));
#endif
}

} } //namespace

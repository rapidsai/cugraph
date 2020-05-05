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
#if ENABLE_OPG
#include <mpi.h>
#include <nccl.h>
#endif

namespace cugraph {
namespace experimental {

enum class ReduceOp { SUM, MAX, MIN };

// basic info about the snmg env setup
class Comm {
private:
  int _p{0};
  int _rank{0};
  bool _finalize_mpi{false};
  bool _finalize_nccl{false};

  int _device_id{0};
  int _device_count{0};

  int _sm_count_per_device{0};
  int _max_grid_dim_1D{0};
  int _max_block_dim_1D{0};
  int _l2_cache_size{0};
  int _shared_memory_size_per_sm{0};

#if ENABLE_OPG
  MPI_Comm _mpi_comm{};
  ncclComm_t _nccl_comm{};
#endif

public:
  Comm(){};
  Comm(int p);
#if ENABLE_OPG
  Comm(ncclComm_t comm, int size, int rank);
#endif
  ~Comm();
  int get_rank() const { return _rank; }
  int get_p() const { return _p; }
  int get_dev() const { return _device_id; }
  int get_dev_count() const { return _device_count; }
  int get_sm_count() const { return _sm_count_per_device; }
  bool is_master() const { return (_rank == 0) ? true : false; }

  void barrier();

  template <typename value_t>
  void allgather(size_t size, value_t *sendbuff, value_t *recvbuff) const;

  template <typename value_t>
  void allreduce(size_t size, value_t *sendbuff, value_t *recvbuff,
                 ReduceOp reduce_op) const;
};

} // namespace experimental
} // namespace cugraph

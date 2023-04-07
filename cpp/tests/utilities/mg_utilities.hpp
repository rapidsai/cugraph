/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>

#include <memory>

namespace cugraph {
namespace test {

void initialize_mpi(int argc, char** argv);

void finalize_mpi();

int query_mpi_comm_world_rank();
int query_mpi_comm_world_size();

std::unique_ptr<raft::handle_t> initialize_mg_handle(size_t pool_size = 64);

// NCCL lazily initializes for P2P, and this enforces P2P initialization for better performance
// measurements
void enforce_p2p_initialization(raft::comms::comms_t const& comm, rmm::cuda_stream_view stream);

}  // namespace test
}  // namespace cugraph

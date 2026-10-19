/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/handle.hpp>

#include <cuda/stream>

#include <memory>

namespace cugraph {
namespace test {

void initialize_mpi(int argc, char** argv);

void finalize_mpi();

int query_mpi_comm_world_rank();
int query_mpi_comm_world_size();

std::unique_ptr<raft::handle_t> initialize_mg_handle(
  size_t pool_size = 8 /* default value of CUDA_DEVICE_MAX_CONNECTIONS */);

// NCCL lazily initializes for P2P, and this enforces P2P initialization for better performance
// measurements
void enforce_p2p_initialization(raft::comms::comms_t const& comm, cuda::stream_ref stream);

}  // namespace test
}  // namespace cugraph

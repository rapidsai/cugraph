/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <string>

namespace cugraph {

/**
 * managed the mapping between graph partitioning and GPU partitioning
 */
class partition_manager {
 public:
  // we 2D partition both a graph adjacency matrix and the GPUs. The graph adjacency matrix is 2D
  // partitioned along the major axis and the minor axis. The GPUs are 2D partitioned to
  // gpu_col_comm_size * gpu_row_comm_size where gpu_col_comm_size is the size of the column
  // direction communicator (GPUs in the same column in the GPU 2D partitioning belongs to the same
  // column sub-communicator) and row_comm_size is the size of the row direction communicator (GPUs
  // in the same row belongs to the same row sub-communicator).  GPUs in the same GPU row
  // communicator have consecutive process IDs (and may be physically closer in hierarchical
  // interconnects). Graph algorithms require communications due to the graph adjacency matrix
  // partitioning along the major axis (major sub-communicator is responsible for this) and along
  // the minor axis (minor sub-communicator is responsible for this). This variable controls whether
  // to map the major sub-communicator to the GPU row communicator or the GPU column communicator.
  static constexpr bool map_major_comm_to_gpu_row_comm = false;  // FIXME: this is for benchmarking, reset to true before merging

#ifdef __CUDACC__
  __host__ __device__
#endif
    static int
    compute_global_comm_rank_from_vertex_partition_id(int major_comm_size,
                                                      int minor_comm_size,
                                                      int vertex_partition_id)
  {
    return map_major_comm_to_gpu_row_comm
             ? vertex_partition_id
             : (vertex_partition_id % major_comm_size) * minor_comm_size +
                 (vertex_partition_id / major_comm_size);
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
    static int
    compute_global_comm_rank_from_graph_subcomm_ranks(int major_comm_size,
                                                      int minor_comm_size,
                                                      int major_comm_rank,
                                                      int minor_comm_rank)
  {
    return map_major_comm_to_gpu_row_comm ? (minor_comm_rank * major_comm_size + major_comm_rank)
                                          : (major_comm_rank * minor_comm_size + minor_comm_rank);
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
    static int
    compute_major_comm_rank_from_global_comm_rank(int major_comm_size,
                                                  int minor_comm_size,
                                                  int comm_rank)
  {
    return map_major_comm_to_gpu_row_comm ? comm_rank % major_comm_size
                                          : comm_rank / minor_comm_size;
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
    static int
    compute_minor_comm_rank_from_global_comm_rank(int major_comm_size,
                                                  int minor_comm_size,
                                                  int comm_rank)
  {
    return map_major_comm_to_gpu_row_comm ? comm_rank / major_comm_size
                                          : comm_rank % minor_comm_size;
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
    static int
    compute_vertex_partition_id_from_graph_subcomm_ranks(int major_comm_size,
                                                         int minor_comm_size,
                                                         int major_comm_rank,
                                                         int minor_comm_rank)
  {
    return map_major_comm_to_gpu_row_comm
             ? compute_global_comm_rank_from_graph_subcomm_ranks(
                 major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank)
             : minor_comm_rank * major_comm_size + major_comm_rank;
  }

  static std::string major_comm_name()
  {
    return std::string(map_major_comm_to_gpu_row_comm ? "gpu_row_comm" : "gpu_col_comm");
  }

  static std::string minor_comm_name()
  {
    return std::string(map_major_comm_to_gpu_row_comm ? "gpu_col_comm" : "gpu_row_comm");
  }

  template <typename vertex_t>
  static std::vector<vertex_t> compute_partition_range_lasts(raft::handle_t const& handle,
                                                             vertex_t local_partition_size)
  {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto const major_comm_rank = major_comm.get_rank();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    auto const minor_comm_rank = minor_comm.get_rank();

    auto vertex_counts = host_scalar_allgather(comm, local_partition_size, handle.get_stream());
    auto vertex_partition_ids =
      host_scalar_allgather(comm,
                            partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
                              major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank),
                            handle.get_stream());

    std::vector<vertex_t> vertex_partition_range_offsets(comm_size + 1, 0);
    for (int i = 0; i < comm_size; ++i) {
      vertex_partition_range_offsets[vertex_partition_ids[i]] = vertex_counts[i];
    }
    std::exclusive_scan(vertex_partition_range_offsets.begin(),
                        vertex_partition_range_offsets.end(),
                        vertex_partition_range_offsets.begin(),
                        vertex_t{0});

    return std::vector<vertex_t>(vertex_partition_range_offsets.begin() + 1,
                                 vertex_partition_range_offsets.end());
  }

  static void init_subcomm(raft::handle_t& handle, int gpu_row_comm_size)
  {
    auto& comm = handle.get_comms();

    auto rank   = comm.get_rank();
    int row_idx = rank / gpu_row_comm_size;
    int col_idx = rank % gpu_row_comm_size;

    handle.set_subcomm("gpu_row_comm",
                       std::make_shared<raft::comms::comms_t>(comm.comm_split(row_idx, col_idx)));
    handle.set_subcomm("gpu_col_comm",
                       std::make_shared<raft::comms::comms_t>(comm.comm_split(col_idx, row_idx)));
  };
};

}  // namespace cugraph

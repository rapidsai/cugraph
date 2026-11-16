/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// cuGraph's exchange for a partitioned run (partition.cuh): one partition per rank (partition r on
// rank r), every collective of the communicator on the run's stream (NCCL orders a rank's
// collectives with its kernels, so no further synchronisation is needed). Data moves as bytes,
// bitmap words as uint32. Included by driver_common.cu only (the engine translation unit; see
// driver.hpp).

#include "community/detail/leiden/partition.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/core/comms.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cugraph::detail::leiden_engine {

class RaftExchange final : public Exchange {
 public:
  RaftExchange(raft::comms::comms_t const& comm, cudaStream_t s)
    : comm_(comm),
      s_(s),
      P_(comm.get_size()),
      rank_(comm.get_rank()),
      host_values_(static_cast<std::size_t>(P_) * kMaxHostValues * sizeof(i64), s)
  {
  }

  int parts() const override { return P_; }
  int n_local() const override { return 1; }
  int global(int) const override { return rank_; }

  void allgather_slices(void* const* bufs, int elem, const i64* bounds) override
  {
    std::vector<std::size_t> counts(P_), displs(P_);
    for (int q = 0; q < P_; ++q) {
      displs[q] = static_cast<std::size_t>(bounds[q]) * elem;
      counts[q] = static_cast<std::size_t>(bounds[q + 1] - bounds[q]) * elem;
    }
    auto* b = static_cast<std::uint8_t*>(bufs[0]);
    comm_.allgatherv(b + displs[rank_], b, counts.data(), displs.data(), s_);  // in place
  }

  void allgatherv(void* const* bufs, int elem, const i64* counts) override
  {
    std::vector<std::size_t> c(P_), displs(P_);
    blocks(elem, counts, c, displs);
    auto* b = static_cast<std::uint8_t*>(bufs[0]);
    comm_.allgatherv(b + displs[rank_], b, c.data(), displs.data(), s_);  // in place
  }

  void allgather_host(const i64* in, int k, i64* out) override
  {
    CUGRAPH_EXPECTS(k >= 0 && k <= kMaxHostValues,
                    "leiden: internal error: too many host values per partition.");
    const std::size_t bytes = static_cast<std::size_t>(k) * sizeof(i64);
    std::uint8_t* d         = host_values_.data();
    RAFT_CUDA_TRY(cudaMemcpyAsync(d + rank_ * bytes, in, bytes, cudaMemcpyHostToDevice, s_));
    comm_.allgather(d + rank_ * bytes, d, bytes, s_);
    RAFT_CUDA_TRY(cudaMemcpyAsync(out, d, P_ * bytes, cudaMemcpyDeviceToHost, s_));
    CUGRAPH_EXPECTS(comm_.sync_stream(s_) == raft::comms::status_t::SUCCESS,
                    "leiden: a collective of the partitioned run failed.");
  }

  // NCCL has no bitwise OR: every partition's words, then OR them locally.
  void allreduce_or(u32* const* in, u32* const* out, u32* const* scratch, i64 words) override
  {
    comm_.allgather(in[0], scratch[0], static_cast<std::size_t>(words), s_);
    part_or_kernel<<<grid_items(words), kBlock, 0, s_>>>(in[0], out[0], scratch[0], P_, words);
    RAFT_CHECK_CUDA(s_);
  }

  void gatherv(int root, const void* const* send, int elem, const i64* counts, void* recv) override
  {
    std::vector<std::size_t> c(P_), displs(P_);
    blocks(elem, counts, c, displs);
    comm_.gatherv(static_cast<const std::uint8_t*>(send[0]),
                  static_cast<std::uint8_t*>(recv),
                  c[rank_],
                  c.data(),
                  displs.data(),
                  root,
                  s_);
  }

  void bcast(int root, void* const* bufs, i64 bytes) override
  {
    comm_.bcast(static_cast<std::uint8_t*>(bufs[0]), static_cast<std::size_t>(bytes), root, s_);
  }

 private:
  // allgather_host: at most this many values per partition (partition.cuh sends P).
  static constexpr int kMaxHostValues = kMaxParts;

  raft::comms::comms_t const& comm_;
  cudaStream_t s_;
  int P_;
  int rank_;
  rmm::device_uvector<std::uint8_t> host_values_;  // P x kMaxHostValues i64

  // bytes and displacements of `counts[q]` elements of `elem` bytes per partition, in order
  void blocks(int elem,
              const i64* counts,
              std::vector<std::size_t>& c,
              std::vector<std::size_t>& displs) const
  {
    std::size_t off = 0;
    for (int q = 0; q < P_; ++q) {
      c[q]      = static_cast<std::size_t>(counts[q]) * elem;
      displs[q] = off;
      off += c[q];
    }
  }
};

}  // namespace cugraph::detail::leiden_engine

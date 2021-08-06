/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cugraph/utilities/host_barrier.hpp>

#include <vector>

namespace cugraph {

// FIXME: a temporary hack till UCC is integrated into RAFT (so we can use UCC barrier for DASK and
// MPI barrier for MPI)
void host_barrier(raft::comms::comms_t const& comm, rmm::cuda_stream_view stream_view)
{
  stream_view.synchronize();

  auto const comm_size = comm.get_size();
  auto const comm_rank = comm.get_rank();

  // k-tree barrier

  int constexpr k = 2;
  static_assert(k >= 2);
  std::vector<raft::comms::request_t> requests(k - 1);
  std::vector<std::byte> dummies(k - 1);

  // up

  int mod = 1;
  while (mod < comm_size) {
    if (comm_rank % mod == 0) {
      auto level_rank = comm_rank / mod;
      if (level_rank % k == 0) {
        auto num_irecvs = 0;
        ;
        for (int i = 1; i < k; ++i) {
          auto src_rank = (level_rank + i) * mod;
          if (src_rank < comm_size) {
            comm.irecv(dummies.data() + (i - 1),
                       sizeof(std::byte),
                       src_rank,
                       int{0} /* tag */,
                       requests.data() + (i - 1));
            ++num_irecvs;
          }
        }
        comm.waitall(num_irecvs, requests.data());
      } else {
        comm.isend(dummies.data(),
                   sizeof(std::byte),
                   (level_rank - (level_rank % k)) * mod,
                   int{0} /* tag */,
                   requests.data());
        comm.waitall(1, requests.data());
      }
    }
    mod *= k;
  }

  // down

  mod /= k;
  while (mod >= 1) {
    if (comm_rank % mod == 0) {
      auto level_rank = comm_rank / mod;
      if (level_rank % k == 0) {
        auto num_isends = 0;
        for (int i = 1; i < k; ++i) {
          auto dst_rank = (level_rank + i) * mod;
          if (dst_rank < comm_size) {
            comm.isend(dummies.data() + (i - 1),
                       sizeof(std::byte),
                       dst_rank,
                       int{0} /* tag */,
                       requests.data() + (i - 1));
            ++num_isends;
          }
        }
        comm.waitall(num_isends, requests.data());
      } else {
        comm.irecv(dummies.data(),
                   sizeof(std::byte),
                   (level_rank - (level_rank % k)) * mod,
                   int{0} /* tag */,
                   requests.data());
        comm.waitall(1, requests.data());
      }
    }
    mod /= k;
  }
}

}  // namespace cugraph

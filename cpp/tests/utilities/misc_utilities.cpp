/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <utilities/test_utilities.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <numeric>
#include <vector>

namespace cugraph {
namespace test {

std::string getFileName(const std::string& s)
{
  char sep = '/';
#ifdef _WIN32
  sep = '\\';
#endif
  size_t i = s.rfind(sep, s.length());
  if (i != std::string::npos) { return (s.substr(i + 1, s.length() - i)); }
  return ("");
}

void enforce_p2p_initialization(raft::handle_t const& handle)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  rmm::device_uvector<int32_t> tx_ints(comm_size, handle.get_stream());
  rmm::device_uvector<int32_t> rx_ints(comm_size, handle.get_stream());
  std::vector<size_t> tx_sizes(comm_size, size_t{1});
  std::vector<size_t> tx_offsets(comm_size);
  std::iota(tx_offsets.begin(), tx_offsets.end(), size_t{0});
  std::vector<int32_t> tx_ranks(comm_size);
  std::iota(tx_ranks.begin(), tx_ranks.end(), int32_t{0});
  auto rx_sizes   = tx_sizes;
  auto rx_offsets = tx_offsets;
  auto rx_ranks   = tx_ranks;

  comm.device_multicast_sendrecv(tx_ints.data(),
                                 tx_sizes,
                                 tx_offsets,
                                 tx_ranks,
                                 rx_ints.data(),
                                 rx_sizes,
                                 rx_offsets,
                                 rx_ranks,
                                 handle.get_stream());

  handle.sync_stream();
}

}  // namespace test
}  // namespace cugraph

/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "device_comm_wrapper.hpp"

#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <numeric>
#include <vector>

namespace cugraph {
namespace test {

template <typename T>
rmm::device_uvector<T> device_gatherv(raft::handle_t const& handle,
                                      raft::device_span<T const> d_input)

{
  bool is_root = handle.get_comms().get_rank() == int{0};
  auto rx_sizes =
    cugraph::host_scalar_gather(handle.get_comms(), d_input.size(), int{0}, handle.get_stream());
  std::vector<size_t> rx_displs(is_root ? static_cast<size_t>(handle.get_comms().get_size())
                                        : size_t{0});
  if (is_root) { std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1); }

  rmm::device_uvector<T> gathered_v(
    is_root ? std::reduce(rx_sizes.begin(), rx_sizes.end()) : size_t{0}, handle.get_stream());

  using comm_datatype_t = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
  cugraph::device_gatherv(handle.get_comms(),
                          reinterpret_cast<comm_datatype_t const*>(d_input.data()),
                          reinterpret_cast<comm_datatype_t*>(gathered_v.data()),
                          d_input.size(),
                          raft::host_span<size_t const>(rx_sizes.data(), rx_sizes.size()),
                          raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                          int{0},
                          handle.get_stream());

  return gathered_v;
}

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::device_span<T const> d_input)
{
  auto rx_sizes =
    cugraph::host_scalar_allgather(handle.get_comms(), d_input.size(), handle.get_stream());
  std::vector<size_t> rx_displs(static_cast<size_t>(handle.get_comms().get_size()));
  std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);

  rmm::device_uvector<T> gathered_v(std::reduce(rx_sizes.begin(), rx_sizes.end()),
                                    handle.get_stream());

  using comm_datatype_t = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
  cugraph::device_allgatherv(handle.get_comms(),
                             reinterpret_cast<comm_datatype_t const*>(d_input.data()),
                             reinterpret_cast<comm_datatype_t*>(gathered_v.data()),
                             raft::host_span<size_t const>(rx_sizes.data(), rx_sizes.size()),
                             raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                             handle.get_stream());

  return gathered_v;
}

// explicit instantiation

template rmm::device_uvector<bool> device_gatherv(raft::handle_t const& handle,
                                                  raft::device_span<bool const> d_input);

template rmm::device_uvector<int32_t> device_gatherv(raft::handle_t const& handle,
                                                     raft::device_span<int32_t const> d_input);

template rmm::device_uvector<int64_t> device_gatherv(raft::handle_t const& handle,
                                                     raft::device_span<int64_t const> d_input);

template rmm::device_uvector<size_t> device_gatherv(raft::handle_t const& handle,
                                                    raft::device_span<size_t const> d_input);

template rmm::device_uvector<float> device_gatherv(raft::handle_t const& handle,
                                                   raft::device_span<float const> d_input);

template rmm::device_uvector<double> device_gatherv(raft::handle_t const& handle,
                                                    raft::device_span<double const> d_input);

template rmm::device_uvector<bool> device_allgatherv(raft::handle_t const& handle,
                                                     raft::device_span<bool const> d_input);

template rmm::device_uvector<int32_t> device_allgatherv(raft::handle_t const& handle,
                                                        raft::device_span<int32_t const> d_input);

template rmm::device_uvector<int64_t> device_allgatherv(raft::handle_t const& handle,
                                                        raft::device_span<int64_t const> d_input);

template rmm::device_uvector<size_t> device_allgatherv(raft::handle_t const& handle,
                                                       raft::device_span<size_t const> d_input);

template rmm::device_uvector<float> device_allgatherv(raft::handle_t const& handle,
                                                      raft::device_span<float const> d_input);

template rmm::device_uvector<double> device_allgatherv(raft::handle_t const& handle,
                                                       raft::device_span<double const> d_input);

}  // namespace test
}  // namespace cugraph

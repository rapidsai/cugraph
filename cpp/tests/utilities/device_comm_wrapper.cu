/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "device_comm_wrapper.hpp"

#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

namespace cugraph {
namespace test {
namespace detail {

// RAFT comms only supports a fixed set of datatypes. gatherv only copy bytes, so transfer as
// uint8_t regardless of the caller's logical element type.
void device_gatherv(raft::handle_t const& handle,
                    raft::device_span<std::byte const> d_input,
                    raft::device_span<std::byte> d_output,
                    raft::host_span<size_t const> rx_sizes)
{
  bool is_root = handle.get_comms().get_rank() == int{0};

  std::vector<size_t> rx_displs{};
  if (is_root) {
    rx_displs.resize(rx_sizes.size());
    if (rx_sizes.size() > 0) {
      std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);
    }
  }

  auto const expected_output_size =
    is_root ? std::reduce(rx_sizes.begin(), rx_sizes.end()) : size_t{0};
  CUGRAPH_EXPECTS(d_output.size() == expected_output_size,
                  "device_gatherv: output span size must match the gathered byte count.");

  cugraph::device_gatherv(handle.get_comms(),
                          reinterpret_cast<uint8_t const*>(d_input.data()),
                          reinterpret_cast<uint8_t*>(d_output.data()),
                          d_input.size(),
                          rx_sizes,
                          raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                          int{0},
                          handle.get_stream());
}

// RAFT comms only supports a fixed set of datatypes. allgatherv only copy bytes, so transfer as
// uint8_t regardless of the caller's logical element type.
void device_allgatherv(raft::handle_t const& handle,
                       raft::device_span<std::byte const> d_input,
                       raft::device_span<std::byte> d_output,
                       raft::host_span<size_t const> rx_sizes)
{
  std::vector<size_t> rx_displs(rx_sizes.size());
  if (rx_sizes.size() > 0) {
    std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);
  }

  auto const expected_output_size = std::reduce(rx_sizes.begin(), rx_sizes.end());
  CUGRAPH_EXPECTS(d_output.size() == expected_output_size,
                  "device_allgatherv: output span size must match the gathered byte count.");

  cugraph::device_allgatherv(handle.get_comms(),
                             reinterpret_cast<uint8_t const*>(d_input.data()),
                             reinterpret_cast<uint8_t*>(d_output.data()),
                             rx_sizes,
                             raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                             handle.get_stream());
}

}  // namespace detail

template <typename T>
rmm::device_uvector<T> device_gatherv(raft::handle_t const& handle,
                                      raft::device_span<T const> d_input)
{
  raft::device_span<std::byte const> d_input_bytes{
    reinterpret_cast<std::byte const*>(d_input.data()), d_input.size() * sizeof(T)};

  bool is_root       = handle.get_comms().get_rank() == int{0};
  auto rx_byte_sizes = cugraph::host_scalar_gather(
    handle.get_comms(), d_input_bytes.size(), int{0}, handle.get_stream());
  auto const output_bytes =
    is_root ? std::reduce(rx_byte_sizes.begin(), rx_byte_sizes.end()) : size_t{0};

  rmm::device_uvector<T> gathered_v(output_bytes / sizeof(T), handle.get_stream());
  detail::device_gatherv(
    handle,
    d_input_bytes,
    raft::device_span<std::byte>{reinterpret_cast<std::byte*>(gathered_v.data()),
                                 gathered_v.size() * sizeof(T)},
    raft::host_span<size_t const>(rx_byte_sizes.data(), rx_byte_sizes.size()));
  return gathered_v;
}

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::device_span<T const> d_input)
{
  raft::device_span<std::byte const> d_input_bytes{
    reinterpret_cast<std::byte const*>(d_input.data()), d_input.size() * sizeof(T)};

  auto rx_byte_sizes =
    cugraph::host_scalar_allgather(handle.get_comms(), d_input_bytes.size(), handle.get_stream());
  auto const output_bytes = std::reduce(rx_byte_sizes.begin(), rx_byte_sizes.end());

  rmm::device_uvector<T> gathered_v(output_bytes / sizeof(T), handle.get_stream());
  detail::device_allgatherv(
    handle,
    d_input_bytes,
    raft::device_span<std::byte>{reinterpret_cast<std::byte*>(gathered_v.data()),
                                 gathered_v.size() * sizeof(T)},
    raft::host_span<size_t const>(rx_byte_sizes.data(), rx_byte_sizes.size()));
  return gathered_v;
}

// explicit instantiation

template rmm::device_uvector<bool> device_gatherv(raft::handle_t const& handle,
                                                  raft::device_span<bool const> d_input);

template rmm::device_uvector<int8_t> device_gatherv(raft::handle_t const& handle,
                                                    raft::device_span<int8_t const> d_input);

template rmm::device_uvector<int16_t> device_gatherv(raft::handle_t const& handle,
                                                     raft::device_span<int16_t const> d_input);

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

template rmm::device_uvector<int8_t> device_allgatherv(raft::handle_t const& handle,
                                                       raft::device_span<int8_t const> d_input);

template rmm::device_uvector<int16_t> device_allgatherv(raft::handle_t const& handle,
                                                        raft::device_span<int16_t const> d_input);

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

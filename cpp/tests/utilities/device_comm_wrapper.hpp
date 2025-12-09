/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace test {

template <typename T>
rmm::device_uvector<T> device_gatherv(raft::handle_t const& handle,
                                      raft::device_span<T const> d_input);

template <typename T>
rmm::device_uvector<T> device_gatherv(raft::handle_t const& handle, T const* d_input, size_t size)
{
  return device_gatherv(handle, raft::device_span<T const>{d_input, size});
}

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::device_span<T const> d_input);

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         T const* d_input,
                                         size_t size)
{
  return device_allgatherv(handle, raft::device_span<T const>{d_input, size});
}

}  // namespace test
}  // namespace cugraph

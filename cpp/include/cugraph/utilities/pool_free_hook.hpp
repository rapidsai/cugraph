/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <functional>

// How a memory-budgeted code path finds out how much device memory it may still spend.
//
// Asking CUDA is not enough. Once an RMM pool has claimed the device, cudaMemGetInfo reports almost
// nothing free even though most of the pool is unused, and libcugraph cannot look inside the pool
// itself: get_current_device_resource() is type-erased and the plain pool exposes neither its
// allocated byte count nor a portable downcast.
//
// So the answer has to come from whoever built the resource stack. An application that wants
// budgeted paths to see the real number wraps its pool in a statistics adaptor and installs a query
// here; library code reads it through query_pool_free_or() and falls back to the CUDA number when
// nothing was installed.
namespace cugraph {
namespace detail {

inline std::function<std::size_t()>& pool_free_hook()
{
  static std::function<std::size_t()> hook{};  // unset by default
  return hook;
}

inline std::size_t query_pool_free_or(std::size_t fallback)
{
  auto& h = pool_free_hook();
  return h ? h() : fallback;
}

}  // namespace detail
}  // namespace cugraph

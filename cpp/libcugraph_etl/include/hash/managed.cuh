/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MANAGED_CUH
#define MANAGED_CUH

#include <new>

struct managed {
  static void* operator new(size_t n)
  {
    void* ptr          = 0;
    cudaError_t result = cudaMallocManaged(&ptr, n);
    if (cudaSuccess != result || 0 == ptr) throw std::bad_alloc();
    return ptr;
  }

  static void operator delete(void* ptr) noexcept
  {
    auto const free_result = cudaFree(ptr);
    assert(free_result == cudaSuccess);
  }
};

inline bool isPtrManaged(cudaPointerAttributes attr)
{
  return (attr.type == cudaMemoryTypeManaged);
}

#endif  // MANAGED_CUH

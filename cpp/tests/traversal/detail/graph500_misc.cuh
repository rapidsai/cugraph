
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/functional>
#include <cuda/std/tuple>

#include <cstdint>

template <typename vertex_t>
struct hash_vertex_pair_t {
  using result_type = uint32_t;

  __device__ result_type operator()(cuda::std::tuple<vertex_t, vertex_t> const& pair) const
  {
    cuda::hash<vertex_t, cuda::hash_algorithm::murmurhash3_32> hash_func{};
    auto hash0 = hash_func(cuda::std::get<0>(pair));
    auto hash1 = hash_func(cuda::std::get<1>(pair));
    return hash0 + hash1;
  }
};


/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/tuple>

#include <cuco/hash_functions.cuh>

template <typename vertex_t>
struct hash_vertex_pair_t {
  using result_type = typename cuco::murmurhash3_32<vertex_t>::result_type;

  __device__ result_type operator()(cuda::std::tuple<vertex_t, vertex_t> const& pair) const
  {
    cuco::murmurhash3_32<vertex_t> hash_func{};
    auto hash0 = hash_func(cuda::std::get<0>(pair));
    auto hash1 = hash_func(cuda::std::get<1>(pair));
    return hash0 + hash1;
  }
};

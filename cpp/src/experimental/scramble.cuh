/* Copyright (C) 2009-2010 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

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
#pragma once

#include <cassert>
#include <cstdint>

/* Apply a permutation to scramble vertex numbers; a randomly generated
 * permutation is not used because applying it at scale is too expensive. */
template <typename vertex_t>
__device__ std::enable_if_t<sizeof(vertex_t) == 8, vertex_t> scramble(vertex_t value, size_t lgN)
{
  assert(std::is_unsigned<vertex_t>::value || lgN < 64);
  assert(value >= 0);

  constexpr uint64_t scramble_value0{606610977102444280};    // randomly generated
  constexpr uint64_t scramble_value1{11680327234415193037};  // randomly generated

  auto v = static_cast<uint64_t>(value);
  v += scramble_value0 + scramble_value1;
  v *= (scramble_value0 | uint64_t{0x4519840211493211});
  v = __brevll(v) >> (64 - lgN);
  v *= (scramble_value1 | uint64_t{0x3050852102C843A5});
  v = __brevll(v) >> (64 - lgN);
  return static_cast<vertex_t>(v);
}

/* Apply a permutation to scramble vertex numbers; a randomly generated
 * permutation is not used because applying it at scale is too expensive. */
template <typename vertex_t>
__device__ std::enable_if_t<sizeof(vertex_t) == 4, vertex_t> scramble(vertex_t value, size_t lgN)
{
  assert(std::is_unsigned<vertex_t>::value || lgN < 32);
  assert(value >= 0);

  constexpr uint32_t scramble_value0{282475248};   // randomly generated
  constexpr uint32_t scramble_value1{2617694917};  // randomly generated

  auto v = static_cast<uint32_t>(value);
  v += scramble_value0 + scramble_value1;
  v *= (scramble_value0 | uint32_t{0x11493211});
  v = __brev(v) >> (32 - lgN);
  v *= (scramble_value1 | uint32_t{0x02C843A5});
  v = __brev(v) >> (32 - lgN);
  return static_cast<vertex_t>(v);
}

/* Apply a permutation to scramble vertex numbers; a randomly generated
 * permutation is not used because applying it at scale is too expensive. */
template <typename vertex_t>
__device__ std::enable_if_t<sizeof(vertex_t) == 2, vertex_t> scramble(vertex_t value, size_t lgN)
{
  assert(std::is_unsigned<vertex_t>::value || lgN < 16);
  assert(value >= 0);

  constexpr uint32_t scramble_value0{0};     // randomly generated
  constexpr uint32_t scramble_value1{8620};  // randomly generated

  auto v = static_cast<uint16_t>(value);
  v += scramble_value0 + scramble_value1;
  v *= (scramble_value0 | uint16_t{0x3211});
  v = static_cast<uint16_t>(__brev(v) >> 16) >> (16 - lgN);
  v *= (scramble_value1 | uint16_t{0x43A5});
  v = static_cast<uint16_t>(__brev(v) >> 16) >> (16 - lgN);
  return static_cast<vertex_t>(v);
}

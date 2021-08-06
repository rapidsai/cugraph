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

namespace cugraph {
namespace detail {

template <typename uvertex_t>
__device__ std::enable_if_t<sizeof(uvertex_t) == 8, uvertex_t> bitreversal(uvertex_t value)
{
  return __brevll(value);
}

template <typename uvertex_t>
__device__ std::enable_if_t<sizeof(uvertex_t) == 4, uvertex_t> bitreversal(uvertex_t value)
{
  return __brev(value);
}

template <typename uvertex_t>
__device__ std::enable_if_t<sizeof(uvertex_t) == 2, uvertex_t> bitreversal(uvertex_t value)
{
  return static_cast<uvertex_t>(__brev(value) >> 16);
}

/* Apply a permutation to scramble vertex numbers; a randomly generated
 * permutation is not used because applying it at scale is too expensive. */
template <typename vertex_t>
__device__ vertex_t scramble(vertex_t value, size_t lgN)
{
  constexpr size_t number_of_bits = sizeof(vertex_t) * 8;

  static_assert((number_of_bits == 64) || (number_of_bits == 32) || (number_of_bits == 16));
  assert((std::is_unsigned<vertex_t>::value && lgN <= number_of_bits) ||
         (!std::is_unsigned<vertex_t>::value && lgN < number_of_bits));
  assert(value >= 0);

  using uvertex_t = typename std::make_unsigned<vertex_t>::type;

  constexpr auto scramble_value0 = static_cast<uvertex_t>(
    sizeof(vertex_t) == 8 ? 606610977102444280 : (sizeof(vertex_t) == 4 ? 282475248 : 0));
  constexpr auto scramble_value1 = static_cast<uvertex_t>(
    sizeof(vertex_t) == 8 ? 11680327234415193037 : (sizeof(vertex_t) == 4 ? 2617694917 : 8620));

  auto v = static_cast<uvertex_t>(value);
  v += scramble_value0 + scramble_value1;
  v *= (scramble_value0 | static_cast<uvertex_t>(0x4519840211493211));
  v = bitreversal(v) >> (number_of_bits - lgN);
  v *= (scramble_value1 | static_cast<uvertex_t>(0x3050852102C843A5));
  v = bitreversal(v) >> (number_of_bits - lgN);
  return static_cast<vertex_t>(v);
}

}  // namespace detail
}  // namespace cugraph

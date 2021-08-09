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

#include <type_traits>

namespace cugraph {

// primary template:
//
template <typename Src, typename... Types>
struct is_one_of;  // purposely empty

// partial specializations:
//
template <typename Src, typename Head, typename... Tail>
struct is_one_of<Src, Head, Tail...> {
  static constexpr bool value = std::is_same<Src, Head>::value || is_one_of<Src, Tail...>::value;
};

template <typename Src>
struct is_one_of<Src> {
  static constexpr bool value = false;
};

// meta-function that constrains
// vertex_t and edge_t template param candidates:
//
template <typename vertex_t, typename edge_t>
struct is_vertex_edge_combo {
  static constexpr bool value = is_one_of<vertex_t, int32_t, int64_t>::value &&
                                is_one_of<edge_t, int32_t, int64_t>::value &&
                                (sizeof(vertex_t) <= sizeof(edge_t));
};

// meta-function that constrains
// all 3 template param candidates:
//
template <typename vertex_t, typename edge_t, typename weight_t>
struct is_candidate {
  static constexpr bool value =
    is_vertex_edge_combo<vertex_t, edge_t>::value && is_one_of<weight_t, float, double>::value;
};

}  // namespace cugraph

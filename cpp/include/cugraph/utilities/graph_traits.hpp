/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
                                (sizeof(vertex_t) == sizeof(edge_t));
};

// meta-function that constrains
// vertex_t and edge_t template param candidates to only int32_t:
//
template <typename vertex_t, typename edge_t>
struct is_vertex_edge_combo_legacy {
  static constexpr bool value = is_one_of<vertex_t, int32_t>::value &&
                                is_one_of<edge_t, int32_t>::value &&
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

// meta-function that constrains
// vertex_t and edge_t are restricted to int32_t:
// FIXME: Drop this functor as it was only used by legacy K-Truss
//
template <typename vertex_t, typename edge_t, typename weight_t>
struct is_candidate_legacy {
  static constexpr bool value = is_vertex_edge_combo_legacy<vertex_t, edge_t>::value &&
                                is_one_of<weight_t, float, double>::value;
};

}  // namespace cugraph

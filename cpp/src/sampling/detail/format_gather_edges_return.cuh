/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cugraph/edge_property.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_properties_t, typename label_t>
struct format_gather_edges_return_t {
  using edge_properties_tup_type =
    std::conditional_t<std::is_same_v<edge_properties_t, cuda::std::nullopt_t>,
                       thrust::tuple<>,
                       std::conditional_t<std::is_arithmetic_v<edge_properties_t>,
                                          thrust::tuple<edge_properties_t>,
                                          edge_properties_t>>;

  using label_tuple_type = std::conditional_t<std::is_same_v<label_t, cuda::std::nullopt_t>,
                                              thrust::tuple<>,
                                              thrust::tuple<label_t>>;

  using return_type = decltype(cugraph::thrust_tuple_cat(
    thrust::tuple<vertex_t, vertex_t>{}, edge_properties_tup_type{}, label_tuple_type{}));

  return_type __device__ format_result(vertex_t src,
                                       vertex_t dst,
                                       edge_properties_t edge_properties,
                                       label_t label) const
  {
    edge_properties_tup_type edge_properties_tup{};

    if constexpr (!std::is_same_v<edge_properties_t, cuda::std::nullopt_t>) {
      if constexpr (std::is_arithmetic_v<edge_properties_t>) {
        thrust::get<0>(edge_properties_tup) = edge_properties;
      } else {
        edge_properties_tup = edge_properties;
      }
    }

    std::conditional_t<std::is_same_v<label_t, cuda::std::nullopt_t>,
                       thrust::tuple<>,
                       thrust::tuple<label_t>>
      label_tup{};
    if constexpr (!std::is_same_v<label_t, cuda::std::nullopt_t>) {
      thrust::get<0>(label_tup) = label;
    }
    return thrust_tuple_cat(thrust::make_tuple(src, dst), edge_properties_tup, label_tup);
  }
};

}  // namespace detail
}  // namespace cugraph

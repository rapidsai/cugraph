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

namespace {

template <typename TupleType>
auto constexpr concatenate_views(TupleType edge_properties)
{
  if constexpr (std::tuple_size_v<TupleType> == 0) {
    return cugraph::edge_dummy_property_view_t{};
  } else if constexpr (std::tuple_size_v<TupleType> == 1) {
    return std::get<0>(edge_properties);
  } else {
    return view_concat(edge_properties);
  }
}

template <size_t input_tuple_pos,
          size_t output_tuple_pos,
          bool Flag,
          bool... Flags,
          typename InputTupleType,
          typename OutputTupleType>
void move_results(InputTupleType& input_tuple, OutputTupleType& output_tuple)
{
  if constexpr (Flag) {
    std::get<output_tuple_pos>(output_tuple) = std::move(std::get<input_tuple_pos>(input_tuple));
  }

  if constexpr (sizeof...(Flags) > 0) {
    if constexpr (Flag) {
      move_results<input_tuple_pos + 1, output_tuple_pos + 1, Flags...>(input_tuple, output_tuple);
    } else {
      move_results<input_tuple_pos, output_tuple_pos + 1, Flags...>(input_tuple, output_tuple);
    }
  }
}

}  // namespace

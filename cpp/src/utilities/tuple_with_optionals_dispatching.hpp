/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/device_span.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

#if 0
template <size_t tuple_pos, use_element_t Flag, use_element_t... Flags, typename TupleType>
auto add_null_opts_impl(TupleType tuple)
{
  if constexpr (sizeof...(Flags) == 0) {
    if constexpr (Flag == use_optional)
      return std::make_tuple(std::make_optional(std::move(std::get<tuple_pos>(tuple))));
    else if constexpr (Flag == not_optional)
      return std::make_tuple(std::move(std::get<tuple_pos>(tuple)));
    else  // skip_optional
      return std::make_tuple(std::nullopt);
  } else {
    if constexpr (Flag == use_optional) {
      auto my_tuple_entry  = std::move(std::get<tuple_pos>(tuple));
      auto remaining_tuple = add_null_opts_impl<tuple_pos + 1, Flags...>(std::move(tuple));
      return std::tuple_cat(std::make_tuple(std::make_optional(std::move(my_tuple_entry))),
                            std::move(remaining_tuple));
    } else if constexpr (Flag == not_optional) {
      auto my_tuple_entry  = std::move(std::get<tuple_pos>(tuple));
      auto remaining_tuple = add_null_opts_impl<tuple_pos + 1, Flags...>(std::move(tuple));
      return std::tuple_cat(std::make_tuple(std::move(my_tuple_entry)), std::move(remaining_tuple));
    } else {
      // skip_optional
      return std::tuple_cat(std::make_tuple(std::nullopt),
                            add_null_opts_impl<tuple_pos, Flags...>(std::move(tuple)));
    }
  }
}
#endif

template <bool... Flags, typename Functor, typename TupleType, typename T1, typename... Ts>
auto tuple_with_optionals_dispatch_impl(Functor f, TupleType tuple, T1 t1, Ts... ts)
{
  if constexpr (sizeof...(Ts) == 0) {
    return t1.has_value() ? f.template operator()<Flags..., true>(
                              std::tuple_cat(tuple, std::move(std::make_tuple(std::move(*t1)))))
                          : f.template operator()<Flags..., false>(std::move(tuple));
  } else {
    return t1.has_value() ? tuple_with_optionals_dispatch_impl<Flags..., true>(
                              f, std::tuple_cat(tuple, std::make_tuple(std::move(*t1))), ts...)
                          : tuple_with_optionals_dispatch_impl<Flags..., false>(f, tuple, ts...);
  }
}

template <typename Functor, typename... Ts>
auto tuple_with_optionals_dispatch(Functor f, Ts... ts)
{
  return tuple_with_optionals_dispatch_impl(f, std::make_tuple(), ts...);
}

}  // namespace detail
}  // namespace cugraph

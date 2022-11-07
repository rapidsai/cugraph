/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

// Andrei Schaffer, aschaffer@nvidia.com
//

#pragma once

#include <cstdint>

namespace cugraph {
namespace visitors {

enum class DTypes { INT32 = 0, INT64, FLOAT32, FLOAT64, SIZE_T, NTYPES };

template <DTypes>
struct DMapType;

template <>
struct DMapType<DTypes::INT32> {
  using type = int32_t;
};

template <>
struct DMapType<DTypes::INT64> {
  using type = int64_t;
};

template <>
struct DMapType<DTypes::FLOAT32> {
  using type = float;
};

template <>
struct DMapType<DTypes::FLOAT64> {
  using type = double;
};

template <>
struct DMapType<DTypes::SIZE_T> {
  using type = size_t;
};

template <typename T>
struct reverse_dmap_t;

template <>
struct reverse_dmap_t<int32_t> {
  static constexpr DTypes type_id = DTypes::INT32;
};

template <>
struct reverse_dmap_t<int64_t> {
  static constexpr DTypes type_id = DTypes::INT64;
};

template <>
struct reverse_dmap_t<float> {
  static constexpr DTypes type_id = DTypes::FLOAT32;
};

template <>
struct reverse_dmap_t<double> {
  static constexpr DTypes type_id = DTypes::FLOAT64;
};

template <>
struct reverse_dmap_t<size_t> {
  static constexpr DTypes type_id = DTypes::SIZE_T;
};

}  // namespace visitors
}  // namespace cugraph

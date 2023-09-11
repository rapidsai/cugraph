/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph/mtmg/detail/device_shared_device_span.hpp>
#include <cugraph/mtmg/handle.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief An MTMG device span for storing a renumber map
 */
template <typename vertex_t>
using renumber_map_view_t = detail::device_shared_device_span_t<vertex_t const>;

}  // namespace mtmg
}  // namespace cugraph

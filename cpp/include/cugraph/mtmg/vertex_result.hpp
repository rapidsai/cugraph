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

#include <cugraph/mtmg/detail/device_shared_device_vector.hpp>
#include <cugraph/mtmg/vertex_result_view.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief An MTMG device vector for storing vertex results
 */
template <typename result_t>
class vertex_result_t : public detail::device_shared_device_vector_t<result_t> {
  using parent_t = detail::device_shared_device_vector_t<result_t>;

 public:
  /**
   * @brief Create a vertex result view (read only)
   */
  auto view() { return static_cast<vertex_result_view_t<result_t>>(this->parent_t::view()); }
};

}  // namespace mtmg
}  // namespace cugraph

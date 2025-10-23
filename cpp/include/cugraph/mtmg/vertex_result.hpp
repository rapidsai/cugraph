/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

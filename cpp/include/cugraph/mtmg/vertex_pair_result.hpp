/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_device_vector_tuple.hpp>
#include <cugraph/mtmg/vertex_pair_result_view.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief An MTMG device vector for storing vertex results
 */
template <typename vertex_t, typename result_t>
class vertex_pair_result_t
  : public detail::device_shared_device_vector_tuple_t<vertex_t, vertex_t, result_t> {
  using parent_t = detail::device_shared_device_vector_tuple_t<vertex_t, vertex_t, result_t>;

 public:
  /**
   * @brief Create a vertex result view (read only)
   */
  auto view() { return vertex_pair_result_view_t<vertex_t, result_t>(this->parent_t::view()); }
};

}  // namespace mtmg
}  // namespace cugraph

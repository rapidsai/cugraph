/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/export.hpp>
#include <cugraph/mtmg/detail/device_shared_device_vector.hpp>
#include <cugraph/mtmg/renumber_map_view.hpp>

namespace CUGRAPH_EXPORT cugraph {
namespace mtmg {

/**
 * @brief An MTMG device vector for storing a renumber map
 */
template <typename vertex_t>
class renumber_map_t : public detail::device_shared_device_vector_t<vertex_t> {
  using parent_t = detail::device_shared_device_vector_t<vertex_t>;

 public:
  /**
   * @brief Return a view (read only) of the renumber map
   */
  auto view() { return static_cast<renumber_map_view_t<vertex_t>>(this->parent_t::view()); }
};

}  // namespace mtmg
}  // namespace CUGRAPH_EXPORT cugraph

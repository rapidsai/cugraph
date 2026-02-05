/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>
#include <cugraph/mtmg/edge_property_view.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Edge property object for each GPU
 */
template <typename edge_t>
class edge_property_t
  : public detail::device_shared_wrapper_t<cugraph::edge_arithmetic_property_t<edge_t>> {
 public:
  using parent_t = detail::device_shared_wrapper_t<cugraph::edge_arithmetic_property_t<edge_t>>;

  /**
   * @brief Return a edge_property_view_t (read only)
   */
  auto view()
  {
    std::lock_guard<std::mutex> lock(parent_t::lock_);

    cugraph::edge_arithmetic_property_view_t<edge_t> result;

    std::for_each(parent_t::objects_.begin(), parent_t::objects_.end(), [&result](auto& p) {
      result.set(p.first,
                 cugraph::variant_type_dispatch(p.second, [](auto& p) { return p.view(); }));
    });

    return result;
  }
};

}  // namespace mtmg
}  // namespace cugraph

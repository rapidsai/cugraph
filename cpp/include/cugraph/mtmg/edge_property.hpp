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

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>
#include <cugraph/mtmg/edge_property_view.hpp>
#include <cugraph/mtmg/handle.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Edge property object for each GPU
 */
template <typename graph_view_t, typename property_t>
class edge_property_t : public detail::device_shared_wrapper_t<
                          cugraph::edge_property_t<typename graph_view_t::wrapped_t, property_t>> {
 public:
  using parent_t = detail::device_shared_wrapper_t<
    cugraph::edge_property_t<typename graph_view_t::wrapped_t, property_t>>;

  /**
   * @brief Return a edge_property_view_t (read only)
   */
  auto view()
  {
    std::lock_guard<std::mutex> lock(parent_t::lock_);

    using edge_t = typename graph_view_t::wrapped_t::edge_type;
    using buffer_t =
      typename cugraph::edge_property_t<typename graph_view_t::wrapped_t, property_t>::buffer_type;
    std::vector<buffer_t> buffers{};
    using const_value_iterator_t = decltype(get_dataframe_buffer_cbegin(buffers[0]));

    edge_property_view_t<edge_t, const_value_iterator_t> result;

    std::for_each(parent_t::objects_.begin(), parent_t::objects_.end(), [&result](auto& p) {
      result.set(p.first, p.second.view());
    });

    return result;
  }
};

}  // namespace mtmg
}  // namespace cugraph

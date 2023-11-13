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
#include <cugraph/mtmg/graph_view.hpp>
#include <cugraph/mtmg/handle.hpp>
#include <cugraph/mtmg/renumber_map.hpp>

#include <optional>

namespace cugraph {
namespace mtmg {

/**
 * @brief An MTMG device span for referencing a vertex result
 */
template <typename result_t>
class vertex_result_view_t : public detail::device_shared_device_span_t<result_t const> {
  using parent_t = detail::device_shared_device_span_t<result_t const>;

 public:
  vertex_result_view_t(parent_t&& other) : parent_t{std::move(other)} {}

  /**
   * @brief Gather results from specified vertices into a device vector
   */
  template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
  rmm::device_uvector<result_t> gather(
    handle_t const& handle,
    raft::device_span<vertex_t const> vertices,
    cugraph::mtmg::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
    std::optional<cugraph::mtmg::renumber_map_view_t<vertex_t>>& renumber_map_view);
};

}  // namespace mtmg
}  // namespace cugraph

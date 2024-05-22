/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cugraph/mtmg/detail/per_device_edgelist.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Edgelist object for each GPU
 */
template <typename vertex_t, typename weight_t, typename edge_t, typename edge_type_t>
class edgelist_t : public detail::device_shared_wrapper_t<
                     detail::per_device_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t>> {
 public:
  /**
   * @brief Create a per_device_edgelist for this GPU
   */
  void set(handle_t const& handle,
           size_t device_buffer_size,
           bool use_weight,
           bool use_edge_id,
           bool use_edge_type)
  {
    detail::per_device_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t> tmp(
      device_buffer_size, use_weight, use_edge_id, use_edge_type, handle.get_stream());

    detail::device_shared_wrapper_t<
      detail::per_device_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t>>::set(handle,
                                                                                   std::move(tmp));
  }

  /**
   * @brief Stop inserting edges into this edgelist so we can use the edges
   */
  void finalize_buffer(handle_t const& handle)
  {
    handle.sync_stream_pool();
    this->get(handle).finalize_buffer(handle.get_stream());
  }

  /**
   * @brief Consolidate for the edgelist edges into a single edgelist and then
   *        shuffle across GPUs.
   */
  void consolidate_and_shuffle(cugraph::mtmg::handle_t const& handle, bool store_transposed)
  {
    this->get(handle).consolidate_and_shuffle(handle, store_transposed);
  }
};

}  // namespace mtmg
}  // namespace cugraph

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
#include <cugraph/mtmg/handle.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Supports creating an edgelist from individual host threads
 *
 * A cugraph edgelist needs to contain all of the edges necessary to create the graph
 * stored in GPU memory (distributed across multiple GPUs in a multi-GPU configuration).
 *
 * This class provides a mechanism for populating the edgelist object from independent CPU threads.
 *
 * Calls to the append() method will take edges (in CPU host memory) and append them to a local
 * buffer.  As the local buffer fills, the buffer will be sent to GPU memory using the flush()
 * method.  This allows the CPU to GPU transfers to be larger (and consequently more efficient).
 */
template <typename vertex_t, typename weight_t, typename edge_t, typename edge_type_t>
class per_thread_edgelist_t {
 public:
  per_thread_edgelist_t()                             = delete;
  per_thread_edgelist_t(per_thread_edgelist_t const&) = delete;

  /**
   * @brief Only constructor
   *
   * @param edgelist            The edge list this thread_edgelist_t should be associated with
   * @param thread_buffer_size  Size of the local buffer for accumulating edges on the CPU
   */
  per_thread_edgelist_t(
    detail::per_device_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t>& edgelist,
    size_t thread_buffer_size)
    : edgelist_{edgelist},
      current_pos_{0},
      src_(thread_buffer_size),
      dst_(thread_buffer_size),
      wgt_{std::nullopt},
      edge_id_{std::nullopt},
      edge_type_{std::nullopt}
  {
    if (edgelist.use_weight()) wgt_ = std::make_optional(std::vector<weight_t>(thread_buffer_size));

    if (edgelist.use_edge_id())
      edge_id_ = std::make_optional(std::vector<edge_t>(thread_buffer_size));

    if (edgelist.use_edge_type())
      edge_type_ = std::make_optional(std::vector<edge_type_t>(thread_buffer_size));
  }

  /**
   * @brief Append an edge to the edge list
   *
   * @param handle     The resource handle
   * @param src        Source vertex id
   * @param dst        Destination vertex id
   * @param wgt        Edge weight
   * @param edge_id    Edge id
   * @param edge_type  Edge type
   */
  void append(handle_t const& handle,
              vertex_t src,
              vertex_t dst,
              std::optional<weight_t> wgt,
              std::optional<edge_t> edge_id,
              std::optional<edge_type_t> edge_type)
  {
    if (current_pos_ == src_.size()) { flush(handle); }

    src_[current_pos_] = src;
    dst_[current_pos_] = dst;
    if (wgt) (*wgt_)[current_pos_] = *wgt;
    if (edge_id) (*edge_id_)[current_pos_] = *edge_id;
    if (edge_type) (*edge_type_)[current_pos_] = *edge_type;

    ++current_pos_;
  }

  /**
   * @brief Append a list of edges to the edge list
   *
   * @param handle     The resource handle
   * @param src        Source vertex id
   * @param dst        Destination vertex id
   * @param wgt        Edge weight
   * @param edge_id    Edge id
   * @param edge_type  Edge type
   */
  void append(handle_t const& handle,
              raft::host_span<vertex_t const> src,
              raft::host_span<vertex_t const> dst,
              std::optional<raft::host_span<weight_t const>> wgt,
              std::optional<raft::host_span<edge_t const>> edge_id,
              std::optional<raft::host_span<edge_type_t const>> edge_type)
  {
    size_t count = src.size();
    size_t pos   = 0;

    while (count > 0) {
      size_t copy_count = std::min(count, (src_.size() - current_pos_));

      std::copy(src.begin() + pos, src.begin() + pos + copy_count, src_.begin() + current_pos_);
      std::copy(dst.begin() + pos, dst.begin() + pos + copy_count, dst_.begin() + current_pos_);
      if (wgt)
        std::copy(wgt.begin() + pos, wgt.begin() + pos + copy_count, wgt_->begin() + current_pos_);
      if (edge_id)
        std::copy(edge_id.begin() + pos,
                  edge_id.begin() + pos + copy_count,
                  edge_id_->begin() + current_pos_);
      if (edge_type)
        std::copy(edge_type.begin() + pos,
                  edge_type.begin() + pos + copy_count,
                  edge_type_->begin() + current_pos_);

      if (current_pos_ == src_.size()) { flush(handle); }

      count -= copy_count;
      pos += copy_count;
    }
  }

  /**
   * @brief Flush thread data from host to GPU memory
   *
   * @param handle     The resource handle
   * @param sync       If true, synchronize the asynchronous copy of data;
   *                   defaults to false.
   */
  void flush(handle_t const& handle, bool sync = false)
  {
    edgelist_.append(
      handle.get_stream(),
      raft::host_span<vertex_t const>{src_.data(), current_pos_},
      raft::host_span<vertex_t const>{dst_.data(), current_pos_},
      wgt_ ? std::make_optional(raft::host_span<weight_t const>{wgt_->data(), current_pos_})
           : std::nullopt,
      edge_id_ ? std::make_optional(raft::host_span<edge_t const>{edge_id_->data(), current_pos_})
               : std::nullopt,
      edge_type_
        ? std::make_optional(raft::host_span<edge_type_t const>{edge_type_->data(), current_pos_})
        : std::nullopt);

    current_pos_ = 0;

    if (sync) handle.sync_stream();
  }

 private:
  detail::per_device_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t>& edgelist_;
  size_t current_pos_{0};
  std::vector<vertex_t> src_{};
  std::vector<vertex_t> dst_{};
  std::optional<std::vector<weight_t>> wgt_{};
  std::optional<std::vector<edge_t>> edge_id_{};
  std::optional<std::vector<edge_type_t>> edge_type_{};
};

}  // namespace mtmg
}  // namespace cugraph

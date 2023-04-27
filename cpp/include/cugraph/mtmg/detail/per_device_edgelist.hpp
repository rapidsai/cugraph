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

#include <cugraph/mtmg/handle.hpp>

// FIXME: Could use std::span once compiler supports C++20
#include <raft/core/span.hpp>

namespace cugraph {
namespace mtmg {
namespace detail {

/**
 * @brief An edgelist for each GPU
 *
 * Manages an edge list for edges associated with a particular GPU.  Multiple threads
 * can call the append() method, possibly concurrently.  To avoid constantly copying
 * when the buffers fill up, the class will create a device buffer containing a
 * number of elements specified in the constructor.  When that device buffer is full
 * we will create a new buffer.
 *
 * When we try and use the edgelist we will consolidate the buffers, since at that
 * time we know the entire size required.
 */
template <typename vertex_t, typename weight_t, typename edge_t, typename edge_type_t>
class per_device_edgelist_t {
 public:
  per_device_edgelist_t(cugraph::mtmg::handle_t const& handle,
                        size_t device_buffer_size,
                        bool use_weight,
                        bool use_edge_id,
                        bool use_edge_type)
    : device_buffer_size_{device_buffer_size},
      current_pos_{0},
      src_{},
      dst_{},
      wgt_{std::nullopt},
      edge_id_{std::nullopt},
      edge_type_{std::nullopt}
  {
    if (use_weight) { wgt_ = std::make_optional(std::vector<rmm::device_uvector<weight_t>>()); }

    if (use_edge_id) { edge_id_ = std::make_optional(std::vector<rmm::device_uvector<edge_t>>()); }

    if (use_edge_type) {
      edge_type_ = std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>());
    }

    create_new_buffers(handle);
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
    // FIXME:  This lock guard could be on a smaller region, but it
    //   would require more careful coding.  The raft::update_device
    //   calls could be done without the lock if we made a local
    //   of the values of *.back() and did an increment of current_pos_
    //   while we hold the lock.
    std::lock_guard<std::mutex> lock(lock_);

    size_t count = src.size();
    size_t pos   = 0;

    while (count > 0) {
      size_t copy_count = std::min(count, (src_.back().size() - current_pos_));

      raft::update_device(src_.back().begin() + current_pos_,
                          src.begin() + pos,
                          copy_count,
                          handle.raft_handle().get_stream());
      raft::update_device(dst_.back().begin() + current_pos_,
                          dst.begin() + pos,
                          copy_count,
                          handle.raft_handle().get_stream());
      if (wgt)
        raft::update_device(wgt_->back().begin() + current_pos_,
                            wgt->begin() + pos,
                            copy_count,
                            handle.raft_handle().get_stream());
      if (edge_id)
        raft::update_device(edge_id_->back().begin() + current_pos_,
                            edge_id->begin() + pos,
                            copy_count,
                            handle.raft_handle().get_stream());
      if (edge_type)
        raft::update_device(edge_type_->back().begin() + current_pos_,
                            edge_type->begin() + pos,
                            copy_count,
                            handle.raft_handle().get_stream());

      if (current_pos_ == src_.size()) { create_new_buffers(handle); }

      count -= copy_count;
      pos += copy_count;
    }
  }

  bool use_weight() const { return wgt_.has_value(); }

  bool use_edge_id() const { return edge_id_.has_value(); }

  bool use_edge_type() const { return edge_type_.has_value(); }

  std::vector<rmm::device_uvector<vertex_t>>& get_src() { return src_; }
  std::vector<rmm::device_uvector<vertex_t>>& get_dst() { return dst_; }
  std::optional<std::vector<rmm::device_uvector<weight_t>>>& get_wgt() { return wgt_; }
  std::optional<std::vector<rmm::device_uvector<edge_t>>>& get_edge_id() { return edge_id_; }
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>& get_edge_type()
  {
    return edge_type_;
  }

 private:
  void create_new_buffers(cugraph::mtmg::handle_t const& handle)
  {
    src_.emplace_back(device_buffer_size_, handle.raft_handle().get_stream());
    dst_.emplace_back(device_buffer_size_, handle.raft_handle().get_stream());

    if (wgt_) { wgt_->emplace_back(device_buffer_size_, handle.raft_handle().get_stream()); }

    if (edge_id_) {
      edge_id_->emplace_back(device_buffer_size_, handle.raft_handle().get_stream());
    }

    if (edge_type_) {
      edge_type_->emplace_back(device_buffer_size_, handle.raft_handle().get_stream());
    }

    current_pos_ = 0;
  }

  mutable std::mutex lock_{};

  size_t current_pos_{0};
  size_t device_buffer_size_;

  std::vector<rmm::device_uvector<vertex_t>> src_;
  std::vector<rmm::device_uvector<vertex_t>> dst_;
  std::optional<std::vector<rmm::device_uvector<weight_t>>> wgt_;
  std::optional<std::vector<rmm::device_uvector<edge_t>>> edge_id_;
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_type_;
};

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph

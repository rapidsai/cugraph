/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <cugraph/shuffle_functions.hpp>

// FIXME: Could use std::span once compiler supports C++20
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

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
 *
 * Important note, the expectation is that this object will be used in two phases:
 *  1) The append() method will be used to fill buffers with edges
 *  2) The edges will be consumed to create a graph
 *
 * These two phases are expected to be disjoint.  The calling process is expected to
 * manage some barrier so that all threads are guaranteed to be completed before changing
 * phases.  If an append() call (part of the filling phase) overlaps with calls to
 * finalize_buffer(), consolidate_and_shuffle(), get_src(), get_dst(), get_wgt(),
 * get_edge_id() and get_edge_type() then the behavior is undefined (data might change
 * in some non-deterministic way).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_t,
          typename edge_type_t,
          typename edge_time_t>
class per_device_edgelist_t {
 public:
  per_device_edgelist_t()                                        = delete;
  per_device_edgelist_t(per_device_edgelist_t const&)            = delete;
  per_device_edgelist_t& operator=(per_device_edgelist_t const&) = delete;
  per_device_edgelist_t& operator=(per_device_edgelist_t&&)      = delete;

  /**
   * @brief Construct a new per device edgelist t object
   *
   * @param device_buffer_size  Number of edges to store in each device buffer
   * @param use_weight          Whether or not the edgelist will have weights
   * @param use_edge_id         Whether or not the edgelist will have edge ids
   * @param use_edge_type       Whether or not the edgelist will have edge types
   * @param use_edge_start_time Whether or not the edgelist will have edge start times
   * @param use_edge_end_time   Whether or not the edgelist will have edge end times
   * @param stream_view         CUDA stream view
   */
  per_device_edgelist_t(size_t device_buffer_size,
                        bool use_weight,
                        bool use_edge_id,
                        bool use_edge_type,
                        bool use_edge_start_time,
                        bool use_edge_end_time,
                        rmm::cuda_stream_view stream_view)
    : device_buffer_size_{device_buffer_size},
      current_pos_{0},
      src_{},
      dst_{},
      wgt_{std::nullopt},
      edge_id_{std::nullopt},
      edge_type_{std::nullopt},
      edge_start_time_{std::nullopt},
      edge_end_time_{std::nullopt}
  {
    if (use_weight) { wgt_ = std::make_optional(std::vector<rmm::device_uvector<weight_t>>()); }

    if (use_edge_id) { edge_id_ = std::make_optional(std::vector<rmm::device_uvector<edge_t>>()); }

    if (use_edge_type) {
      edge_type_ = std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>());
    }

    if (use_edge_start_time) {
      edge_start_time_ = std::make_optional(std::vector<rmm::device_uvector<edge_time_t>>());
    }

    if (use_edge_end_time) {
      edge_end_time_ = std::make_optional(std::vector<rmm::device_uvector<edge_time_t>>());
    }

    create_new_buffers(stream_view);
  }

  /**
   * @brief Move construct a new per device edgelist t object
   *
   * @param other Object to move into this instance
   */
  per_device_edgelist_t(per_device_edgelist_t&& other)
    : device_buffer_size_{other.device_buffer_size_},
      current_pos_{other.current_pos_},
      src_{std::move(other.src_)},
      dst_{std::move(other.dst_)},
      wgt_{std::move(other.wgt_)},
      edge_id_{std::move(other.edge_id_)},
      edge_type_{std::move(other.edge_type_)},
      edge_start_time_{std::move(other.edge_start_time_)},
      edge_end_time_{std::move(other.edge_end_time_)}
  {
  }

  /**
   * @brief Append a list of edges to the edge list
   *
   * @param src             Source vertex id
   * @param dst             Destination vertex id
   * @param wgt             Edge weight
   * @param edge_id         Edge id
   * @param edge_type       Edge type
   * @param edge_start_time Edge start time
   * @param edge_end_time   Edge end time
   * @param stream_view     CUDA stream view
   */
  void append(raft::host_span<vertex_t const> src,
              raft::host_span<vertex_t const> dst,
              std::optional<raft::host_span<weight_t const>> wgt,
              std::optional<raft::host_span<edge_t const>> edge_id,
              std::optional<raft::host_span<edge_type_t const>> edge_type,
              std::optional<raft::host_span<edge_time_t const>> edge_start_time,
              std::optional<raft::host_span<edge_time_t const>> edge_end_time,
              rmm::cuda_stream_view stream_view)
  {
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> copy_positions;

    {
      std::lock_guard<std::mutex> lock(lock_);

      size_t count = src.size();
      size_t pos   = 0;

      while (count > 0) {
        size_t copy_count = std::min(count, (src_.back().size() - current_pos_));

        copy_positions.push_back(std::make_tuple(src_.size() - 1, current_pos_, pos, copy_count));

        count -= copy_count;
        pos += copy_count;
        current_pos_ += copy_count;

        if (current_pos_ == src_.back().size()) { create_new_buffers(stream_view); }
      }
    }

    std::for_each(copy_positions.begin(),
                  copy_positions.end(),
                  [&stream_view,
                   &this_src = src_,
                   &src,
                   &this_dst = dst_,
                   &dst,
                   &this_wgt = wgt_,
                   &wgt,
                   &this_edge_id = edge_id_,
                   &edge_id,
                   &this_edge_type = edge_type_,
                   &edge_type,
                   &this_edge_start_time = edge_start_time_,
                   &edge_start_time,
                   &this_edge_end_time = edge_end_time_,
                   &edge_end_time](auto tuple) {
                    auto [buffer_idx, buffer_pos, input_pos, copy_count] = tuple;

                    raft::update_device(this_src[buffer_idx].begin() + buffer_pos,
                                        src.begin() + input_pos,
                                        copy_count,
                                        stream_view);

                    raft::update_device(this_dst[buffer_idx].begin() + buffer_pos,
                                        dst.begin() + input_pos,
                                        copy_count,
                                        stream_view);

                    if (this_wgt)
                      raft::update_device((*this_wgt)[buffer_idx].begin() + buffer_pos,
                                          wgt->begin() + input_pos,
                                          copy_count,
                                          stream_view);

                    if (this_edge_id)
                      raft::update_device((*this_edge_id)[buffer_idx].begin() + buffer_pos,
                                          edge_id->begin() + input_pos,
                                          copy_count,
                                          stream_view);

                    if (this_edge_type)
                      raft::update_device((*this_edge_type)[buffer_idx].begin() + buffer_pos,
                                          edge_type->begin() + input_pos,
                                          copy_count,
                                          stream_view);

                    if (this_edge_start_time)
                      raft::update_device((*this_edge_start_time)[buffer_idx].begin() + buffer_pos,
                                          edge_start_time->begin() + input_pos,
                                          copy_count,
                                          stream_view);

                    if (this_edge_end_time)
                      raft::update_device((*this_edge_end_time)[buffer_idx].begin() + buffer_pos,
                                          edge_end_time->begin() + input_pos,
                                          copy_count,
                                          stream_view);
                  });
  }

  /**
   * @brief  Mark the edgelist as ready for reading (all writes are complete)
   *
   * @param stream_view  CUDA stream view
   */
  void finalize_buffer(rmm::cuda_stream_view stream_view)
  {
    src_.back().resize(current_pos_, stream_view);
    dst_.back().resize(current_pos_, stream_view);
    if (wgt_) wgt_->back().resize(current_pos_, stream_view);
    if (edge_id_) edge_id_->back().resize(current_pos_, stream_view);
    if (edge_type_) edge_type_->back().resize(current_pos_, stream_view);
    if (edge_start_time_) edge_start_time_->back().resize(current_pos_, stream_view);
    if (edge_end_time_) edge_end_time_->back().resize(current_pos_, stream_view);
  }

  bool use_weight() const { return wgt_.has_value(); }

  bool use_edge_id() const { return edge_id_.has_value(); }

  bool use_edge_type() const { return edge_type_.has_value(); }

  bool use_edge_start_time() const { return edge_start_time_.has_value(); }

  bool use_edge_end_time() const { return edge_end_time_.has_value(); }

  std::vector<rmm::device_uvector<vertex_t>>& get_src() { return src_; }
  std::vector<rmm::device_uvector<vertex_t>>& get_dst() { return dst_; }
  std::optional<std::vector<rmm::device_uvector<weight_t>>>& get_wgt() { return wgt_; }
  std::optional<std::vector<rmm::device_uvector<edge_t>>>& get_edge_id() { return edge_id_; }

  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>& get_edge_type()
  {
    return edge_type_;
  }

  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>& get_edge_start_time()
  {
    return edge_start_time_;
  }

  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>& get_edge_end_time()
  {
    return edge_end_time_;
  }

  /**
   * @brief Consolidate edgelists (if necessary) and shuffle to the proper GPU
   *
   * @param handle    The resource handle
   */
  void consolidate_and_shuffle(cugraph::mtmg::handle_t const& handle, bool store_transposed)
  {
    if (src_.size() > 1) {
      auto stream = handle.raft_handle().get_stream();

      size_t total_size = std::transform_reduce(
        src_.begin(), src_.end(), size_t{0}, std::plus<size_t>(), [](auto& d_vector) {
          return d_vector.size();
        });

      resize_and_copy_buffers(src_, total_size, stream);
      resize_and_copy_buffers(dst_, total_size, stream);
      if (wgt_) resize_and_copy_buffers(*wgt_, total_size, stream);
      if (edge_id_) resize_and_copy_buffers(*edge_id_, total_size, stream);
      if (edge_type_) resize_and_copy_buffers(*edge_type_, total_size, stream);
      if (edge_start_time_) resize_and_copy_buffers(*edge_start_time_, total_size, stream);
      if (edge_end_time_) resize_and_copy_buffers(*edge_end_time_, total_size, stream);
    }

    std::vector<cugraph::arithmetic_device_uvector_t> tmp_edge_properties{};
    if (wgt_) tmp_edge_properties.push_back(std::move((*wgt_)[0]));
    if (edge_id_) tmp_edge_properties.push_back(std::move((*edge_id_)[0]));
    if (edge_type_) tmp_edge_properties.push_back(std::move((*edge_type_)[0]));
    if (edge_start_time_) tmp_edge_properties.push_back(std::move((*edge_start_time_)[0]));
    if (edge_end_time_) tmp_edge_properties.push_back(std::move((*edge_end_time_)[0]));

    std::tie(src_[0], dst_[0], tmp_edge_properties) =
      cugraph::shuffle_ext_edges(handle.raft_handle(),
                                 std::move(src_[0]),
                                 std::move(dst_[0]),
                                 std::move(tmp_edge_properties),
                                 store_transposed);

    size_t pos{0};
    if (wgt_)
      ((*wgt_)[0]) = std::move(std::get<rmm::device_uvector<weight_t>>(tmp_edge_properties[pos++]));
    if (edge_id_)
      ((*edge_id_)[0]) =
        std::move(std::get<rmm::device_uvector<edge_t>>(tmp_edge_properties[pos++]));
    if (edge_type_)
      ((*edge_type_)[0]) =
        std::move(std::get<rmm::device_uvector<edge_type_t>>(tmp_edge_properties[pos++]));
    if (edge_start_time_)
      ((*edge_start_time_)[0]) =
        std::move(std::get<rmm::device_uvector<edge_time_t>>(tmp_edge_properties[pos++]));
    if (edge_end_time_)
      ((*edge_end_time_)[0]) =
        std::move(std::get<rmm::device_uvector<edge_time_t>>(tmp_edge_properties[pos++]));
  }

 private:
  template <typename T>
  void resize_and_copy_buffers(std::vector<rmm::device_uvector<T>>& buffer,
                               size_t total_size,
                               rmm::cuda_stream_view stream)
  {
    size_t pos = buffer[0].size();
    buffer[0].resize(total_size, stream);

    for (size_t i = 1; i < buffer.size(); ++i) {
      raft::copy(buffer[0].data() + pos, buffer[i].data(), buffer[i].size(), stream);
      pos += buffer[i].size();
      buffer[i].resize(0, stream);
      buffer[i].shrink_to_fit(stream);
    }

    std::vector<rmm::device_uvector<T>> new_buffer;
    new_buffer.push_back(std::move(buffer[0]));
    buffer = std::move(new_buffer);
  }

  void create_new_buffers(rmm::cuda_stream_view stream_view)
  {
    src_.emplace_back(device_buffer_size_, stream_view);
    dst_.emplace_back(device_buffer_size_, stream_view);

    if (wgt_) { wgt_->emplace_back(device_buffer_size_, stream_view); }

    if (edge_id_) { edge_id_->emplace_back(device_buffer_size_, stream_view); }

    if (edge_type_) { edge_type_->emplace_back(device_buffer_size_, stream_view); }

    if (edge_start_time_) { edge_start_time_->emplace_back(device_buffer_size_, stream_view); }

    if (edge_end_time_) { edge_end_time_->emplace_back(device_buffer_size_, stream_view); }

    current_pos_ = 0;
  }

  mutable std::mutex lock_{};

  size_t current_pos_{0};
  size_t device_buffer_size_{0};

  std::vector<rmm::device_uvector<vertex_t>> src_{};
  std::vector<rmm::device_uvector<vertex_t>> dst_{};
  std::optional<std::vector<rmm::device_uvector<weight_t>>> wgt_{};
  std::optional<std::vector<rmm::device_uvector<edge_t>>> edge_id_{};
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_type_{};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edge_start_time_{};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edge_end_time_{};
};

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph

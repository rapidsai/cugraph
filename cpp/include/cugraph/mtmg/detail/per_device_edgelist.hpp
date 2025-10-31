/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/mtmg/handle.hpp>
#include <cugraph/shuffle_functions.hpp>

// FIXME: Could use std::span once compiler supports C++20
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace mtmg {

using arithmetic_host_vector_t = std::variant<std::monostate,
                                              std::vector<float>,
                                              std::vector<double>,
                                              std::vector<int32_t>,
                                              std::vector<int64_t>,
                                              std::vector<size_t>>;

using arithmetic_const_host_span_t = std::variant<std::monostate,
                                                  raft::host_span<float const>,
                                                  raft::host_span<double const>,
                                                  raft::host_span<int32_t const>,
                                                  raft::host_span<int64_t const>,
                                                  raft::host_span<size_t const>>;

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
 * finalize_buffer(), consolidate_and_shuffle(), get_src(), get_dst(), get_edge_property_buffers(),
 * then the behavior is undefined (data might change
 * in some non-deterministic way).
 */
template <typename vertex_t>
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
   * @param edge_property_types  List of types of edge properties to store
   * @param stream_view         CUDA stream view
   */
  per_device_edgelist_t(size_t device_buffer_size,
                        std::vector<arithmetic_type_t> const& edge_property_types,
                        rmm::cuda_stream_view stream_view)
    : device_buffer_size_{device_buffer_size},
      current_pos_{0},
      src_{},
      dst_{},
      edge_property_types_{edge_property_types},
      edge_property_buffers_{}
  {
    for (auto const& edge_property_type : edge_property_types_) {
      edge_property_buffers_.emplace_back(std::vector<cugraph::arithmetic_device_uvector_t>());
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
      edge_property_types_{std::move(other.edge_property_types_)},
      edge_property_buffers_{std::move(other.edge_property_buffers_)}
  {
  }

  /**
   * @brief Append a list of edges to the edge list
   *
   * @param src             Source vertex id
   * @param dst             Destination vertex id
   * @param edge_properties Edge properties
   * @param stream_view     CUDA stream view
   */
  void append(raft::host_span<vertex_t const> src,
              raft::host_span<vertex_t const> dst,
              raft::host_span<arithmetic_const_host_span_t> edge_properties,
              rmm::cuda_stream_view stream_view)
  {
    CUGRAPH_EXPECTS(edge_properties.size() == edge_property_buffers_.size(),
                    "Edge property count mismatch");

    for (size_t i = 0; i < edge_properties.size(); ++i) {
      CUGRAPH_EXPECTS(edge_properties[i].index() == edge_property_buffers_[i][0].index(),
                      "Edge property type mismatch");
    }

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

    std::for_each(
      copy_positions.begin(),
      copy_positions.end(),
      [&stream_view,
       &this_src = src_,
       &src,
       &this_dst = dst_,
       &dst,
       &edge_property_buffers = edge_property_buffers_,
       &edge_properties](auto const& tuple) {
        auto [buffer_idx, buffer_pos, input_pos, copy_count] = tuple;

        raft::update_device(this_src[buffer_idx].begin() + buffer_pos,
                            src.begin() + input_pos,
                            copy_count,
                            stream_view);

        raft::update_device(this_dst[buffer_idx].begin() + buffer_pos,
                            dst.begin() + input_pos,
                            copy_count,
                            stream_view);

        for (size_t i = 0; i < edge_properties.size(); ++i) {
          auto& edge_property_buffer = edge_property_buffers[i][buffer_idx];
          auto& edge_property_value  = edge_properties[i];
          variant_type_dispatch(
            edge_property_value,
            [&edge_property_buffer,
             buffer_pos  = buffer_pos,
             input_pos   = input_pos,
             copy_count  = copy_count,
             stream_view = stream_view](auto edge_property_value) {
              using T = typename std::decay_t<decltype(edge_property_value)>::value_type;
              raft::update_device(
                std::get<rmm::device_uvector<T>>(edge_property_buffer).begin() + buffer_pos,
                edge_property_value.begin() + input_pos,
                copy_count,
                stream_view);
            });
        }
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

    std::for_each(edge_property_buffers_.begin(),
                  edge_property_buffers_.end(),
                  [current_pos = current_pos_, stream_view](auto& edge_property_buffer) {
                    cugraph::variant_type_dispatch(
                      edge_property_buffer.back(),
                      [current_pos, stream_view](auto& edge_property_buffer) {
                        edge_property_buffer.resize(current_pos, stream_view);
                      });
                  });
  }

  /**
   * @brief Get the source vertices
   *
   * @return std::vector<rmm::device_uvector<vertex_t>>&
   */
  std::vector<rmm::device_uvector<vertex_t>>& get_src() { return src_; }

  /**
   * @brief Get the destination vertices
   *
   * @return std::vector<rmm::device_uvector<vertex_t>>&
   */
  std::vector<rmm::device_uvector<vertex_t>>& get_dst() { return dst_; }

  /**
   * @brief Get the edge property types
   *
   * @return std::vector<arithmetic_type_t>&
   */
  std::vector<arithmetic_type_t> const& get_edge_property_types() const
  {
    return edge_property_types_;
  }

  /**
   * @brief Get the edge property buffers
   *
   * @return std::vector<std::vector<rmm::device_uvector<arithmetic_type_t>>>&
   */
  std::vector<std::vector<cugraph::arithmetic_device_uvector_t>>& get_edge_property_buffers()
  {
    return edge_property_buffers_;
  }

  /**
   * @brief Consolidate edgelists (if necessary) and shuffle to the proper GPU
   *
   * @param handle    The resource handle
   */
  void consolidate_and_shuffle(cugraph::mtmg::handle_t const& handle, bool store_transposed)
  {
    std::vector<cugraph::arithmetic_device_uvector_t> consolidated_edge_property_buffers{};
    if (src_.size() > 1) {
      auto stream = handle.raft_handle().get_stream();

      size_t total_size = std::transform_reduce(
        src_.begin(), src_.end(), size_t{0}, std::plus<size_t>(), [](auto& d_vector) {
          return d_vector.size();
        });

      src_ = resize_and_copy_buffers(std::move(src_), total_size, stream);
      dst_ = resize_and_copy_buffers(std::move(dst_), total_size, stream);

      for (size_t i = 0; i < edge_property_buffers_.size(); ++i) {
        auto buffer =
          resize_and_copy_buffers(std::move(edge_property_buffers_[i]), total_size, stream);
        consolidated_edge_property_buffers.push_back(std::move(buffer[0]));
      }
    } else {
      for (size_t i = 0; i < edge_property_buffers_.size(); ++i) {
        consolidated_edge_property_buffers.push_back(std::move(edge_property_buffers_[i][0]));
      }
    }

    std::tie(src_[0], dst_[0], consolidated_edge_property_buffers, std::ignore) =
      cugraph::shuffle_ext_edges(handle.raft_handle(),
                                 std::move(src_[0]),
                                 std::move(dst_[0]),
                                 std::move(consolidated_edge_property_buffers),
                                 store_transposed);

    for (size_t i = 0; i < edge_property_buffers_.size(); ++i) {
      edge_property_buffers_[i] = std::vector<cugraph::arithmetic_device_uvector_t>{};
      edge_property_buffers_[i].push_back(std::move(consolidated_edge_property_buffers[i]));
    }
  }

 private:
  std::vector<cugraph::arithmetic_device_uvector_t> resize_and_copy_buffers(
    std::vector<cugraph::arithmetic_device_uvector_t>&& buffer,
    size_t total_size,
    rmm::cuda_stream_view stream)
  {
    return cugraph::variant_type_dispatch(buffer[0], [&buffer, total_size, stream](auto& buffer0) {
      using T = typename std::decay_t<decltype(buffer0)>::value_type;

      rmm::device_uvector<T> new_buffer(total_size, stream);

      size_t pos{0};
      std::for_each(buffer.begin(), buffer.end(), [&new_buffer, &pos, stream](auto& bufferi) {
        auto& buffer = std::get<rmm::device_uvector<T>>(bufferi);
        raft::copy(new_buffer.data() + pos, buffer.data(), buffer.size(), stream);
        pos += buffer.size();
        buffer.resize(0, stream);
        buffer.shrink_to_fit(stream);
      });

      std::vector<cugraph::arithmetic_device_uvector_t> result{};
      result.push_back(std::move(new_buffer));
      return result;
    });
  }

  template <typename T>
  std::vector<rmm::device_uvector<T>> resize_and_copy_buffers(
    std::vector<rmm::device_uvector<T>>&& buffer, size_t total_size, rmm::cuda_stream_view stream)
  {
    rmm::device_uvector<T> new_buffer(total_size, stream);

    size_t pos{0};
    std::for_each(buffer.begin(), buffer.end(), [&new_buffer, &pos, stream](auto& bufferi) {
      raft::copy(new_buffer.data() + pos, bufferi.data(), bufferi.size(), stream);
      pos += bufferi.size();
      bufferi.resize(0, stream);
      bufferi.shrink_to_fit(stream);
    });

    std::vector<rmm::device_uvector<T>> result{};
    result.push_back(std::move(new_buffer));
    return result;
  }

  void create_new_buffers(rmm::cuda_stream_view stream_view)
  {
    src_.emplace_back(device_buffer_size_, stream_view);
    dst_.emplace_back(device_buffer_size_, stream_view);

    for (size_t i = 0; i < edge_property_types_.size(); ++i) {
      cugraph::variant_type_dispatch(
        edge_property_types_[i],
        [device_buffer_size = device_buffer_size_,
         i,
         stream_view,
         &edge_property_buffers = edge_property_buffers_](auto& edge_property_type) {
          using T = std::decay_t<decltype(edge_property_type)>;
          rmm::device_uvector<T> edge_property_buffer(device_buffer_size, stream_view);
          edge_property_buffers[i].push_back(std::move(edge_property_buffer));
        });
    }

    current_pos_ = 0;
  }

  mutable std::mutex lock_{};

  size_t current_pos_{0};
  size_t device_buffer_size_{0};

  std::vector<rmm::device_uvector<vertex_t>> src_{};
  std::vector<rmm::device_uvector<vertex_t>> dst_{};
  std::vector<arithmetic_type_t> edge_property_types_{};
  std::vector<std::vector<cugraph::arithmetic_device_uvector_t>> edge_property_buffers_{};
};

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph

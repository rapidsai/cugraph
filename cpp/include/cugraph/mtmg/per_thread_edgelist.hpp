/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>
#include <cugraph/mtmg/detail/per_device_edgelist.hpp>

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
template <typename vertex_t>
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
  per_thread_edgelist_t(detail::per_device_edgelist_t<vertex_t>& edgelist,
                        size_t thread_buffer_size)
    : edgelist_{edgelist},
      current_pos_{0},
      src_(thread_buffer_size),
      dst_(thread_buffer_size),
      edge_properties_{}
  {
    std::for_each(
      edgelist.get_edge_property_types().begin(),
      edgelist.get_edge_property_types().end(),
      [&edge_properties = edge_properties_, thread_buffer_size](auto& p) {
        cugraph::variant_type_dispatch(p, [&edge_properties, thread_buffer_size](auto& p) {
          using T = std::decay_t<decltype(p)>;
          edge_properties.push_back(arithmetic_host_vector_t{std::vector<T>(thread_buffer_size)});
        });
      });
  }

  /**
   * @brief Append an edge to the edge list
   *
   * @param src             Source vertex id
   * @param dst             Destination vertex id
   * @param edge_properties Edge properties
   * @param stream_view     The cuda stream
   */
  void append(vertex_t src,
              vertex_t dst,
              std::vector<cugraph::arithmetic_type_t> edge_properties,
              rmm::cuda_stream_view stream_view)
  {
    if (current_pos_ == src_.size()) { flush(stream_view); }

    src_[current_pos_] = src;
    dst_[current_pos_] = dst;

    for (size_t i = 0; i < edge_properties.size(); ++i) {
      cugraph::variant_type_dispatch(
        edge_properties[i],
        [&edge_properties = edge_properties_, i, current_pos = current_pos_](auto& p) {
          using T = std::decay_t<decltype(p)>;

          (std::get<std::vector<T>>(edge_properties[i]))[current_pos] = p;
        });
    }

    ++current_pos_;
  }

  /**
   * @brief Append a list of edges to the edge list
   *
   * @param src             Source vertex id
   * @param dst             Destination vertex id
   * @param edge_properties Edge properties
   * @param stream_view     The cuda stream
   */
  void append(raft::host_span<vertex_t const> src,
              raft::host_span<vertex_t const> dst,
              std::vector<raft::host_span<cugraph::arithmetic_type_t const>> edge_properties,
              rmm::cuda_stream_view stream_view)
  {
    size_t count = src.size();
    size_t pos   = 0;

    while (count > 0) {
      size_t copy_count = std::min(count, (src_.size() - current_pos_));

      std::copy(src.begin() + pos, src.begin() + pos + copy_count, src_.begin() + current_pos_);
      std::copy(dst.begin() + pos, dst.begin() + pos + copy_count, dst_.begin() + current_pos_);

      for (size_t i = 0; i < edge_properties.size(); ++i) {
        cugraph::variant_type_dispatch(
          edge_properties[i],
          [&edge_properties = edge_properties_, i, pos, copy_count, current_pos = current_pos_](
            auto& p) {
            using T = std::decay_t<decltype(p)>;
            std::copy(p.begin() + pos,
                      p.begin() + pos + copy_count,
                      (std::get<std::vector<T>>(edge_properties[i])).begin() + current_pos);
          });
      }

      if (current_pos_ == src_.size()) { flush(stream_view); }

      count -= copy_count;
      pos += copy_count;
    }
  }

  /**
   * @brief Flush thread data from host to GPU memory
   *
   * @param stream_view The cuda stream
   * @param sync       If true, synchronize the asynchronous copy of data;
   *                   defaults to false.
   */
  void flush(rmm::cuda_stream_view stream_view, bool sync = false)
  {
    std::vector<arithmetic_const_host_span_t> edge_properties_spans;
    std::for_each(edge_properties_.begin(),
                  edge_properties_.end(),
                  [&edge_properties_spans, current_pos = current_pos_](auto& p) {
                    variant_type_dispatch(p, [&edge_properties_spans](auto& p) {
                      using T = typename std::decay_t<decltype(p)>::value_type;
                      raft::host_span<T const> span{p.data(), p.size()};
                      edge_properties_spans.push_back(span);
                    });
                  });

    edgelist_.append(raft::host_span<vertex_t const>{src_.data(), current_pos_},
                     raft::host_span<vertex_t const>{dst_.data(), current_pos_},
                     raft::host_span<arithmetic_const_host_span_t>{edge_properties_spans.data(),
                                                                   edge_properties_spans.size()},
                     stream_view);

    current_pos_ = 0;

    if (sync) stream_view.synchronize();
  }

 private:
  detail::per_device_edgelist_t<vertex_t>& edgelist_;
  size_t current_pos_{0};
  std::vector<vertex_t> src_{};
  std::vector<vertex_t> dst_{};
  std::vector<arithmetic_host_vector_t> edge_properties_{};
};

}  // namespace mtmg
}  // namespace cugraph

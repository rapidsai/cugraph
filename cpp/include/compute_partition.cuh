/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/binary_search.h>

#include <experimental/graph.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cugraph {
namespace experimental {
namespace detail {

/**
 * @brief  Class to help compute what partition a vertex id or edge id belongs to
 *
 *
 *   FIXME:  This should probably be part of the experimental::partition_t class
 *           rather than having to copy things out of it
 *
 */
template <typename graph_view_type>
class compute_partition_t {
 public:
  using graph_view_t = graph_view_type;
  using vertex_t     = typename graph_view_type::vertex_type;

  compute_partition_t(graph_view_t const &graph_view)
  {
    init<graph_view_t::is_multi_gpu>(graph_view);
  }

 private:
  template <bool is_multi_gpu, typename std::enable_if_t<!is_multi_gpu> * = nullptr>
  void init(graph_view_t const &graph_view)
  {
  }

  template <bool is_multi_gpu, typename std::enable_if_t<is_multi_gpu> * = nullptr>
  void init(graph_view_t const &graph_view)
  {
    auto partition = graph_view.get_partition();
    row_size_      = partition.get_row_size();
    col_size_      = partition.get_col_size();
    size_          = row_size_ * col_size_;

    vertex_partition_offsets_v_.resize(size_ + 1);
    vertex_partition_offsets_v_ = partition.get_vertex_partition_offsets();
  }

 public:
  /**
   * @brief     Compute the partition id for a vertex
   *
   * This is a device view of the partition data that allows for a device
   * function to determine the partition number that is associated with
   * a given vertex id.
   *
   * `vertex_device_view_t` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class vertex_device_view_t {
   public:
    vertex_device_view_t(vertex_t const *d_vertex_partition_offsets, int size)
      : d_vertex_partition_offsets_(d_vertex_partition_offsets), size_(size)
    {
    }

    /**
     * @brief     Compute the partition id for a vertex
     *
     * Given a vertex v, return the partition number to which that vertex is assigned
     *
     */
    __device__ int operator()(vertex_t v) const
    {
      if (graph_view_t::is_multi_gpu) {
        return thrust::distance(d_vertex_partition_offsets_,
                                thrust::upper_bound(thrust::seq,
                                                    d_vertex_partition_offsets_,
                                                    d_vertex_partition_offsets_ + size_ + 1,
                                                    v)) -
               1;
      } else
        return 0;
    }

   private:
    vertex_t const *d_vertex_partition_offsets_;
    int size_;
  };

  class edge_device_view_t {
   public:
    edge_device_view_t(vertex_t const *d_vertex_partition_offsets,
                       int row_size,
                       int col_size,
                       int size)
      : d_vertex_partition_offsets_(d_vertex_partition_offsets),
        row_size_(row_size),
        col_size_(col_size),
        size_(size)
    {
    }

    /**
     * @brief     Compute the partition id for a vertex
     *
     * Given a pair of vertices (src, dst), return the partition number to
     * which an edge between src and dst would be assigned.
     *
     */
    __device__ int operator()(vertex_t src, vertex_t dst) const
    {
      if (graph_view_t::is_multi_gpu) {
        std::size_t src_partition =
          thrust::distance(d_vertex_partition_offsets_,
                           thrust::upper_bound(thrust::seq,
                                               d_vertex_partition_offsets_,
                                               d_vertex_partition_offsets_ + size_ + 1,
                                               src)) -
          1;
        std::size_t dst_partition =
          thrust::distance(d_vertex_partition_offsets_,
                           thrust::upper_bound(thrust::seq,
                                               d_vertex_partition_offsets_,
                                               d_vertex_partition_offsets_ + size_ + 1,
                                               dst)) -
          1;

        std::size_t row = src_partition / row_size_;
        std::size_t col = dst_partition / col_size_;

        return row * row_size_ + col;
      } else {
        return 0;
      }
    }

   private:
    vertex_t const *d_vertex_partition_offsets_;
    int row_size_;
    int col_size_;
    int size_;
  };

  /**
   * @brief get a vertex device view so that device code can identify which
   * gpu a vertex is assigned to
   *
   */
  vertex_device_view_t vertex_device_view() const
  {
    return vertex_device_view_t(vertex_partition_offsets_v_.data().get(), size_);
  }

  /**
   * @brief get an edge device view so that device code can identify which
   * gpu an edge is assigned to
   *
   */
  edge_device_view_t edge_device_view() const
  {
    return edge_device_view_t(
      vertex_partition_offsets_v_.data().get(), row_size_, col_size_, size_);
  }

 private:
  rmm::device_vector<vertex_t> vertex_partition_offsets_v_{};
  int row_size_{1};
  int col_size_{1};
  int size_{1};
};

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph

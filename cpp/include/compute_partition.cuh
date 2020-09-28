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

template <typename graph_view_type>
class compute_partition_t {
 public:
  using graph_view_t = graph_view_type;
  using vertex_t     = typename graph_view_type::vertex_type;

  compute_partition_t(graph_view_t const &graph_view)
  {
    init<graph_view_t::is_multi_gpu>(graph_view);
  }

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

    vertex_partition_offsets_v_.resize(size_);

    // TODO:  Copy from host to device memory...
  }

  class vertex_device_view_t {
   public:
    vertex_device_view_t(vertex_t const *d_vertex_partition_offsets, int row_size, int size)
      : d_vertex_partition_offsets_(d_vertex_partition_offsets), row_size_(row_size), size_(size)
    {
    }

    __device__ int operator()(vertex_t v) const
    {
      if (graph_view_t::is_multi_gpu)
        return thrust::distance(d_vertex_partition_offsets_,
                                thrust::upper_bound(thrust::device,
                                                    d_vertex_partition_offsets_,
                                                    d_vertex_partition_offsets_ + size_ + 1,
                                                    v));
      else
        return 0;
    }

   private:
    vertex_t const *d_vertex_partition_offsets_;
    int row_size_;
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

    __device__ int operator()(vertex_t src, vertex_t dst) const
    {
      if (graph_view_t::is_multi_gpu) {
        std::size_t src_partition =
          thrust::distance(d_vertex_partition_offsets_,
                           thrust::upper_bound(thrust::device,
                                               d_vertex_partition_offsets_,
                                               d_vertex_partition_offsets_ + size_ + 1,
                                               src));
        std::size_t dst_partition =
          thrust::distance(d_vertex_partition_offsets_,
                           thrust::upper_bound(thrust::device,
                                               d_vertex_partition_offsets_,
                                               d_vertex_partition_offsets_ + size_ + 1,
                                               dst));

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

  vertex_device_view_t vertex_device_view() const
  {
    return vertex_device_view_t(vertex_partition_offsets_v_.data().get(), row_size_, size_);
  }

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

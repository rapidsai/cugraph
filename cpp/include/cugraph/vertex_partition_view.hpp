/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t>
class vertex_partition_view_base_t {
 public:
  vertex_partition_view_base_t(vertex_t number_of_vertices)
    : number_of_vertices_(number_of_vertices)
  {
  }

  vertex_t number_of_vertices() const { return number_of_vertices_; }

 private:
  vertex_t number_of_vertices_{0};
};

}  // namespace detail

template <typename vertex_t, bool multi_gpu, typename Enable = void>
class vertex_partition_view_t;

// multi-GPU version
template <typename vertex_t, bool multi_gpu>
class vertex_partition_view_t<vertex_t, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::vertex_partition_view_base_t<vertex_t> {
 public:
  vertex_partition_view_t(vertex_t number_of_vertices,
                          vertex_t local_vertex_partition_range_first,
                          vertex_t local_vertex_partition_range_last)
    : detail::vertex_partition_view_base_t<vertex_t>(number_of_vertices),
      local_vertex_partition_range_first_(local_vertex_partition_range_first),
      local_vertex_partition_range_last_(local_vertex_partition_range_last)
  {
  }

  vertex_t local_vertex_partition_range_first() const
  {
    return local_vertex_partition_range_first_;
  }
  vertex_t local_vertex_partition_range_last() const { return local_vertex_partition_range_last_; }

 private:
  vertex_t local_vertex_partition_range_first_{0};
  vertex_t local_vertex_partition_range_last_{0};
};

// single-GPU version
template <typename vertex_t, bool multi_gpu>
class vertex_partition_view_t<vertex_t, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::vertex_partition_view_base_t<vertex_t> {
 public:
  vertex_partition_view_t(vertex_t number_of_vertices)
    : detail::vertex_partition_view_base_t<vertex_t>(number_of_vertices)
  {
  }

  vertex_t local_vertex_partition_range_first() const { return vertex_t{0}; }
  vertex_t local_vertex_partition_range_last() const { return this->number_of_vertices(); }
};

}  // namespace cugraph

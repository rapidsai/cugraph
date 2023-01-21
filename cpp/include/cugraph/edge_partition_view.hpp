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

#include <raft/core/device_span.hpp>

#include <optional>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t>
class edge_partition_view_base_t {
 public:
  edge_partition_view_base_t(raft::device_span<edge_t const> offsets,
                             raft::device_span<vertex_t const> indices)
    : offsets_(offsets), indices_(indices)
  {
  }

  edge_t number_of_edges() const { return static_cast<edge_t>(indices_.size()); }

  raft::device_span<edge_t const> offsets() const { return offsets_; }
  raft::device_span<vertex_t const> indices() const { return indices_; }

 private:
  raft::device_span<edge_t const> offsets_{};
  raft::device_span<vertex_t const> indices_{};
};

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu, typename Enable = void>
class edge_partition_view_t;

// multi-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
class edge_partition_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>>
  : public detail::edge_partition_view_base_t<vertex_t, edge_t> {
 public:
  edge_partition_view_t(raft::device_span<edge_t const> offsets,
                        raft::device_span<vertex_t const> indices,
                        std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices,
                        std::optional<vertex_t> major_hypersparse_first,
                        vertex_t major_range_first,
                        vertex_t major_range_last,
                        vertex_t minor_range_first,
                        vertex_t minor_range_last,
                        vertex_t major_value_start_offset)
    : detail::edge_partition_view_base_t<vertex_t, edge_t>(offsets, indices),
      dcs_nzd_vertices_(dcs_nzd_vertices),
      major_hypersparse_first_(major_hypersparse_first),
      major_range_first_(major_range_first),
      major_range_last_(major_range_last),
      minor_range_first_(minor_range_first),
      minor_range_last_(minor_range_last),
      major_value_start_offset_(major_value_start_offset)
  {
  }

  std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices() const
  {
    return dcs_nzd_vertices_;
  }

  std::optional<vertex_t> major_hypersparse_first() const { return major_hypersparse_first_; }

  vertex_t major_range_first() const { return major_range_first_; }
  vertex_t major_range_last() const { return major_range_last_; }
  vertex_t minor_range_first() const { return minor_range_first_; }
  vertex_t minor_range_last() const { return minor_range_last_; }

  vertex_t major_value_start_offset() const { return major_value_start_offset_; }

 private:
  // relevant only if we use the CSR + DCSR (or CSC + DCSC) hybrid format
  std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices_{std::nullopt};
  std::optional<vertex_t> major_hypersparse_first_{std::nullopt};

  vertex_t major_range_first_{0};
  vertex_t major_range_last_{0};
  vertex_t minor_range_first_{0};
  vertex_t minor_range_last_{0};

  vertex_t major_value_start_offset_{0};
};

// single-GPU version
template <typename vertex_t, typename edge_t, bool multi_gpu>
class edge_partition_view_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>>
  : public detail::edge_partition_view_base_t<vertex_t, edge_t> {
 public:
  edge_partition_view_t(raft::device_span<edge_t const> offsets,
                        raft::device_span<vertex_t const> indices,
                        vertex_t number_of_vertices)
    : detail::edge_partition_view_base_t<vertex_t, edge_t>(offsets, indices),
      number_of_vertices_(number_of_vertices)
  {
  }

  std::optional<raft::device_span<vertex_t const>> dcs_nzd_vertices() const { return std::nullopt; }
  std::optional<vertex_t> major_hypersparse_first() const { return std::nullopt; }

  vertex_t major_range_first() const { return vertex_t{0}; }
  vertex_t major_range_last() const { return number_of_vertices_; }
  vertex_t minor_range_first() const { return vertex_t{0}; }
  vertex_t minor_range_last() const { return number_of_vertices_; }

 private:
  vertex_t number_of_vertices_{0};
};

}  // namespace cugraph

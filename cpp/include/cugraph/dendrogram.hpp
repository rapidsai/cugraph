/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/vertex_property.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <utility>
#include <vector>

namespace CUGRAPH_EXPORT cugraph {

template <typename vertex_t>
class dendrogram_t {
 public:
  void add_level(vertex_property_t<vertex_t, vertex_t>&& values, vertex_t range_first)
  {
    levels_.push_back(std::move(values));
    level_range_first_.push_back(range_first);
  }

  size_t current_level() const { return levels_.size() - 1; }

  size_t num_levels() const { return levels_.size(); }

  raft::device_span<vertex_t const> level_data(size_t level) const
  {
    return raft::device_span<vertex_t const>(levels_[level].view().value_first(),
                                             level_size(level));
  }

  raft::device_span<vertex_t> level_data(size_t level)
  {
    return raft::device_span<vertex_t>(levels_[level].mutable_view().value_first(),
                                       level_size(level));
  }

  size_t level_size(size_t level) const
  {
    return static_cast<size_t>(levels_[level].view().range_size());
  }

  vertex_t level_range_first(size_t level) const { return level_range_first_[level]; }

  raft::device_span<vertex_t const> current_level_data() const
  {
    return level_data(current_level());
  }

  raft::device_span<vertex_t> current_level_data() { return level_data(current_level()); }

  size_t current_level_size() const { return level_size(current_level()); }

  vertex_t current_level_range_first() const { return level_range_first(current_level()); }

 private:
  std::vector<vertex_property_t<vertex_t, vertex_t>> levels_;
  std::vector<vertex_t> level_range_first_;
};

/** @deprecated Use dendrogram_t. */
template <typename vertex_t>
using Dendrogram = dendrogram_t<vertex_t>;

}  // namespace CUGRAPH_EXPORT cugraph

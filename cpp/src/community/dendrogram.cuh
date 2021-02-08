/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

namespace cugraph {

template <typename vertex_t>
class Dendrogram {
 public:
  void add_level(vertex_t num_verts,
                 cudaStream_t stream                 = 0,
                 rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
  {
    level_ptr_.push_back(
      std::make_unique<rmm::device_buffer>(num_verts * sizeof(vertex_t), stream, mr));
    level_size_.push_back(num_verts);
  }

  size_t current_level() const { return level_size_.size() - 1; }

  size_t num_levels() const { return level_size_.size(); }

  vertex_t const *get_level_ptr_nocheck(size_t level) const
  {
    return static_cast<vertex_t const *>(level_ptr_[level]->data());
  }

  vertex_t *get_level_ptr_nocheck(size_t level)
  {
    return static_cast<vertex_t *>(level_ptr_[level]->data());
  }

  vertex_t get_level_size_nocheck(size_t level) const { return level_size_[level]; }

  vertex_t const *current_level_begin() const { return get_level_ptr_nocheck(current_level()); }

  vertex_t const *current_level_end() const { return current_level_begin() + current_level_size(); }

  vertex_t *current_level_begin() { return get_level_ptr_nocheck(current_level()); }

  vertex_t *current_level_end() { return current_level_begin() + current_level_size(); }

  vertex_t current_level_size() const { return get_level_size_nocheck(current_level()); }

 private:
  std::vector<vertex_t> level_size_;
  std::vector<std::unique_ptr<rmm::device_buffer>> level_ptr_;
};

}  // namespace cugraph

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

#include <rmm/thrust_rmm_allocator.h>
#include <community/dendrogram.cuh>
#include <experimental/graph_functions.hpp>
#include <raft/handle.hpp>

namespace cugraph {

template <typename vertex_t, bool multi_gpu>
void partition_at_level(raft::handle_t const &handle,
                        Dendrogram<vertex_t> const &dendrogram,
                        vertex_t const *d_vertex_ids,
                        vertex_t *d_partition,
                        size_t level)
{
  vertex_t local_num_verts = dendrogram.get_level_size_unsafe(0);

  thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               d_vertex_ids,
               d_vertex_ids + local_num_verts,
               d_partition);

  std::for_each(thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(level),
                [&handle, &dendrogram, d_vertex_ids, &d_partition, local_num_verts](size_t l) {
                  cugraph::experimental::relabel<vertex_t, multi_gpu>(
                    handle,
                    std::tuple<vertex_t const *, vertex_t const *>(
                      d_vertex_ids, dendrogram.get_level_ptr_unsafe(l)),
                    dendrogram.get_level_size_unsafe(l),
                    d_partition,
                    local_num_verts);
                });
}

}  // namespace cugraph

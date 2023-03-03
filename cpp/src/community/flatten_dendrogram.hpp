/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/dendrogram.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cugraph {

template <typename vertex_t, bool multi_gpu>
void partition_at_level(raft::handle_t const& handle,
                        Dendrogram<vertex_t> const& dendrogram,
                        vertex_t const* d_vertex_ids,
                        vertex_t* d_partition,
                        size_t level)
{
  vertex_t local_num_verts = dendrogram.get_level_size_nocheck(0);
  rmm::device_uvector<vertex_t> local_vertex_ids_v(local_num_verts, handle.get_stream());

  raft::copy(d_partition, d_vertex_ids, local_num_verts, handle.get_stream());

  std::for_each(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(level),
    [&handle, &dendrogram, &local_vertex_ids_v, d_vertex_ids, &d_partition, local_num_verts](
      size_t l) {
      detail::sequence_fill(handle.get_stream(),
                            local_vertex_ids_v.begin(),
                            dendrogram.get_level_size_nocheck(l),
                            dendrogram.get_level_first_index_nocheck(l));

      cugraph::relabel<vertex_t, multi_gpu>(
        handle,
        std::tuple<vertex_t const*, vertex_t const*>(local_vertex_ids_v.data(),
                                                     dendrogram.get_level_ptr_nocheck(l)),
        dendrogram.get_level_size_nocheck(l),
        d_partition,
        local_num_verts,
        false);
    });
}

}  // namespace cugraph

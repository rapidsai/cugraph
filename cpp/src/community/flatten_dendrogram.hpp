/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

  // raft::print_device_vector("before relabeling", d_partition + 409, 4, std::cout);
  raft::print_device_vector("before relabeling", d_partition + 1832, 4, std::cout);

  std::for_each(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(level),
    [&handle, &dendrogram, &local_vertex_ids_v, d_vertex_ids, &d_partition, local_num_verts](
      size_t l) {
      detail::sequence_fill(handle.get_stream(),
                            local_vertex_ids_v.begin(),
                            dendrogram.get_level_size_nocheck(l),
                            dendrogram.get_level_first_index_nocheck(l));

      char label[128];
      // snprintf(label, 128, "before relabel+409 %lu", l);
      snprintf(label, 128, "before relabel+1832 %lu", l);
      // raft::print_device_vector(label, d_partition, local_num_verts, std::cout);
      // raft::print_device_vector(label, d_partition + 409, 4, std::cout);
      raft::print_device_vector(label, d_partition + 1832, 4, std::cout);

      std::cout << "dendrogram size: " << dendrogram.get_level_size_nocheck(l) << std::endl;
      if (dendrogram.get_level_size_nocheck(l) > 20)
        raft::print_device_vector(
          "  dendrogram", dendrogram.get_level_ptr_nocheck(l) + 15, 4, std::cout);

      cugraph::relabel<vertex_t, multi_gpu>(
        handle,
        std::tuple<vertex_t const*, vertex_t const*>(local_vertex_ids_v.data(),
                                                     dendrogram.get_level_ptr_nocheck(l)),
        dendrogram.get_level_size_nocheck(l),
        d_partition,
        local_num_verts,
        false);

      // snprintf(label, 128, "after relabel+409 %lu", l);
      snprintf(label, 128, "after relabel+1832 %lu", l);
      // raft::print_device_vector(label, d_partition, local_num_verts, std::cout);
      // raft::print_device_vector(label, d_partition + 409, 4, std::cout);
      raft::print_device_vector(label, d_partition + 1832, 4, std::cout);
    });
}

template <typename vertex_t, bool multi_gpu>
void leiden_partition_at_level(raft::handle_t const& handle,
                               Dendrogram<vertex_t> const& dendrogram,
                               vertex_t* d_partition,
                               size_t level)
{
  vertex_t local_num_verts = dendrogram.get_level_size_nocheck(0);
  raft::copy(
    d_partition, dendrogram.get_level_ptr_nocheck(0), local_num_verts, handle.get_stream());

  rmm::device_uvector<vertex_t> local_vertex_ids_v(local_num_verts, handle.get_stream());

  raft::print_device_vector("before relabeling", d_partition + 409, 4, std::cout);

  std::for_each(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>((level - 1) / 2),
    [&handle, &dendrogram, &local_vertex_ids_v, &d_partition, local_num_verts](size_t l) {
      char label[128];
      snprintf(label, 128, "before relabel+409 %lu", l);
      // raft::print_device_vector(label, d_partition, local_num_verts, std::cout);
      raft::print_device_vector(label, d_partition + 409, 4, std::cout);

      raft::print_device_vector("  p1",
                                dendrogram.get_level_ptr_nocheck(2 * l + 1),
                                dendrogram.get_level_size_nocheck(2 * l + 1),
                                std::cout);
      raft::print_device_vector("  p2",
                                dendrogram.get_level_ptr_nocheck(2 * l + 2),
                                dendrogram.get_level_size_nocheck(2 * l + 2),
                                std::cout);

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(1),
                       [p1   = dendrogram.get_level_ptr_nocheck(2 * l + 1),
                        p2   = dendrogram.get_level_ptr_nocheck(2 * l + 2),
                        size = dendrogram.get_level_size_nocheck(2 * l + 1)] __device__(auto) {
                         for (size_t i = 0; i < size; ++i) {
#if 0
                           if (p1[i] == 2410) {
#else
                           if (p2[i] == 2410) {
#endif
                             printf("%lu: p1 = %d, p2 = %d\n", i, (int)p1[i], (int)p2[i]);
                           }
                         }
                       });

      cugraph::relabel<vertex_t, multi_gpu>(
        handle,
#if 0
        std::tuple<vertex_t const*, vertex_t const*>(dendrogram.get_level_ptr_nocheck(2 * l + 1),
                                                     dendrogram.get_level_ptr_nocheck(2 * l + 2)),
#else
        std::tuple<vertex_t const*, vertex_t const*>(dendrogram.get_level_ptr_nocheck(2 * l + 2),
                                                     dendrogram.get_level_ptr_nocheck(2 * l + 1)),
#endif
        dendrogram.get_level_size_nocheck(2 * l + 1),
        d_partition,
        local_num_verts,
        false);

      snprintf(label, 128, "after relabel+409 %lu", l);
      // raft::print_device_vector(label, d_partition, local_num_verts, std::cout);
      raft::print_device_vector(label, d_partition + 409, 4, std::cout);
    });
}

}  // namespace cugraph

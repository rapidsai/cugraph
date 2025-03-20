/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "sampling_utils.hpp"

#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/core/handle.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <optional>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_time_t, typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           std::optional<rmm::device_uvector<label_t>>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           std::optional<rmm::device_uvector<label_t>>>
temporal_partition_vertices(raft::handle_t const& handle,
                            raft::device_span<vertex_t const> vertices,
                            raft::device_span<edge_time_t const> vertex_times,
                            std::optional<raft::device_span<label_t const>> vertex_labels)
{
  std::cout << "inside temporal_partition_vertices" << std::endl;

  rmm::device_uvector<vertex_t> vertices_p1(vertices.size(), handle.get_stream());
  rmm::device_uvector<edge_time_t> vertex_times_p1(vertex_times.size(), handle.get_stream());
  std::optional<rmm::device_uvector<label_t>> vertex_labels_p1{
    vertex_labels
      ? std::make_optional<rmm::device_uvector<label_t>>(vertex_labels->size(), handle.get_stream())
      : std::nullopt};
  rmm::device_uvector<vertex_t> vertices_p2(0, handle.get_stream());
  rmm::device_uvector<edge_time_t> vertex_times_p2(0, handle.get_stream());
  std::optional<rmm::device_uvector<label_t>> vertex_labels_p2{
    vertex_labels ? std::make_optional<rmm::device_uvector<label_t>>(0, handle.get_stream())
                  : std::nullopt};

#if 0
  std::cout << "calling thrust::copy" << std::endl;

  thrust::for_each(handle.get_thrust_policy(),
thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
[v = vertices.data(), t = vertex_times.data(), sz = vertices.size()] __device__ (auto) {
  for (size_t i = 0 ; i < sz ; ++i) {
    printf("  i = %d, v = %d, time = %d\n", (int) i, (int) v[i], (int) t[i]);
  }
});
#endif

  thrust::copy(handle.get_thrust_policy(), vertices.begin(), vertices.end(), vertices_p1.begin());
  thrust::copy(
    handle.get_thrust_policy(), vertex_times.begin(), vertex_times.end(), vertex_times_p1.begin());

  if (vertex_labels) {
    thrust::copy(handle.get_thrust_policy(),
                 vertex_labels->begin(),
                 vertex_labels->end(),
                 vertex_labels_p1->begin());

    thrust::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()),
      thrust::make_zip_iterator(vertices_p1.end(), vertex_times_p1.end(), vertex_labels_p1->end()));
  } else {
    thrust::sort(handle.get_thrust_policy(),
                 thrust::make_zip_iterator(vertices_p1.begin(), vertex_times_p1.begin()),
                 thrust::make_zip_iterator(vertices_p1.end(), vertex_times_p1.end()));
  }

  rmm::device_uvector<uint32_t> unique_vertex_entries(cugraph::packed_bool_size(vertices_p1.size()),
                                                      handle.get_stream());

  std::cout << "thrust::tabulate" << std::endl;
  thrust::tabulate(
    handle.get_thrust_policy(),
    unique_vertex_entries.begin(),
    unique_vertex_entries.end(),
    [verts = vertices_p1.data(), num_verts = vertices_p1.size()] __device__(size_t idx) {
      auto word                = cugraph::packed_bool_empty_mask();
      size_t start_index       = idx * cugraph::packed_bools_per_word();
      size_t bits_in_this_word = (start_index + cugraph::packed_bools_per_word() < num_verts)
                                   ? cugraph::packed_bools_per_word()
                                   : (num_verts - start_index);

      for (size_t bit = 0; bit < bits_in_this_word; ++bit) {
        size_t vertex_pos = start_index + bit;
        if ((vertex_pos == 0) || (verts[vertex_pos - 1] != verts[vertex_pos]))
          word |= cugraph::packed_bool_mask(bit);
      }

      return word;
    });

  size_t num_unique_vertices =
    detail::count_set_bits(handle, unique_vertex_entries.begin(), vertices_p1.size());

  std::cout << "num_unique_vertices = " << num_unique_vertices
            << ", vertices_p1 size = " << vertices_p1.size() << std::endl;

  if (num_unique_vertices < vertices_p1.size()) {
    vertices_p2.resize(vertices_p1.size() - num_unique_vertices, handle.get_stream());
    vertex_times_p2.resize(vertices_p1.size() - num_unique_vertices, handle.get_stream());

    if (vertex_labels) {
      vertex_labels_p2->resize(vertices_p1.size() - num_unique_vertices, handle.get_stream());

      copy_if_mask_not_set(
        handle,
        thrust::make_zip_iterator(
          vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()),
        thrust::make_zip_iterator(
          vertices_p1.end(), vertex_times_p1.end(), vertex_labels_p1->end()),
        unique_vertex_entries.begin(),
        thrust::make_zip_iterator(
          vertices_p2.begin(), vertex_times_p2.begin(), vertex_labels_p2->begin()));

      vertices_p1.resize(
        thrust::distance(
          thrust::make_zip_iterator(
            vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()),
          copy_if_mask_set(
            handle,
            thrust::make_zip_iterator(
              vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()),
            thrust::make_zip_iterator(
              vertices_p1.end(), vertex_times_p1.end(), vertex_labels_p1->end()),
            unique_vertex_entries.begin(),
            thrust::make_zip_iterator(
              vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()))),
        handle.get_stream());

      vertex_labels_p1->resize(vertices_p1.size(), handle.get_stream());
      vertex_times_p1.resize(vertices_p1.size(), handle.get_stream());

      std::cout << " p1 size = " << vertices_p1.size() << ", p2 size = " << vertices_p2.size()
                << std::endl;
    } else {
      copy_if_mask_not_set(
        handle,
        thrust::make_zip_iterator(
          vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()),
        thrust::make_zip_iterator(
          vertices_p1.end(), vertex_times_p1.end(), vertex_labels_p1->end()),
        unique_vertex_entries.begin(),
        thrust::make_zip_iterator(
          vertices_p2.begin(), vertex_times_p2.begin(), vertex_labels_p2->begin()));

      vertices_p1.resize(
        thrust::distance(
          thrust::make_zip_iterator(
            vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()),
          copy_if_mask_set(
            handle,
            thrust::make_zip_iterator(
              vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()),
            thrust::make_zip_iterator(
              vertices_p1.end(), vertex_times_p1.end(), vertex_labels_p1->end()),
            unique_vertex_entries.begin(),
            thrust::make_zip_iterator(
              vertices_p1.begin(), vertex_times_p1.begin(), vertex_labels_p1->begin()))),
        handle.get_stream());

      vertex_times_p1.resize(vertices_p1.size(), handle.get_stream());
    }
  }

  std::cout << " p1 size = " << vertices_p1.size() << ", p2 size = " << vertices_p2.size()
            << std::endl;
#if 0
  std::cout << "returning make_tuple... p1:" << std::endl;
  raft::print_device_vector("  p1", vertices_p1.data(), vertices_p1.size(), std::cout);
  raft::print_device_vector("  p2", vertices_p2.data(), vertices_p2.size(), std::cout);

  // TODO:   BACKWARDS, p1 and p2 are backwards...
  thrust::for_each(handle.get_thrust_policy(),
thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
[v = vertices_p1.data(), t = vertex_times_p1.data(), sz = vertices_p1.size()] __device__ (auto) {
  printf("p1 SZ = %d\n", (int) sz);
  for (size_t i = 0 ; i < sz ; ++i) {
    printf("  i = %d, v = %d, time = %d\n", (int) i, (int) v[i], (int) t[i]);
  }
});

std::cout << "  p2:" << std::endl;
thrust::for_each(handle.get_thrust_policy(),
thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
[v = vertices_p2.data(), t = vertex_times_p2.data(), sz = vertices_p2.size()] __device__ (auto) {
  printf("p2 SZ = %d\n", (int) sz);
for (size_t i = 0 ; i < sz ; ++i) {
  printf("  i = %d, v = %d, time = %d\n", (int) i, (int) v[i], (int) t[i]);
}
});
#endif

  return std::make_tuple(std::move(vertices_p1),
                         std::move(vertex_times_p1),
                         std::move(vertex_labels_p1),
                         std::move(vertices_p2),
                         std::move(vertex_times_p2),
                         std::move(vertex_labels_p2));
}

}  // namespace detail
}  // namespace cugraph

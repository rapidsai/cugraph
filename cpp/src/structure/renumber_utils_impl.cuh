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

#include <detail/graph_utils.cuh>
#include <prims/kv_store.cuh>
#include <utilities/collect_comm.cuh>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace cugraph {

namespace detail {

template <typename vertex_t>
void unrenumber_local_int_edges(
  raft::handle_t const& handle,
  std::vector<vertex_t*> const& edgelist_majors /* [INOUT] */,
  std::vector<vertex_t*> const& edgelist_minors /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  vertex_t const* renumber_map_labels,
  std::vector<vertex_t> const& vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto const row_comm_rank = row_comm.get_rank();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();
  auto const col_comm_rank = col_comm.get_rank();

  CUGRAPH_EXPECTS(edgelist_majors.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input arguments: erroneous edgelist_majors.size().");
  CUGRAPH_EXPECTS(edgelist_minors.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input arguments: erroneous edgelist_minors.size().");
  CUGRAPH_EXPECTS(edgelist_edge_counts.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input arguments: erroneous edgelist_edge_counts.size().");
  CUGRAPH_EXPECTS(
    std::is_sorted(vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.end()),
    "Invalid input arguments: vertex_partition_range_lasts is not sorted.");
  if (edgelist_intra_partition_segment_offsets) {
    CUGRAPH_EXPECTS(
      (*edgelist_intra_partition_segment_offsets).size() == static_cast<size_t>(col_comm_size),
      "Invalid input arguments: erroneous (*edgelist_intra_partition_segment_offsets).size().");
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      CUGRAPH_EXPECTS(
        (*edgelist_intra_partition_segment_offsets)[i].size() ==
          static_cast<size_t>(row_comm_size + 1),
        "Invalid input arguments: erroneous (*edgelist_intra_partition_segment_offsets)[].size().");
      CUGRAPH_EXPECTS(
        std::is_sorted((*edgelist_intra_partition_segment_offsets)[i].begin(),
                       (*edgelist_intra_partition_segment_offsets)[i].end()),
        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[] is not sorted.");
      CUGRAPH_EXPECTS(
        ((*edgelist_intra_partition_segment_offsets)[i][0] == 0) &&
          ((*edgelist_intra_partition_segment_offsets)[i].back() == edgelist_edge_counts[i]),
        "Invalid input arguments: (*edgelist_intra_partition_segment_offsets)[][0] should be 0 and "
        "(*edgelist_intra_partition_segment_offsets)[].back() should coincide with "
        "edgelist_edge_counts[].");
    }
  }

  if (do_expensive_check) {
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      auto vertex_partition_rank = static_cast<int>(i) * row_comm_size + row_comm_rank;
      auto edge_partition_major_range_first =
        vertex_partition_rank == 0 ? vertex_t{0}
                                   : vertex_partition_range_lasts[vertex_partition_rank - 1];
      auto edge_partition_major_range_last = vertex_partition_range_lasts[vertex_partition_rank];
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edgelist_majors[i],
          edgelist_majors[i] + edgelist_edge_counts[i],
          [edge_partition_major_range_first, edge_partition_major_range_last] __device__(auto v) {
            return v != invalid_vertex_id<vertex_t>::value &&
                   (v < edge_partition_major_range_first || v >= edge_partition_major_range_last);
          }) == 0,
        "Invalid input arguments: there are out-of-range vertices in [edgelist_majors[], "
        "edgelist_majors[] + edgelist_edge_counts[]).");

      if (edgelist_intra_partition_segment_offsets) {
        for (int j = 0; j < row_comm_size; ++j) {
          auto vertex_partition_rank = col_comm_rank * row_comm_size + j;
          auto valid_first           = vertex_partition_rank == 0
                                         ? vertex_t{0}
                                         : vertex_partition_range_lasts[vertex_partition_rank - 1];
          auto valid_last            = vertex_partition_range_lasts[vertex_partition_rank];
          CUGRAPH_EXPECTS(
            thrust::count_if(
              handle.get_thrust_policy(),
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j],
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j + 1],
              [valid_first, valid_last] __device__(auto v) {
                return v != invalid_vertex_id<vertex_t>::value &&
                       (v < valid_first || v >= valid_last);
              }) == 0,
            "Invalid input arguments: there are out-of-range vertices in [edgelist_minors[], "
            "edgelist_minors[] + edgelist_edge_counts[]).");
        }
      } else {
        auto edge_partition_minor_range_first =
          (col_comm_rank * row_comm_size) == 0
            ? vertex_t{0}
            : vertex_partition_range_lasts[col_comm_rank * row_comm_size - 1];
        auto edge_partition_minor_range_last =
          vertex_partition_range_lasts[col_comm_rank * row_comm_size + row_comm_size - 1];
        CUGRAPH_EXPECTS(
          thrust::count_if(
            handle.get_thrust_policy(),
            edgelist_minors[i],
            edgelist_minors[i] + edgelist_edge_counts[i],
            [edge_partition_minor_range_first, edge_partition_minor_range_last] __device__(auto v) {
              return v != invalid_vertex_id<vertex_t>::value &&
                     (v < edge_partition_minor_range_first || v >= edge_partition_minor_range_last);
            }) == 0,
          "Invalid input arguments: there are out-of-range vertices in [edgelist_minors[], "
          "edgelist_minors[] + edgelist_edge_counts[]).");
      }
    }
  }

  auto number_of_edges =
    host_scalar_allreduce(comm,
                          std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end()),
                          raft::comms::op_t::SUM,
                          handle.get_stream());

  // FIXME: compare this hash based approach with a binary search based approach in both memory
  // footprint and execution time

  {
    vertex_t max_edge_partition_major_range_size{0};
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      auto vertex_partition_rank = static_cast<int>(i) * row_comm_size + row_comm_rank;
      auto edge_partition_major_range_size =
        vertex_partition_range_lasts[vertex_partition_rank] -
        (vertex_partition_rank == 0 ? vertex_t{0}
                                    : vertex_partition_range_lasts[vertex_partition_rank - 1]);
      max_edge_partition_major_range_size =
        std::max(max_edge_partition_major_range_size, edge_partition_major_range_size);
    }
    rmm::device_uvector<vertex_t> renumber_map_major_labels(max_edge_partition_major_range_size,
                                                            handle.get_stream());
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      auto vertex_partition_rank = static_cast<int>(i) * row_comm_size + row_comm_rank;
      auto edge_partition_major_range_first =
        vertex_partition_rank == 0 ? vertex_t{0}
                                   : vertex_partition_range_lasts[vertex_partition_rank - 1];
      auto edge_partition_major_range_last = vertex_partition_range_lasts[vertex_partition_rank];
      auto edge_partition_major_range_size =
        edge_partition_major_range_last - edge_partition_major_range_first;
      device_bcast(col_comm,
                   renumber_map_labels,
                   renumber_map_major_labels.data(),
                   edge_partition_major_range_size,
                   i,
                   handle.get_stream());

      kv_store_t<vertex_t, vertex_t, false> renumber_map(
        thrust::make_counting_iterator(edge_partition_major_range_first),
        thrust::make_counting_iterator(edge_partition_major_range_first) +
          edge_partition_major_range_size,
        renumber_map_major_labels.begin(),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value,
        handle.get_stream());
      auto renumber_map_view = renumber_map.view();
      renumber_map_view.find(edgelist_majors[i],
                             edgelist_majors[i] + edgelist_edge_counts[i],
                             edgelist_majors[i],
                             handle.get_stream());
    }
  }

  vertex_t edge_partition_minor_range_size{0};
  for (int i = 0; i < row_comm_size; ++i) {
    auto vertex_partition_rank = col_comm_rank * row_comm_size + i;
    edge_partition_minor_range_size +=
      vertex_partition_range_lasts[vertex_partition_rank] -
      (vertex_partition_rank == 0 ? vertex_t{0}
                                  : vertex_partition_range_lasts[vertex_partition_rank - 1]);
  }
  if ((edge_partition_minor_range_size >= static_cast<vertex_t>(number_of_edges / comm_size)) &&
      edgelist_intra_partition_segment_offsets) {  // memory footprint dominated by the O(V/sqrt(P))
                                                   // part than the O(E/P) part
    vertex_t max_segment_size{0};
    for (int i = 0; i < row_comm_size; ++i) {
      auto vertex_partition_rank = col_comm_rank * row_comm_size + i;
      max_segment_size           = std::max(
        max_segment_size,
        vertex_partition_range_lasts[vertex_partition_rank] -
          (vertex_partition_rank == 0 ? vertex_t{0}
                                                : vertex_partition_range_lasts[vertex_partition_rank - 1]));
    }
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(max_segment_size, handle.get_stream());
    for (int i = 0; i < row_comm_size; ++i) {
      auto vertex_partition_rank = col_comm_rank * row_comm_size + i;
      auto vertex_partition_minor_range_first =
        vertex_partition_rank == 0 ? vertex_t{0}
                                   : vertex_partition_range_lasts[vertex_partition_rank - 1];
      auto vertex_partition_minor_range_last = vertex_partition_range_lasts[vertex_partition_rank];
      auto segment_size = vertex_partition_minor_range_last - vertex_partition_minor_range_first;
      device_bcast(row_comm,
                   renumber_map_labels,
                   renumber_map_minor_labels.data(),
                   segment_size,
                   i,
                   handle.get_stream());

      kv_store_t<vertex_t, vertex_t, false> renumber_map(
        thrust::make_counting_iterator(vertex_partition_minor_range_first),
        thrust::make_counting_iterator(vertex_partition_minor_range_first) + segment_size,
        renumber_map_minor_labels.begin(),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value,
        handle.get_stream());
      auto renumber_map_view = renumber_map.view();
      for (size_t j = 0; j < edgelist_minors.size(); ++j) {
        renumber_map_view.find(
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i],
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i + 1],
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i],
          handle.get_stream());
      }
    }
  } else {
    auto edge_partition_minor_range_first =
      col_comm_rank * row_comm_size == 0
        ? vertex_t{0}
        : vertex_partition_range_lasts[col_comm_rank * row_comm_size - 1];
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(edge_partition_minor_range_size,
                                                            handle.get_stream());
    std::vector<size_t> recvcounts(row_comm_size);
    for (int i = 0; i < row_comm_size; ++i) {
      auto vertex_partition_rank = col_comm_rank * row_comm_size + i;
      recvcounts[i] =
        vertex_partition_range_lasts[vertex_partition_rank] -
        (vertex_partition_rank == 0 ? vertex_t{0}
                                    : vertex_partition_range_lasts[vertex_partition_rank - 1]);
    }
    std::vector<size_t> displacements(recvcounts.size(), 0);
    std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
    device_allgatherv(row_comm,
                      renumber_map_labels,
                      renumber_map_minor_labels.begin(),
                      recvcounts,
                      displacements,
                      handle.get_stream());

    kv_store_t<vertex_t, vertex_t, false> renumber_map(
      thrust::make_counting_iterator(edge_partition_minor_range_first),
      thrust::make_counting_iterator(edge_partition_minor_range_first) +
        renumber_map_minor_labels.size(),
      renumber_map_minor_labels.begin(),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());
    auto renumber_map_view = renumber_map.view();
    for (size_t i = 0; i < edgelist_minors.size(); ++i) {
      renumber_map_view.find(edgelist_minors[i],
                             edgelist_minors[i] + edgelist_edge_counts[i],
                             edgelist_minors[i],
                             handle.get_stream());
    }
  }
}

}  // namespace detail

template <typename vertex_t, bool multi_gpu>
void renumber_ext_vertices(raft::handle_t const& handle,
                           vertex_t* vertices /* [INOUT] */,
                           size_t num_vertices,
                           vertex_t const* renumber_map_labels,
                           vertex_t local_int_vertex_first,
                           vertex_t local_int_vertex_last,
                           bool do_expensive_check)
{
  if (do_expensive_check) {
    rmm::device_uvector<vertex_t> labels(local_int_vertex_last - local_int_vertex_first,
                                         handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 renumber_map_labels,
                 renumber_map_labels + labels.size(),
                 labels.begin());
    thrust::sort(handle.get_thrust_policy(), labels.begin(), labels.end());
    CUGRAPH_EXPECTS(
      thrust::unique(handle.get_thrust_policy(), labels.begin(), labels.end()) == labels.end(),
      "Invalid input arguments: renumber_map_labels have duplicate elements.");
  }

  std::unique_ptr<kv_store_t<vertex_t, vertex_t, false>> renumber_map_ptr{nullptr};
  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    rmm::device_uvector<vertex_t> sorted_unique_ext_vertices(num_vertices, handle.get_stream());
    sorted_unique_ext_vertices.resize(
      thrust::distance(
        sorted_unique_ext_vertices.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        vertices,
                        vertices + num_vertices,
                        sorted_unique_ext_vertices.begin(),
                        [] __device__(auto v) { return v != invalid_vertex_id<vertex_t>::value; })),
      handle.get_stream());
    thrust::sort(handle.get_thrust_policy(),
                 sorted_unique_ext_vertices.begin(),
                 sorted_unique_ext_vertices.end());
    sorted_unique_ext_vertices.resize(
      thrust::distance(sorted_unique_ext_vertices.begin(),
                       thrust::unique(handle.get_thrust_policy(),
                                      sorted_unique_ext_vertices.begin(),
                                      sorted_unique_ext_vertices.end())),
      handle.get_stream());

    kv_store_t<vertex_t, vertex_t, false> local_renumber_map(
      renumber_map_labels,
      renumber_map_labels + (local_int_vertex_last - local_int_vertex_first),
      thrust::make_counting_iterator(local_int_vertex_first),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());  // map local external vertex IDs to local internal vertex IDs

    rmm::device_uvector<vertex_t> int_vertices_for_sorted_unique_ext_vertices(0,
                                                                              handle.get_stream());
    std::tie(sorted_unique_ext_vertices, int_vertices_for_sorted_unique_ext_vertices) =
      collect_values_for_unique_keys(comm,
                                     local_renumber_map.view(),
                                     std::move(sorted_unique_ext_vertices),
                                     detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size},
                                     handle.get_stream());

    renumber_map_ptr = std::make_unique<kv_store_t<vertex_t, vertex_t, false>>(
      sorted_unique_ext_vertices.begin(),
      sorted_unique_ext_vertices.begin() + sorted_unique_ext_vertices.size(),
      int_vertices_for_sorted_unique_ext_vertices.begin(),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());
  } else {
    renumber_map_ptr = std::make_unique<kv_store_t<vertex_t, vertex_t, false>>(
      renumber_map_labels,
      renumber_map_labels + (local_int_vertex_last - local_int_vertex_first),
      thrust::make_counting_iterator(vertex_t{0}),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());
  }
  auto renumber_map_view = renumber_map_ptr->view();

  if (do_expensive_check) {
    rmm::device_uvector<bool> contains(num_vertices, handle.get_stream());
    renumber_map_view.contains(
      vertices, vertices + num_vertices, contains.begin(), handle.get_stream());
    auto vc_pair_first = thrust::make_zip_iterator(thrust::make_tuple(vertices, contains.begin()));
    CUGRAPH_EXPECTS(thrust::count_if(handle.get_thrust_policy(),
                                     vc_pair_first,
                                     vc_pair_first + num_vertices,
                                     [] __device__(auto pair) {
                                       auto v = thrust::get<0>(pair);
                                       auto c = thrust::get<1>(pair);
                                       return v == invalid_vertex_id<vertex_t>::value
                                                ? (c == true)
                                                : (c == false);
                                     }) == 0,
                    "Invalid input arguments: vertices have elements that are missing in "
                    "(aggregate) renumber_map_labels.");
  }

  renumber_map_view.find(vertices, vertices + num_vertices, vertices, handle.get_stream());
}

template <typename vertex_t, bool multi_gpu>
void renumber_local_ext_vertices(raft::handle_t const& handle,
                                 vertex_t* vertices /* [INOUT] */,
                                 size_t num_vertices,
                                 vertex_t const* renumber_map_labels,
                                 vertex_t local_int_vertex_first,
                                 vertex_t local_int_vertex_last,
                                 bool do_expensive_check)
{
  if (do_expensive_check) {
    rmm::device_uvector<vertex_t> labels(local_int_vertex_last - local_int_vertex_first,
                                         handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 renumber_map_labels,
                 renumber_map_labels + labels.size(),
                 labels.begin());
    thrust::sort(handle.get_thrust_policy(), labels.begin(), labels.end());
    CUGRAPH_EXPECTS(
      thrust::unique(handle.get_thrust_policy(), labels.begin(), labels.end()) == labels.end(),
      "Invalid input arguments: renumber_map_labels have duplicate elements.");
  }

  kv_store_t<vertex_t, vertex_t, false> renumber_map(
    renumber_map_labels,
    renumber_map_labels + (local_int_vertex_last - local_int_vertex_first),
    thrust::make_counting_iterator(local_int_vertex_first),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value,
    handle.get_stream());
  auto renumber_map_view = renumber_map.view();

  if (do_expensive_check) {
    rmm::device_uvector<bool> contains(num_vertices, handle.get_stream());
    renumber_map_view.contains(
      vertices, vertices + num_vertices, contains.begin(), handle.get_stream());
    auto vc_pair_first = thrust::make_zip_iterator(thrust::make_tuple(vertices, contains.begin()));
    CUGRAPH_EXPECTS(thrust::count_if(handle.get_thrust_policy(),
                                     vc_pair_first,
                                     vc_pair_first + num_vertices,
                                     [] __device__(auto pair) {
                                       auto v = thrust::get<0>(pair);
                                       auto c = thrust::get<1>(pair);
                                       return v == invalid_vertex_id<vertex_t>::value
                                                ? (c == true)
                                                : (c == false);
                                     }) == 0,
                    "Invalid input arguments: vertices have elements that are missing in "
                    "(aggregate) renumber_map_labels.");
  }

  renumber_map_view.find(vertices, vertices + num_vertices, vertices, handle.get_stream());
}

template <typename vertex_t>
void unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  vertex_t* vertices /* [INOUT] */,
  size_t num_vertices,
  vertex_t const* renumber_map_labels /* size = local_int_vertex_last - local_int_vertex_first */,
  vertex_t local_int_vertex_first,
  vertex_t local_int_vertex_last,
  bool do_expensive_check)
{
  if (do_expensive_check) {
    CUGRAPH_EXPECTS(
      thrust::count_if(handle.get_thrust_policy(),
                       vertices,
                       vertices + num_vertices,
                       [local_int_vertex_first, local_int_vertex_last] __device__(auto v) {
                         return v != invalid_vertex_id<vertex_t>::value &&
                                (v < local_int_vertex_first || v >= local_int_vertex_last);
                       }) == 0,
      "Invalid input arguments: there are non-local vertices in [vertices, vertices "
      "+ num_vertices).");
  }

  thrust::transform(handle.get_thrust_policy(),
                    vertices,
                    vertices + num_vertices,
                    vertices,
                    [renumber_map_labels, local_int_vertex_first] __device__(auto v) {
                      return v == invalid_vertex_id<vertex_t>::value
                               ? v
                               : renumber_map_labels[v - local_int_vertex_first];
                    });
}

template <typename vertex_t, bool multi_gpu>
void unrenumber_int_vertices(raft::handle_t const& handle,
                             vertex_t* vertices /* [INOUT] */,
                             size_t num_vertices,
                             vertex_t const* renumber_map_labels,
                             std::vector<vertex_t> const& vertex_partition_range_lasts,
                             bool do_expensive_check)
{
  if (do_expensive_check) {
    CUGRAPH_EXPECTS(
      thrust::count_if(handle.get_thrust_policy(),
                       vertices,
                       vertices + num_vertices,
                       [int_vertex_last = vertex_partition_range_lasts.back()] __device__(auto v) {
                         return v != invalid_vertex_id<vertex_t>::value &&
                                !is_valid_vertex(int_vertex_last, v);
                       }) == 0,
      "Invalid input arguments: there are out-of-range vertices in [vertices, vertices "
      "+ num_vertices).");
  }

  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto local_int_vertex_first =
      comm_rank == 0 ? vertex_t{0} : vertex_partition_range_lasts[comm_rank - 1];
    auto local_int_vertex_last = vertex_partition_range_lasts[comm_rank];

    rmm::device_uvector<vertex_t> sorted_unique_int_vertices(num_vertices, handle.get_stream());
    sorted_unique_int_vertices.resize(
      thrust::distance(
        sorted_unique_int_vertices.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        vertices,
                        vertices + num_vertices,
                        sorted_unique_int_vertices.begin(),
                        [] __device__(auto v) { return v != invalid_vertex_id<vertex_t>::value; })),
      handle.get_stream());
    thrust::sort(handle.get_thrust_policy(),
                 sorted_unique_int_vertices.begin(),
                 sorted_unique_int_vertices.end());
    sorted_unique_int_vertices.resize(
      thrust::distance(sorted_unique_int_vertices.begin(),
                       thrust::unique(handle.get_thrust_policy(),
                                      sorted_unique_int_vertices.begin(),
                                      sorted_unique_int_vertices.end())),
      handle.get_stream());

    rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
      vertex_partition_range_lasts.size(), handle.get_stream());
    raft::update_device(d_vertex_partition_range_lasts.data(),
                        vertex_partition_range_lasts.data(),
                        vertex_partition_range_lasts.size(),
                        handle.get_stream());
    rmm::device_uvector<size_t> d_tx_int_vertex_offsets(d_vertex_partition_range_lasts.size(),
                                                        handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        sorted_unique_int_vertices.begin(),
                        sorted_unique_int_vertices.end(),
                        d_vertex_partition_range_lasts.begin(),
                        d_vertex_partition_range_lasts.end(),
                        d_tx_int_vertex_offsets.begin());
    std::vector<size_t> h_tx_int_vertex_counts(d_tx_int_vertex_offsets.size());
    raft::update_host(h_tx_int_vertex_counts.data(),
                      d_tx_int_vertex_offsets.data(),
                      d_tx_int_vertex_offsets.size(),
                      handle.get_stream());
    handle.sync_stream();
    std::adjacent_difference(
      h_tx_int_vertex_counts.begin(), h_tx_int_vertex_counts.end(), h_tx_int_vertex_counts.begin());

    rmm::device_uvector<vertex_t> rx_int_vertices(0, handle.get_stream());
    std::vector<size_t> rx_int_vertex_counts{};
    std::tie(rx_int_vertices, rx_int_vertex_counts) = shuffle_values(
      comm, sorted_unique_int_vertices.begin(), h_tx_int_vertex_counts, handle.get_stream());

    auto tx_ext_vertices = std::move(rx_int_vertices);
    thrust::transform(handle.get_thrust_policy(),
                      tx_ext_vertices.begin(),
                      tx_ext_vertices.end(),
                      tx_ext_vertices.begin(),
                      [renumber_map_labels, local_int_vertex_first] __device__(auto v) {
                        return renumber_map_labels[v - local_int_vertex_first];
                      });

    rmm::device_uvector<vertex_t> rx_ext_vertices_for_sorted_unique_int_vertices(
      0, handle.get_stream());
    std::tie(rx_ext_vertices_for_sorted_unique_int_vertices, std::ignore) =
      shuffle_values(comm, tx_ext_vertices.begin(), rx_int_vertex_counts, handle.get_stream());

    kv_store_t<vertex_t, vertex_t, false> renumber_map(
      sorted_unique_int_vertices.begin(),
      sorted_unique_int_vertices.begin() + sorted_unique_int_vertices.size(),
      rx_ext_vertices_for_sorted_unique_int_vertices.begin(),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());
    auto renumber_map_view = renumber_map.view();
    renumber_map_view.find(vertices, vertices + num_vertices, vertices, handle.get_stream());
  } else {
    unrenumber_local_int_vertices(handle,
                                  vertices,
                                  num_vertices,
                                  renumber_map_labels,
                                  vertex_t{0},
                                  vertex_partition_range_lasts[0],
                                  do_expensive_check);
  }
}

template <typename vertex_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<multi_gpu, void> unrenumber_local_int_edges(
  raft::handle_t const& handle,
  std::vector<vertex_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<vertex_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  vertex_t const* renumber_map_labels,
  std::vector<vertex_t> const& vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check)
{
  return detail::unrenumber_local_int_edges(handle,
                                            store_transposed ? edgelist_dsts : edgelist_srcs,
                                            store_transposed ? edgelist_srcs : edgelist_dsts,
                                            edgelist_edge_counts,
                                            renumber_map_labels,
                                            vertex_partition_range_lasts,
                                            edgelist_intra_partition_segment_offsets,
                                            do_expensive_check);
}

template <typename vertex_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<!multi_gpu, void> unrenumber_local_int_edges(raft::handle_t const& handle,
                                                              vertex_t* edgelist_srcs /* [INOUT] */,
                                                              vertex_t* edgelist_dsts /* [INOUT] */,
                                                              size_t num_edgelist_edges,
                                                              vertex_t const* renumber_map_labels,
                                                              vertex_t num_vertices,
                                                              bool do_expensive_check)
{
  unrenumber_local_int_vertices(handle,
                                edgelist_srcs,
                                num_edgelist_edges,
                                renumber_map_labels,
                                vertex_t{0},
                                num_vertices,
                                do_expensive_check);
  unrenumber_local_int_vertices(handle,
                                edgelist_dsts,
                                num_edgelist_edges,
                                renumber_map_labels,
                                vertex_t{0},
                                num_vertices,
                                do_expensive_check);
}

}  // namespace cugraph

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuco/static_map.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace detail {

// returns renumber map, segment_offsets, and # unique edge majors & minors
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<vertex_t>, vertex_t, vertex_t>
compute_renumber_map(raft::handle_t const& handle,
                     std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
                     std::vector<vertex_t const*> const& edgelist_majors,
                     std::vector<vertex_t const*> const& edgelist_minors,
                     std::vector<edge_t> const& edgelist_edge_counts)
{
  // FIXME: compare this sort based approach with hash based approach in both speed and memory
  // footprint

  // 1. acquire (unique major label, count) pairs

  rmm::device_uvector<vertex_t> major_labels(0, handle.get_stream());
  rmm::device_uvector<edge_t> major_counts(0, handle.get_stream());
  vertex_t num_local_unique_edge_majors{0};
  for (size_t i = 0; i < edgelist_majors.size(); ++i) {
    rmm::device_uvector<vertex_t> tmp_major_labels(0, handle.get_stream());
    rmm::device_uvector<edge_t> tmp_major_counts(0, handle.get_stream());
    {
      rmm::device_uvector<vertex_t> sorted_major_labels(edgelist_edge_counts[i],
                                                        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_majors[i],
                   edgelist_majors[i] + edgelist_edge_counts[i],
                   sorted_major_labels.begin());
      // FIXME: better refactor this sort-count_if-reduce_by_key routine for reuse
      thrust::sort(
        handle.get_thrust_policy(), sorted_major_labels.begin(), sorted_major_labels.end());
      auto num_unique_labels =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(sorted_major_labels.size()),
                         [labels = sorted_major_labels.data()] __device__(auto i) {
                           return (i == 0) || (labels[i - 1] != labels[i]);
                         });
      tmp_major_labels.resize(num_unique_labels, handle.get_stream());
      tmp_major_counts.resize(tmp_major_labels.size(), handle.get_stream());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            sorted_major_labels.begin(),
                            sorted_major_labels.end(),
                            thrust::make_constant_iterator(edge_t{1}),
                            tmp_major_labels.begin(),
                            tmp_major_counts.begin());
    }
    num_local_unique_edge_majors += static_cast<vertex_t>(tmp_major_labels.size());

    if (multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      rmm::device_uvector<vertex_t> rx_major_labels(0, handle.get_stream());
      rmm::device_uvector<edge_t> rx_major_counts(0, handle.get_stream());
      auto rx_sizes = host_scalar_gather(
        col_comm, tmp_major_labels.size(), static_cast<int>(i), handle.get_stream());
      std::vector<size_t> rx_displs{};
      if (static_cast<int>(i) == col_comm_rank) {
        rx_displs.assign(col_comm_size, size_t{0});
        std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);
        rx_major_labels.resize(rx_displs.back() + rx_sizes.back(), handle.get_stream());
        rx_major_counts.resize(rx_major_labels.size(), handle.get_stream());
      }
      device_gatherv(col_comm,
                     thrust::make_zip_iterator(
                       thrust::make_tuple(tmp_major_labels.begin(), tmp_major_counts.begin())),
                     thrust::make_zip_iterator(
                       thrust::make_tuple(rx_major_labels.begin(), rx_major_counts.begin())),
                     tmp_major_labels.size(),
                     rx_sizes,
                     rx_displs,
                     static_cast<int>(i),
                     handle.get_stream());
      if (static_cast<int>(i) == col_comm_rank) {
        major_labels = std::move(rx_major_labels);
        major_counts = std::move(rx_major_counts);
      }
    } else {
      assert(i == 0);
      major_labels = std::move(tmp_major_labels);
      major_counts = std::move(tmp_major_counts);
    }
  }
  if (multi_gpu) {
    // FIXME: better refactor this sort-count_if-reduce_by_key routine for reuse
    thrust::sort_by_key(
      handle.get_thrust_policy(), major_labels.begin(), major_labels.end(), major_counts.begin());
    auto num_unique_labels = thrust::count_if(handle.get_thrust_policy(),
                                              thrust::make_counting_iterator(size_t{0}),
                                              thrust::make_counting_iterator(major_labels.size()),
                                              [labels = major_labels.data()] __device__(auto i) {
                                                return (i == 0) || (labels[i - 1] != labels[i]);
                                              });
    rmm::device_uvector<vertex_t> tmp_major_labels(num_unique_labels, handle.get_stream());
    rmm::device_uvector<edge_t> tmp_major_counts(tmp_major_labels.size(), handle.get_stream());
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          major_labels.begin(),
                          major_labels.end(),
                          major_counts.begin(),
                          tmp_major_labels.begin(),
                          tmp_major_counts.begin());
    major_labels = std::move(tmp_major_labels);
    major_counts = std::move(tmp_major_counts);
  }

  // 2. acquire unique minor labels

  std::vector<edge_t> minor_displs(edgelist_minors.size(), edge_t{0});
  std::partial_sum(
    edgelist_edge_counts.begin(), edgelist_edge_counts.end() - 1, minor_displs.begin() + 1);
  rmm::device_uvector<vertex_t> minor_labels(minor_displs.back() + edgelist_edge_counts.back(),
                                             handle.get_stream());
  vertex_t minor_offset{0};
  for (size_t i = 0; i < edgelist_minors.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_minors[i],
                 edgelist_minors[i] + edgelist_edge_counts[i],
                 minor_labels.begin() + minor_offset);
    thrust::sort(handle.get_thrust_policy(),
                 minor_labels.begin() + minor_offset,
                 minor_labels.begin() + minor_offset + edgelist_edge_counts[i]);
    minor_offset += thrust::distance(
      minor_labels.begin() + minor_offset,
      thrust::unique(handle.get_thrust_policy(),
                     minor_labels.begin() + minor_offset,
                     minor_labels.begin() + minor_offset + edgelist_edge_counts[i]));
  }
  minor_labels.resize(minor_offset, handle.get_stream());
  thrust::sort(handle.get_thrust_policy(), minor_labels.begin(), minor_labels.end());
  minor_labels.resize(
    thrust::distance(
      minor_labels.begin(),
      thrust::unique(handle.get_thrust_policy(), minor_labels.begin(), minor_labels.end())),
    handle.get_stream());
  auto num_local_unique_edge_minors = static_cast<vertex_t>(minor_labels.size());
  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();

    if (row_comm_size > 1) {
      rmm::device_uvector<vertex_t> rx_minor_labels(0, handle.get_stream());
      std::tie(rx_minor_labels, std::ignore) = groupby_gpuid_and_shuffle_values(
        row_comm,
        minor_labels.begin(),
        minor_labels.end(),
        [key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{row_comm_size}] __device__(
          auto val) { return key_func(val); },
        handle.get_stream());
      thrust::sort(handle.get_thrust_policy(), rx_minor_labels.begin(), rx_minor_labels.end());
      rx_minor_labels.resize(thrust::distance(rx_minor_labels.begin(),
                                              thrust::unique(handle.get_thrust_policy(),
                                                             rx_minor_labels.begin(),
                                                             rx_minor_labels.end())),
                             handle.get_stream());
      minor_labels = std::move(rx_minor_labels);
    }
  }
  minor_labels.shrink_to_fit(handle.get_stream_view());

  // 3. merge major and minor labels and vertex labels

  rmm::device_uvector<vertex_t> merged_labels(major_labels.size() + minor_labels.size(),
                                              handle.get_stream_view());
  rmm::device_uvector<edge_t> merged_counts(merged_labels.size(), handle.get_stream_view());
  thrust::merge_by_key(handle.get_thrust_policy(),
                       major_labels.begin(),
                       major_labels.end(),
                       minor_labels.begin(),
                       minor_labels.end(),
                       major_counts.begin(),
                       thrust::make_constant_iterator(edge_t{0}),
                       merged_labels.begin(),
                       merged_counts.begin());

  major_labels.resize(0, handle.get_stream());
  major_counts.resize(0, handle.get_stream());
  minor_labels.resize(0, handle.get_stream());
  major_labels.shrink_to_fit(handle.get_stream());
  major_counts.shrink_to_fit(handle.get_stream());
  minor_labels.shrink_to_fit(handle.get_stream());

  rmm::device_uvector<vertex_t> labels(merged_labels.size(), handle.get_stream());
  rmm::device_uvector<edge_t> counts(labels.size(), handle.get_stream());
  auto pair_it = thrust::reduce_by_key(handle.get_thrust_policy(),
                                       merged_labels.begin(),
                                       merged_labels.end(),
                                       merged_counts.begin(),
                                       labels.begin(),
                                       counts.begin());
  merged_labels.resize(0, handle.get_stream());
  merged_counts.resize(0, handle.get_stream());
  merged_labels.shrink_to_fit(handle.get_stream());
  merged_counts.shrink_to_fit(handle.get_stream());
  labels.resize(thrust::distance(labels.begin(), thrust::get<0>(pair_it)), handle.get_stream());
  counts.resize(labels.size(), handle.get_stream());
  labels.shrink_to_fit(handle.get_stream());
  counts.shrink_to_fit(handle.get_stream());

  auto num_non_isolated_vertices = static_cast<vertex_t>(labels.size());

  // 4. if local_vertices.has_value() == true, append isolated vertices

  if (local_vertices) {
    rmm::device_uvector<vertex_t> isolated_vertices(0, handle.get_stream());

    auto num_isolated_vertices = thrust::count_if(
      handle.get_thrust_policy(),
      (*local_vertices).begin(),
      (*local_vertices).end(),
      [label_first = labels.begin(), label_last = labels.end()] __device__(auto v) {
        return !thrust::binary_search(thrust::seq, label_first, label_last, v);
      });
    isolated_vertices.resize(num_isolated_vertices, handle.get_stream());
    thrust::copy_if(handle.get_thrust_policy(),
                    (*local_vertices).begin(),
                    (*local_vertices).end(),
                    isolated_vertices.begin(),
                    [label_first = labels.begin(), label_last = labels.end()] __device__(auto v) {
                      return !thrust::binary_search(thrust::seq, label_first, label_last, v);
                    });
    (*local_vertices).resize(0, handle.get_stream());
    (*local_vertices).shrink_to_fit(handle.get_stream());

    if (isolated_vertices.size() > 0) {
      labels.resize(labels.size() + isolated_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   isolated_vertices.begin(),
                   isolated_vertices.end(),
                   labels.end() - isolated_vertices.size());
    }
  }

  // 5. sort non-isolated vertices by degree

  thrust::sort_by_key(handle.get_thrust_policy(),
                      counts.begin(),
                      counts.begin() + num_non_isolated_vertices,
                      labels.begin(),
                      thrust::greater<edge_t>());

  // 6. compute segment_offsets

  static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
  static_assert((detail::low_degree_threshold <= detail::mid_degree_threshold) &&
                (detail::mid_degree_threshold <= std::numeric_limits<edge_t>::max()));
  size_t mid_degree_threshold{detail::mid_degree_threshold};
  size_t low_degree_threshold{detail::low_degree_threshold};
  size_t hypersparse_degree_threshold{0};
  if (multi_gpu) {
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();
    mid_degree_threshold *= col_comm_size;
    low_degree_threshold *= col_comm_size;
    hypersparse_degree_threshold =
      static_cast<size_t>(col_comm_size * detail::hypersparse_threshold_ratio);
  }
  auto num_segments_per_vertex_partition =
    detail::num_sparse_segments_per_vertex_partition +
    (hypersparse_degree_threshold > 0 ? size_t{1} : size_t{0});
  rmm::device_uvector<edge_t> d_thresholds(num_segments_per_vertex_partition - 1,
                                           handle.get_stream());
  auto h_thresholds = hypersparse_degree_threshold > 0
                        ? std::vector<edge_t>{static_cast<edge_t>(mid_degree_threshold),
                                              static_cast<edge_t>(low_degree_threshold),
                                              static_cast<edge_t>(hypersparse_degree_threshold)}
                        : std::vector<edge_t>{static_cast<edge_t>(mid_degree_threshold),
                                              static_cast<edge_t>(low_degree_threshold)};
  raft::update_device(
    d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());

  rmm::device_uvector<vertex_t> d_segment_offsets(num_segments_per_vertex_partition + 1,
                                                  handle.get_stream());

  auto zero_vertex  = vertex_t{0};
  auto vertex_count = static_cast<vertex_t>(labels.size());
  d_segment_offsets.set_element_async(0, zero_vertex, handle.get_stream());
  d_segment_offsets.set_element_async(
    num_segments_per_vertex_partition, vertex_count, handle.get_stream());

  thrust::upper_bound(handle.get_thrust_policy(),
                      counts.begin(),
                      counts.end(),
                      d_thresholds.begin(),
                      d_thresholds.end(),
                      d_segment_offsets.begin() + 1,
                      thrust::greater<edge_t>{});

  std::vector<vertex_t> h_segment_offsets(d_segment_offsets.size());
  raft::update_host(h_segment_offsets.data(),
                    d_segment_offsets.data(),
                    d_segment_offsets.size(),
                    handle.get_stream());
  handle.get_stream_view().synchronize();

  return std::make_tuple(std::move(labels),
                         h_segment_offsets,
                         num_local_unique_edge_majors,
                         num_local_unique_edge_minors);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void expensive_check_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>> const& local_vertices,
  std::vector<vertex_t const*> const& edgelist_majors,
  std::vector<vertex_t const*> const& edgelist_minors,
  std::vector<edge_t> const& edgelist_edge_counts,
  std::optional<std::vector<std::vector<edge_t>>> const& edgelist_intra_partition_segment_offsets)
{
  std::optional<rmm::device_uvector<vertex_t>> sorted_local_vertices{std::nullopt};
  if (local_vertices) {
    sorted_local_vertices =
      rmm::device_uvector<vertex_t>((*local_vertices).size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*local_vertices).begin(),
                 (*local_vertices).end(),
                 (*sorted_local_vertices).begin());
    thrust::sort(
      handle.get_thrust_policy(), (*sorted_local_vertices).begin(), (*sorted_local_vertices).end());
    CUGRAPH_EXPECTS(
      static_cast<size_t>(thrust::distance((*sorted_local_vertices).begin(),
                                           thrust::unique(handle.get_thrust_policy(),
                                                          (*sorted_local_vertices).begin(),
                                                          (*sorted_local_vertices).end()))) ==
        (*sorted_local_vertices).size(),
      "Invalid input argument: (local_)vertices should not have duplicates.");
  }

  if constexpr (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto const row_comm_rank = row_comm.get_rank();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();
    auto const col_comm_rank = col_comm.get_rank();

    CUGRAPH_EXPECTS((edgelist_majors.size() == edgelist_minors.size()) &&
                      (edgelist_majors.size() == static_cast<size_t>(col_comm_size)),
                    "Invalid input argument: both edgelist_majors.size() & "
                    "edgelist_minors.size() should coincide with col_comm_size.");

    if (sorted_local_vertices) {
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          (*sorted_local_vertices).begin(),
          (*sorted_local_vertices).end(),
          [comm_rank,
           key_func =
             detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(auto val) {
            return key_func(val) != comm_rank;
          }) == 0,
        "Invalid input argument: local_vertices should be pre-shuffled.");
    }

    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[i], edgelist_minors[i]));
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + edgelist_edge_counts[i],
          [comm_size,
           comm_rank,
           row_comm_rank,
           col_comm_size,
           col_comm_rank,
           i,
           gpu_id_key_func =
             detail::compute_gpu_id_from_edge_t<vertex_t>{comm_size, row_comm_size, col_comm_size},
           partition_id_key_func =
             detail::compute_partition_id_from_edge_t<vertex_t>{
               comm_size, row_comm_size, col_comm_size}] __device__(auto edge) {
            return (gpu_id_key_func(thrust::get<0>(edge), thrust::get<1>(edge)) != comm_rank) ||
                   (partition_id_key_func(thrust::get<0>(edge), thrust::get<1>(edge)) !=
                    row_comm_rank * col_comm_size + col_comm_rank + i * comm_size);
          }) == 0,
        "Invalid input argument: edgelist_majors & edgelist_minors should be "
        "pre-shuffled.");

      if (sorted_local_vertices) {
        auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

        rmm::device_uvector<vertex_t> sorted_majors(0, handle.get_stream());
        {
          auto recvcounts =
            host_scalar_allgather(col_comm, (*sorted_local_vertices).size(), handle.get_stream());
          std::vector<size_t> displacements(recvcounts.size(), size_t{0});
          std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
          sorted_majors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
          device_allgatherv(col_comm,
                            (*sorted_local_vertices).data(),
                            sorted_majors.data(),
                            recvcounts,
                            displacements,
                            handle.get_stream());
          thrust::sort(handle.get_thrust_policy(), sorted_majors.begin(), sorted_majors.end());
        }

        rmm::device_uvector<vertex_t> sorted_minors(0, handle.get_stream());
        {
          auto recvcounts =
            host_scalar_allgather(row_comm, (*sorted_local_vertices).size(), handle.get_stream());
          std::vector<size_t> displacements(recvcounts.size(), size_t{0});
          std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
          sorted_minors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
          device_allgatherv(row_comm,
                            (*sorted_local_vertices).data(),
                            sorted_minors.data(),
                            recvcounts,
                            displacements,
                            handle.get_stream());
          thrust::sort(handle.get_thrust_policy(), sorted_minors.begin(), sorted_minors.end());
        }

        auto edge_first =
          thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[i], edgelist_minors[i]));
        CUGRAPH_EXPECTS(
          thrust::count_if(
            handle.get_thrust_policy(),
            edge_first,
            edge_first + edgelist_edge_counts[i],
            [num_majors    = static_cast<vertex_t>(sorted_majors.size()),
             sorted_majors = sorted_majors.data(),
             num_minors    = static_cast<vertex_t>(sorted_minors.size()),
             sorted_minors = sorted_minors.data()] __device__(auto e) {
              return !thrust::binary_search(
                       thrust::seq, sorted_majors, sorted_majors + num_majors, thrust::get<0>(e)) ||
                     !thrust::binary_search(
                       thrust::seq, sorted_minors, sorted_minors + num_minors, thrust::get<1>(e));
            }) == 0,
          "Invalid input argument: edgelist_majors and/or edgelist_minors have "
          "invalid vertex ID(s).");
      }

      if (edgelist_intra_partition_segment_offsets) {
        for (int j = 0; j < row_comm_size; ++j) {
          CUGRAPH_EXPECTS(
            thrust::count_if(
              handle.get_thrust_policy(),
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j],
              edgelist_minors[i] + (*edgelist_intra_partition_segment_offsets)[i][j + 1],
              [row_comm_size,
               col_comm_rank,
               j,
               gpu_id_key_func =
                 detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(auto minor) {
                return gpu_id_key_func(minor) != col_comm_rank * row_comm_size + j;
              }) == 0,
            "Invalid input argument: if edgelist_intra_partition_segment_offsets.has_value() is "
            "true, edgelist_majors & edgelist_minors should be properly grouped "
            "within each local partition.");
        }
      }
    }
  } else {
    assert(edgelist_majors.size() == 1);
    assert(edgelist_minors.size() == 1);

    if (sorted_local_vertices) {
      auto edge_first =
        thrust::make_zip_iterator(thrust::make_tuple(edgelist_majors[0], edgelist_minors[0]));
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + edgelist_edge_counts[0],
          [sorted_local_vertices = (*sorted_local_vertices).data(),
           num_sorted_local_vertices =
             static_cast<vertex_t>((*sorted_local_vertices).size())] __device__(auto e) {
            return !thrust::binary_search(thrust::seq,
                                          sorted_local_vertices,
                                          sorted_local_vertices + num_sorted_local_vertices,
                                          thrust::get<0>(e)) ||
                   !thrust::binary_search(thrust::seq,
                                          sorted_local_vertices,
                                          sorted_local_vertices + num_sorted_local_vertices,
                                          thrust::get<1>(e));
          }) == 0,
        "Invalid input argument: edgelist_majors and/or edgelist_minors have "
        "invalid vertex ID(s).");
    }

    CUGRAPH_EXPECTS(
      edgelist_intra_partition_segment_offsets.has_value() == false,
      "Invalid input argument: edgelist_intra_partition_segment_offsets.has_value() should "
      "be false for single-GPU.");
  }
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
renumber_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<vertex_t*> const& edgelist_majors /* [INOUT] */,
  std::vector<vertex_t*> const& edgelist_minors /* [INOUT] */,
  std::vector<edge_t> const& edgelist_edge_counts,
  std::optional<std::vector<std::vector<edge_t>>> const& edgelist_intra_partition_segment_offsets,
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

  std::vector<vertex_t const*> edgelist_const_majors(edgelist_majors.size());
  std::vector<vertex_t const*> edgelist_const_minors(edgelist_const_majors.size());
  for (size_t i = 0; i < edgelist_const_majors.size(); ++i) {
    edgelist_const_majors[i] = edgelist_majors[i];
    edgelist_const_minors[i] = edgelist_minors[i];
  }

  if (do_expensive_check) {
    detail::expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      local_vertices,
      edgelist_const_majors,
      edgelist_const_minors,
      edgelist_edge_counts,
      edgelist_intra_partition_segment_offsets);
  }

  // 1. compute renumber map

  auto [renumber_map_labels,
        vertex_partition_segment_offsets,
        num_unique_edge_majors,
        num_unique_edge_minors] =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(handle,
                                                              std::move(local_vertices),
                                                              edgelist_const_majors,
                                                              edgelist_const_minors,
                                                              edgelist_edge_counts);

  // 2. initialize partition_t object, number_of_vertices, and number_of_edges for the coarsened
  // graph

  auto vertex_counts = host_scalar_allgather(
    comm, static_cast<vertex_t>(renumber_map_labels.size()), handle.get_stream());
  std::vector<vertex_t> vertex_partition_offsets(comm_size + 1, 0);
  std::partial_sum(
    vertex_counts.begin(), vertex_counts.end(), vertex_partition_offsets.begin() + 1);

  partition_t<vertex_t> partition(
    vertex_partition_offsets, row_comm_size, col_comm_size, row_comm_rank, col_comm_rank);

  auto number_of_vertices = vertex_partition_offsets.back();
  auto number_of_edges    = host_scalar_allreduce(
    comm,
    std::accumulate(edgelist_edge_counts.begin(), edgelist_edge_counts.end(), edge_t{0}),
    raft::comms::op_t::SUM,
    handle.get_stream());

  // 3. renumber edges

  double constexpr load_factor = 0.7;

  // FIXME: compare this hash based approach with a binary search based approach in both memory
  // footprint and execution time

  {
    vertex_t max_matrix_partition_major_size{0};
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      max_matrix_partition_major_size =
        std::max(max_matrix_partition_major_size, partition.get_matrix_partition_major_size(i));
    }
    rmm::device_uvector<vertex_t> renumber_map_major_labels(max_matrix_partition_major_size,
                                                            handle.get_stream());
    for (size_t i = 0; i < edgelist_majors.size(); ++i) {
      device_bcast(col_comm,
                   renumber_map_labels.data(),
                   renumber_map_major_labels.data(),
                   partition.get_matrix_partition_major_size(i),
                   i,
                   handle.get_stream());

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // cuco::static_map currently does not take stream

      auto poly_alloc =
        rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
      auto stream_adapter =
        rmm::mr::make_stream_allocator_adaptor(poly_alloc, cudaStream_t{nullptr});
      cuco::static_map<vertex_t, vertex_t, cuda::thread_scope_device, decltype(stream_adapter)>
        renumber_map{
          // cuco::static_map requires at least one empty slot
          std::max(
            static_cast<size_t>(static_cast<double>(partition.get_matrix_partition_major_size(i)) /
                                load_factor),
            static_cast<size_t>(partition.get_matrix_partition_major_size(i)) + 1),
          invalid_vertex_id<vertex_t>::value,
          invalid_vertex_id<vertex_t>::value,
          stream_adapter};
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        renumber_map_major_labels.begin(),
        thrust::make_counting_iterator(partition.get_matrix_partition_major_first(i))));
      renumber_map.insert(pair_first, pair_first + partition.get_matrix_partition_major_size(i));
      renumber_map.find(
        edgelist_majors[i], edgelist_majors[i] + edgelist_edge_counts[i], edgelist_majors[i]);
    }
  }

  if ((partition.get_matrix_partition_minor_size() >= number_of_edges / comm_size) &&
      edgelist_intra_partition_segment_offsets) {  // memory footprint dominated by the O(V/sqrt(P))
                                                   // part than the O(E/P) part
    vertex_t max_segment_size{0};
    for (int i = 0; i < row_comm_size; ++i) {
      max_segment_size = std::max(
        max_segment_size, partition.get_vertex_partition_size(col_comm_rank * row_comm_size + i));
    }
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(max_segment_size, handle.get_stream());
    for (int i = 0; i < row_comm_size; ++i) {
      auto segment_size = partition.get_vertex_partition_size(col_comm_rank * row_comm_size + i);
      device_bcast(row_comm,
                   renumber_map_labels.data(),
                   renumber_map_minor_labels.data(),
                   segment_size,
                   i,
                   handle.get_stream());

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // cuco::static_map currently does not take stream

      auto poly_alloc =
        rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
      auto stream_adapter =
        rmm::mr::make_stream_allocator_adaptor(poly_alloc, cudaStream_t{nullptr});
      cuco::static_map<vertex_t, vertex_t, cuda::thread_scope_device, decltype(stream_adapter)>
        renumber_map{// cuco::static_map requires at least one empty slot
                     std::max(static_cast<size_t>(static_cast<double>(segment_size) / load_factor),
                              static_cast<size_t>(segment_size) + 1),
                     invalid_vertex_id<vertex_t>::value,
                     invalid_vertex_id<vertex_t>::value,
                     stream_adapter};
      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        renumber_map_minor_labels.begin(),
        thrust::make_counting_iterator(
          partition.get_vertex_partition_first(col_comm_rank * row_comm_size + i))));
      renumber_map.insert(pair_first, pair_first + segment_size);
      for (size_t j = 0; j < edgelist_minors.size(); ++j) {
        renumber_map.find(
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i],
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i + 1],
          edgelist_minors[j] + (*edgelist_intra_partition_segment_offsets)[j][i]);
      }
    }
  } else {
    rmm::device_uvector<vertex_t> renumber_map_minor_labels(
      partition.get_matrix_partition_minor_size(), handle.get_stream());
    std::vector<size_t> recvcounts(row_comm_size);
    for (int i = 0; i < row_comm_size; ++i) {
      recvcounts[i] = partition.get_vertex_partition_size(col_comm_rank * row_comm_size + i);
    }
    std::vector<size_t> displacements(recvcounts.size(), 0);
    std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
    device_allgatherv(row_comm,
                      renumber_map_labels.begin(),
                      renumber_map_minor_labels.begin(),
                      recvcounts,
                      displacements,
                      handle.get_stream());

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // cuco::static_map currently does not take stream

    auto poly_alloc = rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
    auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(poly_alloc, cudaStream_t{nullptr});
    cuco::static_map<vertex_t, vertex_t, cuda::thread_scope_device, decltype(stream_adapter)>
      renumber_map{// cuco::static_map requires at least one empty slot
                   std::max(static_cast<size_t>(
                              static_cast<double>(renumber_map_minor_labels.size()) / load_factor),
                            renumber_map_minor_labels.size() + 1),
                   invalid_vertex_id<vertex_t>::value,
                   invalid_vertex_id<vertex_t>::value,
                   stream_adapter};
    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      renumber_map_minor_labels.begin(),
      thrust::make_counting_iterator(partition.get_matrix_partition_minor_first())));
    renumber_map.insert(pair_first, pair_first + renumber_map_minor_labels.size());
    for (size_t i = 0; i < edgelist_minors.size(); ++i) {
      renumber_map.find(
        edgelist_minors[i], edgelist_minors[i] + edgelist_edge_counts[i], edgelist_minors[i]);
    }
  }

  return std::make_tuple(
    std::move(renumber_map_labels),
    renumber_meta_t<vertex_t, edge_t, multi_gpu>{number_of_vertices,
                                                 number_of_edges,
                                                 partition,
                                                 vertex_partition_segment_offsets,
                                                 num_unique_edge_majors,
                                                 num_unique_edge_minors});
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
renumber_edgelist(raft::handle_t const& handle,
                  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                  vertex_t* edgelist_majors /* [INOUT] */,
                  vertex_t* edgelist_minors /* [INOUT] */,
                  edge_t num_edgelist_edges,
                  bool do_expensive_check)
{
  if (do_expensive_check) {
    detail::expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      vertices,
      std::vector<vertex_t const*>{edgelist_majors},
      std::vector<vertex_t const*>{edgelist_minors},
      std::vector<edge_t>{num_edgelist_edges},
      std::nullopt);
  }

  rmm::device_uvector<vertex_t> renumber_map_labels(0, handle.get_stream());
  std::vector<vertex_t> segment_offsets{};
  std::tie(renumber_map_labels, segment_offsets, std::ignore, std::ignore) =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(
      handle,
      std::move(vertices),
      std::vector<vertex_t const*>{edgelist_majors},
      std::vector<vertex_t const*>{edgelist_minors},
      std::vector<edge_t>{num_edgelist_edges});

  double constexpr load_factor = 0.7;

  // FIXME: compare this hash based approach with a binary search based approach in both memory
  // footprint and execution time

  auto poly_alloc = rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
  auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(poly_alloc, cudaStream_t{nullptr});
  cuco::static_map<vertex_t, vertex_t, cuda::thread_scope_device, decltype(stream_adapter)>
    renumber_map{
      // cuco::static_map requires at least one empty slot
      std::max(static_cast<size_t>(static_cast<double>(renumber_map_labels.size()) / load_factor),
               renumber_map_labels.size() + 1),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      stream_adapter};
  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(renumber_map_labels.begin(), thrust::make_counting_iterator(vertex_t{0})));
  renumber_map.insert(pair_first, pair_first + renumber_map_labels.size());
  renumber_map.find(edgelist_majors, edgelist_majors + num_edgelist_edges, edgelist_majors);
  renumber_map.find(edgelist_minors, edgelist_minors + num_edgelist_edges, edgelist_minors);

  return std::make_tuple(std::move(renumber_map_labels),
                         renumber_meta_t<vertex_t, edge_t, multi_gpu>{segment_offsets});
}

}  // namespace cugraph

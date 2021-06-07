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

#include <cugraph/experimental/detail/graph_utils.cuh>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/experimental/graph_view.hpp>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_barrier.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuco/static_map.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
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
namespace experimental {
namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_renumber_map(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_vertex_span,
  std::vector<vertex_t const*> const& edgelist_major_vertices,
  std::vector<vertex_t const*> const& edgelist_minor_vertices,
  std::vector<edge_t> const& edgelist_edge_counts)
{
  // FIXME: compare this sort based approach with hash based approach in both speed and memory
  // footprint

  // 1. acquire (unique major label, count) pairs

  if (multi_gpu) {
    auto& comm = handle.get_comms();

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (beginning of col_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    ;
    comm.barrier();  // currently, this is ncclAllReduce
#endif
  }

  rmm::device_uvector<vertex_t> major_labels(0, handle.get_stream());
  rmm::device_uvector<edge_t> major_counts(0, handle.get_stream());
  for (size_t i = 0; i < edgelist_major_vertices.size(); ++i) {
    rmm::device_uvector<vertex_t> tmp_major_labels(0, handle.get_stream());
    rmm::device_uvector<edge_t> tmp_major_counts(0, handle.get_stream());
    {
      rmm::device_uvector<vertex_t> sorted_major_labels(edgelist_edge_counts[i],
                                                        handle.get_stream());
      thrust::copy(rmm::exec_policy(handle.get_stream_view()),
                   edgelist_major_vertices[i],
                   edgelist_major_vertices[i] + edgelist_edge_counts[i],
                   sorted_major_labels.begin());
      // FIXME: better refactor this sort-count_if-reduce_by_key routine for reuse
      thrust::sort(rmm::exec_policy(handle.get_stream_view()),
                   sorted_major_labels.begin(),
                   sorted_major_labels.end());
      auto num_unique_labels =
        thrust::count_if(rmm::exec_policy(handle.get_stream_view()),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(sorted_major_labels.size()),
                         [labels = sorted_major_labels.data()] __device__(auto i) {
                           return (i == 0) || (labels[i - 1] != labels[i]);
                         });
      tmp_major_labels.resize(num_unique_labels, handle.get_stream());
      tmp_major_counts.resize(tmp_major_labels.size(), handle.get_stream());
      thrust::reduce_by_key(rmm::exec_policy(handle.get_stream_view()),
                            sorted_major_labels.begin(),
                            sorted_major_labels.end(),
                            thrust::make_constant_iterator(edge_t{1}),
                            tmp_major_labels.begin(),
                            tmp_major_counts.begin());
    }

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
    thrust::sort_by_key(rmm::exec_policy(handle.get_stream_view()),
                        major_labels.begin(),
                        major_labels.end(),
                        major_counts.begin());
    auto num_unique_labels = thrust::count_if(rmm::exec_policy(handle.get_stream_view()),
                                              thrust::make_counting_iterator(size_t{0}),
                                              thrust::make_counting_iterator(major_labels.size()),
                                              [labels = major_labels.data()] __device__(auto i) {
                                                return (i == 0) || (labels[i - 1] != labels[i]);
                                              });
    rmm::device_uvector<vertex_t> tmp_major_labels(num_unique_labels, handle.get_stream());
    rmm::device_uvector<edge_t> tmp_major_counts(tmp_major_labels.size(), handle.get_stream());
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream_view()),
                          major_labels.begin(),
                          major_labels.end(),
                          major_counts.begin(),
                          tmp_major_labels.begin(),
                          tmp_major_counts.begin());
    major_labels = std::move(tmp_major_labels);
    major_counts = std::move(tmp_major_counts);
  }

  // 2. acquire unique minor labels

  std::vector<edge_t> minor_displs(edgelist_minor_vertices.size(), edge_t{0});
  std::partial_sum(
    edgelist_edge_counts.begin(), edgelist_edge_counts.end() - 1, minor_displs.begin() + 1);
  rmm::device_uvector<vertex_t> minor_labels(minor_displs.back() + edgelist_edge_counts.back(),
                                             handle.get_stream());
  for (size_t i = 0; i < edgelist_minor_vertices.size(); ++i) {
    thrust::copy(rmm::exec_policy(handle.get_stream_view()),
                 edgelist_minor_vertices[i],
                 edgelist_minor_vertices[i] + edgelist_edge_counts[i],
                 minor_labels.begin() + minor_displs[i]);
  }
  thrust::sort(
    rmm::exec_policy(handle.get_stream_view()), minor_labels.begin(), minor_labels.end());
  minor_labels.resize(thrust::distance(minor_labels.begin(),
                                       thrust::unique(rmm::exec_policy(handle.get_stream_view()),
                                                      minor_labels.begin(),
                                                      minor_labels.end())),
                      handle.get_stream());
  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (beginning of row_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif

    if (row_comm_size > 1) {
      rmm::device_uvector<vertex_t> rx_minor_labels(0, handle.get_stream());
      std::tie(rx_minor_labels, std::ignore) = groupby_gpuid_and_shuffle_values(
        row_comm,
        minor_labels.begin(),
        minor_labels.end(),
        [key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{row_comm_size}] __device__(
          auto val) { return key_func(val); },
        handle.get_stream());
      thrust::sort(
        rmm::exec_policy(handle.get_stream_view()), rx_minor_labels.begin(), rx_minor_labels.end());
      rx_minor_labels.resize(
        thrust::distance(rx_minor_labels.begin(),
                         thrust::unique(rmm::exec_policy(handle.get_stream_view()),
                                        rx_minor_labels.begin(),
                                        rx_minor_labels.end())),
        handle.get_stream());
      minor_labels = std::move(rx_minor_labels);
    }

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (end of row_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    //
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif
  }
  minor_labels.shrink_to_fit(handle.get_stream_view());

  // 3. merge major and minor labels and vertex labels

  rmm::device_uvector<vertex_t> merged_labels(major_labels.size() + minor_labels.size(),
                                              handle.get_stream_view());
  rmm::device_uvector<edge_t> merged_counts(merged_labels.size(), handle.get_stream_view());
  thrust::merge_by_key(rmm::exec_policy(handle.get_stream_view()),
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
  auto pair_it = thrust::reduce_by_key(rmm::exec_policy(handle.get_stream_view()),
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

  // 4. if vertices != nullptr, add isolated vertices

  rmm::device_uvector<vertex_t> isolated_vertices(0, handle.get_stream());
  if (optional_vertex_span) {
    auto [vertices, num_vertices] = *optional_vertex_span;
    auto num_isolated_vertices    = thrust::count_if(
      rmm::exec_policy(handle.get_stream_view()),
      vertices,
      vertices + num_vertices,
      [label_first = labels.begin(), label_last = labels.end()] __device__(auto v) {
        return !thrust::binary_search(thrust::seq, label_first, label_last, v);
      });
    isolated_vertices.resize(num_isolated_vertices, handle.get_stream());
    thrust::copy_if(rmm::exec_policy(handle.get_stream_view()),
                    vertices,
                    vertices + num_vertices,
                    isolated_vertices.begin(),
                    [label_first = labels.begin(), label_last = labels.end()] __device__(auto v) {
                      return !thrust::binary_search(thrust::seq, label_first, label_last, v);
                    });
  }

  if (isolated_vertices.size() > 0) {
    labels.resize(labels.size() + isolated_vertices.size(), handle.get_stream());
    counts.resize(labels.size(), handle.get_stream());
    thrust::copy(rmm::exec_policy(handle.get_stream_view()),
                 isolated_vertices.begin(),
                 isolated_vertices.end(),
                 labels.end() - isolated_vertices.size());
    thrust::fill(rmm::exec_policy(handle.get_stream_view()),
                 counts.end() - isolated_vertices.size(),
                 counts.end(),
                 edge_t{0});
  }

  // 6. sort by degree

  thrust::sort_by_key(rmm::exec_policy(handle.get_stream_view()),
                      counts.begin(),
                      counts.end(),
                      labels.begin(),
                      thrust::greater<edge_t>());

  return labels;
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void expensive_check_edgelist(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_vertex_span,
  std::vector<vertex_t const*> const& edgelist_major_vertices,
  std::vector<vertex_t const*> const& edgelist_minor_vertices,
  std::vector<edge_t> const& edgelist_edge_counts)
{
  rmm::device_uvector<vertex_t> sorted_local_vertices(size_t{0}, handle.get_stream());
  if (optional_vertex_span) {
    auto [vertices, num_vertices] = *optional_vertex_span;
    sorted_local_vertices.resize(num_vertices, handle.get_stream());
    thrust::copy(rmm::exec_policy(handle.get_stream_view()),
                 vertices,
                 vertices + num_vertices,
                 sorted_local_vertices.begin());
    thrust::sort(rmm::exec_policy(handle.get_stream_view()),
                 sorted_local_vertices.begin(),
                 sorted_local_vertices.end());
    CUGRAPH_EXPECTS(static_cast<size_t>(thrust::distance(
                      sorted_local_vertices.begin(),
                      thrust::unique(rmm::exec_policy(handle.get_stream_view()),
                                     sorted_local_vertices.begin(),
                                     sorted_local_vertices.end()))) == sorted_local_vertices.size(),
                    "Invalid input argument: local_vertices should not have duplicates.");
  }

  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto const row_comm_rank = row_comm.get_rank();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();
    auto const col_comm_rank = col_comm.get_rank();

    CUGRAPH_EXPECTS((edgelist_major_vertices.size() == edgelist_minor_vertices.size()) &&
                      (edgelist_major_vertices.size() == static_cast<size_t>(col_comm_size)),
                    "Invalid input argument: both edgelist_major_vertices.size() & "
                    "edgelist_minor_vertices.size() should coincide with col_comm_size.");

    auto [local_vertices, num_local_vertices] = *optional_vertex_span;
    CUGRAPH_EXPECTS(
      thrust::count_if(
        rmm::exec_policy(handle.get_stream_view()),
        local_vertices,
        local_vertices + num_local_vertices,
        [comm_rank,
         key_func =
           detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(auto val) {
          return key_func(val) != comm_rank;
        }) == 0,
      "Invalid input argument: local_vertices should be pre-shuffled.");

    for (size_t i = 0; i < edgelist_major_vertices.size(); ++i) {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_major_vertices[i], edgelist_minor_vertices[i]));
      CUGRAPH_EXPECTS(
        thrust::count_if(
          rmm::exec_policy(handle.get_stream_view()),
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
        "Invalid input argument: edgelist_major_vertices & edgelist_minor_vertices should be "
        "pre-shuffled.");

      if (optional_vertex_span) {
        auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

        // FIXME: this barrier is unnecessary if the above host_scalar_allreduce is a true host
        // operation (as it serves as a barrier) barrier is necessary here to avoid potential
        // overlap (which can leads to deadlock) between two different communicators (beginning of
        // col_comm)
#if 1
        // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with
        // DASK and MPI barrier with MPI)
        host_barrier(comm, handle.get_stream_view());
#else
        handle.get_stream_view().synchronize();
        comm.barrier();  // currently, this is ncclAllReduce
#endif

        rmm::device_uvector<vertex_t> sorted_major_vertices(0, handle.get_stream());
        {
          auto recvcounts =
            host_scalar_allgather(col_comm, sorted_local_vertices.size(), handle.get_stream());
          std::vector<size_t> displacements(recvcounts.size(), size_t{0});
          std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
          sorted_major_vertices.resize(displacements.back() + recvcounts.back(),
                                       handle.get_stream());
          device_allgatherv(col_comm,
                            sorted_local_vertices.data(),
                            sorted_major_vertices.data(),
                            recvcounts,
                            displacements,
                            handle.get_stream());
          thrust::sort(rmm::exec_policy(handle.get_stream_view()),
                       sorted_major_vertices.begin(),
                       sorted_major_vertices.end());
        }

        // barrier is necessary here to avoid potential overlap (which can leads to deadlock)
        // between two different communicators (beginning of row_comm)
#if 1
        // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with
        // DASK and MPI barrier with MPI)
        host_barrier(comm, handle.get_stream_view());
#else
        handle.get_stream_view().synchronize();
        comm.barrier();  // currently, this is ncclAllReduce
#endif

        rmm::device_uvector<vertex_t> sorted_minor_vertices(0, handle.get_stream());
        {
          auto recvcounts =
            host_scalar_allgather(row_comm, sorted_local_vertices.size(), handle.get_stream());
          std::vector<size_t> displacements(recvcounts.size(), size_t{0});
          std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
          sorted_minor_vertices.resize(displacements.back() + recvcounts.back(),
                                       handle.get_stream());
          device_allgatherv(row_comm,
                            sorted_local_vertices.data(),
                            sorted_minor_vertices.data(),
                            recvcounts,
                            displacements,
                            handle.get_stream());
          thrust::sort(rmm::exec_policy(handle.get_stream_view()),
                       sorted_minor_vertices.begin(),
                       sorted_minor_vertices.end());
        }

        // barrier is necessary here to avoid potential overlap (which can leads to deadlock)
        // between two different communicators (end of row_comm)
#if 1
        // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with
        // DASK and MPI barrier with MPI)
        host_barrier(comm, handle.get_stream_view());
#else
        handle.get_stream_view().synchronize();
        comm.barrier();  // currently, this is ncclAllReduce
#endif

        auto edge_first = thrust::make_zip_iterator(
          thrust::make_tuple(edgelist_major_vertices[i], edgelist_minor_vertices[i]));
        CUGRAPH_EXPECTS(
          thrust::count_if(
            rmm::exec_policy(handle.get_stream_view()),
            edge_first,
            edge_first + edgelist_edge_counts[i],
            [num_major_vertices    = static_cast<vertex_t>(sorted_major_vertices.size()),
             sorted_major_vertices = sorted_major_vertices.data(),
             num_minor_vertices    = static_cast<vertex_t>(sorted_minor_vertices.size()),
             sorted_minor_vertices = sorted_minor_vertices.data()] __device__(auto e) {
              return !thrust::binary_search(thrust::seq,
                                            sorted_major_vertices,
                                            sorted_major_vertices + num_major_vertices,
                                            thrust::get<0>(e)) ||
                     !thrust::binary_search(thrust::seq,
                                            sorted_minor_vertices,
                                            sorted_minor_vertices + num_minor_vertices,
                                            thrust::get<1>(e));
            }) == 0,
          "Invalid input argument: edgelist_major_vertices and/or edgelist_mior_vertices have "
          "invalid vertex ID(s).");
      }
    }
  } else {
    assert(edgelist_major_vertices.size() == 1);
    assert(edgelist_minor_vertices.size() == 1);

    if (optional_vertex_span) {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_major_vertices[0], edgelist_minor_vertices[0]));
      CUGRAPH_EXPECTS(
        thrust::count_if(
          rmm::exec_policy(handle.get_stream_view()),
          edge_first,
          edge_first + edgelist_edge_counts[0],
          [sorted_local_vertices = sorted_local_vertices.data(),
           num_sorted_local_vertices =
             static_cast<vertex_t>(sorted_local_vertices.size())] __device__(auto e) {
            return !thrust::binary_search(thrust::seq,
                                          sorted_local_vertices,
                                          sorted_local_vertices + num_sorted_local_vertices,
                                          thrust::get<0>(e)) ||
                   !thrust::binary_search(thrust::seq,
                                          sorted_local_vertices,
                                          sorted_local_vertices + num_sorted_local_vertices,
                                          thrust::get<1>(e));
          }) == 0,
        "Invalid input argument: edgelist_major_vertices and/or edgelist_minor_vertices have "
        "invalid vertex ID(s).");
    }
  }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_local_vertex_span,
                  std::vector<vertex_t*> const& edgelist_major_vertices /* [INOUT] */,
                  std::vector<vertex_t*> const& edgelist_minor_vertices /* [INOUT] */,
                  std::vector<edge_t> const& edgelist_edge_counts,
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

  std::vector<vertex_t const*> edgelist_const_major_vertices(edgelist_major_vertices.size());
  std::vector<vertex_t const*> edgelist_const_minor_vertices(edgelist_const_major_vertices.size());
  for (size_t i = 0; i < edgelist_const_major_vertices.size(); ++i) {
    edgelist_const_major_vertices[i] = edgelist_major_vertices[i];
    edgelist_const_minor_vertices[i] = edgelist_minor_vertices[i];
  }

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                          optional_local_vertex_span,
                                                          edgelist_const_major_vertices,
                                                          edgelist_const_minor_vertices,
                                                          edgelist_edge_counts);
  }

  // 1. compute renumber map

  auto renumber_map_labels =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(handle,
                                                              optional_local_vertex_span,
                                                              edgelist_const_major_vertices,
                                                              edgelist_const_minor_vertices,
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
    handle.get_stream());

  // 3. renumber edges

  double constexpr load_factor = 0.7;

  // FIXME: compare this hash based approach with a binary search based approach in both memory
  // footprint and execution time

  // FIXME: this barrier is unnecessary if the above host_scalar_allgather is a true host operation
  // (as it serves as a barrier) barrier is necessary here to avoid potential overlap (which can
  // leads to deadlock) between two different communicators (beginning of col_comm)
#if 1
  // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK and
  // MPI barrier with MPI)
  host_barrier(comm, handle.get_stream_view());
#else
  handle.get_stream_view().synchronize();
  comm.barrier();  // currently, this is ncclAllReduce
#endif

  for (size_t i = 0; i < edgelist_major_vertices.size(); ++i) {
    rmm::device_uvector<vertex_t> renumber_map_major_labels(
      col_comm_rank == static_cast<int>(i) ? vertex_t{0}
                                           : partition.get_matrix_partition_major_size(i),
      handle.get_stream());
    device_bcast(col_comm,
                 renumber_map_labels.data(),
                 renumber_map_major_labels.data(),
                 partition.get_matrix_partition_major_size(i),
                 i,
                 handle.get_stream());

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // cuco::static_map currently does not take stream

    auto poly_alloc = rmm::mr::polymorphic_allocator<char>(rmm::mr::get_current_device_resource());
    auto stream_adapter = rmm::mr::make_stream_allocator_adaptor(poly_alloc, cudaStream_t{nullptr});
    cuco::static_map<vertex_t, vertex_t, cuda::thread_scope_device, decltype(stream_adapter)>
      renumber_map{
        // cuco::static_map requires at least one empty slot
        std::max(static_cast<size_t>(
                   static_cast<double>(partition.get_matrix_partition_major_size(i)) / load_factor),
                 static_cast<size_t>(partition.get_matrix_partition_major_size(i)) + 1),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value,
        stream_adapter};
    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
      col_comm_rank == static_cast<int>(i) ? renumber_map_labels.begin()
                                           : renumber_map_major_labels.begin(),
      thrust::make_counting_iterator(partition.get_matrix_partition_major_first(i))));
    renumber_map.insert(pair_first, pair_first + partition.get_matrix_partition_major_size(i));
    renumber_map.find(edgelist_major_vertices[i],
                      edgelist_major_vertices[i] + edgelist_edge_counts[i],
                      edgelist_major_vertices[i]);
  }

  // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between two
  // different communicators (beginning of row_comm)
#if 1
  // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK and
  // MPI barrier with MPI)
  host_barrier(comm, handle.get_stream_view());
#else
  handle.get_stream_view().synchronize();
  comm.barrier();  // currently, this is ncclAllReduce
#endif
  {
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
    for (size_t i = 0; i < edgelist_major_vertices.size(); ++i) {
      renumber_map.find(edgelist_minor_vertices[i],
                        edgelist_minor_vertices[i] + edgelist_edge_counts[i],
                        edgelist_minor_vertices[i]);
    }
  }
  // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between two
  // different communicators (end of row_comm)
#if 1
  // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK and
  // MPI barrier with MPI)
  host_barrier(comm, handle.get_stream_view());
#else
  handle.get_stream_view().synchronize();
  comm.barrier();  // currently, this is ncclAllReduce
#endif

  return std::make_tuple(
    std::move(renumber_map_labels), partition, number_of_vertices, number_of_edges);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_vertex_span,
  vertex_t* edgelist_major_vertices /* [INOUT] */,
  vertex_t* edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool do_expensive_check)
{
  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      optional_vertex_span,
      std::vector<vertex_t const*>{edgelist_major_vertices},
      std::vector<vertex_t const*>{edgelist_minor_vertices},
      std::vector<edge_t>{num_edgelist_edges});
  }

  auto renumber_map_labels = detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(
    handle,
    optional_vertex_span,
    std::vector<vertex_t const*>{edgelist_major_vertices},
    std::vector<vertex_t const*>{edgelist_minor_vertices},
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
  renumber_map.find(
    edgelist_major_vertices, edgelist_major_vertices + num_edgelist_edges, edgelist_major_vertices);
  renumber_map.find(
    edgelist_minor_vertices, edgelist_minor_vertices + num_edgelist_edges, edgelist_minor_vertices);

  return renumber_map_labels;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_local_vertex_span,
                  std::vector<vertex_t*> const& edgelist_major_vertices /* [INOUT] */,
                  std::vector<vertex_t*> const& edgelist_minor_vertices /* [INOUT] */,
                  std::vector<edge_t> const& edgelist_edge_counts,
                  bool do_expensive_check)
{
  return detail::renumber_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                                optional_local_vertex_span,
                                                                edgelist_major_vertices,
                                                                edgelist_minor_vertices,
                                                                edgelist_edge_counts,
                                                                do_expensive_check);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_vertex_span,
  vertex_t* edgelist_major_vertices /* [INOUT] */,
  vertex_t* edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool do_expensive_check)
{
  return detail::renumber_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                                optional_vertex_span,
                                                                edgelist_major_vertices,
                                                                edgelist_minor_vertices,
                                                                num_edgelist_edges,
                                                                do_expensive_check);
}

// explicit instantiation directives (EIDir's):
//

// instantiations for <vertex_t == int32_t, edge_t == int32_t>
//
template std::tuple<rmm::device_uvector<int32_t>, partition_t<int32_t>, int32_t, int32_t>
renumber_edgelist<int32_t, int32_t, true>(
  raft::handle_t const& handle,
  std::optional<std::tuple<int32_t const*, int32_t>> optional_local_vertex_span,
  std::vector<int32_t*> const& edgelist_major_vertices /* [INOUT] */,
  std::vector<int32_t*> const& edgelist_minor_vertices /* [INOUT] */,
  std::vector<int32_t> const& edgelist_edge_counts,
  bool do_expensive_check);

template rmm::device_uvector<int32_t> renumber_edgelist<int32_t, int32_t, false>(
  raft::handle_t const& handle,
  std::optional<std::tuple<int32_t const*, int32_t>> optional_vertex_span,
  int32_t* edgelist_major_vertices /* [INOUT] */,
  int32_t* edgelist_minor_vertices /* [INOUT] */,
  int32_t num_edgelist_edges,
  bool do_expensive_check);

// instantiations for <vertex_t == int32_t, edge_t == int64_t>
//
template std::tuple<rmm::device_uvector<int32_t>, partition_t<int32_t>, int32_t, int64_t>
renumber_edgelist<int32_t, int64_t, true>(
  raft::handle_t const& handle,
  std::optional<std::tuple<int32_t const*, int32_t>> optional_local_vertex_span,
  std::vector<int32_t*> const& edgelist_major_vertices /* [INOUT] */,
  std::vector<int32_t*> const& edgelist_minor_vertices /* [INOUT] */,
  std::vector<int64_t> const& edgelist_edge_counts,
  bool do_expensive_check);

template rmm::device_uvector<int32_t> renumber_edgelist<int32_t, int64_t, false>(
  raft::handle_t const& handle,
  std::optional<std::tuple<int32_t const*, int32_t>> optional_vertex_span,
  int32_t* edgelist_major_vertices /* [INOUT] */,
  int32_t* edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool do_expensive_check);

// instantiations for <vertex_t == int64_t, edge_t == int64_t>
//
template std::tuple<rmm::device_uvector<int64_t>, partition_t<int64_t>, int64_t, int64_t>
renumber_edgelist<int64_t, int64_t, true>(
  raft::handle_t const& handle,
  std::optional<std::tuple<int64_t const*, int64_t>> optional_local_vertex_span,
  std::vector<int64_t*> const& edgelist_major_vertices /* [INOUT] */,
  std::vector<int64_t*> const& edgelist_minor_vertices /* [INOUT] */,
  std::vector<int64_t> const& edgelist_edge_counts,
  bool do_expensive_check);

template rmm::device_uvector<int64_t> renumber_edgelist<int64_t, int64_t, false>(
  raft::handle_t const& handle,
  std::optional<std::tuple<int64_t const*, int64_t>> optional_vertex_span,
  int64_t* edgelist_major_vertices /* [INOUT] */,
  int64_t* edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph

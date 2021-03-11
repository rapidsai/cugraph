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

#include <experimental/include_cuco_static_map.cuh>

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>
#include <utilities/device_comm.cuh>
#include <utilities/error.hpp>
#include <utilities/host_scalar_comm.cuh>
#include <utilities/shuffle_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

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

#ifdef CUCO_STATIC_MAP_DEFINED
template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_renumber_map(
  raft::handle_t const& handle,
  vertex_t const* vertices,
  vertex_t num_local_vertices /* relevant only if vertices != nullptr */,
  vertex_t const* edgelist_major_vertices,
  vertex_t const* edgelist_minor_vertices,
  edge_t num_edgelist_edges)
{
  // FIXME: compare this sort based approach with hash based approach in both speed and memory
  // footprint

  // 1. acquire (unique major label, count) pairs

  rmm::device_uvector<vertex_t> major_labels(0, handle.get_stream());
  rmm::device_uvector<edge_t> major_counts(0, handle.get_stream());
  {
    rmm::device_uvector<vertex_t> tmp_labels(num_edgelist_edges, handle.get_stream());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 edgelist_major_vertices,
                 edgelist_major_vertices + num_edgelist_edges,
                 tmp_labels.begin());
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 tmp_labels.begin(),
                 tmp_labels.end());
    auto num_unique_labels =
      thrust::count_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(tmp_labels.size()),
                       [labels = tmp_labels.data()] __device__(auto i) {
                         return (i == 0) || (labels[i - 1] != labels[i]);
                       });
    major_labels.resize(num_unique_labels, handle.get_stream());
    major_counts.resize(major_labels.size(), handle.get_stream());
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          tmp_labels.begin(),
                          tmp_labels.end(),
                          thrust::make_constant_iterator(edge_t{1}),
                          major_labels.begin(),
                          major_counts.begin());
  }

  // 2. acquire unique minor labels

  rmm::device_uvector<vertex_t> minor_labels(num_edgelist_edges, handle.get_stream());
  thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               edgelist_minor_vertices,
               edgelist_minor_vertices + num_edgelist_edges,
               minor_labels.begin());
  thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               minor_labels.begin(),
               minor_labels.end());
  auto minor_label_it =
    thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   minor_labels.begin(),
                   minor_labels.end());
  minor_labels.resize(thrust::distance(minor_labels.begin(), minor_label_it), handle.get_stream());
  minor_labels.shrink_to_fit(handle.get_stream());

  // 3. merge major and minor labels and vertex labels

  rmm::device_uvector<vertex_t> merged_labels(major_labels.size() + minor_labels.size(),
                                              handle.get_stream());

  rmm::device_uvector<edge_t> merged_counts(merged_labels.size(), handle.get_stream());
  thrust::merge_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
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
  auto pair_it =
    thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
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

  // 4. if multi-GPU, shuffle and reduce (label, count) pairs

  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(labels.begin(), counts.begin()));
    rmm::device_uvector<vertex_t> rx_labels(0, handle.get_stream());
    rmm::device_uvector<edge_t> rx_counts(0, handle.get_stream());
    std::forward_as_tuple(std::tie(rx_labels, rx_counts), std::ignore) =
      groupby_gpuid_and_shuffle_values(
        comm,
        pair_first,
        pair_first + labels.size(),
        [key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(
          auto val) { return key_func(thrust::get<0>(val)); },
        handle.get_stream());

    labels.resize(rx_labels.size(), handle.get_stream());
    counts.resize(labels.size(), handle.get_stream());
    thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        rx_labels.begin(),
                        rx_labels.end(),
                        rx_counts.begin());
    pair_it = thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    rx_labels.begin(),
                                    rx_labels.end(),
                                    rx_counts.begin(),
                                    labels.begin(),
                                    counts.begin());
    rx_labels.resize(0, handle.get_stream());
    rx_counts.resize(0, handle.get_stream());
    rx_labels.shrink_to_fit(handle.get_stream());
    rx_counts.shrink_to_fit(handle.get_stream());
    labels.resize(thrust::distance(labels.begin(), thrust::get<0>(pair_it)), handle.get_stream());
    counts.resize(labels.size(), handle.get_stream());
    labels.shrink_to_fit(handle.get_stream());
    labels.shrink_to_fit(handle.get_stream());
  }

  // 5. if vertices != nullptr, add isolated vertices

  rmm::device_uvector<vertex_t> isolated_vertices(0, handle.get_stream());
  if (vertices != nullptr) {
    auto num_isolated_vertices = thrust::count_if(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      vertices,
      vertices + num_local_vertices,
      [label_first = labels.begin(), label_last = labels.end()] __device__(auto v) {
        return !thrust::binary_search(thrust::seq, label_first, label_last, v);
      });
    isolated_vertices.resize(num_isolated_vertices, handle.get_stream());
    thrust::copy_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    vertices,
                    vertices + num_local_vertices,
                    isolated_vertices.begin(),
                    [label_first = labels.begin(), label_last = labels.end()] __device__(auto v) {
                      return !thrust::binary_search(thrust::seq, label_first, label_last, v);
                    });
  }

  if (isolated_vertices.size() > 0) {
    labels.resize(labels.size() + isolated_vertices.size(), handle.get_stream());
    counts.resize(labels.size(), handle.get_stream());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 isolated_vertices.begin(),
                 isolated_vertices.end(),
                 labels.end() - isolated_vertices.size());
    thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 counts.end() - isolated_vertices.size(),
                 counts.end(),
                 edge_t{0});
  }

  // 6. sort by degree

  thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      counts.begin(),
                      counts.end(),
                      labels.begin(),
                      thrust::greater<edge_t>());

  return labels;
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void expensive_check_edgelist(
  raft::handle_t const& handle,
  vertex_t const* local_vertices,
  vertex_t num_local_vertices /* relevant only if local_vertices != nullptr */,
  vertex_t const* edgelist_major_vertices,
  vertex_t const* edgelist_minor_vertices,
  edge_t num_edgelist_edges,
  bool is_hypergraph_partitioned /* relevant only if multi_gpu == true */)
{
  rmm::device_uvector<vertex_t> sorted_local_vertices(
    local_vertices != nullptr ? num_local_vertices : vertex_t{0}, handle.get_stream());
  thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               local_vertices,
               local_vertices + num_local_vertices,
               sorted_local_vertices.begin());
  thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               sorted_local_vertices.begin(),
               sorted_local_vertices.end());

  if (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();

    CUGRAPH_EXPECTS(
      thrust::count_if(
        rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
        local_vertices,
        local_vertices + num_local_vertices,
        [comm_rank,
         key_func =
           detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(auto val) {
          return key_func(val) != comm_rank;
        }) == 0,
      "Invalid input argument: local_vertices should be pre-shuffled.");

    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices));
    CUGRAPH_EXPECTS(
      thrust::count_if(
        rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
        edge_first,
        edge_first + num_edgelist_edges,
        [comm_rank,
         key_func =
           detail::compute_gpu_id_from_edge_t<vertex_t>{is_hypergraph_partitioned,
                                                        comm_size,
                                                        row_comm_size,
                                                        col_comm_size}] __device__(auto edge) {
          return key_func(thrust::get<0>(edge), thrust::get<1>(edge)) != comm_rank;
        }) == 0,
      "Invalid input argument: edgelist_major_vertices & edgelist_minor_vertices should be "
      "pre-shuffled.");

    if (local_vertices != nullptr) {
      rmm::device_uvector<vertex_t> unique_edge_vertices(num_edgelist_edges * 2,
                                                         handle.get_stream());
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   edgelist_major_vertices,
                   edgelist_major_vertices + num_edgelist_edges,
                   unique_edge_vertices.begin());
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   edgelist_minor_vertices,
                   edgelist_minor_vertices + num_edgelist_edges,
                   unique_edge_vertices.begin() + num_edgelist_edges);
      thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   unique_edge_vertices.begin(),
                   unique_edge_vertices.end());
      unique_edge_vertices.resize(
        thrust::distance(
          unique_edge_vertices.begin(),
          thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                         unique_edge_vertices.begin(),
                         unique_edge_vertices.end())),
        handle.get_stream());

      rmm::device_uvector<vertex_t> rx_unique_edge_vertices(0, handle.get_stream());
      std::tie(rx_unique_edge_vertices, std::ignore) = groupby_gpuid_and_shuffle_values(
        handle.get_comms(),
        unique_edge_vertices.begin(),
        unique_edge_vertices.end(),
        [key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(
          auto val) { return key_func(val); },
        handle.get_stream());

      unique_edge_vertices = std::move(rx_unique_edge_vertices);

      CUGRAPH_EXPECTS(
        thrust::count_if(
          rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
          unique_edge_vertices.begin(),
          unique_edge_vertices.end(),
          [num_local_vertices,
           sorted_local_vertices = sorted_local_vertices.data()] __device__(auto v) {
            return !thrust::binary_search(
              thrust::seq, sorted_local_vertices, sorted_local_vertices + num_local_vertices, v);
          }) == 0,
        "Invalid input argument: edgelist_major_vertices and/or edgelist_minor_vertices have "
        "invalid vertex ID(s).");
    }
  } else {
    if (local_vertices != nullptr) {
      CUGRAPH_EXPECTS(
        thrust::count_if(
          rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
          edgelist_major_vertices,
          edgelist_major_vertices + num_edgelist_edges,
          [num_local_vertices,
           sorted_local_vertices = sorted_local_vertices.data()] __device__(auto v) {
            return !thrust::binary_search(
              thrust::seq, sorted_local_vertices, sorted_local_vertices + num_local_vertices, v);
          }) == 0,
        "Invalid input argument: edgelist_major_vertices has invalid vertex ID(s).");

      CUGRAPH_EXPECTS(
        thrust::count_if(
          rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
          edgelist_major_vertices,
          edgelist_major_vertices + num_edgelist_edges,
          [num_local_vertices,
           sorted_local_vertices = sorted_local_vertices.data()] __device__(auto v) {
            return !thrust::binary_search(
              thrust::seq, sorted_local_vertices, sorted_local_vertices + num_local_vertices, v);
          }) == 0,
        "Invalid input argument: edgelist_major_vertices has invalid vertex ID(s).");
    }
  }
}
#endif

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  vertex_t const* local_vertices,
                  vertex_t num_local_vertices /* relevant only if local_vertices != nullptr */,
                  vertex_t* edgelist_major_vertices /* [INOUT] */,
                  vertex_t* edgelist_minor_vertices /* [INOUT] */,
                  edge_t num_edgelist_edges,
                  bool is_hypergraph_partitioned,
                  bool do_expensive_check)
{
  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "Relabel not supported on Pascal and older architectures.");

#ifdef CUCO_STATIC_MAP_DEFINED
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto const row_comm_rank = row_comm.get_rank();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();
  auto const col_comm_rank = col_comm.get_rank();

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                          local_vertices,
                                                          num_local_vertices,
                                                          edgelist_major_vertices,
                                                          edgelist_minor_vertices,
                                                          num_edgelist_edges,
                                                          is_hypergraph_partitioned);
  }

  // 1. compute renumber map

  auto renumber_map_labels =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(handle,
                                                              local_vertices,
                                                              num_local_vertices,
                                                              edgelist_major_vertices,
                                                              edgelist_minor_vertices,
                                                              num_edgelist_edges);

  // 2. initialize partition_t object, number_of_vertices, and number_of_edges for the coarsened
  // graph

  auto vertex_partition_counts = host_scalar_allgather(
    comm, static_cast<vertex_t>(renumber_map_labels.size()), handle.get_stream());
  std::vector<vertex_t> vertex_partition_offsets(comm_size + 1, 0);
  std::partial_sum(vertex_partition_counts.begin(),
                   vertex_partition_counts.end(),
                   vertex_partition_offsets.begin() + 1);

  partition_t<vertex_t> partition(vertex_partition_offsets,
                                  is_hypergraph_partitioned,
                                  row_comm_size,
                                  col_comm_size,
                                  row_comm_rank,
                                  col_comm_rank);

  auto number_of_vertices = vertex_partition_offsets.back();
  auto number_of_edges    = host_scalar_allreduce(comm, num_edgelist_edges, handle.get_stream());

  // 3. renumber edges

  if (is_hypergraph_partitioned) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    double constexpr load_factor = 0.7;

    // FIXME: compare this hash based approach with a binary search based approach in both memory
    // footprint and execution time

    {
      vertex_t major_first{};
      vertex_t major_last{};
      std::tie(major_first, major_last) = partition.get_matrix_partition_major_range(0);
      rmm::device_uvector<vertex_t> renumber_map_major_labels(major_last - major_first,
                                                              handle.get_stream());
      std::vector<size_t> recvcounts(row_comm_size);
      for (int i = 0; i < row_comm_size; ++i) {
        recvcounts[i] = partition.get_vertex_partition_size(col_comm_rank * row_comm_size + i);
      }
      std::vector<size_t> displacements(row_comm_size, 0);
      std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
      device_allgatherv(row_comm,
                        renumber_map_labels.begin(),
                        renumber_map_major_labels.begin(),
                        recvcounts,
                        displacements,
                        handle.get_stream());

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // cuco::static_map currently does not take stream

      cuco::static_map<vertex_t, vertex_t> renumber_map{
        static_cast<size_t>(static_cast<double>(renumber_map_major_labels.size()) / load_factor),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value};
      auto pair_first = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(renumber_map_major_labels.begin(),
                                                     thrust::make_counting_iterator(major_first))),
        [] __device__(auto val) {
          return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
        });
      renumber_map.insert(pair_first, pair_first + renumber_map_major_labels.size());
      renumber_map.find(edgelist_major_vertices,
                        edgelist_major_vertices + num_edgelist_edges,
                        edgelist_major_vertices);
    }

    {
      vertex_t minor_first{};
      vertex_t minor_last{};
      std::tie(minor_first, minor_last) = partition.get_matrix_partition_minor_range();
      rmm::device_uvector<vertex_t> renumber_map_minor_labels(minor_last - minor_first,
                                                              handle.get_stream());

      // FIXME: this P2P is unnecessary if we apply the partitioning scheme used with hypergraph
      // partitioning
      auto comm_src_rank = row_comm_rank * col_comm_size + col_comm_rank;
      auto comm_dst_rank = (comm_rank % col_comm_size) * row_comm_size + comm_rank / col_comm_size;
      // FIXME: this branch may be no longer necessary with NCCL backend
      if (comm_src_rank == comm_rank) {
        assert(comm_dst_rank == comm_rank);
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     renumber_map_labels.begin(),
                     renumber_map_labels.end(),
                     renumber_map_minor_labels.begin() +
                       (partition.get_vertex_partition_first(comm_src_rank) -
                        partition.get_vertex_partition_first(row_comm_rank * col_comm_size)));
      } else {
        device_sendrecv(comm,
                        renumber_map_labels.begin(),
                        renumber_map_labels.size(),
                        comm_dst_rank,
                        renumber_map_minor_labels.begin() +
                          (partition.get_vertex_partition_first(comm_src_rank) -
                           partition.get_vertex_partition_first(row_comm_rank * col_comm_size)),
                        static_cast<size_t>(partition.get_vertex_partition_size(comm_src_rank)),
                        comm_src_rank,
                        handle.get_stream());
      }

      // FIXME: these broadcast operations can be placed between ncclGroupStart() and
      // ncclGroupEnd()
      for (int i = 0; i < col_comm_size; ++i) {
        auto offset = partition.get_vertex_partition_first(row_comm_rank * col_comm_size + i) -
                      partition.get_vertex_partition_first(row_comm_rank * col_comm_size);
        auto count = partition.get_vertex_partition_size(row_comm_rank * col_comm_size + i);
        device_bcast(col_comm,
                     renumber_map_minor_labels.begin() + offset,
                     renumber_map_minor_labels.begin() + offset,
                     count,
                     i,
                     handle.get_stream());
      }

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // cuco::static_map currently does not take stream

      cuco::static_map<vertex_t, vertex_t> renumber_map{
        static_cast<size_t>(static_cast<double>(renumber_map_minor_labels.size()) / load_factor),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value};
      auto pair_first = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(renumber_map_minor_labels.begin(),
                                                     thrust::make_counting_iterator(minor_first))),
        [] __device__(auto val) {
          return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
        });
      renumber_map.insert(pair_first, pair_first + renumber_map_minor_labels.size());
      renumber_map.find(edgelist_minor_vertices,
                        edgelist_minor_vertices + num_edgelist_edges,
                        edgelist_minor_vertices);
    }
  }

  return std::make_tuple(
    std::move(renumber_map_labels), partition, number_of_vertices, number_of_edges);
#else
  return std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                         partition_t<vertex_t>{},
                         vertex_t{0},
                         edge_t{0});
#endif
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  vertex_t const* vertices,
  vertex_t num_vertices /* relevant only if vertices != nullptr */,
  vertex_t* edgelist_major_vertices /* [INOUT] */,
  vertex_t* edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool do_expensive_check)
{
  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "Relabel not supported on Pascal and older architectures.");

#ifdef CUCO_STATIC_MAP_DEFINED
  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                          vertices,
                                                          num_vertices,
                                                          edgelist_major_vertices,
                                                          edgelist_minor_vertices,
                                                          num_edgelist_edges,
                                                          false);
  }

  auto renumber_map_labels =
    detail::compute_renumber_map<vertex_t, edge_t, multi_gpu>(handle,
                                                              vertices,
                                                              num_vertices,
                                                              edgelist_major_vertices,
                                                              edgelist_minor_vertices,
                                                              num_edgelist_edges);

  double constexpr load_factor = 0.7;

  // FIXME: compare this hash based approach with a binary search based approach in both memory
  // footprint and execution time

  cuco::static_map<vertex_t, vertex_t> renumber_map{
    static_cast<size_t>(static_cast<double>(renumber_map_labels.size()) / load_factor),
    invalid_vertex_id<vertex_t>::value,
    invalid_vertex_id<vertex_t>::value};
  auto pair_first = thrust::make_transform_iterator(
    thrust::make_zip_iterator(
      thrust::make_tuple(renumber_map_labels.begin(), thrust::make_counting_iterator(vertex_t{0}))),
    [] __device__(auto val) {
      return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
    });
  renumber_map.insert(pair_first, pair_first + renumber_map_labels.size());
  renumber_map.find(
    edgelist_major_vertices, edgelist_major_vertices + num_edgelist_edges, edgelist_major_vertices);
  renumber_map.find(
    edgelist_minor_vertices, edgelist_minor_vertices + num_edgelist_edges, edgelist_minor_vertices);

  return renumber_map_labels;
#else
  return rmm::device_uvector<vertex_t>(0, handle.get_stream());
#endif
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  vertex_t* edgelist_major_vertices /* [INOUT] */,
                  vertex_t* edgelist_minor_vertices /* [INOUT] */,
                  edge_t num_edgelist_edges,
                  bool is_hypergraph_partitioned,
                  bool do_expensive_check)
{
  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "Relabel not supported on Pascal and older architectures.");
  return detail::renumber_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                                static_cast<vertex_t*>(nullptr),
                                                                vertex_t{0},
                                                                edgelist_major_vertices,
                                                                edgelist_minor_vertices,
                                                                num_edgelist_edges,
                                                                is_hypergraph_partitioned,
                                                                do_expensive_check);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  vertex_t* edgelist_major_vertices /* [INOUT] */,
  vertex_t* edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool do_expensive_check)
{
  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "Relabel not supported on Pascal and older architectures.");
  return detail::renumber_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                                static_cast<vertex_t*>(nullptr),
                                                                vertex_t{0} /* dummy */,
                                                                edgelist_major_vertices,
                                                                edgelist_minor_vertices,
                                                                num_edgelist_edges,
                                                                do_expensive_check);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  vertex_t const* local_vertices,
                  vertex_t num_local_vertices,
                  vertex_t* edgelist_major_vertices /* [INOUT] */,
                  vertex_t* edgelist_minor_vertices /* [INOUT] */,
                  edge_t num_edgelist_edges,
                  bool is_hypergraph_partitioned,
                  bool do_expensive_check)
{
  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "Relabel not supported on Pascal and older architectures.");
  return detail::renumber_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                                local_vertices,
                                                                num_local_vertices,
                                                                edgelist_major_vertices,
                                                                edgelist_minor_vertices,
                                                                num_edgelist_edges,
                                                                is_hypergraph_partitioned,
                                                                do_expensive_check);
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  vertex_t const* vertices,
  vertex_t num_vertices,
  vertex_t* edgelist_major_vertices /* [INOUT] */,
  vertex_t* edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool do_expensive_check)
{
  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "Relabel not supported on Pascal and older architectures.");
  return detail::renumber_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                                vertices,
                                                                num_vertices,
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
renumber_edgelist<int32_t, int32_t, true>(raft::handle_t const& handle,
                                          int32_t* edgelist_major_vertices /* [INOUT] */,
                                          int32_t* edgelist_minor_vertices /* [INOUT] */,
                                          int32_t num_edgelist_edges,
                                          bool is_hypergraph_partitioned,
                                          bool do_expensive_check);

template rmm::device_uvector<int32_t> renumber_edgelist<int32_t, int32_t, false>(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices /* [INOUT] */,
  int32_t* edgelist_minor_vertices /* [INOUT] */,
  int32_t num_edgelist_edges,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>, partition_t<int32_t>, int32_t, int32_t>
renumber_edgelist<int32_t, int32_t, true>(raft::handle_t const& handle,
                                          int32_t const* local_vertices,
                                          int32_t num_local_vertices,
                                          int32_t* edgelist_major_vertices /* [INOUT] */,
                                          int32_t* edgelist_minor_vertices /* [INOUT] */,
                                          int32_t num_edgelist_edges,
                                          bool is_hypergraph_partitioned,
                                          bool do_expensive_check);

template rmm::device_uvector<int32_t> renumber_edgelist<int32_t, int32_t, false>(
  raft::handle_t const& handle,
  int32_t const* vertices,
  int32_t num_vertices,
  int32_t* edgelist_major_vertices /* [INOUT] */,
  int32_t* edgelist_minor_vertices /* [INOUT] */,
  int32_t num_edgelist_edges,
  bool do_expensive_check);

// instantiations for <vertex_t == int32_t, edge_t == int64_t>
//
template std::tuple<rmm::device_uvector<int32_t>, partition_t<int32_t>, int32_t, int64_t>
renumber_edgelist<int32_t, int64_t, true>(raft::handle_t const& handle,
                                          int32_t* edgelist_major_vertices /* [INOUT] */,
                                          int32_t* edgelist_minor_vertices /* [INOUT] */,
                                          int64_t num_edgelist_edges,
                                          bool is_hypergraph_partitioned,
                                          bool do_expensive_check);

template rmm::device_uvector<int32_t> renumber_edgelist<int32_t, int64_t, false>(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices /* [INOUT] */,
  int32_t* edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>, partition_t<int32_t>, int32_t, int64_t>
renumber_edgelist<int32_t, int64_t, true>(raft::handle_t const& handle,
                                          int32_t const* local_vertices,
                                          int32_t num_local_vertices,
                                          int32_t* edgelist_major_vertices /* [INOUT] */,
                                          int32_t* edgelist_minor_vertices /* [INOUT] */,
                                          int64_t num_edgelist_edges,
                                          bool is_hypergraph_partitioned,
                                          bool do_expensive_check);

template rmm::device_uvector<int32_t> renumber_edgelist<int32_t, int64_t, false>(
  raft::handle_t const& handle,
  int32_t const* vertices,
  int32_t num_vertices,
  int32_t* edgelist_major_vertices /* [INOUT] */,
  int32_t* edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool do_expensive_check);

// instantiations for <vertex_t == int64_t, edge_t == int64_t>
//
template std::tuple<rmm::device_uvector<int64_t>, partition_t<int64_t>, int64_t, int64_t>
renumber_edgelist<int64_t, int64_t, true>(raft::handle_t const& handle,
                                          int64_t* edgelist_major_vertices /* [INOUT] */,
                                          int64_t* edgelist_minor_vertices /* [INOUT] */,
                                          int64_t num_edgelist_edges,
                                          bool is_hypergraph_partitioned,
                                          bool do_expensive_check);

template rmm::device_uvector<int64_t> renumber_edgelist<int64_t, int64_t, false>(
  raft::handle_t const& handle,
  int64_t* edgelist_major_vertices /* [INOUT] */,
  int64_t* edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>, partition_t<int64_t>, int64_t, int64_t>
renumber_edgelist<int64_t, int64_t, true>(raft::handle_t const& handle,
                                          int64_t const* local_vertices,
                                          int64_t num_local_vertices,
                                          int64_t* edgelist_major_vertices /* [INOUT] */,
                                          int64_t* edgelist_minor_vertices /* [INOUT] */,
                                          int64_t num_edgelist_edges,
                                          bool is_hypergraph_partitioned,
                                          bool do_expensive_check);

template rmm::device_uvector<int64_t> renumber_edgelist<int64_t, int64_t, false>(
  raft::handle_t const& handle,
  int64_t const* vertices,
  int64_t num_vertices,
  int64_t* edgelist_major_vertices /* [INOUT] */,
  int64_t* edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph

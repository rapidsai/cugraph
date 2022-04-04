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

#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/handle.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <numeric>

namespace cugraph {

namespace {

template <typename vertex_t>
struct check_edge_t {
  vertex_t const* sorted_valid_major_range_first{nullptr};
  vertex_t const* sorted_valid_major_range_last{nullptr};
  vertex_t const* sorted_valid_minor_range_first{nullptr};
  vertex_t const* sorted_valid_minor_range_last{nullptr};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> const& e) const
  {
    return !thrust::binary_search(thrust::seq,
                                  sorted_valid_major_range_first,
                                  sorted_valid_major_range_last,
                                  thrust::get<0>(e)) ||
           !thrust::binary_search(thrust::seq,
                                  sorted_valid_minor_range_first,
                                  sorted_valid_minor_range_last,
                                  thrust::get<1>(e));
  }
};

template <typename vertex_t, bool multi_gpu>
void expensive_check_edgelist(raft::handle_t const& handle,
                              std::optional<rmm::device_uvector<vertex_t>> const& vertices,
                              rmm::device_uvector<vertex_t> const& edgelist_majors,
                              rmm::device_uvector<vertex_t> const& edgelist_minors)
{
  if (vertices) {
    rmm::device_uvector<vertex_t> sorted_vertices((*vertices).size(), handle.get_stream());
    thrust::copy(
      handle.get_thrust_policy(), (*vertices).begin(), (*vertices).end(), sorted_vertices.begin());
    thrust::sort(handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end());
    CUGRAPH_EXPECTS(static_cast<size_t>(thrust::distance(sorted_vertices.begin(),
                                                         thrust::unique(handle.get_thrust_policy(),
                                                                        sorted_vertices.begin(),
                                                                        sorted_vertices.end()))) ==
                      sorted_vertices.size(),
                    "Invalid input argument: vertices should not have duplicates.");
  }

  if constexpr (multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();

    CUGRAPH_EXPECTS(
      thrust::count_if(
        handle.get_thrust_policy(),
        (*vertices).begin(),
        (*vertices).end(),
        [comm_rank,
         key_func =
           detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(auto val) {
          return key_func(val) != comm_rank;
        }) == 0,
      "Invalid input argument: vertices should be pre-shuffled.");

    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
    CUGRAPH_EXPECTS(
      thrust::count_if(handle.get_thrust_policy(),
                       edge_first,
                       edge_first + edgelist_majors.size(),
                       [comm_rank,
                        gpu_id_key_func =
                          detail::compute_gpu_id_from_edge_t<vertex_t>{
                            comm_size, row_comm_size, col_comm_size}] __device__(auto e) {
                         return (gpu_id_key_func(thrust::get<0>(e), thrust::get<1>(e)) !=
                                 comm_rank);
                       }) == 0,
      "Invalid input argument: edgelist_majors & edgelist_minors should be pre-shuffled.");

    if (vertices) {
      rmm::device_uvector<vertex_t> sorted_majors(0, handle.get_stream());
      {
        auto recvcounts = host_scalar_allgather(col_comm, (*vertices).size(), handle.get_stream());
        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
        sorted_majors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
        device_allgatherv(col_comm,
                          (*vertices).data(),
                          sorted_majors.data(),
                          recvcounts,
                          displacements,
                          handle.get_stream());
        thrust::sort(handle.get_thrust_policy(), sorted_majors.begin(), sorted_majors.end());
      }

      rmm::device_uvector<vertex_t> sorted_minors(0, handle.get_stream());
      {
        auto recvcounts = host_scalar_allgather(row_comm, (*vertices).size(), handle.get_stream());
        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
        sorted_minors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
        device_allgatherv(row_comm,
                          (*vertices).data(),
                          sorted_minors.data(),
                          recvcounts,
                          displacements,
                          handle.get_stream());
        thrust::sort(handle.get_thrust_policy(), sorted_minors.begin(), sorted_minors.end());
      }

      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         edge_first,
                         edge_first + edgelist_majors.size(),
                         check_edge_t<vertex_t>{sorted_majors.data(),
                                                sorted_majors.data() + sorted_majors.size(),
                                                sorted_minors.data(),
                                                sorted_minors.data() + sorted_minors.size()}) == 0,
        "Invalid input argument: edgelist_majors and/or edgelist_minors have invalid vertex "
        "ID(s).");
    }
  } else {
    if (vertices) {
      rmm::device_uvector<vertex_t> sorted_vertices((*vertices).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*vertices).begin(),
                   (*vertices).end(),
                   sorted_vertices.begin());
      thrust::sort(handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end());
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         edge_first,
                         edge_first + edgelist_majors.size(),
                         check_edge_t<vertex_t>{sorted_vertices.data(),
                                                sorted_vertices.data() + sorted_vertices.size(),
                                                sorted_vertices.data(),
                                                sorted_vertices.data() + sorted_vertices.size()}) ==
          0,
        "Invalid input argument: edgelist_majors and/or edgelist_minors have invalid vertex "
        "ID(s).");
    }
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
             std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(raft::handle_t const& handle,
                                std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
                                rmm::device_uvector<vertex_t>&& edgelist_srcs,
                                rmm::device_uvector<vertex_t>&& edgelist_dsts,
                                std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                                graph_properties_t graph_properties,
                                bool renumber,
                                bool do_expensive_check)
{
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  CUGRAPH_EXPECTS(edgelist_srcs.size() == edgelist_dsts.size(),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_dsts.size().");
  CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs.size() == (*edgelist_weights).size()),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_weights.size().");
  CUGRAPH_EXPECTS(renumber, "renumber should be true if multi_gpu is true.");

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, multi_gpu>(handle,
                                                  local_vertices,
                                                  store_transposed ? edgelist_dsts : edgelist_srcs,
                                                  store_transposed ? edgelist_srcs : edgelist_dsts);
  }

  // 1. groupby edges to their target local adjacency matrix partition (and further groupby within
  // the local partition by applying the compute_gpu_id_from_vertex_t to minor vertex IDs).

  auto edge_counts = cugraph::detail::groupby_and_count_edgelist_by_local_partition_id(
    handle,
    store_transposed ? edgelist_dsts : edgelist_srcs,
    store_transposed ? edgelist_srcs : edgelist_dsts,
    edgelist_weights,
    true);

  std::vector<size_t> h_edge_counts(edge_counts.size());
  raft::update_host(
    h_edge_counts.data(), edge_counts.data(), edge_counts.size(), handle.get_stream());
  handle.sync_stream();

  std::vector<edge_t> edgelist_edge_counts(col_comm_size, edge_t{0});
  auto edgelist_intra_partition_segment_offsets =
    std::make_optional<std::vector<std::vector<edge_t>>>(
      col_comm_size, std::vector<edge_t>(row_comm_size + 1, edge_t{0}));
  for (int i = 0; i < col_comm_size; ++i) {
    edgelist_edge_counts[i] = std::accumulate(h_edge_counts.begin() + row_comm_size * i,
                                              h_edge_counts.begin() + row_comm_size * (i + 1),
                                              edge_t{0});
    std::partial_sum(h_edge_counts.begin() + row_comm_size * i,
                     h_edge_counts.begin() + row_comm_size * (i + 1),
                     (*edgelist_intra_partition_segment_offsets)[i].begin() + 1);
  }
  std::vector<edge_t> edgelist_displacements(col_comm_size, edge_t{0});
  std::partial_sum(edgelist_edge_counts.begin(),
                   edgelist_edge_counts.end() - 1,
                   edgelist_displacements.begin() + 1);

  // 2. split the input edges to local partitions

  std::vector<rmm::device_uvector<vertex_t>> edgelist_src_partitions{};
  edgelist_src_partitions.reserve(col_comm_size);
  for (int i = 0; i < col_comm_size; ++i) {
    rmm::device_uvector<vertex_t> tmp_srcs(edgelist_edge_counts[i], handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_srcs.begin() + edgelist_displacements[i],
                 edgelist_srcs.begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
                 tmp_srcs.begin());
    edgelist_src_partitions.push_back(std::move(tmp_srcs));
  }
  edgelist_srcs.resize(0, handle.get_stream());
  edgelist_srcs.shrink_to_fit(handle.get_stream());

  std::vector<rmm::device_uvector<vertex_t>> edgelist_dst_partitions{};
  edgelist_dst_partitions.reserve(col_comm_size);
  for (int i = 0; i < col_comm_size; ++i) {
    rmm::device_uvector<vertex_t> tmp_dsts(edgelist_edge_counts[i], handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_dsts.begin() + edgelist_displacements[i],
                 edgelist_dsts.begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
                 tmp_dsts.begin());
    edgelist_dst_partitions.push_back(std::move(tmp_dsts));
  }
  edgelist_dsts.resize(0, handle.get_stream());
  edgelist_dsts.shrink_to_fit(handle.get_stream());

  std::optional<std::vector<rmm::device_uvector<weight_t>>> edgelist_weight_partitions{};
  if (edgelist_weights) {
    edgelist_weight_partitions = std::vector<rmm::device_uvector<weight_t>>{};
    (*edgelist_weight_partitions).reserve(col_comm_size);
    for (int i = 0; i < col_comm_size; ++i) {
      rmm::device_uvector<weight_t> tmp_weights(edgelist_edge_counts[i], handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        (*edgelist_weights).begin() + edgelist_displacements[i],
        (*edgelist_weights).begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
        tmp_weights.begin());
      (*edgelist_weight_partitions).push_back(std::move(tmp_weights));
    }
    (*edgelist_weights).resize(0, handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }

  // 2. renumber

  std::vector<vertex_t*> src_ptrs(col_comm_size);
  std::vector<vertex_t*> dst_ptrs(src_ptrs.size());
  for (int i = 0; i < col_comm_size; ++i) {
    src_ptrs[i] = edgelist_src_partitions[i].begin();
    dst_ptrs[i] = edgelist_dst_partitions[i].begin();
  }
  auto [renumber_map_labels, meta] = cugraph::renumber_edgelist<vertex_t, edge_t, multi_gpu>(
    handle,
    std::move(local_vertices),
    src_ptrs,
    dst_ptrs,
    edgelist_edge_counts,
    edgelist_intra_partition_segment_offsets,
    store_transposed);

  // 3. create a graph

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_src_partitions),
      std::move(edgelist_dst_partitions),
      std::move(edgelist_weight_partitions),
      cugraph::graph_meta_t<vertex_t, edge_t, multi_gpu>{meta.number_of_vertices,
                                                         meta.number_of_edges,
                                                         graph_properties,
                                                         meta.partition,
                                                         meta.segment_offsets}),
    std::optional<rmm::device_uvector<vertex_t>>{std::move(renumber_map_labels)});
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
             std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(raft::handle_t const& handle,
                                std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                                rmm::device_uvector<vertex_t>&& edgelist_srcs,
                                rmm::device_uvector<vertex_t>&& edgelist_dsts,
                                std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                                graph_properties_t graph_properties,
                                bool renumber,
                                bool do_expensive_check)
{
  CUGRAPH_EXPECTS(edgelist_srcs.size() == edgelist_dsts.size(),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_dsts.size().");
  CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs.size() == (*edgelist_weights).size()),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_weights.size().");

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, multi_gpu>(handle,
                                                  vertices,
                                                  store_transposed ? edgelist_dsts : edgelist_srcs,
                                                  store_transposed ? edgelist_srcs : edgelist_dsts);
  }

  auto input_vertex_list_size = vertices ? static_cast<vertex_t>((*vertices).size()) : vertex_t{0};

  auto renumber_map_labels =
    renumber ? std::make_optional<rmm::device_uvector<vertex_t>>(0, handle.get_stream())
             : std::nullopt;
  renumber_meta_t<vertex_t, edge_t, multi_gpu> meta{};
  if (renumber) {
    std::tie(*renumber_map_labels, meta) = cugraph::renumber_edgelist<vertex_t, edge_t, multi_gpu>(
      handle,
      std::move(vertices),
      edgelist_srcs.data(),
      edgelist_dsts.data(),
      static_cast<edge_t>(edgelist_srcs.size()),
      store_transposed);
  }

  vertex_t num_vertices{};
  if (renumber) {
    num_vertices = static_cast<vertex_t>((*renumber_map_labels).size());
  } else {
    if (vertices) {
      num_vertices = input_vertex_list_size;
    } else {
      num_vertices = 1 + cugraph::detail::compute_maximum_vertex_id(
                           handle.get_stream(), edgelist_srcs, edgelist_dsts);
    }
  }

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      cugraph::graph_meta_t<vertex_t, edge_t, multi_gpu>{
        num_vertices,
        graph_properties,
        renumber ? std::optional<std::vector<vertex_t>>{meta.segment_offsets} : std::nullopt}),
    std::move(renumber_map_labels));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(raft::handle_t const& handle,
                           std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                           rmm::device_uvector<vertex_t>&& edgelist_srcs,
                           rmm::device_uvector<vertex_t>&& edgelist_dsts,
                           std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                           graph_properties_t graph_properties,
                           bool renumber,
                           bool do_expensive_check)
{
  return create_graph_from_edgelist_impl<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle,
    std::move(vertices),
    std::move(edgelist_srcs),
    std::move(edgelist_dsts),
    std::move(edgelist_weights),
    graph_properties,
    renumber,
    do_expensive_check);
}

}  // namespace cugraph

/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <detail/graph_partition_utils.cuh>
#include <structure/detail/structure_utils.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

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
                              rmm::device_uvector<vertex_t> const& edgelist_minors,
                              bool renumber)
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
    if (!renumber) {
      CUGRAPH_EXPECTS(static_cast<size_t>(thrust::count_if(
                        handle.get_thrust_policy(),
                        sorted_vertices.begin(),
                        sorted_vertices.end(),
                        detail::check_out_of_range_t<vertex_t>{
                          vertex_t{0}, std::numeric_limits<vertex_t>::max()})) == size_t{0},
                      "Invalid input argument: vertex IDs should be in [0, "
                      "std::numeric_limits<vertex_t>::max()) if renumber is false.");
      assert(!multi_gpu);  // renumbering is required in multi-GPU
      rmm::device_uvector<vertex_t> sequences(sorted_vertices.size(), handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(), sequences.begin(), sequences.end(), vertex_t{0});
      CUGRAPH_EXPECTS(thrust::equal(handle.get_thrust_policy(),
                                    sorted_vertices.begin(),
                                    sorted_vertices.end(),
                                    sequences.begin()),
                      "Invalid input argument: vertex IDs should be consecutive integers starting "
                      "from 0 if renumber is false.");
    }
  } else if (!renumber) {
    CUGRAPH_EXPECTS(static_cast<size_t>(thrust::count_if(
                      handle.get_thrust_policy(),
                      edgelist_majors.begin(),
                      edgelist_majors.end(),
                      detail::check_out_of_range_t<vertex_t>{
                        vertex_t{0}, std::numeric_limits<vertex_t>::max()})) == size_t{0},
                    "Invalid input argument: vertex IDs should be in [0, "
                    "std::numeric_limits<vertex_t>::max()) if renumber is false.");
    CUGRAPH_EXPECTS(static_cast<size_t>(thrust::count_if(
                      handle.get_thrust_policy(),
                      edgelist_minors.begin(),
                      edgelist_minors.end(),
                      detail::check_out_of_range_t<vertex_t>{
                        vertex_t{0}, std::numeric_limits<vertex_t>::max()})) == size_t{0},
                    "Invalid input argument: vertex IDs should be in [0, "
                    "std::numeric_limits<vertex_t>::max()) if renumber is false.");
  }

  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    if (vertices) {
      auto num_unique_vertices = host_scalar_allreduce(
        comm, (*vertices).size(), raft::comms::op_t::SUM, handle.get_stream());
      CUGRAPH_EXPECTS(
        num_unique_vertices < static_cast<size_t>(std::numeric_limits<vertex_t>::max()),
        "Invalid input arguments: # unique vertex IDs should be smaller than "
        "std::numeric_limits<vertex_t>::Max().");

      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         (*vertices).begin(),
                         (*vertices).end(),
                         [comm_rank,
                          key_func =
                            detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                              comm_size, major_comm_size, minor_comm_size}] __device__(auto val) {
                           return key_func(val) != comm_rank;
                         }) == 0,
        "Invalid input argument: vertices should be pre-shuffled.");
    }

    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_majors.begin(), edgelist_minors.begin()));
    CUGRAPH_EXPECTS(
      thrust::count_if(handle.get_thrust_policy(),
                       edge_first,
                       edge_first + edgelist_majors.size(),
                       [comm_rank,
                        gpu_id_key_func =
                          detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
                            comm_size, major_comm_size, minor_comm_size}] __device__(auto e) {
                         return (gpu_id_key_func(e) != comm_rank);
                       }) == 0,
      "Invalid input argument: edgelist_majors & edgelist_minors should be pre-shuffled.");

    if (vertices) {
      rmm::device_uvector<vertex_t> sorted_majors(0, handle.get_stream());
      {
        auto recvcounts =
          host_scalar_allgather(minor_comm, (*vertices).size(), handle.get_stream());
        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
        sorted_majors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
        device_allgatherv(minor_comm,
                          (*vertices).data(),
                          sorted_majors.data(),
                          recvcounts,
                          displacements,
                          handle.get_stream());
        thrust::sort(handle.get_thrust_policy(), sorted_majors.begin(), sorted_majors.end());
      }

      rmm::device_uvector<vertex_t> sorted_minors(0, handle.get_stream());
      {
        auto recvcounts =
          host_scalar_allgather(major_comm, (*vertices).size(), handle.get_stream());
        std::vector<size_t> displacements(recvcounts.size(), size_t{0});
        std::partial_sum(recvcounts.begin(), recvcounts.end() - 1, displacements.begin() + 1);
        sorted_minors.resize(displacements.back() + recvcounts.back(), handle.get_stream());
        device_allgatherv(major_comm,
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

template <typename vertex_t, bool store_transposed, bool multi_gpu>
bool check_symmetric(raft::handle_t const& handle,
                     raft::device_span<vertex_t const> edgelist_srcs,
                     raft::device_span<vertex_t const> edgelist_dsts)
{
  rmm::device_uvector<vertex_t> org_srcs(edgelist_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> org_dsts(edgelist_dsts.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), edgelist_srcs.begin(), edgelist_srcs.end(), org_srcs.begin());
  thrust::copy(
    handle.get_thrust_policy(), edgelist_dsts.begin(), edgelist_dsts.end(), org_dsts.begin());

  rmm::device_uvector<vertex_t> symmetrized_srcs(org_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> symmetrized_dsts(org_dsts.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), org_srcs.begin(), org_srcs.end(), symmetrized_srcs.begin());
  thrust::copy(
    handle.get_thrust_policy(), org_dsts.begin(), org_dsts.end(), symmetrized_dsts.begin());
  std::tie(symmetrized_srcs, symmetrized_dsts, std::ignore) =
    symmetrize_edgelist<vertex_t, float /* dummy */, store_transposed, multi_gpu>(
      handle, std::move(symmetrized_srcs), std::move(symmetrized_dsts), std::nullopt, true);

  if (org_srcs.size() != symmetrized_srcs.size()) { return false; }

  auto org_edge_first =
    thrust::make_zip_iterator(thrust::make_tuple(org_srcs.begin(), org_dsts.begin()));
  thrust::sort(handle.get_thrust_policy(), org_edge_first, org_edge_first + org_srcs.size());
  auto symmetrized_edge_first = thrust::make_zip_iterator(
    thrust::make_tuple(symmetrized_srcs.begin(), symmetrized_dsts.begin()));
  thrust::sort(handle.get_thrust_policy(),
               symmetrized_edge_first,
               symmetrized_edge_first + symmetrized_srcs.size());

  return thrust::equal(handle.get_thrust_policy(),
                       org_edge_first,
                       org_edge_first + org_srcs.size(),
                       symmetrized_edge_first);
}

template <typename vertex_t>
bool check_no_parallel_edge(raft::handle_t const& handle,
                            raft::device_span<vertex_t const> edgelist_srcs,
                            raft::device_span<vertex_t const> edgelist_dsts)
{
  rmm::device_uvector<vertex_t> org_srcs(edgelist_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> org_dsts(edgelist_dsts.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), edgelist_srcs.begin(), edgelist_srcs.end(), org_srcs.begin());
  thrust::copy(
    handle.get_thrust_policy(), edgelist_dsts.begin(), edgelist_dsts.end(), org_dsts.begin());

  auto org_edge_first =
    thrust::make_zip_iterator(thrust::make_tuple(org_srcs.begin(), org_dsts.begin()));
  thrust::sort(handle.get_thrust_policy(), org_edge_first, org_edge_first + org_srcs.size());
  return thrust::unique(
           handle.get_thrust_policy(), org_edge_first, org_edge_first + edgelist_srcs.size()) ==
         (org_edge_first + edgelist_srcs.size());
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_id_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  CUGRAPH_EXPECTS(edgelist_srcs.size() == edgelist_dsts.size(),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_dsts.size().");
  CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs.size() == (*edgelist_weights).size()),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_weights.size().");
  CUGRAPH_EXPECTS(!edgelist_edge_ids || (edgelist_srcs.size() == (*edgelist_edge_ids).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "std::get<0>((*edgelist_edge_ids)).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_types || (edgelist_srcs.size() == (*edgelist_edge_types).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "std::get<1>((*edgelist_edge_types)).size().");
  CUGRAPH_EXPECTS(renumber,
                  "Invalid input arguments: renumber should be true if multi_gpu is true.");

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, multi_gpu>(handle,
                                                  local_vertices,
                                                  store_transposed ? edgelist_dsts : edgelist_srcs,
                                                  store_transposed ? edgelist_srcs : edgelist_dsts,
                                                  renumber);

    if (graph_properties.is_symmetric) {
      CUGRAPH_EXPECTS(
        (check_symmetric<vertex_t, store_transposed, multi_gpu>(
          handle,
          raft::device_span<vertex_t const>(edgelist_srcs.data(), edgelist_srcs.size()),
          raft::device_span<vertex_t const>(edgelist_dsts.data(), edgelist_dsts.size()))),
        "Invalid input arguments: graph_properties.is_symmetric is true but the input edge list is "
        "not symmetric.");
    }

    if (!graph_properties.is_multigraph) {
      CUGRAPH_EXPECTS(
        check_no_parallel_edge(
          handle,
          raft::device_span<vertex_t const>(edgelist_srcs.data(), edgelist_srcs.size()),
          raft::device_span<vertex_t const>(edgelist_dsts.data(), edgelist_dsts.size())),
        "Invalid input arguments: graph_properties.is_multigraph is false but the input edge list "
        "has parallel edges.");
    }
  }

  // 1. groupby edges to their target local adjacency matrix partition (and further groupby within
  // the local partition by applying the compute_gpu_id_from_vertex_t to minor vertex IDs).

  auto d_edge_counts = cugraph::detail::groupby_and_count_edgelist_by_local_partition_id(
    handle,
    store_transposed ? edgelist_dsts : edgelist_srcs,
    store_transposed ? edgelist_srcs : edgelist_dsts,
    edgelist_weights,
    edgelist_edge_ids,
    edgelist_edge_types,
    true);

  std::vector<size_t> h_edge_counts(d_edge_counts.size());
  raft::update_host(
    h_edge_counts.data(), d_edge_counts.data(), d_edge_counts.size(), handle.get_stream());
  handle.sync_stream();

  std::vector<edge_t> edgelist_edge_counts(minor_comm_size, edge_t{0});
  auto edgelist_intra_partition_segment_offsets =
    std::make_optional<std::vector<std::vector<edge_t>>>(
      minor_comm_size, std::vector<edge_t>(major_comm_size + 1, edge_t{0}));
  for (int i = 0; i < minor_comm_size; ++i) {
    edgelist_edge_counts[i] = std::accumulate(h_edge_counts.begin() + major_comm_size * i,
                                              h_edge_counts.begin() + major_comm_size * (i + 1),
                                              edge_t{0});
    std::partial_sum(h_edge_counts.begin() + major_comm_size * i,
                     h_edge_counts.begin() + major_comm_size * (i + 1),
                     (*edgelist_intra_partition_segment_offsets)[i].begin() + 1);
  }
  std::vector<edge_t> edgelist_displacements(minor_comm_size, edge_t{0});
  std::partial_sum(edgelist_edge_counts.begin(),
                   edgelist_edge_counts.end() - 1,
                   edgelist_displacements.begin() + 1);

  // 2. split the input edges to local partitions

  std::vector<rmm::device_uvector<vertex_t>> edge_partition_edgelist_srcs{};
  edge_partition_edgelist_srcs.reserve(minor_comm_size);
  for (int i = 0; i < minor_comm_size; ++i) {
    rmm::device_uvector<vertex_t> tmp_srcs(edgelist_edge_counts[i], handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_srcs.begin() + edgelist_displacements[i],
                 edgelist_srcs.begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
                 tmp_srcs.begin());
    edge_partition_edgelist_srcs.push_back(std::move(tmp_srcs));
  }
  edgelist_srcs.resize(0, handle.get_stream());
  edgelist_srcs.shrink_to_fit(handle.get_stream());

  std::vector<rmm::device_uvector<vertex_t>> edge_partition_edgelist_dsts{};
  edge_partition_edgelist_dsts.reserve(minor_comm_size);
  for (int i = 0; i < minor_comm_size; ++i) {
    rmm::device_uvector<vertex_t> tmp_dsts(edgelist_edge_counts[i], handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_dsts.begin() + edgelist_displacements[i],
                 edgelist_dsts.begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
                 tmp_dsts.begin());
    edge_partition_edgelist_dsts.push_back(std::move(tmp_dsts));
  }
  edgelist_dsts.resize(0, handle.get_stream());
  edgelist_dsts.shrink_to_fit(handle.get_stream());

  std::optional<std::vector<rmm::device_uvector<weight_t>>> edge_partition_edgelist_weights{};
  if (edgelist_weights) {
    edge_partition_edgelist_weights = std::vector<rmm::device_uvector<weight_t>>{};
    (*edge_partition_edgelist_weights).reserve(minor_comm_size);
    for (int i = 0; i < minor_comm_size; ++i) {
      rmm::device_uvector<weight_t> tmp_weights(edgelist_edge_counts[i], handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        (*edgelist_weights).begin() + edgelist_displacements[i],
        (*edgelist_weights).begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
        tmp_weights.begin());
      (*edge_partition_edgelist_weights).push_back(std::move(tmp_weights));
    }
    (*edgelist_weights).resize(0, handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }

  std::optional<std::vector<rmm::device_uvector<edge_id_t>>> edge_partition_edgelist_edge_ids{};
  if (edgelist_edge_ids) {
    edge_partition_edgelist_edge_ids = std::vector<rmm::device_uvector<edge_id_t>>{};
    (*edge_partition_edgelist_edge_ids).reserve(minor_comm_size);
    for (int i = 0; i < minor_comm_size; ++i) {
      rmm::device_uvector<edge_id_t> tmp_edge_ids(edgelist_edge_counts[i], handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        (*edgelist_edge_ids).begin() + edgelist_displacements[i],
        (*edgelist_edge_ids).begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
        tmp_edge_ids.begin());
      (*edge_partition_edgelist_edge_ids).push_back(std::move(tmp_edge_ids));
    }
    (*edgelist_edge_ids).resize(0, handle.get_stream());
    (*edgelist_edge_ids).shrink_to_fit(handle.get_stream());
  }

  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_partition_edgelist_edge_types{};
  if (edgelist_edge_types) {
    edge_partition_edgelist_edge_types = std::vector<rmm::device_uvector<edge_type_t>>{};
    (*edge_partition_edgelist_edge_types).reserve(minor_comm_size);
    for (int i = 0; i < minor_comm_size; ++i) {
      rmm::device_uvector<edge_type_t> tmp_edge_types(edgelist_edge_counts[i], handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        (*edgelist_edge_types).begin() + edgelist_displacements[i],
        (*edgelist_edge_types).begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
        tmp_edge_types.begin());
      (*edge_partition_edgelist_edge_types).push_back(std::move(tmp_edge_types));
    }
    (*edgelist_edge_types).resize(0, handle.get_stream());
    (*edgelist_edge_types).shrink_to_fit(handle.get_stream());
  }

  // 3. renumber

  std::vector<vertex_t*> src_ptrs(minor_comm_size);
  std::vector<vertex_t*> dst_ptrs(src_ptrs.size());
  for (int i = 0; i < minor_comm_size; ++i) {
    src_ptrs[i] = edge_partition_edgelist_srcs[i].begin();
    dst_ptrs[i] = edge_partition_edgelist_dsts[i].begin();
  }
  auto [renumber_map_labels, meta] = cugraph::renumber_edgelist<vertex_t, edge_t, multi_gpu>(
    handle,
    std::move(local_vertices),
    src_ptrs,
    dst_ptrs,
    edgelist_edge_counts,
    edgelist_intra_partition_segment_offsets,
    store_transposed);

  auto num_segments_per_vertex_partition =
    static_cast<size_t>(meta.edge_partition_segment_offsets.size() / minor_comm_size);
  auto use_dcs =
    num_segments_per_vertex_partition > (detail::num_sparse_segments_per_vertex_partition + 2);

  // 4. sort and compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  size_t element_size   = sizeof(vertex_t) * 2;
  if (edgelist_weights) { element_size += sizeof(weight_t); }
  if (edgelist_edge_ids) { element_size += sizeof(edge_id_t); }
  if (edgelist_edge_types) { element_size += sizeof(edge_type_t); }
  auto constexpr mem_frugal_ratio =
    0.25;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  std::vector<rmm::device_uvector<edge_t>> edge_partition_offsets;
  std::vector<rmm::device_uvector<vertex_t>> edge_partition_indices;
  std::optional<std::vector<rmm::device_uvector<weight_t>>> edge_partition_weights{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_id_t>>> edge_partition_edge_ids{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_partition_edge_types{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<vertex_t>>> edge_partition_dcs_nzd_vertices{
    std::nullopt};

  edge_partition_offsets.reserve(edge_partition_edgelist_srcs.size());
  edge_partition_indices.reserve(edge_partition_edgelist_srcs.size());
  if (edge_partition_edgelist_weights) {
    edge_partition_weights = std::vector<rmm::device_uvector<weight_t>>{};
    (*edge_partition_weights).reserve(edge_partition_edgelist_srcs.size());
  }
  if (edge_partition_edgelist_edge_ids) {
    edge_partition_edge_ids = std::vector<rmm::device_uvector<edge_id_t>>{};
    (*edge_partition_edge_ids).reserve(edge_partition_edgelist_srcs.size());
  }
  if (edge_partition_edgelist_edge_types) {
    edge_partition_edge_types = std::vector<rmm::device_uvector<edge_type_t>>{};
    (*edge_partition_edge_types).reserve(edge_partition_edgelist_srcs.size());
  }
  if (use_dcs) {
    edge_partition_dcs_nzd_vertices = std::vector<rmm::device_uvector<vertex_t>>{};
    (*edge_partition_dcs_nzd_vertices).reserve(edge_partition_edgelist_srcs.size());
  }

  for (size_t i = 0; i < edge_partition_edgelist_srcs.size(); ++i) {
    auto [major_range_first, major_range_last] = meta.partition.local_edge_partition_major_range(i);
    auto [minor_range_first, minor_range_last] = meta.partition.local_edge_partition_minor_range();
    rmm::device_uvector<edge_t> offsets(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> indices(size_t{0}, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    std::optional<rmm::device_uvector<edge_id_t>> edge_ids{std::nullopt};
    std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
    auto major_hypersparse_first =
      use_dcs
        ? std::make_optional<vertex_t>(
            major_range_first +
            meta.edge_partition_segment_offsets[num_segments_per_vertex_partition * i +
                                                detail::num_sparse_segments_per_vertex_partition])
        : std::nullopt;
    if (edgelist_weights) {
      if (edgelist_edge_ids) {
        if (edgelist_edge_types) {
          std::forward_as_tuple(
            offsets, indices, std::tie(weights, edge_ids, edge_types), dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t,
                                               edge_t,
                                               thrust::tuple<weight_t, edge_id_t, edge_type_t>,
                                               store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              std::make_tuple(std::move((*edge_partition_edgelist_weights)[i]),
                              std::move((*edge_partition_edgelist_edge_ids)[i]),
                              std::move((*edge_partition_edgelist_edge_types)[i])),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        } else {
          std::forward_as_tuple(offsets, indices, std::tie(weights, edge_ids), dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t,
                                               edge_t,
                                               thrust::tuple<weight_t, edge_id_t>,
                                               store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              std::make_tuple(std::move((*edge_partition_edgelist_weights)[i]),
                              std::move((*edge_partition_edgelist_edge_ids)[i])),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        }
      } else {
        if (edgelist_edge_types) {
          std::forward_as_tuple(offsets, indices, std::tie(weights, edge_types), dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t,
                                               edge_t,
                                               thrust::tuple<weight_t, edge_type_t>,
                                               store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              std::make_tuple(std::move((*edge_partition_edgelist_weights)[i]),
                              std::move((*edge_partition_edgelist_edge_types)[i])),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        } else {
          std::forward_as_tuple(offsets, indices, weights, dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t, edge_t, weight_t, store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              std::move((*edge_partition_edgelist_weights)[i]),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        }
      }
    } else {
      if (edgelist_edge_ids) {
        if (edgelist_edge_types) {
          std::forward_as_tuple(
            offsets, indices, std::tie(edge_ids, edge_types), dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t,
                                               edge_t,
                                               thrust::tuple<edge_id_t, edge_type_t>,
                                               store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              std::make_tuple(std::move((*edge_partition_edgelist_edge_ids)[i]),
                              std::move((*edge_partition_edgelist_edge_types)[i])),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        } else {
          std::forward_as_tuple(offsets, indices, edge_ids, dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_id_t, store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              std::move((*edge_partition_edgelist_edge_ids)[i]),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        }
      } else {
        if (edgelist_edge_types) {
          std::forward_as_tuple(offsets, indices, edge_types, dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_type_t, store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              std::move((*edge_partition_edgelist_edge_types)[i]),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        } else {
          std::forward_as_tuple(offsets, indices, dcs_nzd_vertices) =
            detail::sort_and_compress_edgelist<vertex_t, edge_t, store_transposed>(
              std::move(edge_partition_edgelist_srcs[i]),
              std::move(edge_partition_edgelist_dsts[i]),
              major_range_first,
              major_hypersparse_first,
              major_range_last,
              minor_range_first,
              minor_range_last,
              mem_frugal_threshold,
              handle.get_stream());
        }
      }
    }

    edge_partition_offsets.push_back(std::move(offsets));
    edge_partition_indices.push_back(std::move(indices));
    if (edge_partition_weights) { (*edge_partition_weights).push_back(std::move(*weights)); }
    if (edge_partition_edge_ids) { (*edge_partition_edge_ids).push_back(std::move(*edge_ids)); }
    if (edge_partition_edge_types) {
      (*edge_partition_edge_types).push_back(std::move(*edge_types));
    }
    if (edge_partition_dcs_nzd_vertices) {
      (*edge_partition_dcs_nzd_vertices).push_back(std::move(*dcs_nzd_vertices));
    }
  }

  // 5. segmented sort neighbors

  for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
    if (edge_partition_weights) {
      if (edge_partition_edge_ids) {
        if (edge_partition_edge_types) {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end(),
            thrust::make_zip_iterator((*edge_partition_weights)[i].begin(),
                                      (*edge_partition_edge_ids)[i].begin(),
                                      (*edge_partition_edge_types)[i].begin()));
        } else {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end(),
            thrust::make_zip_iterator((*edge_partition_weights)[i].begin(),
                                      (*edge_partition_edge_ids)[i].begin()));
        }
      } else {
        if (edge_partition_edge_types) {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end(),
            thrust::make_zip_iterator((*edge_partition_weights)[i].begin(),
                                      (*edge_partition_edge_types)[i].begin()));
        } else {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end(),
            (*edge_partition_weights)[i].begin());
        }
      }
    } else {
      if (edge_partition_edge_ids) {
        if (edge_partition_edge_types) {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end(),
            thrust::make_zip_iterator((*edge_partition_edge_ids)[i].begin(),
                                      (*edge_partition_edge_types)[i].begin()));
        } else {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end(),
            thrust::make_zip_iterator((*edge_partition_edge_ids)[i].begin()));
        }
      } else {
        if (edge_partition_edge_types) {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end(),
            thrust::make_zip_iterator((*edge_partition_edge_types)[i].begin()));
        } else {
          detail::sort_adjacency_list(
            handle,
            raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                            edge_partition_offsets[i].size()),
            edge_partition_indices[i].begin(),
            edge_partition_indices[i].end());
        }
      }
    }
  }

  // 6. create a graph and an edge_property_t object.

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  if (edge_partition_weights) {
    edge_weights =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>(
        std::move(*edge_partition_weights));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_id_t>>
    edge_ids{std::nullopt};
  if (edge_partition_edge_ids) {
    edge_ids =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_id_t>(
        std::move(*edge_partition_edge_ids));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>
    edge_types{std::nullopt};
  if (edge_partition_edge_types) {
    edge_types =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>(
        std::move(*edge_partition_edge_types));
  }

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(
      handle,
      std::move(edge_partition_offsets),
      std::move(edge_partition_indices),
      std::move(edge_partition_dcs_nzd_vertices),
      cugraph::graph_meta_t<vertex_t, edge_t, multi_gpu>{meta.number_of_vertices,
                                                         meta.number_of_edges,
                                                         graph_properties,
                                                         meta.partition,
                                                         meta.edge_partition_segment_offsets}),
    std::move(edge_weights),
    std::move(edge_ids),
    std::move(edge_types),
    std::optional<rmm::device_uvector<vertex_t>>{std::move(renumber_map_labels)});
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_id_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(
    !vertices || ((*vertices).size() < static_cast<size_t>(std::numeric_limits<vertex_t>::max())),
    "Invalid input arguments: # unique vertex IDs should be smaller than "
    "std::numeric_limits<vertex_t>::Max().");
  CUGRAPH_EXPECTS(edgelist_srcs.size() == edgelist_dsts.size(),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_dsts.size().");
  CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs.size() == (*edgelist_weights).size()),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_weights.size().");
  CUGRAPH_EXPECTS(!edgelist_edge_ids || (edgelist_srcs.size() == (*edgelist_edge_ids).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "(*edgelist_edge_ids).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_types || (edgelist_srcs.size() == (*edgelist_edge_types).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "(*edgelist_edge_types).size().");

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, multi_gpu>(handle,
                                                  vertices,
                                                  store_transposed ? edgelist_dsts : edgelist_srcs,
                                                  store_transposed ? edgelist_srcs : edgelist_dsts,
                                                  renumber);

    if (graph_properties.is_symmetric) {
      CUGRAPH_EXPECTS(
        (check_symmetric<vertex_t, store_transposed, multi_gpu>(
          handle,
          raft::device_span<vertex_t const>(edgelist_srcs.data(), edgelist_srcs.size()),
          raft::device_span<vertex_t const>(edgelist_dsts.data(), edgelist_dsts.size()))),
        "Invalid input arguments: graph_properties.is_symmetric is true but the input edge list is "
        "not symmetric.");
    }

    if (!graph_properties.is_multigraph) {
      CUGRAPH_EXPECTS(
        check_no_parallel_edge(
          handle,
          raft::device_span<vertex_t const>(edgelist_srcs.data(), edgelist_srcs.size()),
          raft::device_span<vertex_t const>(edgelist_dsts.data(), edgelist_dsts.size())),
        "Invalid input arguments: graph_properties.is_multigraph is false but the input edge list "
        "has parallel edges.");
    }
  }

  // renumber

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
      num_vertices = (*vertices).size();
    } else {
      num_vertices = 1 + cugraph::detail::compute_maximum_vertex_id(
                           handle.get_stream(), edgelist_srcs, edgelist_dsts);
    }
  }

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  size_t element_size   = sizeof(vertex_t) * 2;
  if (edgelist_weights) { element_size += sizeof(weight_t); }
  if (edgelist_edge_ids) { element_size += sizeof(edge_id_t); }
  if (edgelist_edge_types) { element_size += sizeof(edge_type_t); }
  auto constexpr mem_frugal_ratio =
    0.25;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  rmm::device_uvector<edge_t> offsets(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> indices(size_t{0}, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_id_t>> ids{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> types{std::nullopt};

  if (edgelist_weights) {
    if (edgelist_edge_ids) {
      if (edgelist_edge_types) {
        std::forward_as_tuple(offsets, indices, std::tie(weights, ids, types), std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t,
                                             edge_t,
                                             thrust::tuple<weight_t, edge_id_t, edge_type_t>,
                                             store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights),
                            std::move(*edgelist_edge_ids),
                            std::move(*edgelist_edge_types)),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      } else {
        std::forward_as_tuple(offsets, indices, std::tie(weights, ids), std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t,
                                             edge_t,
                                             thrust::tuple<weight_t, edge_id_t>,
                                             store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights), std::move(*edgelist_edge_ids)),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      }
    } else {
      if (edgelist_edge_types) {
        std::forward_as_tuple(offsets, indices, std::tie(weights, types), std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t,
                                             edge_t,
                                             thrust::tuple<weight_t, edge_type_t>,
                                             store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_weights), std::move(*edgelist_edge_types)),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      } else {
        std::forward_as_tuple(offsets, indices, weights, std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t, edge_t, weight_t, store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::move(*edgelist_weights),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      }
    }
  } else {
    if (edgelist_edge_ids) {
      if (edgelist_edge_types) {
        std::forward_as_tuple(offsets, indices, std::tie(ids, types), std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t,
                                             edge_t,
                                             thrust::tuple<edge_id_t, edge_type_t>,
                                             store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::make_tuple(std::move(*edgelist_edge_ids), std::move(*edgelist_edge_types)),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      } else {
        std::forward_as_tuple(offsets, indices, ids, std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_id_t, store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::move(*edgelist_edge_ids),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      }
    } else {
      if (edgelist_edge_types) {
        std::forward_as_tuple(offsets, indices, types, std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_type_t, store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            std::move(*edgelist_edge_types),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      } else {
        std::forward_as_tuple(offsets, indices, std::ignore) =
          detail::sort_and_compress_edgelist<vertex_t, edge_t, store_transposed>(
            std::move(edgelist_srcs),
            std::move(edgelist_dsts),
            vertex_t{0},
            std::optional<vertex_t>{std::nullopt},
            num_vertices,
            vertex_t{0},
            num_vertices,
            mem_frugal_threshold,
            handle.get_stream());
      }
    }
  }

  // create a graph and an edge_property_t object.

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  if (weights) {
    std::vector<rmm::device_uvector<weight_t>> buffers{};
    buffers.push_back(std::move(*weights));
    edge_weights =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>(
        std::move(buffers));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_id_t>>
    edge_ids{std::nullopt};
  if (ids) {
    std::vector<rmm::device_uvector<edge_id_t>> buffers{};
    buffers.push_back(std::move(*ids));
    edge_ids =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_id_t>(
        std::move(buffers));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>
    edge_types{std::nullopt};
  if (types) {
    std::vector<rmm::device_uvector<edge_type_t>> buffers{};
    buffers.push_back(std::move(*types));
    edge_types =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>(
        std::move(buffers));
  }

  // graph_t constructor

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(
      handle,
      std::move(offsets),
      std::move(indices),
      cugraph::graph_meta_t<vertex_t, edge_t, multi_gpu>{
        num_vertices,
        graph_properties,
        renumber ? std::optional<std::vector<vertex_t>>{meta.segment_offsets} : std::nullopt}),
    std::move(edge_weights),
    std::move(edge_ids),
    std::move(edge_types),
    std::move(renumber_map_labels));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_id_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(raft::handle_t const& handle,
                           std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                           rmm::device_uvector<vertex_t>&& edgelist_srcs,
                           rmm::device_uvector<vertex_t>&& edgelist_dsts,
                           std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                           std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
                           std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                           graph_properties_t graph_properties,
                           bool renumber,
                           bool do_expensive_check)
{
  return create_graph_from_edgelist_impl<vertex_t,
                                         edge_t,
                                         weight_t,
                                         edge_id_t,
                                         edge_type_t,
                                         store_transposed,
                                         multi_gpu>(handle,
                                                    std::move(vertices),
                                                    std::move(edgelist_srcs),
                                                    std::move(edgelist_dsts),
                                                    std::move(edgelist_weights),
                                                    std::move(edgelist_edge_ids),
                                                    std::move(edgelist_edge_types),
                                                    graph_properties,
                                                    renumber,
                                                    do_expensive_check);
}

}  // namespace cugraph

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

#include <detail/graph_utils.cuh>
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
    auto& comm               = handle.get_comms();
    auto const comm_size     = comm.get_size();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_size = col_comm.get_size();

    if (vertices) {
      auto num_unique_vertices = host_scalar_allreduce(
        comm, (*vertices).size(), raft::comms::op_t::SUM, handle.get_stream());
      CUGRAPH_EXPECTS(num_unique_vertices < std::numeric_limits<vertex_t>::max(),
                      "Invalid input arguments: # unique vertex IDs should be smaller than "
                      "std::numeric_limits<vertex_t>::Max().");

      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          (*vertices).begin(),
          (*vertices).end(),
          [comm_rank,
           key_func =
             detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size}] __device__(auto val) {
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
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                  thrust::tuple<edge_t, edge_type_t>>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&&
    edgelist_id_type_pairs,
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
  CUGRAPH_EXPECTS(!edgelist_id_type_pairs ||
                    (edgelist_srcs.size() == std::get<0>((*edgelist_id_type_pairs)).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "std::get<0>((*edgelist_id_type_pairs)).size().");
  CUGRAPH_EXPECTS(!edgelist_id_type_pairs ||
                    (edgelist_srcs.size() == std::get<1>((*edgelist_id_type_pairs)).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "std::get<1>((*edgelist_id_type_pairs)).size().");
  CUGRAPH_EXPECTS(renumber,
                  "Invalid input arguments: renumber should be true if multi_gpu is true.");

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, multi_gpu>(handle,
                                                  local_vertices,
                                                  store_transposed ? edgelist_dsts : edgelist_srcs,
                                                  store_transposed ? edgelist_srcs : edgelist_dsts,
                                                  renumber);
  }

  // 1. groupby edges to their target local adjacency matrix partition (and further groupby within
  // the local partition by applying the compute_gpu_id_from_vertex_t to minor vertex IDs).

  auto d_edge_counts = cugraph::detail::groupby_and_count_edgelist_by_local_partition_id(
    handle,
    store_transposed ? edgelist_dsts : edgelist_srcs,
    store_transposed ? edgelist_srcs : edgelist_dsts,
    edgelist_weights,
    edgelist_id_type_pairs,
    true);

  std::vector<size_t> h_edge_counts(d_edge_counts.size());
  raft::update_host(
    h_edge_counts.data(), d_edge_counts.data(), d_edge_counts.size(), handle.get_stream());
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

  std::vector<rmm::device_uvector<vertex_t>> edge_partition_edgelist_srcs{};
  edge_partition_edgelist_srcs.reserve(col_comm_size);
  for (int i = 0; i < col_comm_size; ++i) {
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
  edge_partition_edgelist_dsts.reserve(col_comm_size);
  for (int i = 0; i < col_comm_size; ++i) {
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
    (*edge_partition_edgelist_weights).reserve(col_comm_size);
    for (int i = 0; i < col_comm_size; ++i) {
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

  std::optional<
    std::vector<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>>
    edge_partition_edgelist_id_type_pairs{};
  if (edgelist_id_type_pairs) {
    edge_partition_edgelist_id_type_pairs =
      std::vector<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>{};
    (*edge_partition_edgelist_id_type_pairs).reserve(col_comm_size);
    for (int i = 0; i < col_comm_size; ++i) {
      auto tmp_id_type_pairs = allocate_dataframe_buffer<thrust::tuple<edge_t, edge_type_t>>(
        edgelist_edge_counts[i], handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(*edgelist_id_type_pairs) + edgelist_displacements[i],
                   get_dataframe_buffer_begin(*edgelist_id_type_pairs) + edgelist_displacements[i] +
                     edgelist_edge_counts[i],
                   get_dataframe_buffer_begin(tmp_id_type_pairs));
      (*edge_partition_edgelist_id_type_pairs).push_back(std::move(tmp_id_type_pairs));
    }
    std::get<0>(*edgelist_id_type_pairs).resize(0, handle.get_stream());
    std::get<0>(*edgelist_id_type_pairs).shrink_to_fit(handle.get_stream());
    std::get<1>(*edgelist_id_type_pairs).resize(0, handle.get_stream());
    std::get<1>(*edgelist_id_type_pairs).shrink_to_fit(handle.get_stream());
  }

  // 2. renumber

  std::vector<vertex_t*> src_ptrs(col_comm_size);
  std::vector<vertex_t*> dst_ptrs(src_ptrs.size());
  for (int i = 0; i < col_comm_size; ++i) {
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
    static_cast<size_t>(meta.edge_partition_segment_offsets.size() / col_comm_size);
  auto use_dcs =
    num_segments_per_vertex_partition > (detail::num_sparse_segments_per_vertex_partition + 2);

  // 3. compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid

  std::vector<rmm::device_uvector<edge_t>> edge_partition_offsets;
  std::vector<rmm::device_uvector<vertex_t>> edge_partition_indices;
  std::optional<std::vector<rmm::device_uvector<weight_t>>> edge_partition_weights{std::nullopt};
  std::optional<
    std::vector<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>>
    edge_partition_id_type_pairs{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<vertex_t>>> edge_partition_dcs_nzd_vertices{
    std::nullopt};

  edge_partition_offsets.reserve(edge_partition_edgelist_srcs.size());
  edge_partition_indices.reserve(edge_partition_edgelist_srcs.size());
  if (edge_partition_edgelist_weights) {
    edge_partition_weights = std::vector<rmm::device_uvector<weight_t>>{};
    (*edge_partition_weights).reserve(edge_partition_edgelist_srcs.size());
  }
  if (edge_partition_edgelist_id_type_pairs) {
    edge_partition_id_type_pairs =
      std::vector<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>{};
    (*edge_partition_id_type_pairs).reserve(edge_partition_edgelist_srcs.size());
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
    std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>
      id_type_pairs{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
    auto major_hypersparse_first =
      use_dcs
        ? std::make_optional<vertex_t>(
            major_range_first +
            meta.edge_partition_segment_offsets[num_segments_per_vertex_partition * i +
                                                detail::num_sparse_segments_per_vertex_partition])
        : std::nullopt;
    if (edgelist_weights && edgelist_id_type_pairs) {
      auto edge_value_first = thrust::make_zip_iterator(
        thrust::make_tuple((*edge_partition_edgelist_weights)[i].begin(),
                           std::get<0>((*edge_partition_edgelist_id_type_pairs)[i]).begin(),
                           std::get<1>((*edge_partition_edgelist_id_type_pairs)[i]).begin()));
      id_type_pairs =
        std::make_tuple(rmm::device_uvector<edge_t>(size_t{0}, handle.get_stream()),
                        rmm::device_uvector<edge_type_t>(size_t{0}, handle.get_stream()));
      std::forward_as_tuple(
        offsets,
        indices,
        std::tie(weights, std::get<0>(*id_type_pairs), std::get<1>(*id_type_pairs)),
        dcs_nzd_vertices) =
        detail::compress_edgelist<edge_t, store_transposed>(edge_partition_edgelist_srcs[i].begin(),
                                                            edge_partition_edgelist_srcs[i].end(),
                                                            edge_partition_edgelist_dsts[i].begin(),
                                                            edge_value_first,
                                                            major_range_first,
                                                            major_hypersparse_first,
                                                            major_range_last,
                                                            minor_range_first,
                                                            minor_range_last,
                                                            handle.get_stream());
    } else if (edge_partition_edgelist_weights) {
      auto edge_value_first = (*edge_partition_edgelist_weights)[i].begin();
      std::tie(offsets, indices, weights, dcs_nzd_vertices) =
        detail::compress_edgelist<edge_t, store_transposed>(edge_partition_edgelist_srcs[i].begin(),
                                                            edge_partition_edgelist_srcs[i].end(),
                                                            edge_partition_edgelist_dsts[i].begin(),
                                                            edge_value_first,
                                                            major_range_first,
                                                            major_hypersparse_first,
                                                            major_range_last,
                                                            minor_range_first,
                                                            minor_range_last,
                                                            handle.get_stream());
    } else if (edge_partition_edgelist_id_type_pairs) {
      auto edge_value_first =
        get_dataframe_buffer_begin((*edge_partition_edgelist_id_type_pairs)[i]);
      std::tie(offsets, indices, id_type_pairs, dcs_nzd_vertices) =
        detail::compress_edgelist<edge_t, store_transposed>(edge_partition_edgelist_srcs[i].begin(),
                                                            edge_partition_edgelist_srcs[i].end(),
                                                            edge_partition_edgelist_dsts[i].begin(),
                                                            edge_value_first,
                                                            major_range_first,
                                                            major_hypersparse_first,
                                                            major_range_last,
                                                            minor_range_first,
                                                            minor_range_last,
                                                            handle.get_stream());
    } else {
      std::tie(offsets, indices, dcs_nzd_vertices) =
        detail::compress_edgelist<edge_t, store_transposed>(edge_partition_edgelist_srcs[i].begin(),
                                                            edge_partition_edgelist_srcs[i].end(),
                                                            edge_partition_edgelist_dsts[i].begin(),
                                                            major_range_first,
                                                            major_hypersparse_first,
                                                            major_range_last,
                                                            minor_range_first,
                                                            minor_range_last,
                                                            handle.get_stream());
    }
    edge_partition_edgelist_srcs[i].resize(0, handle.get_stream());
    edge_partition_edgelist_srcs[i].shrink_to_fit(handle.get_stream());
    edge_partition_edgelist_dsts[i].resize(0, handle.get_stream());
    edge_partition_edgelist_dsts[i].shrink_to_fit(handle.get_stream());
    if (edge_partition_edgelist_weights) {
      (*edge_partition_edgelist_weights)[i].resize(0, handle.get_stream());
      (*edge_partition_edgelist_weights)[i].shrink_to_fit(handle.get_stream());
    }
    if (edge_partition_edgelist_id_type_pairs) {
      std::get<0>((*edge_partition_edgelist_id_type_pairs)[i]).resize(0, handle.get_stream());
      std::get<0>((*edge_partition_edgelist_id_type_pairs)[i]).shrink_to_fit(handle.get_stream());
      std::get<1>((*edge_partition_edgelist_id_type_pairs)[i]).resize(0, handle.get_stream());
      std::get<1>((*edge_partition_edgelist_id_type_pairs)[i]).shrink_to_fit(handle.get_stream());
    }

    edge_partition_offsets.push_back(std::move(offsets));
    edge_partition_indices.push_back(std::move(indices));
    if (edge_partition_weights) { (*edge_partition_weights).push_back(std::move(*weights)); }
    if (edge_partition_id_type_pairs) {
      (*edge_partition_id_type_pairs).push_back(std::move(*id_type_pairs));
    }
    if (edge_partition_dcs_nzd_vertices) {
      (*edge_partition_dcs_nzd_vertices).push_back(std::move(*dcs_nzd_vertices));
    }
  }

  // 4. segmented sort neighbors

  for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
    if (edge_partition_weights && edge_partition_id_type_pairs) {
      detail::sort_adjacency_list(handle,
                                  raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                                                  edge_partition_offsets[i].size()),
                                  edge_partition_indices[i].begin(),
                                  edge_partition_indices[i].end(),
                                  thrust::make_zip_iterator(thrust::make_tuple(
                                    (*edge_partition_weights)[i].begin(),
                                    std::get<0>((*edge_partition_id_type_pairs)[i]).begin(),
                                    std::get<1>((*edge_partition_id_type_pairs)[i]).begin())));
    } else if (edge_partition_weights) {
      detail::sort_adjacency_list(handle,
                                  raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                                                  edge_partition_offsets[i].size()),
                                  edge_partition_indices[i].begin(),
                                  edge_partition_indices[i].end(),
                                  (*edge_partition_weights)[i].begin());
    } else if (edge_partition_id_type_pairs) {
      detail::sort_adjacency_list(handle,
                                  raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                                                  edge_partition_offsets[i].size()),
                                  edge_partition_indices[i].begin(),
                                  edge_partition_indices[i].end(),
                                  thrust::make_zip_iterator(thrust::make_tuple(
                                    std::get<0>((*edge_partition_id_type_pairs)[i]).begin(),
                                    std::get<1>((*edge_partition_id_type_pairs)[i]).begin())));
    } else {
      detail::sort_adjacency_list(handle,
                                  raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                                                  edge_partition_offsets[i].size()),
                                  edge_partition_indices[i].begin(),
                                  edge_partition_indices[i].end());
    }
  }

  // 5. create a graph and an edge_property_t object.

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  if (edge_partition_weights) {
    edge_weights =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>(
        std::move(*edge_partition_weights));
  }

  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                thrust::tuple<edge_t, edge_type_t>>>
    edge_id_type_pairs{std::nullopt};
  if (edge_partition_id_type_pairs) {
    edge_id_type_pairs =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                      thrust::tuple<edge_t, edge_type_t>>(std::move(*edge_partition_id_type_pairs));
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
    std::move(edge_id_type_pairs),
    std::optional<rmm::device_uvector<vertex_t>>{std::move(renumber_map_labels)});
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                  thrust::tuple<edge_t, edge_type_t>>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!vertices || ((*vertices).size() < std::numeric_limits<vertex_t>::max()),
                  "Invalid input arguments: # unique vertex IDs should be smaller than "
                  "std::numeric_limits<vertex_t>::Max().");
  CUGRAPH_EXPECTS(edgelist_srcs.size() == edgelist_dsts.size(),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_dsts.size().");
  CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs.size() == (*edgelist_weights).size()),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_weights.size().");
  CUGRAPH_EXPECTS(!edgelist_id_type_pairs ||
                    (edgelist_srcs.size() == std::get<0>((*edgelist_id_type_pairs)).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "std::get<0>((*edgelist_id_type_pairs)).size().");
  CUGRAPH_EXPECTS(!edgelist_id_type_pairs ||
                    (edgelist_srcs.size() == std::get<1>((*edgelist_id_type_pairs)).size()),
                  "Invalid input arguments: edgelist_srcs.size() != "
                  "std::get<1>((*edgelist_id_type_pairs)).size().");

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, multi_gpu>(handle,
                                                  vertices,
                                                  store_transposed ? edgelist_dsts : edgelist_srcs,
                                                  store_transposed ? edgelist_srcs : edgelist_dsts,
                                                  renumber);
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

  rmm::device_uvector<edge_t> offsets(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> indices(size_t{0}, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>
    id_type_pairs{std::nullopt};
  if (edgelist_weights && edgelist_id_type_pairs) {
    auto edge_value_first =
      thrust::make_zip_iterator(thrust::make_tuple((*edgelist_weights).begin(),
                                                   std::get<0>(*edgelist_id_type_pairs).begin(),
                                                   std::get<1>(*edgelist_id_type_pairs).begin()));
    id_type_pairs =
      std::make_tuple(rmm::device_uvector<edge_t>(size_t{0}, handle.get_stream()),
                      rmm::device_uvector<edge_type_t>(size_t{0}, handle.get_stream()));
    std::forward_as_tuple(
      offsets,
      indices,
      std::tie(weights, std::get<0>(*id_type_pairs), std::get<1>(*id_type_pairs)),
      std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(edgelist_srcs.begin(),
                                                          edgelist_srcs.end(),
                                                          edgelist_dsts.begin(),
                                                          edge_value_first,
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          num_vertices,
                                                          vertex_t{0},
                                                          num_vertices,
                                                          handle.get_stream());
  } else if (edgelist_weights) {
    auto edge_value_first = (*edgelist_weights).begin();
    std::tie(offsets, indices, weights, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(edgelist_srcs.begin(),
                                                          edgelist_srcs.end(),
                                                          edgelist_dsts.begin(),
                                                          edge_value_first,
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          num_vertices,
                                                          vertex_t{0},
                                                          num_vertices,
                                                          handle.get_stream());
  } else if (edgelist_id_type_pairs) {
    auto edge_value_first = get_dataframe_buffer_begin(*edgelist_id_type_pairs);
    std::tie(offsets, indices, id_type_pairs, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(edgelist_srcs.begin(),
                                                          edgelist_srcs.end(),
                                                          edgelist_dsts.begin(),
                                                          edge_value_first,
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          num_vertices,
                                                          vertex_t{0},
                                                          num_vertices,
                                                          handle.get_stream());
  } else {
    std::tie(offsets, indices, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(edgelist_srcs.begin(),
                                                          edgelist_srcs.end(),
                                                          edgelist_dsts.begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          num_vertices,
                                                          vertex_t{0},
                                                          num_vertices,
                                                          handle.get_stream());
  }
  edgelist_srcs.resize(0, handle.get_stream());
  edgelist_srcs.shrink_to_fit(handle.get_stream());
  edgelist_dsts.resize(0, handle.get_stream());
  edgelist_dsts.shrink_to_fit(handle.get_stream());
  if (edgelist_weights) {
    (*edgelist_weights).resize(0, handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }
  if (edgelist_id_type_pairs) {
    std::get<0>(*edgelist_id_type_pairs).resize(0, handle.get_stream());
    std::get<0>(*edgelist_id_type_pairs).shrink_to_fit(handle.get_stream());
    std::get<1>(*edgelist_id_type_pairs).resize(0, handle.get_stream());
    std::get<1>(*edgelist_id_type_pairs).shrink_to_fit(handle.get_stream());
  }

  // segmented sort neighbors

  if (weights && id_type_pairs) {
    detail::sort_adjacency_list(
      handle,
      raft::device_span<edge_t const>(offsets.data(), offsets.size()),
      indices.begin(),
      indices.end(),
      thrust::make_zip_iterator(thrust::make_tuple((*weights).begin(),
                                                   std::get<0>(*id_type_pairs).begin(),
                                                   std::get<1>(*id_type_pairs).begin())));
  } else if (weights) {
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(offsets.data(), offsets.size()),
                                indices.begin(),
                                indices.end(),
                                (*weights).begin());
  } else if (id_type_pairs) {
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(offsets.data(), offsets.size()),
                                indices.begin(),
                                indices.end(),
                                get_dataframe_buffer_begin(*id_type_pairs));
  } else {
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(offsets.data(), offsets.size()),
                                indices.begin(),
                                indices.end());
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

  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                thrust::tuple<edge_t, edge_type_t>>>
    edge_id_type_pairs{std::nullopt};
  if (id_type_pairs) {
    std::vector<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>
      buffers{};
    buffers.push_back(std::move(*id_type_pairs));
    edge_id_type_pairs =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                      thrust::tuple<edge_t, edge_type_t>>(std::move(buffers));
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
    std::move(edge_id_type_pairs),
    std::move(renumber_map_labels));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                thrust::tuple<edge_t, edge_type_t>>>,
  std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  return create_graph_from_edgelist_impl<vertex_t,
                                         edge_t,
                                         weight_t,
                                         edge_type_t,
                                         store_transposed,
                                         multi_gpu>(handle,
                                                    std::move(vertices),
                                                    std::move(edgelist_srcs),
                                                    std::move(edgelist_dsts),
                                                    std::move(edgelist_weights),
                                                    std::move(edgelist_id_type_pairs),
                                                    graph_properties,
                                                    renumber,
                                                    do_expensive_check);
}

}  // namespace cugraph

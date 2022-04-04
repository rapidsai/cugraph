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

#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/device_atomics.cuh>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <tuple>

namespace cugraph {

namespace {

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t>
struct out_of_range_t {
  vertex_t major_range_first{};
  vertex_t major_range_last{};
  vertex_t minor_range_first{};
  vertex_t minor_range_last{};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> t) const
  {
    auto major = thrust::get<0>(t);
    auto minor = thrust::get<1>(t);
    return (major < major_range_first) || (major >= major_range_last) ||
           (minor < minor_range_first) || (minor >= minor_range_last);
  }
};

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t, typename edge_t>
struct has_nzd_t {
  edge_t const* offsets{nullptr};
  vertex_t major_range_first{};

  __device__ bool operator()(vertex_t major) const
  {
    auto major_offset = major - major_range_first;
    return offsets[major_offset + 1] - offsets[major_offset] > 0;
  }
};

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t>
struct atomic_or_bitmap_t {
  uint32_t* bitmaps{nullptr};
  vertex_t minor_range_first{};

  __device__ void operator()(vertex_t minor) const
  {
    auto minor_offset = minor - minor_range_first;
    auto mask         = uint32_t{1} << (minor_offset % (sizeof(uint32_t) * 8));
    atomicOr(bitmaps + (minor_offset / (sizeof(uint32_t) * 8)), mask);
  }
};

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t>
struct popc_t {
  __device__ vertex_t operator()(uint32_t bitmap) const
  {
    return static_cast<vertex_t>(__popc(bitmap));
  }
};

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename edge_t>
struct rebase_offset_t {
  edge_t base_offset{};
  __device__ edge_t operator()(edge_t offset) const { return offset - base_offset; }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
bool check_symmetric(raft::handle_t const& handle,
                     std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const& edgelists)
{
  size_t number_of_local_edges{0};
  for (size_t i = 0; i < edgelists.size(); ++i) {
    number_of_local_edges += edgelists[i].number_of_edges;
  }

  auto is_weighted = edgelists[0].p_edge_weights.has_value();

  rmm::device_uvector<vertex_t> org_srcs(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> org_dsts(number_of_local_edges, handle.get_stream());
  auto org_weights = is_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                     number_of_local_edges, handle.get_stream())
                                 : std::nullopt;
  size_t offset{0};
  for (size_t i = 0; i < edgelists.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].p_src_vertices,
                 edgelists[i].p_src_vertices + edgelists[i].number_of_edges,
                 org_srcs.begin() + offset);
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].p_dst_vertices,
                 edgelists[i].p_dst_vertices + edgelists[i].number_of_edges,
                 org_dsts.begin() + offset);
    if (is_weighted) {
      thrust::copy(handle.get_thrust_policy(),
                   *(edgelists[i].p_edge_weights),
                   *(edgelists[i].p_edge_weights) + edgelists[i].number_of_edges,
                   (*org_weights).begin() + offset);
    }
    offset += edgelists[i].number_of_edges;
  }
  if constexpr (multi_gpu) {
    std::tie(
      store_transposed ? org_dsts : org_srcs, store_transposed ? org_srcs : org_dsts, org_weights) =
      detail::shuffle_edgelist_by_gpu_id(handle,
                                         std::move(store_transposed ? org_dsts : org_srcs),
                                         std::move(store_transposed ? org_srcs : org_dsts),
                                         std::move(org_weights));
  }

  rmm::device_uvector<vertex_t> symmetrized_srcs(org_srcs.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> symmetrized_dsts(org_dsts.size(), handle.get_stream());
  auto symmetrized_weights = org_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                             (*org_weights).size(), handle.get_stream())
                                         : std::nullopt;
  thrust::copy(
    handle.get_thrust_policy(), org_srcs.begin(), org_srcs.end(), symmetrized_srcs.begin());
  thrust::copy(
    handle.get_thrust_policy(), org_dsts.begin(), org_dsts.end(), symmetrized_dsts.begin());
  if (org_weights) {
    thrust::copy(handle.get_thrust_policy(),
                 (*org_weights).begin(),
                 (*org_weights).end(),
                 (*symmetrized_weights).begin());
  }
  std::tie(symmetrized_srcs, symmetrized_dsts, symmetrized_weights) =
    symmetrize_edgelist<vertex_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(symmetrized_srcs),
      std::move(symmetrized_dsts),
      std::move(symmetrized_weights),
      true);

  if (org_srcs.size() != symmetrized_srcs.size()) { return false; }

  if (org_weights) {
    auto org_edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(org_srcs.begin(), org_dsts.begin(), (*org_weights).begin()));
    thrust::sort(handle.get_thrust_policy(), org_edge_first, org_edge_first + org_srcs.size());
    auto symmetrized_edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      symmetrized_srcs.begin(), symmetrized_dsts.begin(), (*symmetrized_weights).begin()));
    thrust::sort(handle.get_thrust_policy(),
                 symmetrized_edge_first,
                 symmetrized_edge_first + symmetrized_srcs.size());

    return thrust::equal(handle.get_thrust_policy(),
                         org_edge_first,
                         org_edge_first + org_srcs.size(),
                         symmetrized_edge_first);
  } else {
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
}

template <typename vertex_t, typename edge_t, typename weight_t>
bool check_no_parallel_edge(raft::handle_t const& handle,
                            std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const& edgelists)
{
  size_t number_of_local_edges{0};
  for (size_t i = 0; i < edgelists.size(); ++i) {
    number_of_local_edges += edgelists[i].number_of_edges;
  }

  auto is_weighted = edgelists[0].p_edge_weights.has_value();

  rmm::device_uvector<vertex_t> edgelist_srcs(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(number_of_local_edges, handle.get_stream());
  auto edgelist_weights = is_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                          number_of_local_edges, handle.get_stream())
                                      : std::nullopt;
  size_t offset{0};
  for (size_t i = 0; i < edgelists.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].p_src_vertices,
                 edgelists[i].p_src_vertices + edgelists[i].number_of_edges,
                 edgelist_srcs.begin() + offset);
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].p_dst_vertices,
                 edgelists[i].p_dst_vertices + edgelists[i].number_of_edges,
                 edgelist_dsts.begin() + offset);
    if (is_weighted) {
      thrust::copy(handle.get_thrust_policy(),
                   *(edgelists[i].p_edge_weights),
                   *(edgelists[i].p_edge_weights) + edgelists[i].number_of_edges,
                   (*edgelist_weights).begin() + offset);
    }
    offset += edgelists[i].number_of_edges;
  }

  if (edgelist_weights) {
    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      edgelist_srcs.begin(), edgelist_dsts.begin(), (*edgelist_weights).begin()));
    thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
    return thrust::unique(handle.get_thrust_policy(),
                          edge_first,
                          edge_first + edgelist_srcs.size()) == (edge_first + edgelist_srcs.size());
  } else {
    auto edge_first =
      thrust::make_zip_iterator(thrust::make_tuple(edgelist_srcs.begin(), edgelist_dsts.begin()));
    thrust::sort(handle.get_thrust_policy(), edge_first, edge_first + edgelist_srcs.size());
    return thrust::unique(handle.get_thrust_policy(),
                          edge_first,
                          edge_first + edgelist_srcs.size()) == (edge_first + edgelist_srcs.size());
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<multi_gpu, void> check_graph_constructor_input_arguments(
  raft::handle_t const& handle,
  std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const& edgelists,
  graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
  bool do_expensive_check)
{
  // cheap error checks

  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  CUGRAPH_EXPECTS(edgelists.size() == static_cast<size_t>(col_comm_size),
                  "Invalid input argument: erroneous edgelists.size().");
  CUGRAPH_EXPECTS(
    !(meta.segment_offsets).has_value() ||
      ((*(meta.segment_offsets)).size() ==
       (detail::num_sparse_segments_per_vertex_partition + 1)) ||
      ((*(meta.segment_offsets)).size() == (detail::num_sparse_segments_per_vertex_partition + 2)),
    "Invalid input argument: (*(meta.segment_offsets)).size() returns an invalid value.");

  auto is_weighted = edgelists[0].p_edge_weights.has_value();

  CUGRAPH_EXPECTS(
    std::any_of(edgelists.begin(),
                edgelists.end(),
                [is_weighted](auto edgelist) {
                  return ((edgelist.number_of_edges > 0) && (edgelist.p_src_vertices == nullptr)) ||
                         ((edgelist.number_of_edges > 0) && (edgelist.p_dst_vertices == nullptr)) ||
                         (is_weighted && (edgelist.number_of_edges > 0) &&
                          ((edgelist.p_edge_weights.has_value() == false) ||
                           (*(edgelist.p_edge_weights) == nullptr)));
                }) == false,
    "Invalid input argument: edgelists[].p_src_vertices and edgelists[].p_dst_vertices should not "
    "be nullptr if edgelists[].number_of_edges > 0 and edgelists[].p_edge_weights should be "
    "neither std::nullopt nor nullptr if weighted and edgelists[].number_of_edges >  0.");

  // optional expensive checks

  if (do_expensive_check) {
    edge_t number_of_local_edges{0};
    for (size_t i = 0; i < edgelists.size(); ++i) {
      auto [major_range_first, major_range_last] =
        meta.partition.local_edge_partition_major_range(i);
      auto [minor_range_first, minor_range_last] =
        meta.partition.local_edge_partition_minor_range();

      number_of_local_edges += edgelists[i].number_of_edges;

      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        store_transposed ? edgelists[i].p_dst_vertices : edgelists[i].p_src_vertices,
        store_transposed ? edgelists[i].p_src_vertices : edgelists[i].p_dst_vertices));
      // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + edgelists[i].number_of_edges,
          out_of_range_t<vertex_t>{
            major_range_first, major_range_last, minor_range_first, minor_range_last}) == 0,
        "Invalid input argument: edgelists[] have out-of-range values.");
    }
    auto number_of_local_edges_sum = host_scalar_allreduce(
      comm, number_of_local_edges, raft::comms::op_t::SUM, handle.get_stream());
    CUGRAPH_EXPECTS(number_of_local_edges_sum == meta.number_of_edges,
                    "Invalid input argument: the sum of local edge counts does not match with "
                    "meta.number_of_edges.");

    CUGRAPH_EXPECTS(
      meta.partition.vertex_partition_range_last(comm_size - 1) == meta.number_of_vertices,
      "Invalid input argument: vertex partition should cover [0, meta.number_of_vertices).");

    if (meta.properties.is_symmetric) {
      CUGRAPH_EXPECTS(
        (check_symmetric<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(handle,
                                                                                  edgelists)),
        "Invalid input argument: meta.property.is_symmetric is true but the input edge list is not "
        "symmetric.");
    }
    if (!meta.properties.is_multigraph) {
      CUGRAPH_EXPECTS(
        check_no_parallel_edge(handle, edgelists),
        "Invalid input argument: meta.property.is_multigraph is false but the input edge list has "
        "parallel edges.");
    }
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<!multi_gpu, void> check_graph_constructor_input_arguments(
  raft::handle_t const& handle,
  edgelist_t<vertex_t, edge_t, weight_t> const& edgelist,
  graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
  bool do_expensive_check)
{
  // cheap error checks

  auto is_weighted = edgelist.p_edge_weights.has_value();

  CUGRAPH_EXPECTS(
    ((edgelist.number_of_edges == 0) || (edgelist.p_src_vertices != nullptr)) &&
      ((edgelist.number_of_edges == 0) || (edgelist.p_dst_vertices != nullptr)) &&
      (!is_weighted || (is_weighted && ((edgelist.number_of_edges == 0) ||
                                        (*(edgelist.p_edge_weights) != nullptr)))),
    "Invalid input argument: edgelist.p_src_vertices and edgelist.p_dst_vertices should not be "
    "nullptr if edgelist.number_of_edges > 0 and edgelist.p_edge_weights should be neither "
    "std::nullopt nor nullptr if weighted and edgelist.number_of_edges > 0.");

  CUGRAPH_EXPECTS(
    !meta.segment_offsets.has_value() ||
      ((*(meta.segment_offsets)).size() == (detail::num_sparse_segments_per_vertex_partition + 1)),
    "Invalid input argument: (*(meta.segment_offsets)).size() returns an invalid value.");

  // optional expensive checks

  if (do_expensive_check) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(store_transposed ? edgelist.p_dst_vertices : edgelist.p_src_vertices,
                         store_transposed ? edgelist.p_src_vertices : edgelist.p_dst_vertices));
    // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
    CUGRAPH_EXPECTS(
      thrust::count_if(
        handle.get_thrust_policy(),
        edge_first,
        edge_first + edgelist.number_of_edges,
        out_of_range_t<vertex_t>{0, meta.number_of_vertices, 0, meta.number_of_vertices}) == 0,
      "Invalid input argument: edgelist have out-of-range values.");

    if (meta.properties.is_symmetric) {
      CUGRAPH_EXPECTS(
        (check_symmetric<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
          handle, std::vector<edgelist_t<vertex_t, edge_t, weight_t>>{edgelist})),
        "Invalid input argument: meta.property.is_symmetric is true but the input edge list is not "
        "symmetric.");
    }
    if (!meta.properties.is_multigraph) {
      CUGRAPH_EXPECTS(
        check_no_parallel_edge(handle,
                               std::vector<edgelist_t<vertex_t, edge_t, weight_t>>{edgelist}),
        "Invalid input argument: meta.property.is_multigraph is false but the input edge list has "
        "parallel edges.");
    }
  }
}

template <typename vertex_t>
std::vector<vertex_t> aggregate_segment_offsets(raft::handle_t const& handle,
                                                std::vector<vertex_t> const& segment_offsets)
{
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  rmm::device_uvector<vertex_t> d_segment_offsets(segment_offsets.size(), handle.get_stream());
  raft::update_device(
    d_segment_offsets.data(), segment_offsets.data(), segment_offsets.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_aggregate_segment_offsets(
    col_comm_size * d_segment_offsets.size(), handle.get_stream());
  col_comm.allgather(d_segment_offsets.data(),
                     d_aggregate_segment_offsets.data(),
                     d_segment_offsets.size(),
                     handle.get_stream());

  std::vector<vertex_t> h_aggregate_segment_offsets(d_aggregate_segment_offsets.size(),
                                                    vertex_t{0});
  raft::update_host(h_aggregate_segment_offsets.data(),
                    d_aggregate_segment_offsets.data(),
                    d_aggregate_segment_offsets.size(),
                    handle.get_stream());

  handle.sync_stream();  // this is necessary as h_aggregate_offsets can be used right after return.

  return h_aggregate_segment_offsets;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<std::optional<rmm::device_uvector<vertex_t>>,
                            std::optional<std::vector<vertex_t>>,
                            std::optional<rmm::device_uvector<vertex_t>>,
                            std::optional<std::vector<vertex_t>>>>
update_local_sorted_unique_edge_majors_minors(
  raft::handle_t const& handle,
  graph_meta_t<vertex_t, edge_t, multi_gpu> const& meta,
  std::optional<std::vector<vertex_t>> const& edge_partition_segment_offsets,
  std::vector<rmm::device_uvector<edge_t>> const& edge_partition_offsets,
  std::vector<rmm::device_uvector<vertex_t>> const& edge_partition_indices,
  std::optional<std::vector<rmm::device_uvector<vertex_t>>> const& edge_partition_dcs_nzd_vertices,
  std::optional<std::vector<vertex_t>> const& edge_partition_dcs_nzd_vertex_counts)
{
  auto& comm               = handle.get_comms();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();

  auto use_dcs =
    meta.segment_offsets
      ? ((*(meta.segment_offsets)).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
      : false;

  std::optional<rmm::device_uvector<vertex_t>> local_sorted_unique_edge_majors{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_major_offsets{std::nullopt};

  std::optional<rmm::device_uvector<vertex_t>> local_sorted_unique_edge_minors{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_minor_offsets{std::nullopt};

  // if # unique edge majors/minors << V / row_comm_size|col_comm_size, store unique edge
  // majors/minors to support storing edge major/minor properties in (key, value) pairs.

  vertex_t num_local_unique_edge_majors{0};
  for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
    num_local_unique_edge_majors += thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(vertex_t{0}),
      thrust::make_counting_iterator(static_cast<vertex_t>(edge_partition_offsets[i].size() - 1)),
      has_nzd_t<vertex_t, edge_t>{edge_partition_offsets[i].data(), vertex_t{0}});
  }

  auto [minor_range_first, minor_range_last] = meta.partition.local_edge_partition_minor_range();
  rmm::device_uvector<uint32_t> minor_bitmaps(
    ((minor_range_last - minor_range_first) + sizeof(uint32_t) * 8 - 1) / (sizeof(uint32_t) * 8),
    handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), minor_bitmaps.begin(), minor_bitmaps.end(), uint32_t{0});
  for (size_t i = 0; i < edge_partition_indices.size(); ++i) {
    thrust::for_each(handle.get_thrust_policy(),
                     edge_partition_indices[i].begin(),
                     edge_partition_indices[i].end(),
                     atomic_or_bitmap_t<vertex_t>{minor_bitmaps.data(), minor_range_first});
  }

  auto count_first = thrust::make_transform_iterator(minor_bitmaps.begin(), popc_t<vertex_t>{});
  auto num_local_unique_edge_minors = thrust::reduce(
    handle.get_thrust_policy(), count_first, count_first + minor_bitmaps.size(), vertex_t{0});

  minor_bitmaps.resize(0, handle.get_stream());
  minor_bitmaps.shrink_to_fit(handle.get_stream());

  vertex_t aggregate_major_range_size{0};
  for (size_t i = 0; i < meta.partition.number_of_local_edge_partitions(); ++i) {
    aggregate_major_range_size += meta.partition.local_edge_partition_major_range_size(i);
  }
  auto minor_size = meta.partition.local_edge_partition_minor_range_size();
  auto max_major_properties_fill_ratio =
    host_scalar_allreduce(comm,
                          static_cast<double>(num_local_unique_edge_majors) /
                            static_cast<double>(aggregate_major_range_size),
                          raft::comms::op_t::MAX,
                          handle.get_stream());
  auto max_minor_properties_fill_ratio = host_scalar_allreduce(
    comm,
    static_cast<double>(num_local_unique_edge_minors) / static_cast<double>(minor_size),
    raft::comms::op_t::MAX,
    handle.get_stream());

  if (max_major_properties_fill_ratio <
      detail::edge_partition_src_dst_property_values_kv_pair_fill_ratio_threshold) {
    local_sorted_unique_edge_majors =
      rmm::device_uvector<vertex_t>(num_local_unique_edge_majors, handle.get_stream());
    size_t cur_size{0};
    for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
      auto [major_range_first, major_range_last] =
        meta.partition.local_edge_partition_major_range(i);
      auto major_hypersparse_first =
        use_dcs ? std::optional<vertex_t>{major_range_first +
                                          (*edge_partition_segment_offsets)
                                            [(*(meta.segment_offsets)).size() * i +
                                             detail::num_sparse_segments_per_vertex_partition]}
                : std::nullopt;
      cur_size += thrust::distance(
        (*local_sorted_unique_edge_majors).data() + cur_size,
        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(major_range_first),
          thrust::make_counting_iterator(use_dcs ? *major_hypersparse_first : major_range_last),
          (*local_sorted_unique_edge_majors).data() + cur_size,
          has_nzd_t<vertex_t, edge_t>{edge_partition_offsets[i].data(), major_range_first}));
      if (use_dcs) {
        thrust::copy(handle.get_thrust_policy(),
                     (*edge_partition_dcs_nzd_vertices)[i].begin(),
                     (*edge_partition_dcs_nzd_vertices)[i].begin() +
                       (*edge_partition_dcs_nzd_vertex_counts)[i],
                     (*local_sorted_unique_edge_majors).data() + cur_size);
        cur_size += (*edge_partition_dcs_nzd_vertex_counts)[i];
      }
    }
    assert(cur_size == num_local_unique_edge_majors);

    std::vector<vertex_t> h_vertex_partition_firsts(col_comm_size - 1);
    for (int i = 1; i < col_comm_size; ++i) {
      h_vertex_partition_firsts[i - 1] =
        meta.partition.vertex_partition_range_first(i * row_comm_size + row_comm_rank);
    }
    rmm::device_uvector<vertex_t> d_vertex_partition_firsts(h_vertex_partition_firsts.size(),
                                                            handle.get_stream());
    raft::update_device(d_vertex_partition_firsts.data(),
                        h_vertex_partition_firsts.data(),
                        h_vertex_partition_firsts.size(),
                        handle.get_stream());
    rmm::device_uvector<vertex_t> d_key_offsets(d_vertex_partition_firsts.size(),
                                                handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        (*local_sorted_unique_edge_majors).begin(),
                        (*local_sorted_unique_edge_majors).end(),
                        d_vertex_partition_firsts.begin(),
                        d_vertex_partition_firsts.end(),
                        d_key_offsets.begin());
    std::vector<vertex_t> h_key_offsets(col_comm_size + 1, vertex_t{0});
    h_key_offsets.back() = static_cast<vertex_t>((*local_sorted_unique_edge_majors).size());
    raft::update_host(
      h_key_offsets.data() + 1, d_key_offsets.data(), d_key_offsets.size(), handle.get_stream());

    local_sorted_unique_edge_major_offsets = std::move(h_key_offsets);
  }

  if (max_minor_properties_fill_ratio <
      detail::edge_partition_src_dst_property_values_kv_pair_fill_ratio_threshold) {
    local_sorted_unique_edge_minors = rmm::device_uvector<vertex_t>(0, handle.get_stream());
    for (size_t i = 0; i < edge_partition_indices.size(); ++i) {
      rmm::device_uvector<vertex_t> tmp_minors(edge_partition_indices[i].size(),
                                               handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edge_partition_indices[i].begin(),
                   edge_partition_indices[i].end(),
                   tmp_minors.begin());
      thrust::sort(handle.get_thrust_policy(), tmp_minors.begin(), tmp_minors.end());
      tmp_minors.resize(
        thrust::distance(
          tmp_minors.begin(),
          thrust::unique(handle.get_thrust_policy(), tmp_minors.begin(), tmp_minors.end())),
        handle.get_stream());
      auto cur_size = (*local_sorted_unique_edge_minors).size();
      if (cur_size == 0) {
        (*local_sorted_unique_edge_minors) = std::move(tmp_minors);
      } else {
        (*local_sorted_unique_edge_minors)
          .resize((*local_sorted_unique_edge_minors).size() + tmp_minors.size(),
                  handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     tmp_minors.begin(),
                     tmp_minors.end(),
                     (*local_sorted_unique_edge_minors).begin() + cur_size);
      }
    }
    thrust::sort(handle.get_thrust_policy(),
                 (*local_sorted_unique_edge_minors).begin(),
                 (*local_sorted_unique_edge_minors).end());
    (*local_sorted_unique_edge_minors)
      .resize(thrust::distance((*local_sorted_unique_edge_minors).begin(),
                               thrust::unique(handle.get_thrust_policy(),
                                              (*local_sorted_unique_edge_minors).begin(),
                                              (*local_sorted_unique_edge_minors).end())),
              handle.get_stream());
    (*local_sorted_unique_edge_minors).shrink_to_fit(handle.get_stream());

    std::vector<vertex_t> h_vertex_partition_firsts(row_comm_size - 1);
    for (int i = 1; i < row_comm_size; ++i) {
      h_vertex_partition_firsts[i - 1] =
        meta.partition.vertex_partition_range_first(col_comm_rank * row_comm_size + i);
    }
    rmm::device_uvector<vertex_t> d_vertex_partition_firsts(h_vertex_partition_firsts.size(),
                                                            handle.get_stream());
    raft::update_device(d_vertex_partition_firsts.data(),
                        h_vertex_partition_firsts.data(),
                        h_vertex_partition_firsts.size(),
                        handle.get_stream());
    rmm::device_uvector<vertex_t> d_key_offsets(d_vertex_partition_firsts.size(),
                                                handle.get_stream());
    thrust::lower_bound(handle.get_thrust_policy(),
                        (*local_sorted_unique_edge_minors).begin(),
                        (*local_sorted_unique_edge_minors).end(),
                        d_vertex_partition_firsts.begin(),
                        d_vertex_partition_firsts.end(),
                        d_key_offsets.begin());
    std::vector<vertex_t> h_key_offsets(row_comm_size + 1, vertex_t{0});
    h_key_offsets.back() = static_cast<vertex_t>((*local_sorted_unique_edge_minors).size());
    raft::update_host(
      h_key_offsets.data() + 1, d_key_offsets.data(), d_key_offsets.size(), handle.get_stream());

    local_sorted_unique_edge_minor_offsets = std::move(h_key_offsets);
  }

  return std::make_tuple(std::move(local_sorted_unique_edge_majors),
                         std::move(local_sorted_unique_edge_major_offsets),
                         std::move(local_sorted_unique_edge_minors),
                         std::move(local_sorted_unique_edge_minor_offsets));
}

// compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
std::tuple<rmm::device_uvector<edge_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
compress_edgelist(edgelist_t<vertex_t, edge_t, weight_t> const& edgelist,
                  vertex_t major_range_first,
                  std::optional<vertex_t> major_hypersparse_first,
                  vertex_t major_range_last,
                  vertex_t /* minor_range_first */,
                  vertex_t /* minor_range_last */,
                  rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<edge_t> offsets((major_range_last - major_range_first) + 1, stream_view);
  rmm::device_uvector<vertex_t> indices(edgelist.number_of_edges, stream_view);
  auto weights = edgelist.p_edge_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                             edgelist.number_of_edges, stream_view)
                                         : std::nullopt;
  thrust::fill(rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), edge_t{0});
  thrust::fill(rmm::exec_policy(stream_view), indices.begin(), indices.end(), vertex_t{0});

  auto p_offsets = offsets.data();
  thrust::for_each(rmm::exec_policy(stream_view),
                   store_transposed ? edgelist.p_dst_vertices : edgelist.p_src_vertices,
                   store_transposed ? edgelist.p_dst_vertices + edgelist.number_of_edges
                                    : edgelist.p_src_vertices + edgelist.number_of_edges,
                   [p_offsets, major_range_first] __device__(auto v) {
                     atomicAdd(p_offsets + (v - major_range_first), edge_t{1});
                   });
  thrust::exclusive_scan(
    rmm::exec_policy(stream_view), offsets.begin(), offsets.end(), offsets.begin());

  auto p_indices = indices.data();
  if (edgelist.p_edge_weights) {
    auto p_weights = (*weights).data();

    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      edgelist.p_src_vertices, edgelist.p_dst_vertices, *(edgelist.p_edge_weights)));
    thrust::for_each(rmm::exec_policy(stream_view),
                     edge_first,
                     edge_first + edgelist.number_of_edges,
                     [p_offsets, p_indices, p_weights, major_range_first] __device__(auto e) {
                       auto s      = thrust::get<0>(e);
                       auto d      = thrust::get<1>(e);
                       auto w      = thrust::get<2>(e);
                       auto major  = store_transposed ? d : s;
                       auto minor  = store_transposed ? s : d;
                       auto start  = p_offsets[major - major_range_first];
                       auto degree = p_offsets[(major - major_range_first) + 1] - start;
                       auto idx    = atomicAdd(p_indices + (start + degree - 1),
                                            vertex_t{1});  // use the last element as a counter
                       // FIXME: we can actually store minor - minor_range_first instead of minor to
                       // save memory if minor can be larger than 32 bit but minor -
                       // minor_range_first fits within 32 bit
                       p_indices[start + idx] =
                         minor;  // overwrite the counter only if idx == degree - 1 (no race)
                       p_weights[start + idx] = w;
                     });
  } else {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist.p_src_vertices, edgelist.p_dst_vertices));
    thrust::for_each(rmm::exec_policy(stream_view),
                     edge_first,
                     edge_first + edgelist.number_of_edges,
                     [p_offsets, p_indices, major_range_first] __device__(auto e) {
                       auto s      = thrust::get<0>(e);
                       auto d      = thrust::get<1>(e);
                       auto major  = store_transposed ? d : s;
                       auto minor  = store_transposed ? s : d;
                       auto start  = p_offsets[major - major_range_first];
                       auto degree = p_offsets[(major - major_range_first) + 1] - start;
                       auto idx    = atomicAdd(p_indices + (start + degree - 1),
                                            vertex_t{1});  // use the last element as a counter
                       // FIXME: we can actually store minor - minor_range_first instead of minor to
                       // save memory if minor can be larger than 32 bit but minor -
                       // minor_range_first fits within 32 bit
                       p_indices[start + idx] =
                         minor;  // overwrite the counter only if idx == degree - 1 (no race)
                     });
  }

  auto dcs_nzd_vertices = major_hypersparse_first
                            ? std::make_optional<rmm::device_uvector<vertex_t>>(
                                major_range_last - *major_hypersparse_first, stream_view)
                            : std::nullopt;
  if (dcs_nzd_vertices) {
    auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

    thrust::transform(
      rmm::exec_policy(stream_view),
      thrust::make_counting_iterator(*major_hypersparse_first),
      thrust::make_counting_iterator(major_range_last),
      (*dcs_nzd_vertices).begin(),
      [major_range_first, offsets = offsets.data()] __device__(auto major) {
        auto major_offset = major - major_range_first;
        return offsets[major_offset + 1] - offsets[major_offset] > 0 ? major : invalid_vertex;
      });

    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple((*dcs_nzd_vertices).begin(),
                         offsets.begin() + (*major_hypersparse_first - major_range_first)));
    (*dcs_nzd_vertices)
      .resize(thrust::distance(pair_first,
                               thrust::remove_if(rmm::exec_policy(stream_view),
                                                 pair_first,
                                                 pair_first + (*dcs_nzd_vertices).size(),
                                                 [] __device__(auto pair) {
                                                   return thrust::get<0>(pair) == invalid_vertex;
                                                 })),
              stream_view);
    (*dcs_nzd_vertices).shrink_to_fit(stream_view);
    if (static_cast<vertex_t>((*dcs_nzd_vertices).size()) <
        major_range_last - *major_hypersparse_first) {
      thrust::copy(rmm::exec_policy(stream_view),
                   offsets.begin() + (major_range_last - major_range_first),
                   offsets.end(),
                   offsets.begin() + (*major_hypersparse_first - major_range_first) +
                     (*dcs_nzd_vertices).size());
      offsets.resize(
        (*major_hypersparse_first - major_range_first) + (*dcs_nzd_vertices).size() + 1,
        stream_view);
      offsets.shrink_to_fit(stream_view);
    }
  }

  return std::make_tuple(
    std::move(offsets), std::move(indices), std::move(weights), std::move(dcs_nzd_vertices));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sort_adjacency_list(raft::handle_t const& handle,
                         edge_t const* offsets,
                         vertex_t* indices /* [INOUT} */,
                         std::optional<weight_t*> weights /* [INOUT] */,
                         vertex_t num_vertices,
                         edge_t num_edges)
{
  // 1. Check if there is anything to sort

  if (num_edges == 0) { return; }

  // 2. We segmented sort edges in chunks, and we need to adjust chunk offsets as we need to sort
  // each vertex's neighbors at once.

  // to limit memory footprint ((1 << 20) is a tuning parameter)
  auto approx_edges_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 20);
  auto [h_vertex_offsets, h_edge_offsets] = detail::compute_offset_aligned_edge_chunks(
    handle, offsets, num_vertices, num_edges, approx_edges_to_sort_per_iteration);
  auto num_chunks = h_vertex_offsets.size() - 1;

  // 3. Segmented sort each vertex's neighbors

  size_t max_chunk_size{0};
  for (size_t i = 0; i < num_chunks; ++i) {
    max_chunk_size =
      std::max(max_chunk_size, static_cast<size_t>(h_edge_offsets[i + 1] - h_edge_offsets[i]));
  }
  rmm::device_uvector<vertex_t> segment_sorted_indices(max_chunk_size, handle.get_stream());
  auto segment_sorted_weights =
    weights ? std::make_optional<rmm::device_uvector<weight_t>>(max_chunk_size, handle.get_stream())
            : std::nullopt;
  rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
  for (size_t i = 0; i < num_chunks; ++i) {
    size_t tmp_storage_bytes{0};
    auto offset_first = thrust::make_transform_iterator(offsets + h_vertex_offsets[i],
                                                        rebase_offset_t<edge_t>{h_edge_offsets[i]});
    if (weights) {
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          indices + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          (*weights) + h_edge_offsets[i],
                                          (*segment_sorted_weights).data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
    } else {
      cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                         tmp_storage_bytes,
                                         indices + h_edge_offsets[i],
                                         segment_sorted_indices.data(),
                                         h_edge_offsets[i + 1] - h_edge_offsets[i],
                                         h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                         offset_first,
                                         offset_first + 1,
                                         handle.get_stream());
    }
    if (tmp_storage_bytes > d_tmp_storage.size()) {
      d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
    }
    if (weights) {
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          indices + h_edge_offsets[i],
                                          segment_sorted_indices.data(),
                                          (*weights) + h_edge_offsets[i],
                                          (*segment_sorted_weights).data(),
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
    } else {
      cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                         tmp_storage_bytes,
                                         indices + h_edge_offsets[i],
                                         segment_sorted_indices.data(),
                                         h_edge_offsets[i + 1] - h_edge_offsets[i],
                                         h_vertex_offsets[i + 1] - h_vertex_offsets[i],
                                         offset_first,
                                         offset_first + 1,
                                         handle.get_stream());
    }
    thrust::copy(handle.get_thrust_policy(),
                 segment_sorted_indices.begin(),
                 segment_sorted_indices.begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                 indices + h_edge_offsets[i]);
    if (weights) {
      thrust::copy(handle.get_thrust_policy(),
                   (*segment_sorted_weights).begin(),
                   (*segment_sorted_weights).begin() + (h_edge_offsets[i + 1] - h_edge_offsets[i]),
                   (*weights) + h_edge_offsets[i]);
    }
  }
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_t(raft::handle_t const& handle,
          std::vector<edgelist_t<vertex_t, edge_t, weight_t>> const& edgelists,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, meta.number_of_edges, meta.properties),
    partition_(meta.partition)
{
  auto is_weighted = edgelists[0].p_edge_weights.has_value();
  auto use_dcs =
    meta.segment_offsets
      ? ((*(meta.segment_offsets)).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
      : false;

  check_graph_constructor_input_arguments<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle, edgelists, meta, do_expensive_check);

  if (meta.segment_offsets) {
    edge_partition_segment_offsets_ = aggregate_segment_offsets(handle, (*meta.segment_offsets));
  }

  // compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid

  edge_partition_offsets_.reserve(edgelists.size());
  edge_partition_indices_.reserve(edgelists.size());
  if (is_weighted) {
    edge_partition_weights_ = std::vector<rmm::device_uvector<weight_t>>{};
    (*edge_partition_weights_).reserve(edgelists.size());
  }
  if (use_dcs) {
    edge_partition_dcs_nzd_vertices_      = std::vector<rmm::device_uvector<vertex_t>>{};
    edge_partition_dcs_nzd_vertex_counts_ = std::vector<vertex_t>{};
    (*edge_partition_dcs_nzd_vertices_).reserve(edgelists.size());
    (*edge_partition_dcs_nzd_vertex_counts_).reserve(edgelists.size());
  }
  for (size_t i = 0; i < edgelists.size(); ++i) {
    auto [major_range_first, major_range_last] = partition_.local_edge_partition_major_range(i);
    auto [minor_range_first, minor_range_last] = partition_.local_edge_partition_minor_range();
    auto major_hypersparse_first =
      use_dcs ? std::optional<vertex_t>{major_range_first +
                                        (*edge_partition_segment_offsets_)
                                          [(*(meta.segment_offsets)).size() * i +
                                           detail::num_sparse_segments_per_vertex_partition]}
              : std::nullopt;
    auto [offsets, indices, weights, dcs_nzd_vertices] =
      compress_edgelist<store_transposed>(edgelists[i],
                                          major_range_first,
                                          major_hypersparse_first,
                                          major_range_last,
                                          minor_range_first,
                                          minor_range_last,
                                          handle.get_stream());

    edge_partition_offsets_.push_back(std::move(offsets));
    edge_partition_indices_.push_back(std::move(indices));
    if (is_weighted) { (*edge_partition_weights_).push_back(std::move(*weights)); }
    if (use_dcs) {
      auto dcs_nzd_vertex_count = static_cast<vertex_t>((*dcs_nzd_vertices).size());
      (*edge_partition_dcs_nzd_vertices_).push_back(std::move(*dcs_nzd_vertices));
      (*edge_partition_dcs_nzd_vertex_counts_).push_back(dcs_nzd_vertex_count);
    }
  }

  // segmented sort neighbors

  for (size_t i = 0; i < edge_partition_offsets_.size(); ++i) {
    sort_adjacency_list(handle,
                        edge_partition_offsets_[i].data(),
                        edge_partition_indices_[i].data(),
                        edge_partition_weights_
                          ? std::optional<weight_t*>{(*edge_partition_weights_)[i].data()}
                          : std::nullopt,
                        static_cast<vertex_t>(edge_partition_offsets_[i].size() - 1),
                        static_cast<edge_t>(edge_partition_indices_[i].size()));
  }

  // update local sorted unique edge sources/destinations (only if key, value pair will be used)

  std::tie(store_transposed ? local_sorted_unique_edge_dsts_ : local_sorted_unique_edge_srcs_,
           store_transposed ? local_sorted_unique_edge_dst_offsets_
                            : local_sorted_unique_edge_src_offsets_,
           store_transposed ? local_sorted_unique_edge_srcs_ : local_sorted_unique_edge_dsts_,
           store_transposed ? local_sorted_unique_edge_src_offsets_
                            : local_sorted_unique_edge_dst_offsets_) =
    update_local_sorted_unique_edge_majors_minors<vertex_t, edge_t, store_transposed, multi_gpu>(
      handle,
      meta,
      edge_partition_segment_offsets_,
      edge_partition_offsets_,
      edge_partition_indices_,
      edge_partition_dcs_nzd_vertices_,
      edge_partition_dcs_nzd_vertex_counts_);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_t(raft::handle_t const& handle,
          std::vector<rmm::device_uvector<vertex_t>>&& edgelist_src_partitions,
          std::vector<rmm::device_uvector<vertex_t>>&& edgelist_dst_partitions,
          std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edgelist_weight_partitions,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, meta.number_of_edges, meta.properties),
    partition_(meta.partition)
{
  CUGRAPH_EXPECTS(
    edgelist_src_partitions.size() == edgelist_dst_partitions.size(),
    "Invalid input argument: edgelist_src_partitions.size() != edgelist_dst_partitions.size().");
  CUGRAPH_EXPECTS(!edgelist_weight_partitions.has_value() ||
                    (edgelist_src_partitions.size() == (*edgelist_weight_partitions).size()),
                  "Invalid input argument: edgelist_weight_partitions.has_value() && "
                  "edgelist_src_partitions.size() != (*edgelist_weight_partitions).size().");
  for (size_t i = 0; i < edgelist_src_partitions.size(); ++i) {
    CUGRAPH_EXPECTS(edgelist_src_partitions[i].size() == edgelist_dst_partitions[i].size(),
                    "Invalid input argument: edgelist_src_partitions[].size() != "
                    "edgelist_dst_partitions[].size().");
    CUGRAPH_EXPECTS(
      !edgelist_weight_partitions.has_value() ||
        (edgelist_src_partitions[i].size() == (*edgelist_weight_partitions)[i].size()),
      "Invalid input argument: edgelist_weight_partitions.has_value() && "
      "edgelist_src_partitions[].size() != (*edgelist_weight_partitions)[].size().");
  }

  auto is_weighted = edgelist_weight_partitions.has_value();
  auto use_dcs =
    meta.segment_offsets
      ? ((*(meta.segment_offsets)).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
      : false;

  std::vector<edgelist_t<vertex_t, edge_t, weight_t>> edgelists(edgelist_src_partitions.size());
  for (size_t i = 0; i < edgelists.size(); ++i) {
    edgelists[i] = edgelist_t<vertex_t, edge_t, weight_t>{
      edgelist_src_partitions[i].data(),
      edgelist_dst_partitions[i].data(),
      edgelist_weight_partitions
        ? std::optional<weight_t const*>{(*edgelist_weight_partitions)[i].data()}
        : std::nullopt,
      static_cast<edge_t>(edgelist_src_partitions[i].size())};
  }

  check_graph_constructor_input_arguments<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle, edgelists, meta, do_expensive_check);

  if (meta.segment_offsets) {
    edge_partition_segment_offsets_ = aggregate_segment_offsets(handle, (*meta.segment_offsets));
  }

  // compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid

  edge_partition_offsets_.reserve(edgelists.size());
  edge_partition_indices_.reserve(edgelists.size());
  if (is_weighted) {
    edge_partition_weights_ = std::vector<rmm::device_uvector<weight_t>>{};
    (*edge_partition_weights_).reserve(edgelists.size());
  }
  if (use_dcs) {
    edge_partition_dcs_nzd_vertices_      = std::vector<rmm::device_uvector<vertex_t>>{};
    edge_partition_dcs_nzd_vertex_counts_ = std::vector<vertex_t>{};
    (*edge_partition_dcs_nzd_vertices_).reserve(edgelists.size());
    (*edge_partition_dcs_nzd_vertex_counts_).reserve(edgelists.size());
  }
  for (size_t i = 0; i < edgelists.size(); ++i) {
    auto [major_range_first, major_range_last] = partition_.local_edge_partition_major_range(i);
    auto [minor_range_first, minor_range_last] = partition_.local_edge_partition_minor_range();
    auto major_hypersparse_first =
      use_dcs ? std::optional<vertex_t>{major_range_first +
                                        (*edge_partition_segment_offsets_)
                                          [(*(meta.segment_offsets)).size() * i +
                                           detail::num_sparse_segments_per_vertex_partition]}
              : std::nullopt;
    auto [offsets, indices, weights, dcs_nzd_vertices] =
      compress_edgelist<store_transposed>(edgelists[i],
                                          major_range_first,
                                          major_hypersparse_first,
                                          major_range_last,
                                          minor_range_first,
                                          minor_range_last,
                                          handle.get_stream());
    edgelist_src_partitions[i].resize(0, handle.get_stream());
    edgelist_src_partitions[i].shrink_to_fit(handle.get_stream());
    edgelist_dst_partitions[i].resize(0, handle.get_stream());
    edgelist_dst_partitions[i].shrink_to_fit(handle.get_stream());
    if (edgelist_weight_partitions) {
      (*edgelist_weight_partitions)[i].resize(0, handle.get_stream());
      (*edgelist_weight_partitions)[i].shrink_to_fit(handle.get_stream());
    }

    edge_partition_offsets_.push_back(std::move(offsets));
    edge_partition_indices_.push_back(std::move(indices));
    if (is_weighted) { (*edge_partition_weights_).push_back(std::move(*weights)); }
    if (use_dcs) {
      auto dcs_nzd_vertex_count = static_cast<vertex_t>((*dcs_nzd_vertices).size());
      (*edge_partition_dcs_nzd_vertices_).push_back(std::move(*dcs_nzd_vertices));
      (*edge_partition_dcs_nzd_vertex_counts_).push_back(dcs_nzd_vertex_count);
    }
  }

  // segmented sort neighbors

  for (size_t i = 0; i < edge_partition_offsets_.size(); ++i) {
    sort_adjacency_list(handle,
                        edge_partition_offsets_[i].data(),
                        edge_partition_indices_[i].data(),
                        edge_partition_weights_
                          ? std::optional<weight_t*>{(*edge_partition_weights_)[i].data()}
                          : std::nullopt,
                        static_cast<vertex_t>(edge_partition_offsets_[i].size() - 1),
                        static_cast<edge_t>(edge_partition_indices_[i].size()));
  }

  // update local sorted unique edge sources/destinations (only if key, value pair will be used)

  std::tie(store_transposed ? local_sorted_unique_edge_dsts_ : local_sorted_unique_edge_srcs_,
           store_transposed ? local_sorted_unique_edge_dst_offsets_
                            : local_sorted_unique_edge_src_offsets_,
           store_transposed ? local_sorted_unique_edge_srcs_ : local_sorted_unique_edge_dsts_,
           store_transposed ? local_sorted_unique_edge_src_offsets_
                            : local_sorted_unique_edge_dst_offsets_) =
    update_local_sorted_unique_edge_majors_minors<vertex_t, edge_t, store_transposed, multi_gpu>(
      handle,
      meta,
      edge_partition_segment_offsets_,
      edge_partition_offsets_,
      edge_partition_indices_,
      edge_partition_dcs_nzd_vertices_,
      edge_partition_dcs_nzd_vertex_counts_);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  graph_t(raft::handle_t const& handle,
          edgelist_t<vertex_t, edge_t, weight_t> const& edgelist,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, edgelist.number_of_edges, meta.properties),
    offsets_(rmm::device_uvector<edge_t>(0, handle.get_stream())),
    indices_(rmm::device_uvector<vertex_t>(0, handle.get_stream())),
    segment_offsets_(meta.segment_offsets)
{
  check_graph_constructor_input_arguments<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle, edgelist, meta, do_expensive_check);

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  std::tie(offsets_, indices_, weights_, std::ignore) =
    compress_edgelist<store_transposed>(edgelist,
                                        vertex_t{0},
                                        std::optional<vertex_t>{std::nullopt},
                                        this->number_of_vertices(),
                                        vertex_t{0},
                                        this->number_of_vertices(),
                                        handle.get_stream());

  // segmented sort neighbors

  sort_adjacency_list(handle,
                      offsets_.data(),
                      indices_.data(),
                      weights_ ? std::optional<weight_t*>{(*weights_).data()} : std::nullopt,
                      static_cast<vertex_t>(offsets_.size() - 1),
                      static_cast<edge_t>(indices_.size()));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  graph_t(raft::handle_t const& handle,
          rmm::device_uvector<vertex_t>&& edgelist_srcs,
          rmm::device_uvector<vertex_t>&& edgelist_dsts,
          std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, static_cast<edge_t>(edgelist_srcs.size()), meta.properties),
    offsets_(rmm::device_uvector<edge_t>(0, handle.get_stream())),
    indices_(rmm::device_uvector<vertex_t>(0, handle.get_stream())),
    segment_offsets_(meta.segment_offsets)
{
  CUGRAPH_EXPECTS(edgelist_srcs.size() == edgelist_dsts.size(),
                  "Invalid input argument: edgelist_srcs.size() != edgelist_dsts.size().");
  CUGRAPH_EXPECTS(
    !edgelist_weights.has_value() || (edgelist_srcs.size() == (*edgelist_weights).size()),
    "Invalid input argument: edgelist_weights.has_value() && edgelist_srcs.size() != "
    "(*edgelist_weights).size().");

  edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    edgelist_srcs.data(),
    edgelist_dsts.data(),
    edgelist_weights ? std::optional<weight_t const*>{(*edgelist_weights).data()} : std::nullopt,
    static_cast<edge_t>(edgelist_srcs.size())};

  check_graph_constructor_input_arguments<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle, edgelist, meta, do_expensive_check);

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  std::tie(offsets_, indices_, weights_, std::ignore) =
    compress_edgelist<store_transposed>(edgelist,
                                        vertex_t{0},
                                        std::optional<vertex_t>{std::nullopt},
                                        this->number_of_vertices(),
                                        vertex_t{0},
                                        this->number_of_vertices(),
                                        handle.get_stream());
  edgelist_srcs.resize(0, handle.get_stream());
  edgelist_srcs.shrink_to_fit(handle.get_stream());
  edgelist_dsts.resize(0, handle.get_stream());
  edgelist_dsts.shrink_to_fit(handle.get_stream());
  if (edgelist_weights) {
    (*edgelist_weights).resize(0, handle.get_stream());
    (*edgelist_weights).shrink_to_fit(handle.get_stream());
  }

  // segmented sort neighbors

  sort_adjacency_list(handle,
                      offsets_.data(),
                      indices_.data(),
                      weights_ ? std::optional<weight_t*>{(*weights_).data()} : std::nullopt,
                      static_cast<vertex_t>(offsets_.size() - 1),
                      static_cast<edge_t>(indices_.size()));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<vertex_t>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  symmetrize(raft::handle_t const& handle,
             rmm::device_uvector<vertex_t>&& renumber_map,
             bool reciprocal)
{
  if (this->is_symmetric()) { return std::move(renumber_map); }

  auto is_multigraph = this->is_multigraph();

  auto wrapped_renumber_map = std::optional<rmm::device_uvector<vertex_t>>(std::move(renumber_map));

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    this->decompress_to_edgelist(handle, wrapped_renumber_map, true);

  std::tie(edgelist_srcs, edgelist_dsts, edgelist_weights) =
    symmetrize_edgelist<vertex_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      reciprocal);

  auto [symmetrized_graph, new_renumber_map] =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(*wrapped_renumber_map),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      graph_properties_t{is_multigraph, true},
      true);
  *this = std::move(symmetrized_graph);

  return std::move(*new_renumber_map);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::optional<rmm::device_uvector<vertex_t>>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  symmetrize(raft::handle_t const& handle,
             std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
             bool reciprocal)
{
  if (this->is_symmetric()) { return std::move(renumber_map); }

  auto number_of_vertices = this->number_of_vertices();
  auto is_multigraph      = this->is_multigraph();
  bool renumber           = renumber_map.has_value();

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    this->decompress_to_edgelist(handle, renumber_map, true);

  std::tie(edgelist_srcs, edgelist_dsts, edgelist_weights) =
    symmetrize_edgelist<vertex_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      reciprocal);

  auto vertex_span = renumber ? std::move(renumber_map)
                              : std::make_optional<rmm::device_uvector<vertex_t>>(
                                  number_of_vertices, handle.get_stream());
  if (!renumber) {
    thrust::sequence(
      handle.get_thrust_policy(), (*vertex_span).begin(), (*vertex_span).end(), vertex_t{0});
  }

  auto [symmetrized_graph, new_renumber_map] =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(vertex_span),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      graph_properties_t{is_multigraph, true},
      renumber);
  *this = std::move(symmetrized_graph);

  return std::move(new_renumber_map);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<vertex_t>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  transpose(raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& renumber_map)
{
  if (this->is_symmetric()) { return std::move(renumber_map); }

  auto is_multigraph = this->is_multigraph();

  auto wrapped_renumber_map = std::optional<rmm::device_uvector<vertex_t>>(std::move(renumber_map));

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    this->decompress_to_edgelist(handle, wrapped_renumber_map, true);

  std::tie(store_transposed ? edgelist_srcs : edgelist_dsts,
           store_transposed ? edgelist_dsts : edgelist_srcs,
           edgelist_weights) =
    detail::shuffle_edgelist_by_gpu_id(handle,
                                       std::move(store_transposed ? edgelist_srcs : edgelist_dsts),
                                       std::move(store_transposed ? edgelist_dsts : edgelist_srcs),
                                       std::move(edgelist_weights));

  auto [transposed_graph, new_renumber_map] =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(*wrapped_renumber_map),
      std::move(edgelist_dsts),
      std::move(edgelist_srcs),
      std::move(edgelist_weights),
      graph_properties_t{is_multigraph, false},
      true);
  *this = std::move(transposed_graph);

  return std::move(*new_renumber_map);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::optional<rmm::device_uvector<vertex_t>>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  transpose(raft::handle_t const& handle,
            std::optional<rmm::device_uvector<vertex_t>>&& renumber_map)
{
  if (this->is_symmetric()) { return std::move(renumber_map); }

  auto number_of_vertices = this->number_of_vertices();
  auto is_multigraph      = this->is_multigraph();
  bool renumber           = renumber_map.has_value();

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    this->decompress_to_edgelist(handle, renumber_map, true);
  auto vertex_span = renumber ? std::move(renumber_map)
                              : std::make_optional<rmm::device_uvector<vertex_t>>(
                                  number_of_vertices, handle.get_stream());
  if (!renumber) {
    thrust::sequence(
      handle.get_thrust_policy(), (*vertex_span).begin(), (*vertex_span).end(), vertex_t{0});
  }

  auto [transposed_graph, new_renumber_map] =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(vertex_span),
      std::move(edgelist_dsts),
      std::move(edgelist_srcs),
      std::move(edgelist_weights),
      graph_properties_t{is_multigraph, false},
      renumber);
  *this = std::move(transposed_graph);

  return std::move(new_renumber_map);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<graph_t<vertex_t, edge_t, weight_t, !store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  transpose_storage(raft::handle_t const& handle,
                    rmm::device_uvector<vertex_t>&& renumber_map,
                    bool destroy)
{
  auto is_multigraph = this->is_multigraph();

  auto wrapped_renumber_map = std::optional<rmm::device_uvector<vertex_t>>(std::move(renumber_map));

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    this->decompress_to_edgelist(handle, wrapped_renumber_map, destroy);

  std::tie(!store_transposed ? edgelist_dsts : edgelist_srcs,
           !store_transposed ? edgelist_srcs : edgelist_dsts,
           edgelist_weights) =
    detail::shuffle_edgelist_by_gpu_id(handle,
                                       std::move(!store_transposed ? edgelist_dsts : edgelist_srcs),
                                       std::move(!store_transposed ? edgelist_srcs : edgelist_dsts),
                                       std::move(edgelist_weights));

  auto [storage_transposed_graph, new_renumber_map] =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, !store_transposed, multi_gpu>(
      handle,
      std::move(*wrapped_renumber_map),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      graph_properties_t{is_multigraph, false},
      true);

  return std::make_tuple(std::move(storage_transposed_graph), std::move(*new_renumber_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<graph_t<vertex_t, edge_t, weight_t, !store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  transpose_storage(raft::handle_t const& handle,
                    std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
                    bool destroy)
{
  auto number_of_vertices = this->number_of_vertices();
  auto is_multigraph      = this->is_multigraph();
  bool renumber           = renumber_map.has_value();

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    this->decompress_to_edgelist(handle, renumber_map, destroy);
  auto vertex_span = renumber ? std::move(renumber_map)
                              : std::make_optional<rmm::device_uvector<vertex_t>>(
                                  number_of_vertices, handle.get_stream());
  if (!renumber) {
    thrust::sequence(
      handle.get_thrust_policy(), (*vertex_span).begin(), (*vertex_span).end(), vertex_t{0});
  }

  return create_graph_from_edgelist<vertex_t, edge_t, weight_t, !store_transposed, multi_gpu>(
    handle,
    std::move(vertex_span),
    std::move(edgelist_srcs),
    std::move(edgelist_dsts),
    std::move(edgelist_weights),
    graph_properties_t{is_multigraph, false},
    renumber);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  decompress_to_edgelist(raft::handle_t const& handle,
                         std::optional<rmm::device_uvector<vertex_t>> const& renumber_map,
                         bool destroy)
{
  auto result = this->view().decompress_to_edgelist(handle, renumber_map);

  if (destroy) { *this = graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(handle); }

  return result;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  decompress_to_edgelist(raft::handle_t const& handle,
                         std::optional<rmm::device_uvector<vertex_t>> const& renumber_map,
                         bool destroy)
{
  auto result = this->view().decompress_to_edgelist(handle, renumber_map);

  if (destroy) { *this = graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(handle); }

  return result;
}

}  // namespace cugraph

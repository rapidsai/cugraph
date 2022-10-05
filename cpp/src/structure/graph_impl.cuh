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
#include <structure/detail/structure_utils.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/handle.hpp>
#include <raft/util/device_atomics.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

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
    number_of_local_edges += edgelists[i].srcs.size();
  }

  auto is_weighted = edgelists[0].weights.has_value();

  rmm::device_uvector<vertex_t> org_srcs(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> org_dsts(number_of_local_edges, handle.get_stream());
  auto org_weights = is_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                     number_of_local_edges, handle.get_stream())
                                 : std::nullopt;
  size_t offset{0};
  for (size_t i = 0; i < edgelists.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].srcs.begin(),
                 edgelists[i].srcs.end(),
                 org_srcs.begin() + offset);
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].dsts.begin(),
                 edgelists[i].dsts.end(),
                 org_dsts.begin() + offset);
    if (is_weighted) {
      thrust::copy(handle.get_thrust_policy(),
                   (*(edgelists[i].weights)).begin(),
                   (*(edgelists[i].weights)).end(),
                   (*org_weights).begin() + offset);
    }
    offset += edgelists[i].srcs.size();
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
    number_of_local_edges += edgelists[i].srcs.size();
  }

  auto is_weighted = edgelists[0].weights.has_value();

  rmm::device_uvector<vertex_t> edgelist_srcs(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(number_of_local_edges, handle.get_stream());
  auto edgelist_weights = is_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                          number_of_local_edges, handle.get_stream())
                                      : std::nullopt;
  size_t offset{0};
  for (size_t i = 0; i < edgelists.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].srcs.begin(),
                 edgelists[i].srcs.end(),
                 edgelist_srcs.begin() + offset);
    thrust::copy(handle.get_thrust_policy(),
                 edgelists[i].dsts.begin(),
                 edgelists[i].dsts.end(),
                 edgelist_dsts.begin() + offset);
    if (is_weighted) {
      thrust::copy(handle.get_thrust_policy(),
                   (*(edgelists[i].weights)).begin(),
                   (*(edgelists[i].weights)).end(),
                   (*edgelist_weights).begin() + offset);
    }
    offset += edgelists[i].srcs.size();
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
    (meta.edge_partition_segment_offsets.size() ==
     (detail::num_sparse_segments_per_vertex_partition + 2) * col_comm_size) ||
      (meta.edge_partition_segment_offsets.size() ==
       (detail::num_sparse_segments_per_vertex_partition + 3) * col_comm_size),
    "Invalid input argument: meta.edge_partition_segment_offsets.size() returns an invalid value.");

  auto is_weighted = edgelists[0].weights.has_value();

  CUGRAPH_EXPECTS(
    std::any_of(edgelists.begin(),
                edgelists.end(),
                [is_weighted](auto edgelist) {
                  return (edgelist.srcs.size() != edgelist.dsts.size()) ||
                         (is_weighted && (edgelist.srcs.size() != (*(edgelist.weights)).size()));
                }) == false,
    "Invalid input argument: edgelists[].srcs.size() and edgelists[].dsts.size() should coincide, "
    "if edgelists[].weights.has_value(), (*(edgelists[].weights)).size() should also coincide.");

  // optional expensive checks

  if (do_expensive_check) {
    edge_t number_of_local_edges{0};
    for (size_t i = 0; i < edgelists.size(); ++i) {
      auto [major_range_first, major_range_last] =
        meta.partition.local_edge_partition_major_range(i);
      auto [minor_range_first, minor_range_last] =
        meta.partition.local_edge_partition_minor_range();

      number_of_local_edges += static_cast<edge_t>(edgelists[i].srcs.size());

      auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
        store_transposed ? edgelists[i].dsts.begin() : edgelists[i].srcs.begin(),
        store_transposed ? edgelists[i].srcs.begin() : edgelists[i].dsts.begin()));
      // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
      CUGRAPH_EXPECTS(
        thrust::count_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + edgelists[i].srcs.size(),
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

  auto is_weighted = edgelist.weights.has_value();

  CUGRAPH_EXPECTS(
    (edgelist.srcs.size() == edgelist.dsts.size()) &&
      (!is_weighted || (edgelist.srcs.size() == (*(edgelist.weights)).size())),
    "Invalid input argument: edgelists.srcs.size() and edgelists.dsts.size() should coincide, if "
    "edgelists.weights.has_value(), (*(edgelists.weights)).size() should also coincide.");

  CUGRAPH_EXPECTS(
    !meta.segment_offsets.has_value() ||
      ((*(meta.segment_offsets)).size() == (detail::num_sparse_segments_per_vertex_partition + 2)),
    "Invalid input argument: (*(meta.segment_offsets)).size() returns an invalid value.");

  // optional expensive checks

  if (do_expensive_check) {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(store_transposed ? edgelist.dsts.begin() : edgelist.srcs.begin(),
                         store_transposed ? edgelist.srcs.begin() : edgelist.dsts.begin()));
    // better use thrust::any_of once https://github.com/thrust/thrust/issues/1016 is resolved
    CUGRAPH_EXPECTS(
      thrust::count_if(
        handle.get_thrust_policy(),
        edge_first,
        edge_first + edgelist.srcs.size(),
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

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<std::optional<std::vector<rmm::device_uvector<vertex_t>>>,
                            std::optional<std::vector<rmm::device_uvector<vertex_t>>>,
                            std::optional<vertex_t>,
                            std::optional<rmm::device_uvector<vertex_t>>,
                            std::optional<rmm::device_uvector<vertex_t>>,
                            std::optional<vertex_t>,
                            std::optional<std::vector<vertex_t>>>>
update_local_sorted_unique_edge_majors_minors(
  raft::handle_t const& handle,
  graph_meta_t<vertex_t, edge_t, multi_gpu> const& meta,
  std::vector<rmm::device_uvector<edge_t>> const& edge_partition_offsets,
  std::vector<rmm::device_uvector<vertex_t>> const& edge_partition_indices,
  std::optional<std::vector<rmm::device_uvector<vertex_t>>> const& edge_partition_dcs_nzd_vertices,
  std::optional<std::vector<vertex_t>> const& edge_partition_dcs_nzd_vertex_counts)
{
  auto& comm               = handle.get_comms();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();

  auto num_segments_per_vertex_partition =
    static_cast<size_t>(meta.edge_partition_segment_offsets.size() / col_comm_size);
  auto use_dcs =
    num_segments_per_vertex_partition > (detail::num_sparse_segments_per_vertex_partition + 2);

  std::optional<std::vector<rmm::device_uvector<vertex_t>>> local_sorted_unique_edge_majors{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<vertex_t>>>
    local_sorted_unique_edge_major_chunk_start_offsets{std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_major_chunk_size{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> local_sorted_unique_edge_minors{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> local_sorted_unique_edge_minor_chunk_start_offsets{
    std::nullopt};
  std::optional<vertex_t> local_sorted_unique_edge_minor_chunk_size{std::nullopt};
  std::optional<std::vector<vertex_t>> local_sorted_unique_edge_minor_vertex_partition_offsets{
    std::nullopt};

  // if # unique edge majors/minors << V / row_comm_size|col_comm_size, store unique edge
  // majors/minors to support storing edge major/minor properties in (key, value) pairs.

  // 1. Update local_sorted_unique_edge_minors & local_sorted_unique_edge_minor_offsets

  {
    auto [minor_range_first, minor_range_last] = meta.partition.local_edge_partition_minor_range();
    auto minor_range_size = meta.partition.local_edge_partition_minor_range_size();
    rmm::device_uvector<uint32_t> minor_bitmaps(
      (minor_range_size + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8),
      handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), minor_bitmaps.begin(), minor_bitmaps.end(), uint32_t{0});
    for (size_t i = 0; i < edge_partition_indices.size(); ++i) {
      thrust::for_each(handle.get_thrust_policy(),
                       edge_partition_indices[i].begin(),
                       edge_partition_indices[i].end(),
                       atomic_or_bitmap_t<vertex_t>{minor_bitmaps.data(), minor_range_first});
    }

    auto count_first = thrust::make_transform_iterator(minor_bitmaps.begin(), popc_t<vertex_t>{});
    auto num_local_unique_edge_minors = thrust::reduce(
      handle.get_thrust_policy(), count_first, count_first + minor_bitmaps.size(), vertex_t{0});

    auto max_minor_properties_fill_ratio = host_scalar_allreduce(
      comm,
      static_cast<double>(num_local_unique_edge_minors) / static_cast<double>(minor_range_size),
      raft::comms::op_t::MAX,
      handle.get_stream());

    if (max_minor_properties_fill_ratio <
        detail::edge_partition_src_dst_property_values_kv_pair_fill_ratio_threshold) {
      auto const chunk_size =
        std::min(static_cast<size_t>(1.0 / max_minor_properties_fill_ratio), size_t{1024});

      rmm::device_uvector<vertex_t> unique_edge_minors(num_local_unique_edge_minors,
                                                       handle.get_stream());
#if 1  // FIXME: work-around for the 32 bit integer overflow issue in thrust::remove,
       // thrust::remove_if, and thrust::copy_if (https://github.com/NVIDIA/thrust/issues/1302)
      size_t num_copied{0};
      size_t num_scanned{0};
      while (num_scanned < static_cast<size_t>(minor_range_size)) {
        size_t this_scan_size =
          std::min(size_t{1} << 30,
                   static_cast<size_t>(minor_range_last - (minor_range_first + num_scanned)));
        num_copied += static_cast<size_t>(thrust::distance(
          unique_edge_minors.begin() + num_copied,
          thrust::copy_if(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(minor_range_first + num_scanned),
            thrust::make_counting_iterator(minor_range_first + num_scanned + this_scan_size),
            unique_edge_minors.begin() + num_copied,
            cugraph::detail::check_bit_set_t<vertex_t>{minor_bitmaps.data(), minor_range_first})));
        num_scanned += this_scan_size;
      }
#else
      thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(minor_range_first),
        thrust::make_counting_iterator(minor_range_last),
        unique_edge_minors.begin(),
        cugraph::detail::check_bit_set_t<vertex_t>{minor_bitmaps.data(), minor_range_first});
#endif

      auto num_chunks =
        static_cast<size_t>((minor_range_size + (chunk_size - size_t{1})) / chunk_size);
      rmm::device_uvector<vertex_t> unique_edge_minor_chunk_start_offsets(num_chunks + size_t{1},
                                                                          handle.get_stream());

      auto chunk_start_vertex_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(vertex_t{0}),
        detail::multiply_and_add_t<vertex_t>{static_cast<vertex_t>(chunk_size), minor_range_first});
      thrust::lower_bound(handle.get_thrust_policy(),
                          unique_edge_minors.begin(),
                          unique_edge_minors.end(),
                          chunk_start_vertex_first,
                          chunk_start_vertex_first + num_chunks,
                          unique_edge_minor_chunk_start_offsets.begin());
      unique_edge_minor_chunk_start_offsets.set_element(
        num_chunks, static_cast<vertex_t>(unique_edge_minors.size()), handle.get_stream());

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
                          unique_edge_minors.begin(),
                          unique_edge_minors.end(),
                          d_vertex_partition_firsts.begin(),
                          d_vertex_partition_firsts.end(),
                          d_key_offsets.begin());
      std::vector<vertex_t> h_key_offsets(row_comm_size + 1, vertex_t{0});
      h_key_offsets.back() = static_cast<vertex_t>(unique_edge_minors.size());
      raft::update_host(
        h_key_offsets.data() + 1, d_key_offsets.data(), d_key_offsets.size(), handle.get_stream());

      local_sorted_unique_edge_minors = std::move(unique_edge_minors);
      local_sorted_unique_edge_minor_chunk_start_offsets =
        std::move(unique_edge_minor_chunk_start_offsets);
      local_sorted_unique_edge_minor_chunk_size               = chunk_size;
      local_sorted_unique_edge_minor_vertex_partition_offsets = std::move(h_key_offsets);
    }
  }

  // 2. Update local_sorted_unique_edge_majors & local_sorted_unique_edge_major_offsets

  std::vector<vertex_t> num_local_unique_edge_major_counts(edge_partition_offsets.size());
  for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
    num_local_unique_edge_major_counts[i] += thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(vertex_t{0}),
      thrust::make_counting_iterator(static_cast<vertex_t>(edge_partition_offsets[i].size() - 1)),
      has_nzd_t<vertex_t, edge_t>{edge_partition_offsets[i].data(), vertex_t{0}});
  }
  auto num_local_unique_edge_majors = std::reduce(num_local_unique_edge_major_counts.begin(),
                                                  num_local_unique_edge_major_counts.end());

  vertex_t aggregate_major_range_size{0};
  for (size_t i = 0; i < meta.partition.number_of_local_edge_partitions(); ++i) {
    aggregate_major_range_size += meta.partition.local_edge_partition_major_range_size(i);
  }

  auto max_major_properties_fill_ratio =
    host_scalar_allreduce(comm,
                          static_cast<double>(num_local_unique_edge_majors) /
                            static_cast<double>(aggregate_major_range_size),
                          raft::comms::op_t::MAX,
                          handle.get_stream());

  if (max_major_properties_fill_ratio <
      detail::edge_partition_src_dst_property_values_kv_pair_fill_ratio_threshold) {
    auto const chunk_size =
      std::min(static_cast<size_t>(1.0 / max_major_properties_fill_ratio), size_t{1024});

    local_sorted_unique_edge_majors = std::vector<rmm::device_uvector<vertex_t>>{};
    local_sorted_unique_edge_major_chunk_start_offsets =
      std::vector<rmm::device_uvector<vertex_t>>{};

    (*local_sorted_unique_edge_majors).reserve(edge_partition_offsets.size());
    (*local_sorted_unique_edge_major_chunk_start_offsets).reserve(edge_partition_offsets.size());
    for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
      auto [major_range_first, major_range_last] =
        meta.partition.local_edge_partition_major_range(i);
      auto sparse_range_last =
        use_dcs
          ? (major_range_first +
             meta.edge_partition_segment_offsets[num_segments_per_vertex_partition * i +
                                                 detail::num_sparse_segments_per_vertex_partition])
          : major_range_last;

      rmm::device_uvector<vertex_t> unique_edge_majors(num_local_unique_edge_major_counts[i],
                                                       handle.get_stream());
      CUGRAPH_EXPECTS(
        sparse_range_last - major_range_first < std::numeric_limits<int32_t>::max(),
        "copy_if will fail (https://github.com/NVIDIA/thrust/issues/1302), work-around required.");
      auto cur_size = thrust::distance(
        unique_edge_majors.begin(),
        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(major_range_first),
          thrust::make_counting_iterator(sparse_range_last),
          unique_edge_majors.begin(),
          has_nzd_t<vertex_t, edge_t>{edge_partition_offsets[i].data(), major_range_first}));
      if (use_dcs) {
        thrust::copy(handle.get_thrust_policy(),
                     (*edge_partition_dcs_nzd_vertices)[i].begin(),
                     (*edge_partition_dcs_nzd_vertices)[i].begin() +
                       (*edge_partition_dcs_nzd_vertex_counts)[i],
                     unique_edge_majors.begin() + cur_size);
      }

      auto num_chunks = static_cast<size_t>(
        ((major_range_last - major_range_first) + (chunk_size - size_t{1})) / chunk_size);
      rmm::device_uvector<vertex_t> unique_edge_major_chunk_start_offsets(num_chunks + size_t{1},
                                                                          handle.get_stream());

      auto chunk_start_vertex_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(vertex_t{0}),
        detail::multiply_and_add_t<vertex_t>{static_cast<vertex_t>(chunk_size), major_range_first});
      thrust::lower_bound(handle.get_thrust_policy(),
                          unique_edge_majors.begin(),
                          unique_edge_majors.end(),
                          chunk_start_vertex_first,
                          chunk_start_vertex_first + num_chunks,
                          unique_edge_major_chunk_start_offsets.begin());
      unique_edge_major_chunk_start_offsets.set_element(
        num_chunks, static_cast<vertex_t>(unique_edge_majors.size()), handle.get_stream());

      (*local_sorted_unique_edge_majors).push_back(std::move(unique_edge_majors));
      (*local_sorted_unique_edge_major_chunk_start_offsets)
        .push_back(std::move(unique_edge_major_chunk_start_offsets));
    }
    local_sorted_unique_edge_major_chunk_size = chunk_size;
  }

  return std::make_tuple(std::move(local_sorted_unique_edge_majors),
                         std::move(local_sorted_unique_edge_major_chunk_start_offsets),
                         std::move(local_sorted_unique_edge_major_chunk_size),
                         std::move(local_sorted_unique_edge_minors),
                         std::move(local_sorted_unique_edge_minor_chunk_start_offsets),
                         std::move(local_sorted_unique_edge_minor_chunk_size),
                         std::move(local_sorted_unique_edge_minor_vertex_partition_offsets));
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
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  auto is_weighted = edgelists[0].weights.has_value();
  auto num_segments_per_vertex_partition =
    static_cast<size_t>(meta.edge_partition_segment_offsets.size() / col_comm_size);
  auto use_dcs =
    num_segments_per_vertex_partition > (detail::num_sparse_segments_per_vertex_partition + 2);

  check_graph_constructor_input_arguments<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle, edgelists, meta, do_expensive_check);

  edge_partition_segment_offsets_ = meta.edge_partition_segment_offsets;

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
    rmm::device_uvector<edge_t> offsets(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> indices(size_t{0}, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
    auto major_hypersparse_first =
      use_dcs ? std::make_optional<vertex_t>(
                  major_range_first +
                  edge_partition_segment_offsets_[num_segments_per_vertex_partition * i +
                                                  detail::num_sparse_segments_per_vertex_partition])
              : std::nullopt;
    if (edgelists[i].weights) {
      std::tie(offsets, indices, weights, dcs_nzd_vertices) =
        detail::compress_edgelist<edge_t, store_transposed>(edgelists[i].srcs.begin(),
                                                            edgelists[i].srcs.end(),
                                                            edgelists[i].dsts.begin(),
                                                            (*(edgelists[i].weights)).begin(),
                                                            major_range_first,
                                                            major_hypersparse_first,
                                                            major_range_last,
                                                            minor_range_first,
                                                            minor_range_last,
                                                            handle.get_stream());
    } else {
      std::tie(offsets, indices, dcs_nzd_vertices) =
        detail::compress_edgelist<edge_t, store_transposed>(edgelists[i].srcs.begin(),
                                                            edgelists[i].srcs.end(),
                                                            edgelists[i].dsts.begin(),
                                                            major_range_first,
                                                            major_hypersparse_first,
                                                            major_range_last,
                                                            minor_range_first,
                                                            minor_range_last,
                                                            handle.get_stream());
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
    if (edge_partition_weights_) {
      detail::sort_adjacency_list(
        handle,
        raft::device_span<edge_t const>(edge_partition_offsets_[i].data(),
                                        edge_partition_offsets_[i].size()),
        edge_partition_indices_[i].begin(),
        edge_partition_indices_[i].end(),
        (*edge_partition_weights_)[i].begin());
    } else {
      detail::sort_adjacency_list(
        handle,
        raft::device_span<edge_t const>(edge_partition_offsets_[i].data(),
                                        edge_partition_offsets_[i].size()),
        edge_partition_indices_[i].begin(),
        edge_partition_indices_[i].end());
    }
  }

  // update local sorted unique edge sources/destinations (only if key, value pair will be used)

  if constexpr (store_transposed) {
    std::tie(local_sorted_unique_edge_dsts_,
             local_sorted_unique_edge_dst_chunk_start_offsets_,
             local_sorted_unique_edge_dst_chunk_size_,
             local_sorted_unique_edge_srcs_,
             local_sorted_unique_edge_src_chunk_start_offsets_,
             local_sorted_unique_edge_src_chunk_size_,
             local_sorted_unique_edge_src_vertex_partition_offsets_) =
      update_local_sorted_unique_edge_majors_minors<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        meta,
        edge_partition_offsets_,
        edge_partition_indices_,
        edge_partition_dcs_nzd_vertices_,
        edge_partition_dcs_nzd_vertex_counts_);
  } else {
    std::tie(local_sorted_unique_edge_srcs_,
             local_sorted_unique_edge_src_chunk_start_offsets_,
             local_sorted_unique_edge_src_chunk_size_,
             local_sorted_unique_edge_dsts_,
             local_sorted_unique_edge_dst_chunk_start_offsets_,
             local_sorted_unique_edge_dst_chunk_size_,
             local_sorted_unique_edge_dst_vertex_partition_offsets_) =
      update_local_sorted_unique_edge_majors_minors<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        meta,
        edge_partition_offsets_,
        edge_partition_indices_,
        edge_partition_dcs_nzd_vertices_,
        edge_partition_dcs_nzd_vertex_counts_);
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_t(
    raft::handle_t const& handle,
    std::vector<rmm::device_uvector<edge_t>>&& edge_partition_offsets,
    std::vector<rmm::device_uvector<vertex_t>>&& edge_partition_indices,
    std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edge_partition_weights,
    std::optional<std::vector<rmm::device_uvector<vertex_t>>>&& edge_partition_dcs_nzd_vertices,
    graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
    bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, meta.number_of_edges, meta.properties),
    partition_(meta.partition)
{
  CUGRAPH_EXPECTS(
    edge_partition_offsets.size() == edge_partition_indices.size(),
    "Invalid input argument: edge_partition_offsets.size() != edge_partition_indices.size().");
  CUGRAPH_EXPECTS(!edge_partition_weights.has_value() ||
                    (edge_partition_indices.size() == (*edge_partition_weights).size()),
                  "Invalid input argument: edge_partition_weights.has_value() && "
                  "edge_partition_indices.size() != (*edge_partition_weights).size().");
  CUGRAPH_EXPECTS(!edge_partition_dcs_nzd_vertices.has_value() ||
                    (edge_partition_indices.size() == (*edge_partition_dcs_nzd_vertices).size()),
                  "Invalid input argument: edge_partition_dcs_nzd_vertices.has_value() && "
                  "edge_partition_indices.size() != (*edge_partition_dcs_nzd_vertices).size().");
  for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
    CUGRAPH_EXPECTS(!edge_partition_weights.has_value() ||
                      (edge_partition_indices[i].size() == (*edge_partition_weights)[i].size()),
                    "Invalid input argument: edge_partition_weights.has_value() && "
                    "edge_partition_indices[].size() != (*edge_partition_weights)[].size().");
  }

  edge_partition_segment_offsets_ = meta.edge_partition_segment_offsets;

  // compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid

  edge_partition_offsets_          = std::move(edge_partition_offsets);
  edge_partition_indices_          = std::move(edge_partition_indices);
  edge_partition_weights_          = std::move(edge_partition_weights);
  edge_partition_dcs_nzd_vertices_ = std::move(edge_partition_dcs_nzd_vertices);
  if (edge_partition_dcs_nzd_vertices_) {
    edge_partition_dcs_nzd_vertex_counts_ =
      std::vector<vertex_t>((*edge_partition_dcs_nzd_vertices_).size());
    for (size_t i = 0; i < (*edge_partition_dcs_nzd_vertex_counts_).size(); ++i) {
      (*edge_partition_dcs_nzd_vertex_counts_)[i] =
        static_cast<vertex_t>((*edge_partition_dcs_nzd_vertices_)[i].size());
    }
  }

  // update local sorted unique edge sources/destinations (only if key, value pair will be used)

  if constexpr (store_transposed) {
    std::tie(local_sorted_unique_edge_dsts_,
             local_sorted_unique_edge_dst_chunk_start_offsets_,
             local_sorted_unique_edge_dst_chunk_size_,
             local_sorted_unique_edge_srcs_,
             local_sorted_unique_edge_src_chunk_start_offsets_,
             local_sorted_unique_edge_src_chunk_size_,
             local_sorted_unique_edge_src_vertex_partition_offsets_) =
      update_local_sorted_unique_edge_majors_minors<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        meta,
        edge_partition_offsets_,
        edge_partition_indices_,
        edge_partition_dcs_nzd_vertices_,
        edge_partition_dcs_nzd_vertex_counts_);
  } else {
    std::tie(local_sorted_unique_edge_srcs_,
             local_sorted_unique_edge_src_chunk_start_offsets_,
             local_sorted_unique_edge_src_chunk_size_,
             local_sorted_unique_edge_dsts_,
             local_sorted_unique_edge_dst_chunk_start_offsets_,
             local_sorted_unique_edge_dst_chunk_size_,
             local_sorted_unique_edge_dst_vertex_partition_offsets_) =
      update_local_sorted_unique_edge_majors_minors<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        meta,
        edge_partition_offsets_,
        edge_partition_indices_,
        edge_partition_dcs_nzd_vertices_,
        edge_partition_dcs_nzd_vertex_counts_);
  }
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
      handle, meta.number_of_vertices, static_cast<edge_t>(edgelist.srcs.size()), meta.properties),
    offsets_(rmm::device_uvector<edge_t>(0, handle.get_stream())),
    indices_(rmm::device_uvector<vertex_t>(0, handle.get_stream())),
    segment_offsets_(meta.segment_offsets)
{
  check_graph_constructor_input_arguments<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle, edgelist, meta, do_expensive_check);

  // convert edge list (COO) to compressed sparse format (CSR or CSC)

  if (edgelist.weights) {
    std::tie(offsets_, indices_, weights_, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(edgelist.srcs.begin(),
                                                          edgelist.srcs.end(),
                                                          edgelist.dsts.begin(),
                                                          (*(edgelist.weights)).begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          this->number_of_vertices(),
                                                          vertex_t{0},
                                                          this->number_of_vertices(),
                                                          handle.get_stream());
  } else {
    std::tie(offsets_, indices_, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(edgelist.srcs.begin(),
                                                          edgelist.srcs.end(),
                                                          edgelist.dsts.begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          this->number_of_vertices(),
                                                          vertex_t{0},
                                                          this->number_of_vertices(),
                                                          handle.get_stream());
  }

  // segmented sort neighbors

  if (weights_) {
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(offsets_.data(), offsets_.size()),
                                indices_.begin(),
                                indices_.end(),
                                (*weights_).begin());
  } else {
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(offsets_.data(), offsets_.size()),
                                indices_.begin(),
                                indices_.end());
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  graph_t(raft::handle_t const& handle,
          rmm::device_uvector<edge_t>&& offsets,
          rmm::device_uvector<vertex_t>&& indices,
          std::optional<rmm::device_uvector<weight_t>>&& weights,
          graph_meta_t<vertex_t, edge_t, multi_gpu> meta,
          bool do_expensive_check)
  : detail::graph_base_t<vertex_t, edge_t, weight_t>(
      handle, meta.number_of_vertices, static_cast<edge_t>(indices.size()), meta.properties),
    offsets_(std::move(offsets)),
    indices_(std::move(indices)),
    weights_(std::move(weights)),
    segment_offsets_(meta.segment_offsets)
{
  CUGRAPH_EXPECTS(
    !weights.has_value() || (indices.size() == (*weights).size()),
    "Invalid input argument: weights.has_value() && indices.size() != (*weights).size().");
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

  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> symmetrized_graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(symmetrized_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(*wrapped_renumber_map),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
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

  auto vertex_span = renumber ? std::move(renumber_map) : std::nullopt;

  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> symmetrized_graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(symmetrized_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(vertex_span),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
      graph_properties_t{is_multigraph, true},
      renumber);
  *this = std::move(symmetrized_graph);

  return new_renumber_map;
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

  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> transposed_graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(transposed_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(*wrapped_renumber_map),
      std::move(edgelist_dsts),
      std::move(edgelist_srcs),
      std::move(edgelist_weights),
      std::nullopt,
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
  auto vertex_span = renumber ? std::move(renumber_map) : std::nullopt;

  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> transposed_graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(transposed_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(vertex_span),
      std::move(edgelist_dsts),
      std::move(edgelist_srcs),
      std::move(edgelist_weights),
      std::nullopt,
      graph_properties_t{is_multigraph, false},
      renumber);
  *this = std::move(transposed_graph);

  return new_renumber_map;
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

  graph_t<vertex_t, edge_t, weight_t, !store_transposed, multi_gpu> storage_transposed_graph(
    handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(storage_transposed_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, !store_transposed, multi_gpu>(
      handle,
      std::move(*wrapped_renumber_map),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
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
  auto vertex_span = renumber ? std::move(renumber_map) : std::nullopt;

  graph_t<vertex_t, edge_t, weight_t, !store_transposed, multi_gpu> storage_transposed_graph(
    handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(storage_transposed_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, !store_transposed, multi_gpu>(
      handle,
      std::move(vertex_span),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
      graph_properties_t{is_multigraph, false},
      renumber);

  return std::make_tuple(std::move(storage_transposed_graph), std::move(*new_renumber_map));
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

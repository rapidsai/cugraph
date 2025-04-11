/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "cugraph/edge_partition_view.hpp"
#include "detail/graph_partition_utils.cuh"
#include "structure/detail/structure_utils.cuh"

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

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cstdint>
#include <numeric>
#include <variant>

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
    CUGRAPH_EXPECTS(static_cast<size_t>(cuda::std::distance(
                      sorted_vertices.begin(),
                      thrust::unique(handle.get_thrust_policy(),
                                     sorted_vertices.begin(),
                                     sorted_vertices.end()))) == sorted_vertices.size(),
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
                          raft::host_span<size_t const>(recvcounts.data(), recvcounts.size()),
                          raft::host_span<size_t const>(displacements.data(), displacements.size()),
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
                          raft::host_span<size_t const>(recvcounts.data(), recvcounts.size()),
                          raft::host_span<size_t const>(displacements.data(), displacements.size()),
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
  std::tie(symmetrized_srcs,
           symmetrized_dsts,
           std::ignore,
           std::ignore,
           std::ignore,
           std::ignore,
           std::ignore) = symmetrize_edgelist<vertex_t,
                                              vertex_t,
                                              float /* dummy */,
                                              int32_t,
                                              int32_t,
                                              store_transposed,
                                              multi_gpu>(handle,
                                                         std::move(symmetrized_srcs),
                                                         std::move(symmetrized_dsts),
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         true);

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

template <typename edge_t>
std::vector<rmm::device_uvector<std::byte>>
split_edge_chunk_compressed_elements_to_local_edge_partitions(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<std::byte>>&& edgelist_compressed_elements,
  std::vector<std::vector<edge_t>> const& edgelist_edge_offset_vectors,
  std::vector<edge_t> const& edge_partition_edge_counts,
  std::vector<std::vector<edge_t>> const& edge_partition_intra_partition_segment_offset_vectors,
  std::vector<std::vector<edge_t>> const&
    edge_partition_intra_segment_copy_output_displacement_vectors,
  size_t element_size)
{
  auto num_chunks          = edgelist_compressed_elements.size();
  auto num_edge_partitions = edge_partition_edge_counts.size();
  auto num_segments        = edge_partition_intra_partition_segment_offset_vectors[0].size() - 1;
  for (size_t i = 0; i < edge_partition_intra_partition_segment_offset_vectors.size(); ++i) {
    assert(edge_partition_intra_partition_segment_offset_vectors[i].size() == (num_segments + 1));
  }

  std::vector<rmm::device_uvector<std::byte>> edge_partition_compressed_elements{};
  edge_partition_compressed_elements.reserve(num_edge_partitions);
  for (size_t i = 0; i < num_edge_partitions; ++i) {
    edge_partition_compressed_elements.push_back(rmm::device_uvector<std::byte>(
      edge_partition_edge_counts[i] * element_size, handle.get_stream()));
  }

  for (size_t i = 0; i < num_edge_partitions; ++i) {
    for (size_t j = 0; j < num_segments; ++j) {
      for (size_t k = 0; k < num_chunks; ++k) {
        auto segment_offset = edgelist_edge_offset_vectors[k][i * num_segments + j];
        auto segment_size   = edgelist_edge_offset_vectors[k][i * num_segments + j + 1] -
                            edgelist_edge_offset_vectors[k][i * num_segments + j];
        auto output_offset =
          edge_partition_intra_partition_segment_offset_vectors[i][j] +
          edge_partition_intra_segment_copy_output_displacement_vectors[i][j * num_chunks + k];
        thrust::copy(
          handle.get_thrust_policy(),
          edgelist_compressed_elements[k].begin() + segment_offset * element_size,
          edgelist_compressed_elements[k].begin() + (segment_offset + segment_size) * element_size,
          edge_partition_compressed_elements[i].begin() + output_offset * element_size);
      }
    }
  }
  edgelist_compressed_elements.clear();

  return edge_partition_compressed_elements;
}

template <typename edge_t, typename T>
std::vector<rmm::device_uvector<T>> split_edge_chunk_elements_to_local_edge_partitions(
  raft::handle_t const& handle,
  std::vector<rmm::device_uvector<T>>&& edgelist_elements,
  std::vector<std::vector<edge_t>> const& edgelist_edge_offset_vectors,
  std::vector<edge_t> const& edge_partition_edge_counts,
  std::vector<std::vector<edge_t>> const& edge_partition_intra_partition_segment_offset_vectors,
  std::vector<std::vector<edge_t>> const&
    edge_partition_intra_segment_copy_output_displacement_vectors)
{
  static_assert(std::is_arithmetic_v<T>);  // otherwise, unimplemented.
  auto num_chunks          = edgelist_elements.size();
  auto num_edge_partitions = edge_partition_edge_counts.size();
  auto num_segments        = edge_partition_intra_partition_segment_offset_vectors[0].size() - 1;
  for (size_t i = 0; i < edge_partition_intra_partition_segment_offset_vectors.size(); ++i) {
    assert(edge_partition_intra_partition_segment_offset_vectors[i].size() == (num_segments + 1));
  }

  std::vector<rmm::device_uvector<T>> edge_partition_elements{};
  edge_partition_elements.reserve(num_edge_partitions);
  for (size_t i = 0; i < num_edge_partitions; ++i) {
    edge_partition_elements.push_back(
      rmm::device_uvector<T>(edge_partition_edge_counts[i], handle.get_stream()));
  }

  for (size_t i = 0; i < num_edge_partitions; ++i) {
    for (size_t j = 0; j < num_segments; ++j) {
      for (size_t k = 0; k < num_chunks; ++k) {
        auto segment_offset = edgelist_edge_offset_vectors[k][i * num_segments + j];
        auto segment_size   = edgelist_edge_offset_vectors[k][i * num_segments + j + 1] -
                            edgelist_edge_offset_vectors[k][i * num_segments + j];
        auto output_offset =
          edge_partition_intra_partition_segment_offset_vectors[i][j] +
          edge_partition_intra_segment_copy_output_displacement_vectors[i][j * num_chunks + k];

        thrust::copy(handle.get_thrust_policy(),
                     edgelist_elements[k].begin() + segment_offset,
                     edgelist_elements[k].begin() + (segment_offset + segment_size),
                     edge_partition_elements[i].begin() + output_offset);
      }
    }
  }
  edgelist_elements.clear();

  return edge_partition_elements;
}

template <typename vertex_t>
void decompress_vertices(raft::handle_t const& handle,
                         raft::device_span<std::byte const> compressed_vertices,
                         raft::device_span<vertex_t> vertices,
                         size_t compressed_v_size)
{
  auto input_v_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}),
    cuda::proclaim_return_type<vertex_t>(
      [byte_first = compressed_vertices.begin(), compressed_v_size] __device__(size_t i) {
        uint64_t v{0};
        for (size_t j = 0; j < compressed_v_size; ++j) {
          auto b = *(byte_first + i * compressed_v_size + j);
          v |= static_cast<uint64_t>(b) << (8 * j);
        }
        return static_cast<vertex_t>(v);
      }));
  thrust::copy(
    handle.get_thrust_policy(), input_v_first, input_v_first + vertices.size(), vertices.begin());
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_partitioned_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<rmm::device_uvector<vertex_t>>&& edge_partition_edgelist_srcs,
  std::vector<rmm::device_uvector<vertex_t>>&& edge_partition_edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edge_partition_edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edge_partition_edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edge_partition_edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&&
    edge_partition_edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&&
    edge_partition_edgelist_edge_end_times,
  std::vector<std::vector<edge_t>> const& edgelist_intra_partition_segment_offset_vectors,
  graph_properties_t graph_properties,
  bool renumber)
{
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  // 1. renumber

  std::vector<edge_t> edgelist_edge_counts(minor_comm_size, edge_t{0});
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    edgelist_edge_counts[i] = static_cast<edge_t>(edge_partition_edgelist_srcs[i].size());
  }

  std::vector<vertex_t*> src_ptrs(minor_comm_size);
  std::vector<vertex_t*> dst_ptrs(src_ptrs.size());
  for (int i = 0; i < minor_comm_size; ++i) {
    src_ptrs[i] = edge_partition_edgelist_srcs[i].begin();
    dst_ptrs[i] = edge_partition_edgelist_dsts[i].begin();
  }
  auto [renumber_map_labels, meta] = cugraph::renumber_edgelist<vertex_t, edge_t, true>(
    handle,
    std::move(local_vertices),
    src_ptrs,
    dst_ptrs,
    edgelist_edge_counts,
    edgelist_intra_partition_segment_offset_vectors,
    store_transposed);

  auto num_segments_per_vertex_partition =
    static_cast<size_t>(meta.edge_partition_segment_offsets.size() / minor_comm_size);
  auto use_dcs =
    num_segments_per_vertex_partition > (detail::num_sparse_segments_per_vertex_partition + 2);

  // 2. sort and compress edge list (COO) to CSR (or CSC) or CSR + DCSR (CSC + DCSC) hybrid
  int edge_property_count = 0;
  size_t element_size     = sizeof(vertex_t) * 2;

  if (edge_partition_edgelist_weights) {
    ++edge_property_count;
    element_size += sizeof(weight_t);
  }

  if (edge_partition_edgelist_edge_ids) {
    ++edge_property_count;
    element_size += sizeof(edge_t);
  }
  if (edge_partition_edgelist_edge_types) {
    ++edge_property_count;
    element_size += sizeof(edge_type_t);
  }
  if (edge_partition_edgelist_edge_start_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }
  if (edge_partition_edgelist_edge_end_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }

  if (edge_property_count > 1) { element_size = sizeof(vertex_t) * 2 + sizeof(size_t); }

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto constexpr mem_frugal_ratio =
    0.05;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  std::vector<rmm::device_uvector<edge_t>> edge_partition_offsets;
  std::vector<rmm::device_uvector<vertex_t>> edge_partition_indices;
  std::optional<std::vector<rmm::device_uvector<weight_t>>> edge_partition_weights{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_t>>> edge_partition_edge_ids{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_partition_edge_types{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edge_partition_edge_start_times{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>> edge_partition_edge_end_times{
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
    edge_partition_edge_ids = std::vector<rmm::device_uvector<edge_t>>{};
    (*edge_partition_edge_ids).reserve(edge_partition_edgelist_srcs.size());
  }
  if (edge_partition_edgelist_edge_types) {
    edge_partition_edge_types = std::vector<rmm::device_uvector<edge_type_t>>{};
    (*edge_partition_edge_types).reserve(edge_partition_edgelist_srcs.size());
  }
  if (edge_partition_edgelist_edge_start_times) {
    edge_partition_edge_start_times = std::vector<rmm::device_uvector<edge_time_t>>{};
    (*edge_partition_edge_start_times).reserve(edge_partition_edgelist_srcs.size());
  }
  if (edge_partition_edgelist_edge_end_times) {
    edge_partition_edge_end_times = std::vector<rmm::device_uvector<edge_time_t>>{};
    (*edge_partition_edge_end_times).reserve(edge_partition_edgelist_srcs.size());
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
    std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
    std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
    std::optional<rmm::device_uvector<edge_time_t>> edge_start_times{std::nullopt};
    std::optional<rmm::device_uvector<edge_time_t>> edge_end_times{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> dcs_nzd_vertices{std::nullopt};
    auto major_hypersparse_first =
      use_dcs
        ? std::make_optional<vertex_t>(
            major_range_first +
            meta.edge_partition_segment_offsets[num_segments_per_vertex_partition * i +
                                                detail::num_sparse_segments_per_vertex_partition])
        : std::nullopt;

    if (edge_property_count == 0) {
      std::tie(offsets, indices, dcs_nzd_vertices) =
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
    } else if (edge_property_count == 1) {
      if (edge_partition_edgelist_weights) {
        std::tie(offsets, indices, weights, dcs_nzd_vertices) =
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
      } else if (edge_partition_edgelist_edge_ids) {
        std::tie(offsets, indices, edge_ids, dcs_nzd_vertices) =
          detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_t, store_transposed>(
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
      } else if (edge_partition_edgelist_edge_types) {
        std::tie(offsets, indices, edge_types, dcs_nzd_vertices) =
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
      } else if (edge_partition_edgelist_edge_start_times) {
        std::tie(offsets, indices, edge_start_times, dcs_nzd_vertices) =
          detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_time_t, store_transposed>(
            std::move(edge_partition_edgelist_srcs[i]),
            std::move(edge_partition_edgelist_dsts[i]),
            std::move((*edge_partition_edgelist_edge_start_times)[i]),
            major_range_first,
            major_hypersparse_first,
            major_range_last,
            minor_range_first,
            minor_range_last,
            mem_frugal_threshold,
            handle.get_stream());
      } else if (edge_partition_edgelist_edge_end_times) {
        std::tie(offsets, indices, edge_end_times, dcs_nzd_vertices) =
          detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_time_t, store_transposed>(
            std::move(edge_partition_edgelist_srcs[i]),
            std::move(edge_partition_edgelist_dsts[i]),
            std::move((*edge_partition_edgelist_edge_end_times)[i]),
            major_range_first,
            major_hypersparse_first,
            major_range_last,
            minor_range_first,
            minor_range_last,
            mem_frugal_threshold,
            handle.get_stream());
      }
    } else {
      rmm::device_uvector<edge_t> property_position(edge_partition_edgelist_srcs[i].size(),
                                                    handle.get_stream());
      detail::sequence_fill(
        handle.get_stream(), property_position.data(), property_position.size(), edge_t{0});

      std::tie(offsets, indices, property_position, dcs_nzd_vertices) =
        detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_t, store_transposed>(
          std::move(edge_partition_edgelist_srcs[i]),
          std::move(edge_partition_edgelist_dsts[i]),
          std::move(property_position),
          major_range_first,
          major_hypersparse_first,
          major_range_last,
          minor_range_first,
          minor_range_last,
          mem_frugal_threshold,
          handle.get_stream());

      if (edge_partition_edgelist_weights) {
        weights = rmm::device_uvector<weight_t>(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_weights)[i].begin(),
                       weights->begin());
      }

      if (edge_partition_edgelist_edge_ids) {
        edge_ids = rmm::device_uvector<edge_t>(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_ids)[i].begin(),
                       edge_ids->begin());
      }

      if (edge_partition_edgelist_edge_types) {
        edge_types =
          rmm::device_uvector<edge_type_t>(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_types)[i].begin(),
                       edge_types->begin());
      }

      if (edge_partition_edgelist_edge_start_times) {
        edge_start_times =
          rmm::device_uvector<edge_time_t>(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_start_times)[i].begin(),
                       edge_start_times->begin());
      }

      if (edge_partition_edgelist_edge_end_times) {
        edge_end_times =
          rmm::device_uvector<edge_time_t>(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_end_times)[i].begin(),
                       edge_end_times->begin());
      }
    }

    edge_partition_offsets.push_back(std::move(offsets));
    edge_partition_indices.push_back(std::move(indices));
    if (edge_partition_weights) { (*edge_partition_weights).push_back(std::move(*weights)); }
    if (edge_partition_edge_ids) { (*edge_partition_edge_ids).push_back(std::move(*edge_ids)); }
    if (edge_partition_edge_types) {
      (*edge_partition_edge_types).push_back(std::move(*edge_types));
    }
    if (edge_partition_edge_start_times) {
      (*edge_partition_edge_start_times).push_back(std::move(*edge_start_times));
    }
    if (edge_partition_edge_end_times) {
      (*edge_partition_edge_end_times).push_back(std::move(*edge_end_times));
    }

    if (edge_partition_dcs_nzd_vertices) {
      (*edge_partition_dcs_nzd_vertices).push_back(std::move(*dcs_nzd_vertices));
    }
  }

  // 3. segmented sort neighbors

  for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
    if (edge_property_count == 0) {
      detail::sort_adjacency_list(handle,
                                  raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                                                  edge_partition_offsets[i].size()),
                                  edge_partition_indices[i].begin(),
                                  edge_partition_indices[i].end());
    } else if (edge_property_count == 1) {
      if (edge_partition_edgelist_weights) {
        detail::sort_adjacency_list(
          handle,
          raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                          edge_partition_offsets[i].size()),
          edge_partition_indices[i].begin(),
          edge_partition_indices[i].end(),
          (*edge_partition_weights)[i].begin());
      } else if (edge_partition_edgelist_edge_ids) {
        detail::sort_adjacency_list(
          handle,
          raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                          edge_partition_offsets[i].size()),
          edge_partition_indices[i].begin(),
          edge_partition_indices[i].end(),
          (*edge_partition_edge_ids)[i].begin());
      } else if (edge_partition_edgelist_edge_types) {
        detail::sort_adjacency_list(
          handle,
          raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                          edge_partition_offsets[i].size()),
          edge_partition_indices[i].begin(),
          edge_partition_indices[i].end(),
          (*edge_partition_edge_types)[i].begin());
      } else if (edge_partition_edgelist_edge_start_times) {
        detail::sort_adjacency_list(
          handle,
          raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                          edge_partition_offsets[i].size()),
          edge_partition_indices[i].begin(),
          edge_partition_indices[i].end(),
          (*edge_partition_edge_start_times)[i].begin());
      } else if (edge_partition_edgelist_edge_end_times) {
        detail::sort_adjacency_list(
          handle,
          raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                          edge_partition_offsets[i].size()),
          edge_partition_indices[i].begin(),
          edge_partition_indices[i].end(),
          (*edge_partition_edge_end_times)[i].begin());
      }
    } else {
      rmm::device_uvector<edge_t> property_position(edge_partition_indices[i].size(),
                                                    handle.get_stream());
      detail::sequence_fill(
        handle.get_stream(), property_position.data(), property_position.size(), edge_t{0});

      detail::sort_adjacency_list(handle,
                                  raft::device_span<edge_t const>(edge_partition_offsets[i].data(),
                                                                  edge_partition_offsets[i].size()),
                                  edge_partition_indices[i].begin(),
                                  edge_partition_indices[i].end(),
                                  property_position.begin());

      if (edge_partition_edgelist_weights) {
        rmm::device_uvector<weight_t> tmp(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_weights)[i].begin(),
                       tmp.begin());
        (*edge_partition_edgelist_weights)[i] = std::move(tmp);
      }
      if (edge_partition_edgelist_edge_ids) {
        rmm::device_uvector<edge_t> tmp(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_ids)[i].begin(),
                       tmp.begin());
        (*edge_partition_edgelist_edge_ids)[i] = std::move(tmp);
      }
      if (edge_partition_edgelist_edge_types) {
        rmm::device_uvector<edge_type_t> tmp(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_types)[i].begin(),
                       tmp.begin());
        (*edge_partition_edgelist_edge_types)[i] = std::move(tmp);
      }
      if (edge_partition_edgelist_edge_end_times) {
        rmm::device_uvector<edge_time_t> tmp(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_end_times)[i].begin(),
                       tmp.begin());
        (*edge_partition_edgelist_edge_end_times)[i] = std::move(tmp);
      }
      if (edge_partition_edgelist_edge_end_times) {
        rmm::device_uvector<edge_time_t> tmp(property_position.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       (*edge_partition_edgelist_edge_end_times)[i].begin(),
                       tmp.begin());
        (*edge_partition_edgelist_edge_end_times)[i] = std::move(tmp);
      }
    }
  }

  // 4. create a graph and an edge_property_t object.

  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, weight_t>>
    edge_weights{std::nullopt};
  if (edge_partition_weights) {
    edge_weights =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, weight_t>(
        std::move(*edge_partition_weights));
  }

  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_t>>
    edge_ids{std::nullopt};
  if (edge_partition_edge_ids) {
    edge_ids = edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_t>(
      std::move(*edge_partition_edge_ids));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_type_t>>
    edge_types{std::nullopt};
  if (edge_partition_edge_types) {
    edge_types =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_type_t>(
        std::move(*edge_partition_edge_types));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_time_t>>
    edge_start_times{std::nullopt};
  if (edge_partition_edge_start_times) {
    edge_start_times =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_time_t>(
        std::move(*edge_partition_edge_start_times));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_time_t>>
    edge_end_times{std::nullopt};
  if (edge_partition_edge_end_times) {
    edge_end_times =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, true>, edge_time_t>(
        std::move(*edge_partition_edge_end_times));
  }

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, store_transposed, true>(
      handle,
      std::move(edge_partition_offsets),
      std::move(edge_partition_indices),
      std::move(edge_partition_dcs_nzd_vertices),
      cugraph::graph_meta_t<vertex_t, edge_t, true>{
        meta.number_of_vertices,
        meta.number_of_edges,
        graph_properties,
        meta.partition,
        meta.edge_partition_segment_offsets,
        meta.edge_partition_hypersparse_degree_offsets}),
    std::move(edge_weights),
    std::move(edge_ids),
    std::move(edge_types),
    std::move(edge_start_times),
    std::move(edge_end_times),
    std::optional<rmm::device_uvector<vertex_t>>{std::move(renumber_map_labels)});
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_end_times,
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
                  "Invalid input arguments: edgelist_weights.has_value() is true and "
                  "edgelist_srcs.size() != (*edgelist_weights).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_ids || (edgelist_srcs.size() == (*edgelist_edge_ids).size()),
                  "Invalid input arguments: edgelist_edge_ids.has_value() is true and "
                  "edgelist_srcs.size() != (*edgelist_edge_ids).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_types || (edgelist_srcs.size() == (*edgelist_edge_types).size()),
                  "Invalid input arguments: edgelist_edge_types.has_value() is true, "
                  "edgelist_srcs.size() != (*edgelist_edge_types).size().");
  CUGRAPH_EXPECTS(
    !edgelist_edge_start_times || (edgelist_srcs.size() == (*edgelist_edge_start_times).size()),
    "Invalid input arguments: edgelist_edge_start_times.has_value() is true, "
    "edgelist_srcs.size() != (*edgelist_edge_start_times).size().");
  CUGRAPH_EXPECTS(
    !edgelist_edge_end_times || (edgelist_srcs.size() == (*edgelist_edge_end_times).size()),
    "Invalid input arguments: edgelist_edge_end_times.has_value() is true, "
    "edgelist_srcs.size() != (*edgelist_edge_end_times).size().");
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
    edgelist_edge_start_times,
    edgelist_edge_end_times,
    true);

  std::vector<size_t> h_edge_counts(d_edge_counts.size());
  raft::update_host(
    h_edge_counts.data(), d_edge_counts.data(), d_edge_counts.size(), handle.get_stream());
  handle.sync_stream();

  std::vector<edge_t> edgelist_edge_counts(minor_comm_size, edge_t{0});
  auto edgelist_intra_partition_segment_offset_vectors = std::vector<std::vector<edge_t>>(
    minor_comm_size, std::vector<edge_t>(major_comm_size + 1, edge_t{0}));
  for (int i = 0; i < minor_comm_size; ++i) {
    edgelist_edge_counts[i] = std::accumulate(h_edge_counts.begin() + major_comm_size * i,
                                              h_edge_counts.begin() + major_comm_size * (i + 1),
                                              edge_t{0});
    std::partial_sum(h_edge_counts.begin() + major_comm_size * i,
                     h_edge_counts.begin() + major_comm_size * (i + 1),
                     edgelist_intra_partition_segment_offset_vectors[i].begin() + 1);
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

  std::optional<std::vector<rmm::device_uvector<edge_t>>> edge_partition_edgelist_edge_ids{};
  if (edgelist_edge_ids) {
    edge_partition_edgelist_edge_ids = std::vector<rmm::device_uvector<edge_t>>{};
    (*edge_partition_edgelist_edge_ids).reserve(minor_comm_size);
    for (int i = 0; i < minor_comm_size; ++i) {
      rmm::device_uvector<edge_t> tmp_edge_ids(edgelist_edge_counts[i], handle.get_stream());
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

  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>
    edge_partition_edgelist_edge_start_times{};
  if (edgelist_edge_start_times) {
    edge_partition_edgelist_edge_start_times = std::vector<rmm::device_uvector<edge_time_t>>{};
    (*edge_partition_edgelist_edge_start_times).reserve(minor_comm_size);
    for (int i = 0; i < minor_comm_size; ++i) {
      rmm::device_uvector<edge_time_t> tmp_edge_start_times(edgelist_edge_counts[i],
                                                            handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        (*edgelist_edge_start_times).begin() + edgelist_displacements[i],
        (*edgelist_edge_start_times).begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
        tmp_edge_start_times.begin());
      (*edge_partition_edgelist_edge_start_times).push_back(std::move(tmp_edge_start_times));
    }
    (*edgelist_edge_start_times).resize(0, handle.get_stream());
    (*edgelist_edge_start_times).shrink_to_fit(handle.get_stream());
  }

  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>
    edge_partition_edgelist_edge_end_times{};
  if (edgelist_edge_end_times) {
    edge_partition_edgelist_edge_end_times = std::vector<rmm::device_uvector<edge_time_t>>{};
    (*edge_partition_edgelist_edge_end_times).reserve(minor_comm_size);
    for (int i = 0; i < minor_comm_size; ++i) {
      rmm::device_uvector<edge_time_t> tmp_edge_end_times(edgelist_edge_counts[i],
                                                          handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(),
        (*edgelist_edge_end_times).begin() + edgelist_displacements[i],
        (*edgelist_edge_end_times).begin() + edgelist_displacements[i] + edgelist_edge_counts[i],
        tmp_edge_end_times.begin());
      (*edge_partition_edgelist_edge_end_times).push_back(std::move(tmp_edge_end_times));
    }
    (*edgelist_edge_end_times).resize(0, handle.get_stream());
    (*edgelist_edge_end_times).shrink_to_fit(handle.get_stream());
  }
  return create_graph_from_partitioned_edgelist<vertex_t,
                                                edge_t,
                                                weight_t,
                                                edge_type_t,
                                                edge_time_t,
                                                store_transposed,
                                                multi_gpu>(
    handle,
    std::move(local_vertices),
    std::move(edge_partition_edgelist_srcs),
    std::move(edge_partition_edgelist_dsts),
    std::move(edge_partition_edgelist_weights),
    std::move(edge_partition_edgelist_edge_ids),
    std::move(edge_partition_edgelist_edge_types),
    std::move(edge_partition_edgelist_edge_start_times),
    std::move(edge_partition_edgelist_edge_end_times),
    edgelist_intra_partition_segment_offset_vectors,
    graph_properties,
    renumber);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_end_times,
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
                  "Invalid input arguments: edgelist_weights.has_value() is true and "
                  "edgelist_srcs.size() != (*edgelist_weights).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_ids || (edgelist_srcs.size() == (*edgelist_edge_ids).size()),
                  "Invalid input arguments: edgelist_edge_ids.has_value() is true and "
                  "edgelist_srcs.size() != (*edgelist_edge_ids).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_types || (edgelist_srcs.size() == (*edgelist_edge_types).size()),
                  "Invalid input arguments: edgelist_edge_types.has_value() is true, "
                  "edgelist_srcs.size() != (*edgelist_edge_types).size().");
  CUGRAPH_EXPECTS(
    !edgelist_edge_start_times || (edgelist_srcs.size() == (*edgelist_edge_start_times).size()),
    "Invalid input arguments: edgelist_edge_start_times.has_value() is true, "
    "edgelist_srcs.size() != (*edgelist_edge_start_times).size().");
  CUGRAPH_EXPECTS(
    !edgelist_edge_end_times || (edgelist_srcs.size() == (*edgelist_edge_end_times).size()),
    "Invalid input arguments: edgelist_edge_end_times.has_value() is true, "
    "edgelist_srcs.size() != (*edgelist_edge_end_times).size().");
  for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
    CUGRAPH_EXPECTS(edgelist_srcs[i].size() == edgelist_dsts[i].size(),
                    "Invalid input arguments: edgelist_srcs[i].size() != edgelist_dsts[i].size().");
    CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs[i].size() == (*edgelist_weights)[i].size()),
                    "Invalid input arguments: edgelist_weights.has_value() is true and "
                    "edgelist_srcs[i].size() != (*edgelist_weights)[i].size().");
    CUGRAPH_EXPECTS(
      !edgelist_edge_ids || (edgelist_srcs[i].size() == (*edgelist_edge_ids)[i].size()),
      "Invalid input arguments: edgelist_edge_ids.has_value() is true and "
      "edgelist_srcs[i].size() != (*edgelist_edge_ids)[i].size().");
    CUGRAPH_EXPECTS(
      !edgelist_edge_types || (edgelist_srcs[i].size() == (*edgelist_edge_types)[i].size()),
      "Invalid input arguments: edgelist_edge_types.has_value() is true, "
      "edgelist_srcs[i].size() != (*edgelist_edge_types)[i].size().");
    CUGRAPH_EXPECTS(!edgelist_edge_start_times ||
                      (edgelist_srcs[i].size() == (*edgelist_edge_start_times)[i].size()),
                    "Invalid input arguments: edgelist_edge_start_times.has_value() is true, "
                    "edgelist_srcs[i].size() != (*edgelist_edge_start_times)[i].size().");
    CUGRAPH_EXPECTS(
      !edgelist_edge_end_times || (edgelist_srcs[i].size() == (*edgelist_edge_end_times)[i].size()),
      "Invalid input arguments: edgelist_edge_end_times.has_value() is true, "
      "edgelist_srcs[i].size() != (*edgelist_edge_end_times)[i].size().");
  }
  CUGRAPH_EXPECTS(renumber,
                  "Invalid input arguments: renumber should be true if multi_gpu is true.");

  if (do_expensive_check) {
    edge_t aggregate_edge_count{0};
    for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
      aggregate_edge_count += edgelist_srcs[i].size();
    }

    rmm::device_uvector<vertex_t> aggregate_edgelist_srcs(aggregate_edge_count,
                                                          handle.get_stream());
    rmm::device_uvector<vertex_t> aggregate_edgelist_dsts(aggregate_edge_count,
                                                          handle.get_stream());
    edge_t output_offset{0};
    for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_srcs[i].begin(),
                   edgelist_srcs[i].end(),
                   aggregate_edgelist_srcs.begin() + output_offset);
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_dsts[i].begin(),
                   edgelist_dsts[i].end(),
                   aggregate_edgelist_dsts.begin() + output_offset);
      output_offset += edgelist_srcs[i].size();
    }

    expensive_check_edgelist<vertex_t, multi_gpu>(
      handle,
      local_vertices,
      store_transposed ? aggregate_edgelist_dsts : aggregate_edgelist_srcs,
      store_transposed ? aggregate_edgelist_srcs : aggregate_edgelist_dsts,
      renumber);

    if (graph_properties.is_symmetric) {
      CUGRAPH_EXPECTS(
        (check_symmetric<vertex_t, store_transposed, multi_gpu>(
          handle,
          raft::device_span<vertex_t const>(aggregate_edgelist_srcs.data(),
                                            aggregate_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(aggregate_edgelist_dsts.data(),
                                            aggregate_edgelist_dsts.size()))),
        "Invalid input arguments: graph_properties.is_symmetric is true but the input edge list is "
        "not symmetric.");
    }

    if (!graph_properties.is_multigraph) {
      CUGRAPH_EXPECTS(
        check_no_parallel_edge(handle,
                               raft::device_span<vertex_t const>(aggregate_edgelist_srcs.data(),
                                                                 aggregate_edgelist_srcs.size()),
                               raft::device_span<vertex_t const>(aggregate_edgelist_dsts.data(),
                                                                 aggregate_edgelist_dsts.size())),
        "Invalid input arguments: graph_properties.is_multigraph is false but the input edge list "
        "has parallel edges.");
    }
  }

  auto num_chunks = edgelist_srcs.size();

  // 1. set whether to temporarily compress vertex IDs or not in splitting edge chunks

  size_t compressed_v_size =
    sizeof(vertex_t);  // if set to a value smaller than sizeof(vertex_t), temporarily store vertex
                       // IDs in compressed_v_size byte variables

  static_assert((sizeof(vertex_t) == 4) || (sizeof(vertex_t) == 8));
  if constexpr (sizeof(vertex_t) == 8) {        // 64 bit vertex ID
    static_assert(std::is_signed_v<vertex_t>);  // __clzll takes a signed integer

    auto total_global_mem = handle.get_device_properties().totalGlobalMem;
    size_t element_size   = sizeof(vertex_t) * 2;
    if (edgelist_weights) { element_size += sizeof(weight_t); }
    if (edgelist_edge_ids) { element_size += sizeof(edge_t); }
    if (edgelist_edge_types) { element_size += sizeof(edge_type_t); }
    if (edgelist_edge_start_times) { element_size += sizeof(edge_time_t); }
    if (edgelist_edge_end_times) { element_size += sizeof(edge_time_t); }
    edge_t num_edges{0};
    for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
      num_edges += edgelist_srcs[i].size();
    }
    bool compress{false};
    if (static_cast<size_t>(num_edges) * element_size >
        static_cast<size_t>(total_global_mem * 0.5 /* tuning parameter */)) {
      compress = true;
    }

    if (compress) {
      size_t min_clz{sizeof(vertex_t) * 8};
      for (size_t i = 0; i < num_chunks; ++i) {
        auto min_clz_first = thrust::make_transform_iterator(
          thrust::make_zip_iterator(edgelist_srcs[i].begin(), edgelist_dsts[i].begin()),
          cuda::proclaim_return_type<size_t>([] __device__(auto pair) {
            return static_cast<size_t>(
              cuda::std::min(__clzll(thrust::get<0>(pair)), __clzll(thrust::get<1>(pair))));
          }));
        min_clz = thrust::reduce(handle.get_thrust_policy(),
                                 min_clz_first,
                                 min_clz_first + edgelist_srcs[i].size(),
                                 min_clz,
                                 thrust::minimum<size_t>{});
      }
      compressed_v_size = sizeof(vertex_t) - (min_clz / 8);
      compressed_v_size = std::max(compressed_v_size, size_t{1});
    }
  }

  // 2. groupby each edge chunks to their target local adjacency matrix partition (and further
  // groupby within the local partition by applying the compute_gpu_id_from_vertex_t to minor vertex
  // IDs).
  std::vector<std::vector<edge_t>> edgelist_edge_offset_vectors(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {  // iterate over input edge chunks
    std::optional<rmm::device_uvector<weight_t>> this_chunk_weights{std::nullopt};
    if (edgelist_weights) { this_chunk_weights = std::move((*edgelist_weights)[i]); }
    std::optional<rmm::device_uvector<edge_t>> this_chunk_edge_ids{std::nullopt};
    if (edgelist_edge_ids) { this_chunk_edge_ids = std::move((*edgelist_edge_ids)[i]); }
    std::optional<rmm::device_uvector<edge_type_t>> this_chunk_edge_types{std::nullopt};
    if (edgelist_edge_types) { this_chunk_edge_types = std::move((*edgelist_edge_types)[i]); }
    std::optional<rmm::device_uvector<edge_time_t>> this_chunk_edge_start_times{std::nullopt};
    if (edgelist_edge_start_times) {
      this_chunk_edge_start_times = std::move((*edgelist_edge_start_times)[i]);
    }
    std::optional<rmm::device_uvector<edge_time_t>> this_chunk_edge_end_times{std::nullopt};
    if (edgelist_edge_end_times) {
      this_chunk_edge_end_times = std::move((*edgelist_edge_end_times)[i]);
    }

    auto d_this_chunk_edge_counts =
      cugraph::detail::groupby_and_count_edgelist_by_local_partition_id(
        handle,
        store_transposed ? edgelist_dsts[i] : edgelist_srcs[i],
        store_transposed ? edgelist_srcs[i] : edgelist_dsts[i],
        this_chunk_weights,
        this_chunk_edge_ids,
        this_chunk_edge_types,
        this_chunk_edge_start_times,
        this_chunk_edge_end_times,
        true);
    if (this_chunk_weights) { (*edgelist_weights)[i] = std::move(*this_chunk_weights); }
    if (this_chunk_edge_ids) { (*edgelist_edge_ids)[i] = std::move(*this_chunk_edge_ids); }
    if (this_chunk_edge_types) { (*edgelist_edge_types)[i] = std::move(*this_chunk_edge_types); }
    if (this_chunk_edge_start_times) {
      (*edgelist_edge_start_times)[i] = std::move(*this_chunk_edge_start_times);
    }
    if (this_chunk_edge_end_times) {
      (*edgelist_edge_end_times)[i] = std::move(*this_chunk_edge_end_times);
    }

    std::vector<size_t> h_this_chunk_edge_counts(d_this_chunk_edge_counts.size());
    raft::update_host(h_this_chunk_edge_counts.data(),
                      d_this_chunk_edge_counts.data(),
                      d_this_chunk_edge_counts.size(),
                      handle.get_stream());
    handle.sync_stream();
    std::vector<edge_t> h_this_chunk_edge_offsets(
      h_this_chunk_edge_counts.size() + 1,
      0);  // size = minor_comm_size (# local edge partitions) * major_comm_size (# segments in the
           // local minor range)
    std::inclusive_scan(h_this_chunk_edge_counts.begin(),
                        h_this_chunk_edge_counts.end(),
                        h_this_chunk_edge_offsets.begin() + 1);
    edgelist_edge_offset_vectors[i] = std::move(h_this_chunk_edge_offsets);
  }

  // 3. compress edge chunk source/destination vertices to cut intermediate peak memory requirement
  std::optional<std::vector<rmm::device_uvector<std::byte>>> edgelist_compressed_srcs{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<std::byte>>> edgelist_compressed_dsts{std::nullopt};
  if (compressed_v_size < sizeof(vertex_t)) {
    edgelist_compressed_srcs = std::vector<rmm::device_uvector<std::byte>>{};
    edgelist_compressed_dsts = std::vector<rmm::device_uvector<std::byte>>{};
    (*edgelist_compressed_srcs).reserve(num_chunks);
    (*edgelist_compressed_dsts).reserve(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {  // iterate over input edge chunks
                                               // compress source values
      auto tmp_srcs = rmm::device_uvector<std::byte>(edgelist_srcs[i].size() * compressed_v_size,
                                                     handle.get_stream());
      auto input_src_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<std::byte>(
          [src_first = edgelist_srcs[i].begin(), compressed_v_size] __device__(size_t i) {
            auto v = static_cast<uint64_t>(*(src_first + (i / compressed_v_size)));
            return static_cast<std::byte>((v >> (8 * (i % compressed_v_size))) & uint64_t{0xff});
          }));
      thrust::copy(handle.get_thrust_policy(),
                   input_src_first,
                   input_src_first + edgelist_srcs[i].size() * compressed_v_size,
                   tmp_srcs.begin());
      edgelist_srcs[i].resize(0, handle.get_stream());
      edgelist_srcs[i].shrink_to_fit(handle.get_stream());
      (*edgelist_compressed_srcs).push_back(std::move(tmp_srcs));

      // compress destination values

      auto tmp_dsts = rmm::device_uvector<std::byte>(edgelist_dsts[i].size() * compressed_v_size,
                                                     handle.get_stream());
      auto input_dst_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<std::byte>(
          [dst_first = edgelist_dsts[i].begin(), compressed_v_size] __device__(size_t i) {
            auto v = static_cast<uint64_t>(*(dst_first + (i / compressed_v_size)));
            return static_cast<std::byte>((v >> (8 * (i % compressed_v_size))) & uint64_t{0xff});
          }));
      thrust::copy(handle.get_thrust_policy(),
                   input_dst_first,
                   input_dst_first + edgelist_dsts[i].size() * compressed_v_size,
                   tmp_dsts.begin());
      edgelist_dsts[i].resize(0, handle.get_stream());
      edgelist_dsts[i].shrink_to_fit(handle.get_stream());
      (*edgelist_compressed_dsts).push_back(std::move(tmp_dsts));
    }
  }

  // 4. compute additional copy_offset vectors
  std::vector<edge_t> edge_partition_edge_counts(minor_comm_size);
  std::vector<std::vector<edge_t>> edge_partition_intra_partition_segment_offset_vectors(
    minor_comm_size);
  std::vector<std::vector<edge_t>> edge_partition_intra_segment_copy_output_displacement_vectors(
    minor_comm_size);
  for (int i = 0; i < minor_comm_size; ++i) {
    edge_t edge_count{0};
    std::vector<edge_t> intra_partition_segment_sizes(major_comm_size, 0);
    std::vector<edge_t> intra_segment_copy_output_displacements(major_comm_size * num_chunks);
    for (int j = 0; j < major_comm_size /* # segments in the local minor range */; ++j) {
      edge_t displacement{0};
      for (size_t k = 0; k < num_chunks; ++k) {
        auto segment_size = edgelist_edge_offset_vectors[k][i * major_comm_size + j + 1] -
                            edgelist_edge_offset_vectors[k][i * major_comm_size + j];
        edge_count += segment_size;
        intra_partition_segment_sizes[j] += segment_size;
        intra_segment_copy_output_displacements[j * num_chunks + k] = displacement;
        displacement += segment_size;
      }
    }
    std::vector<edge_t> intra_partition_segment_offsets(major_comm_size + 1, 0);
    std::inclusive_scan(intra_partition_segment_sizes.begin(),
                        intra_partition_segment_sizes.end(),
                        intra_partition_segment_offsets.begin() + 1);

    edge_partition_edge_counts[i] = edge_count;
    edge_partition_intra_partition_segment_offset_vectors[i] =
      std::move(intra_partition_segment_offsets);
    edge_partition_intra_segment_copy_output_displacement_vectors[i] =
      std::move(intra_segment_copy_output_displacements);
  }

  // 5. split the grouped edge chunks to local partitions
  std::vector<rmm::device_uvector<vertex_t>> edge_partition_edgelist_srcs{};
  std::vector<rmm::device_uvector<vertex_t>> edge_partition_edgelist_dsts{};
  std::optional<std::vector<rmm::device_uvector<weight_t>>> edge_partition_edgelist_weights{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_t>>> edge_partition_edgelist_edge_ids{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>> edge_partition_edgelist_edge_types{
    std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>
    edge_partition_edgelist_edge_start_times{std::nullopt};
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>
    edge_partition_edgelist_edge_end_times{std::nullopt};

  std::optional<std::vector<rmm::device_uvector<std::byte>>>
    edge_partition_edgelist_compressed_srcs{};
  std::optional<std::vector<rmm::device_uvector<std::byte>>>
    edge_partition_edgelist_compressed_dsts{};

  if (compressed_v_size < sizeof(vertex_t)) {
    edge_partition_edgelist_compressed_srcs =
      split_edge_chunk_compressed_elements_to_local_edge_partitions<edge_t>(
        handle,
        std::move(*edgelist_compressed_srcs),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors,
        compressed_v_size);

    edge_partition_edgelist_compressed_dsts =
      split_edge_chunk_compressed_elements_to_local_edge_partitions<edge_t>(
        handle,
        std::move(*edgelist_compressed_dsts),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors,
        compressed_v_size);
  } else {
    edge_partition_edgelist_srcs =
      split_edge_chunk_elements_to_local_edge_partitions<edge_t, vertex_t>(
        handle,
        std::move(edgelist_srcs),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors);

    edge_partition_edgelist_dsts =
      split_edge_chunk_elements_to_local_edge_partitions<edge_t, vertex_t>(
        handle,
        std::move(edgelist_dsts),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors);
  }

  if (edgelist_weights) {
    edge_partition_edgelist_weights =
      split_edge_chunk_elements_to_local_edge_partitions<edge_t, weight_t>(
        handle,
        std::move(*edgelist_weights),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors);
  }
  if (edgelist_edge_ids) {
    edge_partition_edgelist_edge_ids =
      split_edge_chunk_elements_to_local_edge_partitions<edge_t, edge_t>(
        handle,
        std::move(*edgelist_edge_ids),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors);
  }
  if (edgelist_edge_types) {
    edge_partition_edgelist_edge_types =
      split_edge_chunk_elements_to_local_edge_partitions<edge_t, edge_type_t>(
        handle,
        std::move(*edgelist_edge_types),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors);
  }
  if (edgelist_edge_start_times) {
    edge_partition_edgelist_edge_start_times =
      split_edge_chunk_elements_to_local_edge_partitions<edge_t, edge_time_t>(
        handle,
        std::move(*edgelist_edge_start_times),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors);
  }
  if (edgelist_edge_end_times) {
    edge_partition_edgelist_edge_end_times =
      split_edge_chunk_elements_to_local_edge_partitions<edge_t, edge_time_t>(
        handle,
        std::move(*edgelist_edge_end_times),
        edgelist_edge_offset_vectors,
        edge_partition_edge_counts,
        edge_partition_intra_partition_segment_offset_vectors,
        edge_partition_intra_segment_copy_output_displacement_vectors);
  }

  // 6. decompress edge chunk source/destination vertices to cut intermediate peak memory
  // requirement
  if (compressed_v_size < sizeof(vertex_t)) {
    assert(edge_partition_edgelist_compressed_srcs);
    assert(edge_partition_edgelist_compressed_dsts);

    edge_partition_edgelist_srcs.reserve(minor_comm_size);
    edge_partition_edgelist_dsts.reserve(minor_comm_size);

    for (int i = 0; i < minor_comm_size; ++i) {
      rmm::device_uvector<vertex_t> tmp_srcs(edge_partition_edge_counts[i], handle.get_stream());
      decompress_vertices(
        handle,
        raft::device_span<std::byte const>((*edge_partition_edgelist_compressed_srcs)[i].data(),
                                           (*edge_partition_edgelist_compressed_srcs)[i].size()),
        raft::device_span<vertex_t>(tmp_srcs.data(), tmp_srcs.size()),
        compressed_v_size);
      edge_partition_edgelist_srcs.push_back(std::move(tmp_srcs));
      (*edge_partition_edgelist_compressed_srcs)[i].resize(0, handle.get_stream());
      (*edge_partition_edgelist_compressed_srcs)[i].shrink_to_fit(handle.get_stream());

      rmm::device_uvector<vertex_t> tmp_dsts(edge_partition_edge_counts[i], handle.get_stream());
      decompress_vertices(
        handle,
        raft::device_span<std::byte const>((*edge_partition_edgelist_compressed_dsts)[i].data(),
                                           (*edge_partition_edgelist_compressed_dsts)[i].size()),
        raft::device_span<vertex_t>(tmp_dsts.data(), tmp_dsts.size()),
        compressed_v_size);
      edge_partition_edgelist_dsts.push_back(std::move(tmp_dsts));
      (*edge_partition_edgelist_compressed_dsts)[i].resize(0, handle.get_stream());
      (*edge_partition_edgelist_compressed_dsts)[i].shrink_to_fit(handle.get_stream());
    }
  }

  return create_graph_from_partitioned_edgelist<vertex_t,
                                                edge_t,
                                                weight_t,
                                                edge_type_t,
                                                edge_time_t,
                                                store_transposed,
                                                multi_gpu>(
    handle,
    std::move(local_vertices),
    std::move(edge_partition_edgelist_srcs),
    std::move(edge_partition_edgelist_dsts),
    std::move(edge_partition_edgelist_weights),
    std::move(edge_partition_edgelist_edge_ids),
    std::move(edge_partition_edgelist_edge_types),
    std::move(edge_partition_edgelist_edge_start_times),
    std::move(edge_partition_edgelist_edge_end_times),
    edge_partition_intra_partition_segment_offset_vectors,
    graph_properties,
    renumber);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_end_times,
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
  CUGRAPH_EXPECTS(
    !edgelist_edge_start_times || (edgelist_srcs.size() == (*edgelist_edge_start_times).size()),
    "Invalid input arguments: edgelist_srcs.size() != "
    "(*edgelist_edge_start_times).size().");
  CUGRAPH_EXPECTS(
    !edgelist_edge_end_times || (edgelist_srcs.size() == (*edgelist_edge_end_times).size()),
    "Invalid input arguments: edgelist_srcs.size() != "
    "(*edgelist_edge_end_times).size().");

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
        "Invalid input arguments: graph_properties.is_symmetric is true but the input edge "
        "list is "
        "not symmetric.");
    }

    if (!graph_properties.is_multigraph) {
      CUGRAPH_EXPECTS(
        check_no_parallel_edge(
          handle,
          raft::device_span<vertex_t const>(edgelist_srcs.data(), edgelist_srcs.size()),
          raft::device_span<vertex_t const>(edgelist_dsts.data(), edgelist_dsts.size())),
        "Invalid input arguments: graph_properties.is_multigraph is false but the input edge "
        "list "
        "has parallel edges.");
    }
  }

  // 1. renumber
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
      if (edgelist_srcs.size() > 0) {
        num_vertices = 1 + cugraph::detail::compute_maximum_vertex_id(
                             handle.get_stream(), edgelist_srcs, edgelist_dsts);
      } else {
        num_vertices = 0;
      }
    }
  }

  // 2. convert edge list (COO) to compressed sparse format (CSR or CSC)
  int edge_property_count = 0;
  size_t element_size     = sizeof(vertex_t) * 2;

  if (edgelist_weights) {
    ++edge_property_count;
    element_size += sizeof(weight_t);
  }

  if (edgelist_edge_ids) {
    ++edge_property_count;
    element_size += sizeof(edge_t);
  }
  if (edgelist_edge_types) {
    ++edge_property_count;
    element_size += sizeof(edge_type_t);
  }
  if (edgelist_edge_start_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }
  if (edgelist_edge_end_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }

  if (edge_property_count > 1) { element_size = sizeof(vertex_t) * 2 + sizeof(size_t); }

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto constexpr mem_frugal_ratio =
    0.25;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  rmm::device_uvector<edge_t> offsets(size_t{0}, handle.get_stream());
  rmm::device_uvector<vertex_t> indices(size_t{0}, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> ids{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> types{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> start_times{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> end_times{std::nullopt};

  if (edge_property_count == 0) {
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
  } else if (edge_property_count == 1) {
    if (edgelist_weights) {
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
    } else if (edgelist_edge_ids) {
      std::forward_as_tuple(offsets, indices, ids, std::ignore) =
        detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_t, store_transposed>(
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
    } else if (edgelist_edge_types) {
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
    } else if (edgelist_edge_start_times) {
      std::forward_as_tuple(offsets, indices, start_times, std::ignore) =
        detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_time_t, store_transposed>(
          std::move(edgelist_srcs),
          std::move(edgelist_dsts),
          std::move(*edgelist_edge_start_times),
          vertex_t{0},
          std::optional<vertex_t>{std::nullopt},
          num_vertices,
          vertex_t{0},
          num_vertices,
          mem_frugal_threshold,
          handle.get_stream());
    } else if (edgelist_edge_end_times) {
      std::forward_as_tuple(offsets, indices, end_times, std::ignore) =
        detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_time_t, store_transposed>(
          std::move(edgelist_srcs),
          std::move(edgelist_dsts),
          std::move(*edgelist_edge_end_times),
          vertex_t{0},
          std::optional<vertex_t>{std::nullopt},
          num_vertices,
          vertex_t{0},
          num_vertices,
          mem_frugal_threshold,
          handle.get_stream());
    }
  } else {
    rmm::device_uvector<edge_t> property_position(edgelist_srcs.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), edge_t{0});

    std::tie(offsets, indices, property_position, std::ignore) =
      detail::sort_and_compress_edgelist<vertex_t, edge_t, edge_t, store_transposed>(
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::move(property_position),
        vertex_t{0},
        std::optional<vertex_t>{std::nullopt},
        num_vertices,
        vertex_t{0},
        num_vertices,
        mem_frugal_threshold,
        handle.get_stream());

    if (edgelist_weights) {
      weights = std::make_optional<rmm::device_uvector<weight_t>>(property_position.size(),
                                                                  handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_weights->begin(),
                     weights->begin());
    }
    if (edgelist_edge_ids) {
      ids = std::make_optional<rmm::device_uvector<edge_t>>(property_position.size(),
                                                            handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_ids->begin(),
                     ids->begin());
    }
    if (edgelist_edge_types) {
      types = std::make_optional<rmm::device_uvector<edge_type_t>>(property_position.size(),
                                                                   handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_types->begin(),
                     types->begin());
    }
    if (edgelist_edge_start_times) {
      start_times = std::make_optional<rmm::device_uvector<edge_time_t>>(property_position.size(),
                                                                         handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_start_times->begin(),
                     start_times->begin());
    }
    if (edgelist_edge_end_times) {
      end_times = std::make_optional<rmm::device_uvector<edge_time_t>>(property_position.size(),
                                                                       handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edgelist_edge_end_times->begin(),
                     end_times->begin());
    }
  }

  // 3. create a graph and an edge_property_t object.

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
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>
    edge_ids{std::nullopt};
  if (ids) {
    std::vector<rmm::device_uvector<edge_t>> buffers{};
    buffers.push_back(std::move(*ids));
    edge_ids = edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>(
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

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>
    edge_start_times{std::nullopt};
  if (start_times) {
    std::vector<rmm::device_uvector<edge_time_t>> buffers{};
    buffers.push_back(std::move(*start_times));
    edge_start_times =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>(
        std::move(buffers));
  }

  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>
    edge_end_times{std::nullopt};
  if (end_times) {
    std::vector<rmm::device_uvector<edge_time_t>> buffers{};
    buffers.push_back(std::move(*end_times));
    edge_end_times =
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>(
        std::move(buffers));
  }

  // 4. graph_t constructor

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(
      handle,
      std::move(offsets),
      std::move(indices),
      cugraph::graph_meta_t<vertex_t, edge_t, multi_gpu>{
        num_vertices,
        graph_properties,
        renumber ? std::optional<std::vector<vertex_t>>{meta.segment_offsets} : std::nullopt,
        meta.hypersparse_degree_offsets}),
    std::move(edge_weights),
    std::move(edge_ids),
    std::move(edge_types),
    std::move(edge_start_times),
    std::move(edge_end_times),
    std::move(renumber_map_labels));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<
    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_end_times,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(edgelist_srcs.size() == edgelist_dsts.size(),
                  "Invalid input arguments: edgelist_srcs.size() != edgelist_dsts.size().");
  CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs.size() == (*edgelist_weights).size()),
                  "Invalid input arguments: edgelist_weights.has_value() is true and "
                  "edgelist_srcs.size() != (*edgelist_weights).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_ids || (edgelist_srcs.size() == (*edgelist_edge_ids).size()),
                  "Invalid input arguments: edgelist_edge_ids.has_value() is true and "
                  "edgelist_srcs.size() != (*edgelist_edge_ids).size().");
  CUGRAPH_EXPECTS(!edgelist_edge_types || (edgelist_srcs.size() == (*edgelist_edge_types).size()),
                  "Invalid input arguments: edgelist_edge_types.has_value() is true, "
                  "edgelist_srcs.size() != (*edgelist_edge_types).size().");
  CUGRAPH_EXPECTS(
    !edgelist_edge_start_times || (edgelist_srcs.size() == (*edgelist_edge_start_times).size()),
    "Invalid input arguments: edgelist_edge_start_times.has_value() is true, "
    "edgelist_srcs.size() != (*edgelist_edge_start_times).size().");
  CUGRAPH_EXPECTS(
    !edgelist_edge_end_times || (edgelist_srcs.size() == (*edgelist_edge_end_times).size()),
    "Invalid input arguments: edgelist_edge_end_times.has_value() is true, "
    "edgelist_srcs.size() != (*edgelist_edge_end_times).size().");
  for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
    CUGRAPH_EXPECTS(edgelist_srcs[i].size() == edgelist_dsts[i].size(),
                    "Invalid input arguments: edgelist_srcs[i].size() != edgelist_dsts[i].size().");
    CUGRAPH_EXPECTS(!edgelist_weights || (edgelist_srcs[i].size() == (*edgelist_weights)[i].size()),
                    "Invalid input arguments: edgelist_weights.has_value() is true and "
                    "edgelist_srcs[i].size() != (*edgelist_weights)[i].size().");
    CUGRAPH_EXPECTS(
      !edgelist_edge_ids || (edgelist_srcs[i].size() == (*edgelist_edge_ids)[i].size()),
      "Invalid input arguments: edgelist_edge_ids.has_value() is true and "
      "edgelist_srcs[i].size() != (*edgelist_edge_ids)[i].size().");
    CUGRAPH_EXPECTS(
      !edgelist_edge_types || (edgelist_srcs[i].size() == (*edgelist_edge_types)[i].size()),
      "Invalid input arguments: edgelist_edge_types.has_value() is true, "
      "edgelist_srcs[i].size() != (*edgelist_edge_types)[i].size().");
    CUGRAPH_EXPECTS(!edgelist_edge_start_times ||
                      (edgelist_srcs[i].size() == (*edgelist_edge_start_times)[i].size()),
                    "Invalid input arguments: edgelist_edge_start_times.has_value() is true, "
                    "edgelist_srcs[i].size() != (*edgelist_edge_start_times)[i].size().");
    CUGRAPH_EXPECTS(
      !edgelist_edge_end_times || (edgelist_srcs[i].size() == (*edgelist_edge_end_times)[i].size()),
      "Invalid input arguments: edgelist_edge_end_times.has_value() is true, "
      "edgelist_srcs[i].size() != (*edgelist_edge_end_times)[i].size().");
  }

  std::vector<edge_t> chunk_edge_counts(edgelist_srcs.size());
  for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
    chunk_edge_counts[i] = edgelist_srcs[i].size();
  }
  std::vector<edge_t> chunk_edge_displacements(chunk_edge_counts.size());
  std::exclusive_scan(chunk_edge_counts.begin(),
                      chunk_edge_counts.end(),
                      chunk_edge_displacements.begin(),
                      edge_t{0});
  auto aggregate_edge_count = chunk_edge_displacements.back() + chunk_edge_counts.back();

  rmm::device_uvector<vertex_t> aggregate_edgelist_srcs(aggregate_edge_count, handle.get_stream());
  for (size_t i = 0; i < edgelist_srcs.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_srcs[i].begin(),
                 edgelist_srcs[i].end(),
                 aggregate_edgelist_srcs.begin() + chunk_edge_displacements[i]);
    edgelist_srcs[i].resize(0, handle.get_stream());
    edgelist_srcs[i].shrink_to_fit(handle.get_stream());
  }
  edgelist_srcs.clear();

  rmm::device_uvector<vertex_t> aggregate_edgelist_dsts(aggregate_edge_count, handle.get_stream());
  for (size_t i = 0; i < edgelist_dsts.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_dsts[i].begin(),
                 edgelist_dsts[i].end(),
                 aggregate_edgelist_dsts.begin() + chunk_edge_displacements[i]);
    edgelist_dsts[i].resize(0, handle.get_stream());
    edgelist_dsts[i].shrink_to_fit(handle.get_stream());
  }
  edgelist_dsts.clear();

  auto aggregate_edgelist_weights =
    edgelist_weights
      ? std::make_optional<rmm::device_uvector<weight_t>>(aggregate_edge_count, handle.get_stream())
      : std::nullopt;
  if (aggregate_edgelist_weights) {
    for (size_t i = 0; i < (*edgelist_weights).size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_weights)[i].begin(),
                   (*edgelist_weights)[i].end(),
                   (*aggregate_edgelist_weights).begin() + chunk_edge_displacements[i]);
      (*edgelist_weights)[i].resize(0, handle.get_stream());
      (*edgelist_weights)[i].shrink_to_fit(handle.get_stream());
    }
    (*edgelist_weights).clear();
  }

  auto aggregate_edgelist_edge_ids =
    edgelist_edge_ids
      ? std::make_optional<rmm::device_uvector<edge_t>>(aggregate_edge_count, handle.get_stream())
      : std::nullopt;
  if (aggregate_edgelist_edge_ids) {
    for (size_t i = 0; i < (*edgelist_edge_ids).size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_edge_ids)[i].begin(),
                   (*edgelist_edge_ids)[i].end(),
                   (*aggregate_edgelist_edge_ids).begin() + chunk_edge_displacements[i]);
      (*edgelist_edge_ids)[i].resize(0, handle.get_stream());
      (*edgelist_edge_ids)[i].shrink_to_fit(handle.get_stream());
    }
    (*edgelist_edge_ids).clear();
  }

  auto aggregate_edgelist_edge_types = edgelist_edge_types
                                         ? std::make_optional<rmm::device_uvector<edge_type_t>>(
                                             aggregate_edge_count, handle.get_stream())
                                         : std::nullopt;
  if (aggregate_edgelist_edge_types) {
    for (size_t i = 0; i < (*edgelist_edge_types).size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_edge_types)[i].begin(),
                   (*edgelist_edge_types)[i].end(),
                   (*aggregate_edgelist_edge_types).begin() + chunk_edge_displacements[i]);
      (*edgelist_edge_types)[i].resize(0, handle.get_stream());
      (*edgelist_edge_types)[i].shrink_to_fit(handle.get_stream());
    }
    (*edgelist_edge_types).clear();
  }

  auto aggregate_edgelist_edge_start_times =
    edgelist_edge_start_times ? std::make_optional<rmm::device_uvector<edge_time_t>>(
                                  aggregate_edge_count, handle.get_stream())
                              : std::nullopt;
  if (aggregate_edgelist_edge_start_times) {
    for (size_t i = 0; i < (*edgelist_edge_start_times).size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_edge_start_times)[i].begin(),
                   (*edgelist_edge_start_times)[i].end(),
                   (*aggregate_edgelist_edge_start_times).begin() + chunk_edge_displacements[i]);
      (*edgelist_edge_start_times)[i].resize(0, handle.get_stream());
      (*edgelist_edge_start_times)[i].shrink_to_fit(handle.get_stream());
    }
    (*edgelist_edge_start_times).clear();
  }

  auto aggregate_edgelist_edge_end_times = edgelist_edge_end_times
                                             ? std::make_optional<rmm::device_uvector<edge_time_t>>(
                                                 aggregate_edge_count, handle.get_stream())
                                             : std::nullopt;
  if (aggregate_edgelist_edge_end_times) {
    for (size_t i = 0; i < (*edgelist_edge_end_times).size(); ++i) {
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_edge_end_times)[i].begin(),
                   (*edgelist_edge_end_times)[i].end(),
                   (*aggregate_edgelist_edge_end_times).begin() + chunk_edge_displacements[i]);
      (*edgelist_edge_end_times)[i].resize(0, handle.get_stream());
      (*edgelist_edge_end_times)[i].shrink_to_fit(handle.get_stream());
    }
    (*edgelist_edge_end_times).clear();
  }

  if (do_expensive_check) {
    expensive_check_edgelist<vertex_t, multi_gpu>(
      handle,
      local_vertices,
      store_transposed ? aggregate_edgelist_dsts : aggregate_edgelist_srcs,
      store_transposed ? aggregate_edgelist_srcs : aggregate_edgelist_dsts,
      renumber);

    if (graph_properties.is_symmetric) {
      CUGRAPH_EXPECTS((check_symmetric<vertex_t, store_transposed, multi_gpu>(
                        handle,
                        raft::device_span<vertex_t const>(aggregate_edgelist_srcs.data(),
                                                          aggregate_edgelist_srcs.size()),
                        raft::device_span<vertex_t const>(aggregate_edgelist_dsts.data(),
                                                          aggregate_edgelist_dsts.size()))),
                      "Invalid input arguments: graph_properties.is_symmetric is true but the "
                      "input edge list is "
                      "not symmetric.");
    }

    if (!graph_properties.is_multigraph) {
      CUGRAPH_EXPECTS(
        check_no_parallel_edge(handle,
                               raft::device_span<vertex_t const>(aggregate_edgelist_srcs.data(),
                                                                 aggregate_edgelist_srcs.size()),
                               raft::device_span<vertex_t const>(aggregate_edgelist_dsts.data(),
                                                                 aggregate_edgelist_dsts.size())),
        "Invalid input arguments: graph_properties.is_multigraph is false but "
        "the input edge list "
        "has parallel edges.");
    }
  }

  return create_graph_from_edgelist_impl<vertex_t,
                                         edge_t,
                                         weight_t,
                                         edge_type_t,
                                         edge_time_t,
                                         store_transposed,
                                         multi_gpu>(handle,
                                                    std::move(local_vertices),
                                                    std::move(aggregate_edgelist_srcs),
                                                    std::move(aggregate_edgelist_dsts),
                                                    std::move(aggregate_edgelist_weights),
                                                    std::move(aggregate_edgelist_edge_ids),
                                                    std::move(aggregate_edgelist_edge_types),
                                                    std::move(aggregate_edgelist_edge_start_times),
                                                    std::move(aggregate_edgelist_edge_end_times),
                                                    graph_properties,
                                                    renumber,
                                                    do_expensive_check);
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(raft::handle_t const& handle,
                           std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                           rmm::device_uvector<vertex_t>&& edgelist_srcs,
                           rmm::device_uvector<vertex_t>&& edgelist_dsts,
                           std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                           std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
                           std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                           graph_properties_t graph_properties,
                           bool renumber,
                           bool do_expensive_check)
{
  auto [graph, weights, edge_ids, edge_types, edge_start_times, edge_end_times, number_map] =
    create_graph_from_edgelist_impl<vertex_t,
                                    edge_t,
                                    weight_t,
                                    edge_type_t,
                                    int32_t,
                                    store_transposed,
                                    multi_gpu>(handle,
                                               std::move(vertices),
                                               std::move(edgelist_srcs),
                                               std::move(edgelist_dsts),
                                               std::move(edgelist_weights),
                                               std::move(edgelist_edge_ids),
                                               std::move(edgelist_edge_types),
                                               std::nullopt,
                                               std::nullopt,
                                               graph_properties,
                                               renumber,
                                               do_expensive_check);

  return std::make_tuple(std::move(graph),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(number_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edgelist_edge_types,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  auto [graph, weights, edge_ids, edge_types, edge_start_times, edge_end_times, number_map] =
    create_graph_from_edgelist_impl<vertex_t,
                                    edge_t,
                                    weight_t,
                                    edge_type_t,
                                    int32_t,
                                    store_transposed,
                                    multi_gpu>(handle,
                                               std::move(vertices),
                                               std::move(edgelist_srcs),
                                               std::move(edgelist_dsts),
                                               std::move(edgelist_weights),
                                               std::move(edgelist_edge_ids),
                                               std::move(edgelist_edge_types),
                                               std::nullopt,
                                               std::nullopt,
                                               graph_properties,
                                               renumber,
                                               do_expensive_check);

  return std::make_tuple(std::move(graph),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(number_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edgelist_edge_end_times,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  return create_graph_from_edgelist_impl<vertex_t,
                                         edge_t,
                                         weight_t,
                                         edge_type_t,
                                         edge_time_t,
                                         store_transposed,
                                         multi_gpu>(handle,
                                                    std::move(vertices),
                                                    std::move(edgelist_srcs),
                                                    std::move(edgelist_dsts),
                                                    std::move(edgelist_weights),
                                                    std::move(edgelist_edge_ids),
                                                    std::move(edgelist_edge_types),
                                                    std::move(edgelist_edge_start_times),
                                                    std::move(edgelist_edge_end_times),
                                                    graph_properties,
                                                    renumber,
                                                    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_type_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<vertex_t>>&& edgelist_dsts,
  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& edgelist_weights,
  std::optional<std::vector<rmm::device_uvector<edge_t>>>&& edgelist_edge_ids,
  std::optional<std::vector<rmm::device_uvector<edge_type_t>>>&& edgelist_edge_types,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_start_times,
  std::optional<std::vector<rmm::device_uvector<edge_time_t>>>&& edgelist_edge_end_times,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check)
{
  return create_graph_from_edgelist_impl<vertex_t,
                                         edge_t,
                                         weight_t,
                                         edge_type_t,
                                         edge_time_t,
                                         store_transposed,
                                         multi_gpu>(handle,
                                                    std::move(vertices),
                                                    std::move(edgelist_srcs),
                                                    std::move(edgelist_dsts),
                                                    std::move(edgelist_weights),
                                                    std::move(edgelist_edge_ids),
                                                    std::move(edgelist_edge_types),
                                                    std::move(edgelist_edge_start_times),
                                                    std::move(edgelist_edge_end_times),
                                                    graph_properties,
                                                    renumber,
                                                    do_expensive_check);
}

}  // namespace cugraph

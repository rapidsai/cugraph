/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "prims/kv_store.cuh"

#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <optional>

namespace cugraph {

namespace {

template <typename vertex_t, typename edge_type_t>
struct edge_order_t {
  cuda::std::optional<raft::device_span<size_t const>> edgelist_label_offsets{cuda::std::nullopt};
  cuda::std::optional<raft::device_span<edge_type_t const>> edgelist_edge_types{cuda::std::nullopt};
  cuda::std::optional<raft::device_span<int32_t const>> edgelist_hops{cuda::std::nullopt};
  raft::device_span<vertex_t const> edgelist_majors{};
  raft::device_span<vertex_t const> edgelist_minors{};

  __device__ bool operator()(size_t l_idx, size_t r_idx) const
  {
    if (edgelist_label_offsets) {
      auto l_label = cuda::std::distance((*edgelist_label_offsets).begin() + 1,
                                         thrust::upper_bound(thrust::seq,
                                                             (*edgelist_label_offsets).begin() + 1,
                                                             (*edgelist_label_offsets).end(),
                                                             (*edgelist_label_offsets)[0] + l_idx));
      auto r_label = cuda::std::distance((*edgelist_label_offsets).begin() + 1,
                                         thrust::upper_bound(thrust::seq,
                                                             (*edgelist_label_offsets).begin() + 1,
                                                             (*edgelist_label_offsets).end(),
                                                             (*edgelist_label_offsets)[0] + r_idx));
      if (l_label != r_label) { return l_label < r_label; }
    }

    if (edgelist_edge_types) {
      auto l_type = (*edgelist_edge_types)[l_idx];
      auto r_type = (*edgelist_edge_types)[r_idx];
      if (l_type != r_type) { return l_type < r_type; }
    }

    if (edgelist_hops) {
      auto l_hop = (*edgelist_hops)[l_idx];
      auto r_hop = (*edgelist_hops)[r_idx];
      if (l_hop != r_hop) { return l_hop < r_hop; }
    }

    auto l_major = edgelist_majors[l_idx];
    auto r_major = edgelist_majors[r_idx];
    if (l_major != r_major) { return l_major < r_major; }

    auto l_minor = edgelist_minors[l_idx];
    auto r_minor = edgelist_minors[r_idx];
    if (l_minor != r_minor) { return l_minor < r_minor; }

    return l_idx < r_idx;
  }
};

template <typename vertex_t>
struct is_first_triplet_in_run_t {
  cuda::std::optional<raft::device_span<size_t const>> edgelist_label_offsets{cuda::std::nullopt};
  cuda::std::optional<raft::device_span<int32_t const>> edgelist_hops{cuda::std::nullopt};
  raft::device_span<vertex_t const> edgelist_majors{};

  __device__ bool operator()(size_t i) const
  {
    if (i == 0) return true;
    if (edgelist_label_offsets) {
      auto prev_label =
        cuda::std::distance((*edgelist_label_offsets).begin() + 1,
                            thrust::upper_bound(thrust::seq,
                                                (*edgelist_label_offsets).begin() + 1,
                                                (*edgelist_label_offsets).end(),
                                                i - 1));
      auto this_label = cuda::std::distance(
        (*edgelist_label_offsets).begin() + 1,
        thrust::upper_bound(
          thrust::seq, (*edgelist_label_offsets).begin() + 1, (*edgelist_label_offsets).end(), i));
      if (this_label != prev_label) { return true; }
    }
    if (edgelist_hops) {
      auto prev_hop = (*edgelist_hops)[i - 1];
      auto this_hop = (*edgelist_hops)[i];
      if (this_hop != prev_hop) { return true; }
    }
    return edgelist_majors[i] != edgelist_majors[i - 1];
  }
};

template <typename label_index_t>
struct compute_label_index_t {
  raft::device_span<size_t const> edgelist_label_offsets{};

  __device__ label_index_t operator()(size_t i) const
  {
    return static_cast<label_index_t>(cuda::std::distance(
      edgelist_label_offsets.begin() + 1,
      thrust::upper_bound(
        thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), i)));
  }
};

template <typename label_index_t>
struct optionally_compute_label_index_t {
  cuda::std::optional<raft::device_span<size_t const>> edgelist_label_offsets{cuda::std::nullopt};

  __device__ label_index_t operator()(size_t i) const
  {
    return edgelist_label_offsets ? static_cast<label_index_t>(cuda::std::distance(
                                      (*edgelist_label_offsets).begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          (*edgelist_label_offsets).begin() + 1,
                                                          (*edgelist_label_offsets).end(),
                                                          i)))
                                  : label_index_t{0};
  }
};

template <typename label_index_t,
          typename vertex_t,
          typename vertex_type_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
void check_input_edges(raft::handle_t const& handle,
                       rmm::device_uvector<vertex_t> const& edgelist_majors,
                       rmm::device_uvector<vertex_t> const& edgelist_minors,
                       std::optional<rmm::device_uvector<weight_t>> const& edgelist_weights,
                       std::optional<rmm::device_uvector<edge_id_t>> const& edgelist_edge_ids,
                       std::optional<rmm::device_uvector<edge_type_t>> const& edgelist_edge_types,
                       std::optional<rmm::device_uvector<int32_t>> const& edgelist_hops,
                       std::optional<raft::device_span<vertex_t const>> seed_vertices,
                       std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
                       std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
                       std::optional<raft::device_span<vertex_t const>> vertex_type_offsets,
                       size_t num_labels,
                       size_t num_hops,
                       size_t num_vertex_types,
                       std::optional<size_t> num_edge_types,
                       bool do_expensive_check)
{
  CUGRAPH_EXPECTS(
    edgelist_majors.size() == edgelist_minors.size(),
    "Invalid input arguments: edgelist_srcs.size() and edgelist_dsts.size() should coincide.");
  CUGRAPH_EXPECTS(
    !edgelist_weights.has_value() || (edgelist_majors.size() == (*edgelist_weights).size()),
    "Invalid input arguments: if edgelist_weights is valid, (*edgelist_weights).size() and "
    "edgelist_(srcs|dsts).size() should coincide.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_ids.has_value() || (edgelist_majors.size() == (*edgelist_edge_ids).size()),
    "Invalid input arguments: if edgelist_edge_ids is valid, (*edgelist_edge_ids).size() and "
    "edgelist_(srcs|dsts).size() should coincide.");
  CUGRAPH_EXPECTS(
    !edgelist_edge_types.has_value() || (edgelist_majors.size() == (*edgelist_edge_types).size()),
    "Invalid input arguments: if edgelist_edge_types is valid, (*edgelist_edge_types).size() and "
    "edgelist_(srcs|dsts).size() should coincide.");
  CUGRAPH_EXPECTS(!edgelist_hops.has_value() || (edgelist_majors.size() == (*edgelist_hops).size()),
                  "Invalid input arguments: if edgelist_hops is valid, (*edgelist_hops).size() and "
                  "edgelist_(srcs|dsts).size() should coincide.");

  CUGRAPH_EXPECTS(
    !edgelist_label_offsets.has_value() || ((*edgelist_label_offsets).size() == num_labels + 1),
    "Invalid input arguments: if edgelist_label_offsets is valid, (*edgelist_label_offsets).size() "
    "(size of the offset array) should be num_labels + 1.");

  if (edgelist_majors.size() > 0) {
    static_assert(std::is_same_v<label_index_t, uint32_t>);
    CUGRAPH_EXPECTS((num_labels >= 1) && (num_labels <= std::numeric_limits<label_index_t>::max()),
                    "Invalid input arguments: num_labels should be a positive integer and the "
                    "current implementation assumes that the number of unique labels is no larger "
                    "than std::numeric_limits<uint32_t>::max().");
    CUGRAPH_EXPECTS((num_labels == 1) || edgelist_label_offsets.has_value(),
                    "Invalid input arguments: edgelist_label_offsets.has_value() should be true if "
                    "num_labels >= 2.");

    CUGRAPH_EXPECTS(
      (num_hops >= 1) && (num_hops <= std::numeric_limits<int32_t>::max()),
      "Invalid input arguments: num_hops should be a positive integer and the current "
      "implementation "
      "assumes that the number of hops is no larger than std::numeric_limits<int32_t>::max().");
    CUGRAPH_EXPECTS(
      (num_hops == 1) || edgelist_hops.has_value(),
      "Invalid input arguments: edgelist_hops.has_value() should be true if num_hops >= 2.");

    static_assert(std::is_same_v<vertex_type_t, uint32_t>);
    CUGRAPH_EXPECTS(
      (num_vertex_types >= 1) && (num_vertex_types <= std::numeric_limits<vertex_type_t>::max()),
      "Invalid input arguments: num_vertex_types should be a positive integer and the "
      "current implementation assumes that the number of vertex types is no larger "
      "than std::numeric_limits<uint32_t>::max().");
    CUGRAPH_EXPECTS((num_vertex_types == 1) || vertex_type_offsets.has_value(),
                    "Invalid input arguments: vertex_type_offsets.has_value() should be true if "
                    "num_vertex_types >= 2.");
  }

  CUGRAPH_EXPECTS((!seed_vertices.has_value() && !seed_vertex_label_offsets.has_value()) ||
                    (seed_vertices.has_value() &&
                     (edgelist_label_offsets.has_value() == seed_vertex_label_offsets.has_value())),
                  "Invaild input arguments: if seed_vertices.has_value() is false, "
                  "seed_vertex_label_offsets.has_value() should be false as well. If "
                  "seed_vertices.has_value( ) is true, seed_vertex_label_offsets.has_value() "
                  "should coincide with edgelist_label_offsets.has_value().");
  CUGRAPH_EXPECTS(
    !seed_vertex_label_offsets.has_value() ||
      ((*seed_vertex_label_offsets).size() == num_labels + 1),
    "Invalid input arguments: if seed_vertex_label_offsets is valid, "
    "(*seed_vertex_label_offsets).size() (size of the offset array) should be num_labels + 1.");

  if (do_expensive_check) {
    if (edgelist_label_offsets) {
      CUGRAPH_EXPECTS(thrust::is_sorted(handle.get_thrust_policy(),
                                        (*edgelist_label_offsets).begin(),
                                        (*edgelist_label_offsets).end()),
                      "Invalid input arguments: if edgelist_label_offsets is valid, "
                      "*edgelist_label_offsets should be sorted.");
      size_t front_element{};
      raft::update_host(
        &front_element, (*edgelist_label_offsets).data(), size_t{1}, handle.get_stream());
      size_t back_element{};
      raft::update_host(&back_element,
                        (*edgelist_label_offsets).data() + num_labels,
                        size_t{1},
                        handle.get_stream());
      handle.sync_stream();
      CUGRAPH_EXPECTS(
        front_element == size_t{0},
        "Invalid input arguments: if edgelist_label_offsets is valid, the first element of "
        "*edgelist_label_offsets should be 0.");
      CUGRAPH_EXPECTS(
        back_element == edgelist_majors.size(),
        "Invalid input arguments: if edgelist_label_offsets is valid, the last element of "
        "*edgelist_label_offsets and edgelist_(srcs|dsts).size() should coincide.");
    }

    if (edgelist_edge_types && num_edge_types) {
      CUGRAPH_EXPECTS(
        thrust::count_if(handle.get_thrust_policy(),
                         (*edgelist_edge_types).begin(),
                         (*edgelist_edge_types).end(),
                         [num_edge_types = static_cast<edge_type_t>(*num_edge_types)] __device__(
                           edge_type_t edge_type) { return edge_type >= num_edge_types; }) == 0,
        "Invalid input arguments: edgelist_edge_type is valid but contains out-of-range edge type "
        "values.");
      if constexpr (std::is_signed_v<edge_type_t>) {
        CUGRAPH_EXPECTS(thrust::count_if(handle.get_thrust_policy(),
                                         (*edgelist_edge_types).begin(),
                                         (*edgelist_edge_types).end(),
                                         [] __device__(edge_type_t edge_type) {
                                           return edge_type < edge_type_t{0};
                                         }) == 0,
                        "Invalid input arguments: edgelist_edge_type is valid but contains "
                        "negative edge type values.");
      }
    }

    if (vertex_type_offsets) {
      CUGRAPH_EXPECTS(
        thrust::is_sorted(
          handle.get_thrust_policy(), (*vertex_type_offsets).begin(), (*vertex_type_offsets).end()),
        "Invalid input arguments: if vertex_type_offsets is valid, "
        "*vertex_type_offsets should be sorted.");
      vertex_t front_element{};
      raft::update_host(
        &front_element, (*vertex_type_offsets).data(), size_t{1}, handle.get_stream());
      vertex_t back_element{};
      raft::update_host(&back_element,
                        (*vertex_type_offsets).data() + num_vertex_types,
                        size_t{1},
                        handle.get_stream());
      handle.sync_stream();
      CUGRAPH_EXPECTS(
        front_element == vertex_t{0},
        "Invalid input arguments: if vertex_type_offsets is valid, the first element of "
        "*vertex_type_offsets should be 0.");
      vertex_t max_v = std::max(thrust::reduce(handle.get_thrust_policy(),
                                               edgelist_majors.begin(),
                                               edgelist_majors.end(),
                                               vertex_t{0},
                                               thrust::maximum<vertex_t>{}),
                                thrust::reduce(handle.get_thrust_policy(),
                                               edgelist_minors.begin(),
                                               edgelist_minors.end(),
                                               vertex_t{0},
                                               thrust::maximum<vertex_t>{}));
      CUGRAPH_EXPECTS(
        back_element > max_v,
        "Invalid input arguments: if vertex_type_offsets is valid, the last element of "
        "*vertex_type_offsets should be larger than the maximum vertex ID in edgelist_majors & "
        "edgelist_minors.");

      rmm::device_uvector<vertex_t> tmp_majors(edgelist_majors.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> tmp_minors(edgelist_minors.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_majors.begin(),
                   edgelist_majors.end(),
                   tmp_majors.begin());
      thrust::copy(handle.get_thrust_policy(),
                   edgelist_minors.begin(),
                   edgelist_minors.end(),
                   tmp_minors.begin());
      if (edgelist_edge_types) {
        rmm::device_uvector<edge_type_t> tmp_edge_types((*edgelist_edge_types).size(),
                                                        handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     (*edgelist_edge_types).begin(),
                     (*edgelist_edge_types).end(),
                     tmp_edge_types.begin());
        auto triplet_first =
          thrust::make_zip_iterator(tmp_edge_types.begin(), tmp_majors.begin(), tmp_minors.begin());
        thrust::sort(handle.get_thrust_policy(), triplet_first, triplet_first + tmp_majors.size());
        CUGRAPH_EXPECTS(
          thrust::count_if(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(tmp_majors.size()),
            [vertex_type_offsets = *vertex_type_offsets, triplet_first] __device__(size_t i) {
              if (i > 0) {
                auto prev = *(triplet_first + i - 1);
                auto cur  = *(triplet_first + i);
                if (thrust::get<0>(prev) == thrust::get<0>(cur)) {  // same edge type
                  auto prev_major_v_type =
                    cuda::std::distance(vertex_type_offsets.begin() + 1,
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_type_offsets.begin() + 1,
                                                            vertex_type_offsets.end(),
                                                            thrust::get<1>(prev)));
                  auto cur_major_v_type =
                    cuda::std::distance(vertex_type_offsets.begin() + 1,
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_type_offsets.begin() + 1,
                                                            vertex_type_offsets.end(),
                                                            thrust::get<1>(cur)));
                  if (prev_major_v_type != cur_major_v_type) { return true; }
                  auto prev_minor_v_type =
                    cuda::std::distance(vertex_type_offsets.begin() + 1,
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_type_offsets.begin() + 1,
                                                            vertex_type_offsets.end(),
                                                            thrust::get<2>(prev)));
                  auto cur_minor_v_type =
                    cuda::std::distance(vertex_type_offsets.begin() + 1,
                                        thrust::upper_bound(thrust::seq,
                                                            vertex_type_offsets.begin() + 1,
                                                            vertex_type_offsets.end(),
                                                            thrust::get<2>(cur)));
                  if (prev_minor_v_type != cur_minor_v_type) { return true; }
                }
              }
              return false;
            }) == 0,
          "Invalid input arguments: if vertex_type_offsets and edgelist_edge_types are valid, the "
          "entire set of input edge source vertices for each edge type should have an identical "
          "vertex type, and the entire set of input edge destination vertices for each type should "
          "have an identical vertex type.");
      } else {
        auto pair_first = thrust::make_zip_iterator(tmp_majors.begin(), tmp_minors.begin());
        thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + tmp_majors.size());
        CUGRAPH_EXPECTS(
          thrust::count_if(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(tmp_majors.size()),
            [vertex_type_offsets = *vertex_type_offsets, pair_first] __device__(size_t i) {
              if (i > 0) {
                auto prev = *(pair_first + i - 1);
                auto cur  = *(pair_first + i);
                auto prev_src_v_type =
                  cuda::std::distance(vertex_type_offsets.begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          vertex_type_offsets.begin() + 1,
                                                          vertex_type_offsets.end(),
                                                          thrust::get<0>(prev)));
                auto cur_src_v_type =
                  cuda::std::distance(vertex_type_offsets.begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          vertex_type_offsets.begin() + 1,
                                                          vertex_type_offsets.end(),
                                                          thrust::get<0>(cur)));
                if (prev_src_v_type != cur_src_v_type) { return true; }
                auto prev_dst_v_type =
                  cuda::std::distance(vertex_type_offsets.begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          vertex_type_offsets.begin() + 1,
                                                          vertex_type_offsets.end(),
                                                          thrust::get<1>(prev)));
                auto cur_dst_v_type =
                  cuda::std::distance(vertex_type_offsets.begin() + 1,
                                      thrust::upper_bound(thrust::seq,
                                                          vertex_type_offsets.begin() + 1,
                                                          vertex_type_offsets.end(),
                                                          thrust::get<1>(cur)));
                if (prev_dst_v_type != cur_dst_v_type) { return true; }
              }
              return false;
            }) == 0,
          "Invalid input arguments: if vertex_type_offsets is valid (but "
          "edgelist_edge_types is invalid), the entire set of input edge source "
          "vertices should have an identical vertex type, and the entire set of "
          "input edge destination vertices should have an identical vertex type.");
      }
    }

    if (seed_vertices) {
      for (size_t i = 0; i < num_labels; ++i) {
        rmm::device_uvector<vertex_t> this_label_seed_vertices(0, handle.get_stream());
        {
          size_t start_offset{0};
          auto end_offset = (*seed_vertices).size();
          if (seed_vertex_label_offsets) {
            raft::update_host(
              &start_offset, (*seed_vertex_label_offsets).data() + i, 1, handle.get_stream());
            raft::update_host(
              &end_offset, (*seed_vertex_label_offsets).data() + (i + 1), 1, handle.get_stream());
            handle.sync_stream();
          }
          this_label_seed_vertices.resize(end_offset - start_offset, handle.get_stream());
          thrust::copy(handle.get_thrust_policy(),
                       (*seed_vertices).begin() + start_offset,
                       (*seed_vertices).begin() + end_offset,
                       this_label_seed_vertices.begin());
          thrust::sort(handle.get_thrust_policy(),
                       this_label_seed_vertices.begin(),
                       this_label_seed_vertices.end());
          this_label_seed_vertices.resize(
            cuda::std::distance(this_label_seed_vertices.begin(),
                                thrust::unique(handle.get_thrust_policy(),
                                               this_label_seed_vertices.begin(),
                                               this_label_seed_vertices.end())),
            handle.get_stream());
        }

        rmm::device_uvector<vertex_t> this_label_zero_hop_majors(0, handle.get_stream());
        {
          size_t start_offset{0};
          auto end_offset = edgelist_majors.size();
          if (edgelist_label_offsets) {
            raft::update_host(
              &start_offset, (*edgelist_label_offsets).data() + i, 1, handle.get_stream());
            raft::update_host(
              &end_offset, (*edgelist_label_offsets).data() + (i + 1), 1, handle.get_stream());
            handle.sync_stream();
          }
          this_label_zero_hop_majors.resize(end_offset - start_offset, handle.get_stream());
          if (edgelist_hops) {
            this_label_zero_hop_majors.resize(
              cuda::std::distance(this_label_zero_hop_majors.begin(),
                                  thrust::copy_if(handle.get_thrust_policy(),
                                                  edgelist_majors.begin() + start_offset,
                                                  edgelist_majors.begin() + end_offset,
                                                  (*edgelist_hops).begin() + start_offset,
                                                  this_label_zero_hop_majors.begin(),
                                                  detail::is_equal_t<int32_t>{0})),
              handle.get_stream());
          } else {
            thrust::copy(handle.get_thrust_policy(),
                         edgelist_majors.begin() + start_offset,
                         edgelist_majors.begin() + end_offset,
                         this_label_zero_hop_majors.begin());
          }
          thrust::sort(handle.get_thrust_policy(),
                       this_label_zero_hop_majors.begin(),
                       this_label_zero_hop_majors.end());
          this_label_zero_hop_majors.resize(
            cuda::std::distance(this_label_zero_hop_majors.begin(),
                                thrust::unique(handle.get_thrust_policy(),
                                               this_label_zero_hop_majors.begin(),
                                               this_label_zero_hop_majors.end())),
            handle.get_stream());
        }

        rmm::device_uvector<vertex_t> zero_hop_majors_minus_seed_vertices(
          this_label_zero_hop_majors.size(), handle.get_stream());
        CUGRAPH_EXPECTS(cuda::std::distance(
                          zero_hop_majors_minus_seed_vertices.begin(),
                          thrust::set_difference(handle.get_thrust_policy(),
                                                 this_label_zero_hop_majors.begin(),
                                                 this_label_zero_hop_majors.end(),
                                                 this_label_seed_vertices.begin(),
                                                 this_label_seed_vertices.end(),
                                                 zero_hop_majors_minus_seed_vertices.begin())) == 0,
                        "Invalid input arguments: if seed_vertices.has_value() is true, "
                        "seed_vertices should include all zero-hop majors.");
      }
    }
  }
}

// output sorted by (primary key:label_index, secondary key:vertex)
template <typename vertex_t, typename label_index_t>
std::tuple<std::optional<rmm::device_uvector<label_index_t>> /* label indices */,
           rmm::device_uvector<vertex_t> /* vertices */,
           std::optional<rmm::device_uvector<int32_t>> /* minimum hops for the vertices */,
           std::optional<rmm::device_uvector<size_t>> /* label offsets for the output */>
compute_min_hop_for_unique_label_vertex_pairs(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> edgelist_vertices,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets)
{
  auto approx_items_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
    (1 << 18) /* tuning parameter */;  // for segmented sort

  if (edgelist_label_offsets) {
    rmm::device_uvector<label_index_t> tmp_label_indices(0, handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_vertices(0, handle.get_stream());
    std::optional<rmm::device_uvector<int32_t>> tmp_hops{std::nullopt};

    auto [h_label_offsets, h_edge_offsets] =
      detail::compute_offset_aligned_element_chunks(handle,
                                                    *edgelist_label_offsets,
                                                    edgelist_vertices.size(),
                                                    approx_items_to_sort_per_iteration);
    auto num_chunks = h_label_offsets.size() - 1;

    if (edgelist_hops) {
      rmm::device_uvector<size_t> tmp_indices(edgelist_vertices.size(), handle.get_stream());
      thrust::sequence(
        handle.get_thrust_policy(), tmp_indices.begin(), tmp_indices.end(), size_t{0});

      // cub::DeviceSegmentedSort currently does not suuport thrust::tuple type keys, sorting in
      // chunks still helps in limiting the binary search range and improving memory locality
      for (size_t i = 0; i < num_chunks; ++i) {
        thrust::sort(
          handle.get_thrust_policy(),
          tmp_indices.begin() + h_edge_offsets[i],
          tmp_indices.begin() + h_edge_offsets[i + 1],
          [edgelist_label_offsets =
             raft::device_span<size_t const>((*edgelist_label_offsets).data() + h_label_offsets[i],
                                             (h_label_offsets[i + 1] - h_label_offsets[i]) + 1),
           edgelist_vertices,
           edgelist_hops = *edgelist_hops] __device__(size_t l_idx, size_t r_idx) {
            auto l_it = thrust::upper_bound(
              thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), l_idx);
            auto r_it = thrust::upper_bound(
              thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), r_idx);
            if (l_it != r_it) { return l_it < r_it; }

            auto l_vertex = edgelist_vertices[l_idx];
            auto r_vertex = edgelist_vertices[r_idx];
            if (l_vertex != r_vertex) { return l_vertex < r_vertex; }

            auto l_hop = edgelist_hops[l_idx];
            auto r_hop = edgelist_hops[r_idx];
            return l_hop < r_hop;
          });
      }

      tmp_indices.resize(
        cuda::std::distance(
          tmp_indices.begin(),
          thrust::unique(handle.get_thrust_policy(),
                         tmp_indices.begin(),
                         tmp_indices.end(),
                         [edgelist_label_offsets = *edgelist_label_offsets,
                          edgelist_vertices] __device__(size_t l_idx, size_t r_idx) {
                           auto l_it = thrust::upper_bound(thrust::seq,
                                                           edgelist_label_offsets.begin() + 1,
                                                           edgelist_label_offsets.end(),
                                                           l_idx);
                           auto r_it = thrust::upper_bound(thrust::seq,
                                                           edgelist_label_offsets.begin() + 1,
                                                           edgelist_label_offsets.end(),
                                                           r_idx);
                           if (l_it != r_it) { return false; }

                           auto l_vertex = edgelist_vertices[l_idx];
                           auto r_vertex = edgelist_vertices[r_idx];
                           return l_vertex == r_vertex;
                         })),
        handle.get_stream());

      tmp_label_indices.resize(tmp_indices.size(), handle.get_stream());
      tmp_vertices.resize(tmp_indices.size(), handle.get_stream());
      tmp_hops = rmm::device_uvector<int32_t>(tmp_indices.size(), handle.get_stream());

      auto triplet_first = thrust::make_transform_iterator(
        tmp_indices.begin(),
        cuda::proclaim_return_type<thrust::tuple<label_index_t, vertex_t, int32_t>>(
          [edgelist_label_offsets = *edgelist_label_offsets,
           edgelist_vertices,
           edgelist_hops = *edgelist_hops] __device__(size_t i) {
            auto label_idx = static_cast<label_index_t>(cuda::std::distance(
              edgelist_label_offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), i)));
            return thrust::make_tuple(label_idx, edgelist_vertices[i], edgelist_hops[i]);
          }));
      thrust::copy(handle.get_thrust_policy(),
                   triplet_first,
                   triplet_first + tmp_indices.size(),
                   thrust::make_zip_iterator(
                     tmp_label_indices.begin(), tmp_vertices.begin(), (*tmp_hops).begin()));
    } else {
      rmm::device_uvector<vertex_t> segment_sorted_vertices(edgelist_vertices.size(),
                                                            handle.get_stream());

      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
      for (size_t i = 0; i < num_chunks; ++i) {
        size_t tmp_storage_bytes{0};

        auto offset_first =
          thrust::make_transform_iterator((*edgelist_label_offsets).data() + h_label_offsets[i],
                                          detail::shift_left_t<size_t>{h_edge_offsets[i]});
        cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                           tmp_storage_bytes,
                                           edgelist_vertices.begin() + h_edge_offsets[i],
                                           segment_sorted_vertices.begin() + h_edge_offsets[i],
                                           h_edge_offsets[i + 1] - h_edge_offsets[i],
                                           h_label_offsets[i + 1] - h_label_offsets[i],
                                           offset_first,
                                           offset_first + 1,
                                           handle.get_stream());

        if (tmp_storage_bytes > d_tmp_storage.size()) {
          d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
        }

        cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                           tmp_storage_bytes,
                                           edgelist_vertices.begin() + h_edge_offsets[i],
                                           segment_sorted_vertices.begin() + h_edge_offsets[i],
                                           h_edge_offsets[i + 1] - h_edge_offsets[i],
                                           h_label_offsets[i + 1] - h_label_offsets[i],
                                           offset_first,
                                           offset_first + 1,
                                           handle.get_stream());
      }
      d_tmp_storage.resize(0, handle.get_stream());
      d_tmp_storage.shrink_to_fit(handle.get_stream());

      tmp_label_indices.resize(segment_sorted_vertices.size(), handle.get_stream());
      tmp_vertices.resize(segment_sorted_vertices.size(), handle.get_stream());

      auto input_pair_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<thrust::tuple<label_index_t, vertex_t>>(
          [edgelist_label_offsets = *edgelist_label_offsets,
           edgelist_vertices      = raft::device_span<vertex_t const>(
             segment_sorted_vertices.data(), segment_sorted_vertices.size())] __device__(size_t i) {
            auto label_idx = static_cast<label_index_t>(cuda::std::distance(
              edgelist_label_offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), i)));
            return thrust::make_tuple(label_idx, edgelist_vertices[i]);
          }));
      auto output_pair_first =
        thrust::make_zip_iterator(tmp_label_indices.begin(), tmp_vertices.begin());
      auto num_uniques =
        cuda::std::distance(output_pair_first,
                            thrust::unique_copy(handle.get_thrust_policy(),
                                                input_pair_first,
                                                input_pair_first + segment_sorted_vertices.size(),
                                                output_pair_first));
      tmp_label_indices.resize(num_uniques, handle.get_stream());
      tmp_vertices.resize(num_uniques, handle.get_stream());
      tmp_label_indices.shrink_to_fit(handle.get_stream());
      tmp_vertices.shrink_to_fit(handle.get_stream());
    }

    if (seed_vertices) {
      /* label segmented sort */

      rmm::device_uvector<vertex_t> segment_sorted_vertices((*seed_vertices).size(),
                                                            handle.get_stream());

      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
      size_t tmp_storage_bytes{0};

      cub::DeviceSegmentedSort::SortKeys(static_cast<void*>(nullptr),
                                         tmp_storage_bytes,
                                         (*seed_vertices).begin(),
                                         segment_sorted_vertices.begin(),
                                         (*seed_vertices).size(),
                                         (*seed_vertex_label_offsets).size() - 1,
                                         (*seed_vertex_label_offsets).begin(),
                                         (*seed_vertex_label_offsets).begin() + 1,
                                         handle.get_stream());

      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }

      cub::DeviceSegmentedSort::SortKeys(d_tmp_storage.data(),
                                         tmp_storage_bytes,
                                         (*seed_vertices).begin(),
                                         segment_sorted_vertices.begin(),
                                         (*seed_vertices).size(),
                                         (*seed_vertex_label_offsets).size() - 1,
                                         (*seed_vertex_label_offsets).begin(),
                                         (*seed_vertex_label_offsets).begin() + 1,
                                         handle.get_stream());

      /* enumerate unique (label, vertex) pairs */

      rmm::device_uvector<label_index_t> unique_seed_vertex_label_indices((*seed_vertices).size(),
                                                                          handle.get_stream());
      rmm::device_uvector<vertex_t> unique_seed_vertices((*seed_vertices).size(),
                                                         handle.get_stream());
      auto input_pair_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<thrust::tuple<label_index_t, vertex_t>>(
          [seed_vertex_label_offsets = *seed_vertex_label_offsets,
           seed_vertices             = raft::device_span<vertex_t const>(
             segment_sorted_vertices.data(), segment_sorted_vertices.size())] __device__(size_t i) {
            auto label_idx = static_cast<label_index_t>(
              cuda::std::distance(seed_vertex_label_offsets.begin() + 1,
                                  thrust::upper_bound(thrust::seq,
                                                      seed_vertex_label_offsets.begin() + 1,
                                                      seed_vertex_label_offsets.end(),
                                                      i)));
            return thrust::make_tuple(label_idx, seed_vertices[i]);
          }));
      auto output_pair_first = thrust::make_zip_iterator(unique_seed_vertex_label_indices.begin(),
                                                         unique_seed_vertices.begin());
      auto num_uniques =
        cuda::std::distance(output_pair_first,
                            thrust::unique_copy(handle.get_thrust_policy(),
                                                input_pair_first,
                                                input_pair_first + segment_sorted_vertices.size(),
                                                output_pair_first));
      unique_seed_vertex_label_indices.resize(
        cuda::std::distance(output_pair_first,
                            thrust::unique_copy(handle.get_thrust_policy(),
                                                input_pair_first,
                                                input_pair_first + segment_sorted_vertices.size(),
                                                output_pair_first)),
        handle.get_stream());
      unique_seed_vertices.resize(unique_seed_vertex_label_indices.size(), handle.get_stream());

      /* merge with the (label, vertex, min. hop) triplets from the edgelist */

      if (edgelist_hops) {
        auto triplet_from_edgelist_first = thrust::make_zip_iterator(
          tmp_label_indices.begin(), tmp_vertices.begin(), (*tmp_hops).begin());
        auto key_pair_from_seed_vertex_first = thrust::make_zip_iterator(
          unique_seed_vertex_label_indices.begin(), unique_seed_vertices.begin());
        thrust::for_each(
          handle.get_thrust_policy(),
          key_pair_from_seed_vertex_first,
          key_pair_from_seed_vertex_first + unique_seed_vertex_label_indices.size(),
          [triplet_from_edgelist_first,
           triplet_from_edgelist_last =
             triplet_from_edgelist_first + tmp_label_indices.size()] __device__(auto pair) {
            auto it = thrust::lower_bound(
              thrust::seq,
              triplet_from_edgelist_first,
              triplet_from_edgelist_last,
              thrust::make_tuple(thrust::get<0>(pair), thrust::get<1>(pair), int32_t{0}));
            if ((it != triplet_from_edgelist_last) &&
                (thrust::get<0>(*it) == thrust::get<0>(pair)) &&
                (thrust::get<1>(*it) == thrust::get<1>(pair))) {
              // update min. hop to 0
              if (thrust::get<2>(*it) != int32_t{0}) { thrust::get<2>(*it) = int32_t{0}; }
            }
          });

        unique_seed_vertex_label_indices.resize(
          cuda::std::distance(
            key_pair_from_seed_vertex_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              key_pair_from_seed_vertex_first,
              key_pair_from_seed_vertex_first + unique_seed_vertices.size(),
              [triplet_from_edgelist_first,
               triplet_from_edgelist_last =
                 triplet_from_edgelist_first + tmp_label_indices.size()] __device__(auto pair) {
                auto it = thrust::lower_bound(
                  thrust::seq,
                  triplet_from_edgelist_first,
                  triplet_from_edgelist_last,
                  thrust::make_tuple(thrust::get<0>(pair), thrust::get<1>(pair), int32_t{0}));
                return (it != triplet_from_edgelist_last) &&
                       (thrust::get<0>(*it) == thrust::get<0>(pair)) &&
                       (thrust::get<1>(*it) == thrust::get<1>(pair));
              })),
          handle.get_stream());
        unique_seed_vertices.resize(unique_seed_vertex_label_indices.size(), handle.get_stream());
        if (unique_seed_vertex_label_indices.size() > 0) {
          rmm::device_uvector<label_index_t> merged_label_indices(
            tmp_label_indices.size() + unique_seed_vertex_label_indices.size(),
            handle.get_stream());
          rmm::device_uvector<vertex_t> merged_vertices(merged_label_indices.size(),
                                                        handle.get_stream());
          rmm::device_uvector<int32_t> merged_hops(merged_label_indices.size(),
                                                   handle.get_stream());
          auto triplet_from_seed_vertex_first =
            thrust::make_zip_iterator(unique_seed_vertex_label_indices.begin(),
                                      unique_seed_vertices.begin(),
                                      thrust::make_constant_iterator(int32_t{0}));
          thrust::merge(
            handle.get_thrust_policy(),
            triplet_from_edgelist_first,
            triplet_from_edgelist_first + tmp_label_indices.size(),
            triplet_from_seed_vertex_first,
            triplet_from_seed_vertex_first + unique_seed_vertex_label_indices.size(),
            thrust::make_zip_iterator(
              merged_label_indices.begin(), merged_vertices.begin(), merged_hops.begin()));
          tmp_label_indices = std::move(merged_label_indices);
          tmp_vertices      = std::move(merged_vertices);
          tmp_hops          = std::move(merged_hops);
        }
      } else {
        auto pair_from_edgelist_first =
          thrust::make_zip_iterator(tmp_label_indices.begin(), tmp_vertices.begin());
        auto pair_from_seed_vertex_first = thrust::make_zip_iterator(
          unique_seed_vertex_label_indices.begin(), unique_seed_vertices.begin());
        unique_seed_vertex_label_indices.resize(
          cuda::std::distance(
            pair_from_seed_vertex_first,
            thrust::remove_if(
              handle.get_thrust_policy(),
              pair_from_seed_vertex_first,
              pair_from_seed_vertex_first + unique_seed_vertex_label_indices.size(),
              [pair_from_edgelist_first,
               pair_from_edgelist_last =
                 pair_from_edgelist_first + tmp_label_indices.size()] __device__(auto pair) {
                auto it = thrust::lower_bound(
                  thrust::seq, pair_from_edgelist_first, pair_from_edgelist_last, pair);
                return (it != pair_from_edgelist_last) && (*it == pair);
              })),
          handle.get_stream());
        unique_seed_vertices.resize(unique_seed_vertex_label_indices.size(), handle.get_stream());
        if (unique_seed_vertex_label_indices.size() > 0) {
          rmm::device_uvector<label_index_t> merged_label_indices(
            tmp_label_indices.size() + unique_seed_vertex_label_indices.size(),
            handle.get_stream());
          rmm::device_uvector<vertex_t> merged_vertices(merged_label_indices.size(),
                                                        handle.get_stream());
          pair_from_seed_vertex_first = thrust::make_zip_iterator(
            unique_seed_vertex_label_indices.begin(), unique_seed_vertices.begin());
          thrust::merge(
            handle.get_thrust_policy(),
            pair_from_edgelist_first,
            pair_from_edgelist_first + tmp_label_indices.size(),
            pair_from_seed_vertex_first,
            pair_from_seed_vertex_first + unique_seed_vertex_label_indices.size(),
            thrust::make_zip_iterator(merged_label_indices.begin(), merged_vertices.begin()));
          tmp_label_indices = std::move(merged_label_indices);
          tmp_vertices      = std::move(merged_vertices);
        }
      }
    }

    rmm::device_uvector<size_t> tmp_label_offsets((*edgelist_label_offsets).size(),
                                                  handle.get_stream());
    tmp_label_offsets.set_element_to_zero_async(0, handle.get_stream());
    thrust::upper_bound(handle.get_thrust_policy(),
                        tmp_label_indices.begin(),
                        tmp_label_indices.end(),
                        thrust::make_counting_iterator(size_t{0}),
                        thrust::make_counting_iterator(tmp_label_offsets.size() - 1),
                        tmp_label_offsets.begin() + 1);

    return std::make_tuple(std::move(tmp_label_indices),
                           std::move(tmp_vertices),
                           std::move(tmp_hops),
                           std::move(tmp_label_offsets));
  } else {
    rmm::device_uvector<vertex_t> tmp_vertices(edgelist_vertices.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_vertices.begin(),
                 edgelist_vertices.end(),
                 tmp_vertices.begin());
    std::optional<rmm::device_uvector<int32_t>> tmp_hops{std::nullopt};

    if (edgelist_hops) {
      tmp_hops = rmm::device_uvector<int32_t>((*edgelist_hops).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_hops).begin(),
                   (*edgelist_hops).end(),
                   (*tmp_hops).begin());

      auto pair_first = thrust::make_zip_iterator(
        tmp_vertices.begin(),
        (*tmp_hops).begin());  // vertex is a primary key, hop is a secondary key
      thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + tmp_vertices.size());
      tmp_vertices.resize(
        cuda::std::distance(tmp_vertices.begin(),
                            thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                                 tmp_vertices.begin(),
                                                                 tmp_vertices.end(),
                                                                 (*tmp_hops).begin()))),
        handle.get_stream());
      (*tmp_hops).resize(tmp_vertices.size(), handle.get_stream());
      tmp_vertices.shrink_to_fit(handle.get_stream());
      (*tmp_hops).shrink_to_fit(handle.get_stream());
    } else {
      thrust::sort(handle.get_thrust_policy(), tmp_vertices.begin(), tmp_vertices.end());
      tmp_vertices.resize(
        cuda::std::distance(
          tmp_vertices.begin(),
          thrust::unique(handle.get_thrust_policy(), tmp_vertices.begin(), tmp_vertices.end())),
        handle.get_stream());
      tmp_vertices.shrink_to_fit(handle.get_stream());
    }

    if (seed_vertices) {
      /* sort and enumerate unique verties */

      rmm::device_uvector<vertex_t> unique_seed_vertices((*seed_vertices).size(),
                                                         handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*seed_vertices).begin(),
                   (*seed_vertices).end(),
                   unique_seed_vertices.begin());
      thrust::sort(
        handle.get_thrust_policy(), unique_seed_vertices.begin(), unique_seed_vertices.end());
      unique_seed_vertices.resize(cuda::std::distance(unique_seed_vertices.begin(),
                                                      thrust::unique(handle.get_thrust_policy(),
                                                                     unique_seed_vertices.begin(),
                                                                     unique_seed_vertices.end())),
                                  handle.get_stream());

      /* merge with the (vertex, min. hop) pairs from the edgelist */

      if (edgelist_hops) {
        auto pair_from_edgelist_first =
          thrust::make_zip_iterator(tmp_vertices.begin(), (*tmp_hops).begin());
        thrust::for_each(handle.get_thrust_policy(),
                         unique_seed_vertices.begin(),
                         unique_seed_vertices.end(),
                         [pair_from_edgelist_first,
                          pair_from_edgelist_last =
                            pair_from_edgelist_first + tmp_vertices.size()] __device__(auto v) {
                           auto it = thrust::lower_bound(thrust::seq,
                                                         pair_from_edgelist_first,
                                                         pair_from_edgelist_last,
                                                         thrust::make_tuple(v, int32_t{0}));
                           if ((it != pair_from_edgelist_last) && (thrust::get<0>(*it) == v)) {
                             // update min. hop to 0
                             if (thrust::get<1>(*it) != int32_t{0}) {
                               thrust::get<1>(*it) = int32_t{0};
                             }
                           }
                         });

        unique_seed_vertices.resize(
          cuda::std::distance(
            unique_seed_vertices.begin(),
            thrust::remove_if(handle.get_thrust_policy(),
                              unique_seed_vertices.begin(),
                              unique_seed_vertices.end(),
                              [pair_from_edgelist_first,
                               pair_from_edgelist_last = pair_from_edgelist_first +
                                                         tmp_vertices.size()] __device__(auto v) {
                                auto it = thrust::lower_bound(thrust::seq,
                                                              pair_from_edgelist_first,
                                                              pair_from_edgelist_last,
                                                              thrust::make_tuple(v, int32_t{0}));
                                return (it != pair_from_edgelist_last) &&
                                       (thrust::get<0>(*it) == v);
                              })),
          handle.get_stream());
        if (unique_seed_vertices.size() > 0) {
          rmm::device_uvector<vertex_t> merged_vertices(
            tmp_vertices.size() + unique_seed_vertices.size(), handle.get_stream());
          rmm::device_uvector<int32_t> merged_hops(merged_vertices.size(), handle.get_stream());
          auto pair_from_seed_vertex_first = thrust::make_zip_iterator(
            unique_seed_vertices.begin(), thrust::make_constant_iterator(int32_t{0}));
          thrust::merge(handle.get_thrust_policy(),
                        pair_from_edgelist_first,
                        pair_from_edgelist_first + tmp_vertices.size(),
                        pair_from_seed_vertex_first,
                        pair_from_seed_vertex_first + unique_seed_vertices.size(),
                        thrust::make_zip_iterator(merged_vertices.begin(), merged_hops.begin()));
          tmp_vertices = std::move(merged_vertices);
          tmp_hops     = std::move(merged_hops);
        }
      } else {
        unique_seed_vertices.resize(
          cuda::std::distance(
            unique_seed_vertices.begin(),
            thrust::remove_if(handle.get_thrust_policy(),
                              unique_seed_vertices.begin(),
                              unique_seed_vertices.end(),
                              [tmp_vertices = raft::device_span<vertex_t const>(
                                 tmp_vertices.data(), tmp_vertices.size())] __device__(auto v) {
                                auto it = thrust::lower_bound(
                                  thrust::seq, tmp_vertices.begin(), tmp_vertices.end(), v);
                                return (it != tmp_vertices.end()) && (*it == v);
                              })),
          handle.get_stream());
        if (unique_seed_vertices.size() > 0) {
          rmm::device_uvector<vertex_t> merged_vertices(
            tmp_vertices.size() + unique_seed_vertices.size(), handle.get_stream());
          thrust::merge(handle.get_thrust_policy(),
                        tmp_vertices.begin(),
                        tmp_vertices.end(),
                        unique_seed_vertices.begin(),
                        unique_seed_vertices.end(),
                        merged_vertices.begin());
          tmp_vertices = std::move(merged_vertices);
        }
      }
    }

    return std::make_tuple(
      std::nullopt, std::move(tmp_vertices), std::move(tmp_hops), std::nullopt);
  }
}

// returns renumber map & optional (label, type) offsets
// indices are non-descedning)
template <typename vertex_t, typename vertex_type_t, typename label_index_t>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<size_t>>>
compute_vertex_renumber_map(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> edgelist_majors,
  raft::device_span<vertex_t const> edgelist_minors,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  std::optional<raft::device_span<vertex_t const>> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types)
{
  auto approx_items_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
    (1 << 20) /* tuning parameter */;  // for segmented sort

  auto [unique_label_major_pair_label_indices,
        unique_label_major_pair_vertices,
        unique_label_major_pair_hops,
        unique_label_major_pair_label_offsets] =
    compute_min_hop_for_unique_label_vertex_pairs<vertex_t, label_index_t>(
      handle,
      edgelist_majors,
      edgelist_hops,
      seed_vertices,
      seed_vertex_label_offsets,
      edgelist_label_offsets);

  auto [unique_label_minor_pair_label_indices,
        unique_label_minor_pair_vertices,
        unique_label_minor_pair_hops,
        unique_label_minor_pair_label_offsets] =
    compute_min_hop_for_unique_label_vertex_pairs<vertex_t, label_index_t>(
      handle, edgelist_minors, edgelist_hops, std::nullopt, std::nullopt, edgelist_label_offsets);

  rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> renumber_map_label_type_offsets{std::nullopt};
  if (edgelist_label_offsets) {
    rmm::device_uvector<label_index_t> renumber_map_label_indices(0, handle.get_stream());

    renumber_map.reserve((*unique_label_major_pair_label_indices).size() +
                           (*unique_label_minor_pair_label_indices).size(),
                         handle.get_stream());
    renumber_map_label_indices.reserve(renumber_map.capacity(), handle.get_stream());

    auto num_chunks = (edgelist_majors.size() + (approx_items_to_sort_per_iteration - 1)) /
                      approx_items_to_sort_per_iteration;
    auto chunk_size = (num_chunks > 0) ? ((num_labels + (num_chunks - 1)) / num_chunks) : 0;

    size_t copy_offset{0};
    for (size_t i = 0; i < num_chunks; ++i) {
      auto major_start_offset =
        (*unique_label_major_pair_label_offsets).element(chunk_size * i, handle.get_stream());
      auto major_end_offset =
        (*unique_label_major_pair_label_offsets)
          .element(std::min(chunk_size * (i + 1), num_labels), handle.get_stream());
      auto minor_start_offset =
        (*unique_label_minor_pair_label_offsets).element(chunk_size * i, handle.get_stream());
      auto minor_end_offset =
        (*unique_label_minor_pair_label_offsets)
          .element(std::min(chunk_size * (i + 1), num_labels), handle.get_stream());

      rmm::device_uvector<label_index_t> merged_label_indices(
        (major_end_offset - major_start_offset) + (minor_end_offset - minor_start_offset),
        handle.get_stream());
      rmm::device_uvector<vertex_t> merged_vertices(merged_label_indices.size(),
                                                    handle.get_stream());
      rmm::device_uvector<int8_t> merged_flags(merged_label_indices.size(), handle.get_stream());

      if (edgelist_hops) {
        rmm::device_uvector<int32_t> merged_hops(merged_label_indices.size(), handle.get_stream());
        auto major_quad_first =
          thrust::make_zip_iterator((*unique_label_major_pair_label_indices).begin(),
                                    unique_label_major_pair_vertices.begin(),
                                    (*unique_label_major_pair_hops).begin(),
                                    thrust::make_constant_iterator(int8_t{0}));
        auto minor_quad_first =
          thrust::make_zip_iterator((*unique_label_minor_pair_label_indices).begin(),
                                    unique_label_minor_pair_vertices.begin(),
                                    (*unique_label_minor_pair_hops).begin(),
                                    thrust::make_constant_iterator(int8_t{1}));
        thrust::merge(handle.get_thrust_policy(),
                      major_quad_first + major_start_offset,
                      major_quad_first + major_end_offset,
                      minor_quad_first + minor_start_offset,
                      minor_quad_first + minor_end_offset,
                      thrust::make_zip_iterator(merged_label_indices.begin(),
                                                merged_vertices.begin(),
                                                merged_hops.begin(),
                                                merged_flags.begin()));

        auto unique_key_first =
          thrust::make_zip_iterator(merged_label_indices.begin(), merged_vertices.begin());
        merged_label_indices.resize(
          cuda::std::distance(
            unique_key_first,
            thrust::get<0>(thrust::unique_by_key(
              handle.get_thrust_policy(),
              unique_key_first,
              unique_key_first + merged_label_indices.size(),
              thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin())))),
          handle.get_stream());
        merged_vertices.resize(merged_label_indices.size(), handle.get_stream());
        merged_hops.resize(merged_label_indices.size(), handle.get_stream());
        merged_flags.resize(merged_label_indices.size(), handle.get_stream());
        if (vertex_type_offsets) {
          auto quadraplet_first = thrust::make_zip_iterator(merged_label_indices.begin(),
                                                            merged_vertices.begin(),
                                                            merged_hops.begin(),
                                                            merged_flags.begin());
          thrust::sort(
            handle.get_thrust_policy(),
            quadraplet_first,
            quadraplet_first + merged_vertices.size(),
            [offsets = *vertex_type_offsets] __device__(auto lhs, auto rhs) {
              auto lhs_v_type = cuda::std::distance(
                offsets.begin() + 1,
                thrust::upper_bound(
                  thrust::seq, offsets.begin() + 1, offsets.end(), thrust::get<1>(lhs)));
              auto rhs_v_type = cuda::std::distance(
                offsets.begin() + 1,
                thrust::upper_bound(
                  thrust::seq, offsets.begin() + 1, offsets.end(), thrust::get<1>(rhs)));
              return thrust::make_tuple(
                       thrust::get<0>(lhs), lhs_v_type, thrust::get<2>(lhs), thrust::get<3>(lhs)) <
                     thrust::make_tuple(
                       thrust::get<0>(rhs), rhs_v_type, thrust::get<2>(rhs), thrust::get<3>(rhs));
            });
        } else {
          auto sort_key_first = thrust::make_zip_iterator(
            merged_label_indices.begin(), merged_hops.begin(), merged_flags.begin());
          thrust::sort_by_key(handle.get_thrust_policy(),
                              sort_key_first,
                              sort_key_first + merged_label_indices.size(),
                              merged_vertices.begin());
        }
      } else {
        auto major_triplet_first =
          thrust::make_zip_iterator((*unique_label_major_pair_label_indices).begin(),
                                    unique_label_major_pair_vertices.begin(),
                                    thrust::make_constant_iterator(int8_t{0}));
        auto minor_triplet_first =
          thrust::make_zip_iterator((*unique_label_minor_pair_label_indices).begin(),
                                    unique_label_minor_pair_vertices.begin(),
                                    thrust::make_constant_iterator(int8_t{1}));
        thrust::merge(
          handle.get_thrust_policy(),
          major_triplet_first + major_start_offset,
          major_triplet_first + major_end_offset,
          minor_triplet_first + minor_start_offset,
          minor_triplet_first + minor_end_offset,
          thrust::make_zip_iterator(
            merged_label_indices.begin(), merged_vertices.begin(), merged_flags.begin()));

        auto unique_key_first =
          thrust::make_zip_iterator(merged_label_indices.begin(), merged_vertices.begin());
        merged_label_indices.resize(
          cuda::std::distance(
            unique_key_first,
            thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                 unique_key_first,
                                                 unique_key_first + merged_label_indices.size(),
                                                 merged_flags.begin()))),
          handle.get_stream());
        merged_vertices.resize(merged_label_indices.size(), handle.get_stream());
        merged_flags.resize(merged_label_indices.size(), handle.get_stream());
        if (vertex_type_offsets) {
          auto triplet_first = thrust::make_zip_iterator(
            merged_label_indices.begin(), merged_vertices.begin(), merged_flags.begin());
          thrust::sort(
            handle.get_thrust_policy(),
            triplet_first,
            triplet_first + merged_vertices.size(),
            [offsets = *vertex_type_offsets] __device__(auto lhs, auto rhs) {
              auto lhs_v_type = cuda::std::distance(
                offsets.begin() + 1,
                thrust::upper_bound(
                  thrust::seq, offsets.begin() + 1, offsets.end(), thrust::get<1>(lhs)));
              auto rhs_v_type = cuda::std::distance(
                offsets.begin() + 1,
                thrust::upper_bound(
                  thrust::seq, offsets.begin() + 1, offsets.end(), thrust::get<1>(rhs)));
              return thrust::make_tuple(thrust::get<0>(lhs), lhs_v_type, thrust::get<2>(lhs)) <
                     thrust::make_tuple(thrust::get<0>(rhs), rhs_v_type, thrust::get<2>(rhs));
            });
        } else {
          auto sort_key_first =
            thrust::make_zip_iterator(merged_label_indices.begin(), merged_flags.begin());
          thrust::sort_by_key(handle.get_thrust_policy(),
                              sort_key_first,
                              sort_key_first + merged_label_indices.size(),
                              merged_vertices.begin());
        }
      }

      renumber_map.resize(copy_offset + merged_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   merged_vertices.begin(),
                   merged_vertices.end(),
                   renumber_map.begin() + copy_offset);
      renumber_map_label_indices.resize(copy_offset + merged_label_indices.size(),
                                        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   merged_label_indices.begin(),
                   merged_label_indices.end(),
                   renumber_map_label_indices.begin() + copy_offset);

      copy_offset += merged_vertices.size();
    }

    renumber_map.shrink_to_fit(handle.get_stream());
    renumber_map_label_indices.shrink_to_fit(handle.get_stream());

    renumber_map_label_type_offsets =
      rmm::device_uvector<size_t>(num_labels * num_vertex_types + 1, handle.get_stream());
    (*renumber_map_label_type_offsets).set_element_to_zero_async(0, handle.get_stream());
    if (vertex_type_offsets) {
      auto label_type_pair_first = thrust::make_zip_iterator(
        renumber_map_label_indices.begin(),
        thrust::make_transform_iterator(
          renumber_map.begin(),
          cuda::proclaim_return_type<vertex_type_t>(
            [offsets = *vertex_type_offsets] __device__(auto v) {
              return static_cast<vertex_type_t>(cuda::std::distance(
                offsets.begin() + 1,
                thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), v)));
            })));
      auto value_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<thrust::tuple<label_index_t, vertex_type_t>>(
          [num_vertex_types] __device__(size_t i) {
            return thrust::make_tuple(static_cast<label_index_t>(i / num_vertex_types),
                                      static_cast<vertex_type_t>(i % num_vertex_types));
          }));
      thrust::upper_bound(handle.get_thrust_policy(),
                          label_type_pair_first,
                          label_type_pair_first + renumber_map.size(),
                          value_first,
                          value_first + (num_labels * num_vertex_types),
                          (*renumber_map_label_type_offsets).begin() + 1);
    } else {
      thrust::upper_bound(handle.get_thrust_policy(),
                          renumber_map_label_indices.begin(),
                          renumber_map_label_indices.end(),
                          thrust::make_counting_iterator(label_index_t{0}),
                          thrust::make_counting_iterator(static_cast<label_index_t>(num_labels)),
                          (*renumber_map_label_type_offsets).begin() + 1);
    }
  } else {
    if (edgelist_hops) {
      rmm::device_uvector<vertex_t> merged_vertices(
        unique_label_major_pair_vertices.size() + unique_label_minor_pair_vertices.size(),
        handle.get_stream());
      rmm::device_uvector<int32_t> merged_hops(merged_vertices.size(), handle.get_stream());
      rmm::device_uvector<int8_t> merged_flags(merged_vertices.size(), handle.get_stream());
      auto major_triplet_first =
        thrust::make_zip_iterator(unique_label_major_pair_vertices.begin(),
                                  (*unique_label_major_pair_hops).begin(),
                                  thrust::make_constant_iterator(int8_t{0}));
      auto minor_triplet_first =
        thrust::make_zip_iterator(unique_label_minor_pair_vertices.begin(),
                                  (*unique_label_minor_pair_hops).begin(),
                                  thrust::make_constant_iterator(int8_t{1}));
      thrust::merge(handle.get_thrust_policy(),
                    major_triplet_first,
                    major_triplet_first + unique_label_major_pair_vertices.size(),
                    minor_triplet_first,
                    minor_triplet_first + unique_label_minor_pair_vertices.size(),
                    thrust::make_zip_iterator(
                      merged_vertices.begin(), merged_hops.begin(), merged_flags.begin()));

      unique_label_major_pair_vertices.resize(0, handle.get_stream());
      unique_label_major_pair_vertices.shrink_to_fit(handle.get_stream());
      unique_label_major_pair_hops = std::nullopt;
      unique_label_minor_pair_vertices.resize(0, handle.get_stream());
      unique_label_minor_pair_vertices.shrink_to_fit(handle.get_stream());
      unique_label_minor_pair_hops = std::nullopt;

      merged_vertices.resize(
        cuda::std::distance(
          merged_vertices.begin(),
          thrust::get<0>(thrust::unique_by_key(
            handle.get_thrust_policy(),
            merged_vertices.begin(),
            merged_vertices.end(),
            thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin())))),
        handle.get_stream());
      merged_hops.resize(merged_vertices.size(), handle.get_stream());
      merged_flags.resize(merged_vertices.size(), handle.get_stream());

      if (vertex_type_offsets) {
        auto triplet_first = thrust::make_zip_iterator(
          merged_vertices.begin(), merged_hops.begin(), merged_flags.begin());
        thrust::sort(
          handle.get_thrust_policy(),
          triplet_first,
          triplet_first + merged_vertices.size(),
          [offsets = *vertex_type_offsets] __device__(auto lhs, auto rhs) {
            auto lhs_v_type = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, offsets.begin() + 1, offsets.end(), thrust::get<0>(lhs)));
            auto rhs_v_type = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, offsets.begin() + 1, offsets.end(), thrust::get<0>(rhs)));
            return thrust::make_tuple(lhs_v_type, thrust::get<1>(lhs), thrust::get<2>(lhs)) <
                   thrust::make_tuple(rhs_v_type, thrust::get<1>(rhs), thrust::get<2>(rhs));
          });
      } else {
        auto sort_key_first = thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin());
        thrust::sort_by_key(handle.get_thrust_policy(),
                            sort_key_first,
                            sort_key_first + merged_hops.size(),
                            merged_vertices.begin());
      }

      renumber_map = std::move(merged_vertices);
    } else {
      rmm::device_uvector<vertex_t> output_vertices(unique_label_minor_pair_vertices.size(),
                                                    handle.get_stream());
      auto output_last = thrust::set_difference(handle.get_thrust_policy(),
                                                unique_label_minor_pair_vertices.begin(),
                                                unique_label_minor_pair_vertices.end(),
                                                unique_label_major_pair_vertices.begin(),
                                                unique_label_major_pair_vertices.end(),
                                                output_vertices.begin());

      auto num_unique_majors = unique_label_major_pair_vertices.size();
      renumber_map           = std::move(unique_label_major_pair_vertices);
      renumber_map.resize(
        renumber_map.size() + cuda::std::distance(output_vertices.begin(), output_last),
        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   output_vertices.begin(),
                   output_last,
                   renumber_map.begin() + num_unique_majors);

      if (vertex_type_offsets) {
        thrust::stable_sort(
          handle.get_thrust_policy(),
          renumber_map.begin(),
          renumber_map.end(),
          [offsets = *vertex_type_offsets] __device__(auto lhs, auto rhs) {
            auto lhs_v_type = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), lhs));
            auto rhs_v_type = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), rhs));
            return lhs_v_type < rhs_v_type;
          });
      }
    }

    if (vertex_type_offsets) {
      renumber_map_label_type_offsets =
        rmm::device_uvector<size_t>(num_vertex_types + 1, handle.get_stream());
      (*renumber_map_label_type_offsets).set_element_to_zero_async(0, handle.get_stream());
      auto type_first = thrust::make_transform_iterator(
        renumber_map.begin(),
        cuda::proclaim_return_type<vertex_type_t>(
          [offsets = *vertex_type_offsets] __device__(auto v) {
            return static_cast<vertex_type_t>(cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), v)));
          }));
      thrust::upper_bound(
        handle.get_thrust_policy(),
        type_first,
        type_first + renumber_map.size(),
        thrust::make_counting_iterator(vertex_type_t{0}),
        thrust::make_counting_iterator(static_cast<vertex_type_t>(num_vertex_types)),
        (*renumber_map_label_type_offsets).begin() + 1);
    }
  }

  return std::make_tuple(std::move(renumber_map), std::move(renumber_map_label_type_offsets));
}

// returns renumber map & optional (label, type) offsets
template <typename vertex_t, typename edge_id_t, typename edge_type_t, typename label_index_t>
std::tuple<rmm::device_uvector<edge_id_t>, std::optional<rmm::device_uvector<size_t>>>
compute_edge_id_renumber_map(
  raft::handle_t const& handle,
  raft::device_span<edge_id_t const> edgelist_edge_ids,
  std::optional<raft::device_span<edge_type_t const>> edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  size_t num_labels,
  size_t num_edge_types)
{
  rmm::device_uvector<edge_id_t> renumber_map(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> renumber_map_label_type_offsets{std::nullopt};
  if (edgelist_label_offsets) {
    auto approx_items_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 20) /* tuning parameter */;  // for segmented sort

    auto [h_label_offsets, h_edge_offsets] =
      detail::compute_offset_aligned_element_chunks(handle,
                                                    *edgelist_label_offsets,
                                                    edgelist_edge_ids.size(),
                                                    approx_items_to_sort_per_iteration);
    auto num_chunks = h_label_offsets.size() - 1;

    rmm::device_uvector<size_t> tmp_indices(edgelist_edge_ids.size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), tmp_indices.begin(), tmp_indices.end(), size_t{0});

    // cub::DeviceSegmentedSort currently does not suuport thrust::tuple type keys, sorting in
    // chunks still helps in limiting the binary search range and improving memory locality
    for (size_t i = 0; i < num_chunks; ++i) {
      // sort by (label, (type), id, (hop))

      thrust::sort(
        handle.get_thrust_policy(),
        tmp_indices.begin() + h_edge_offsets[i],
        tmp_indices.begin() + h_edge_offsets[i + 1],
        [edgelist_label_offsets =
           raft::device_span<size_t const>((*edgelist_label_offsets).data() + h_label_offsets[i],
                                           (h_label_offsets[i + 1] - h_label_offsets[i]) + 1),
         edgelist_edge_types = detail::to_thrust_optional(edgelist_edge_types),
         edgelist_edge_ids,
         edgelist_hops = detail::to_thrust_optional(edgelist_hops)] __device__(size_t l_idx,
                                                                               size_t r_idx) {
          auto l_it = thrust::upper_bound(
            thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), l_idx);
          auto r_it = thrust::upper_bound(
            thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), r_idx);
          if (l_it != r_it) { return l_it < r_it; }

          if (edgelist_edge_types) {
            auto l_type = (*edgelist_edge_types)[l_idx];
            auto r_type = (*edgelist_edge_types)[r_idx];
            if (l_type != r_type) { return l_type < r_type; }
          }

          auto l_id = edgelist_edge_ids[l_idx];
          auto r_id = edgelist_edge_ids[r_idx];
          if (l_id != r_id) { return l_id < r_id; }

          if (edgelist_hops) {
            auto l_hop = (*edgelist_hops)[l_idx];
            auto r_hop = (*edgelist_hops)[r_idx];
            return l_hop < r_hop;
          }

          return false;
        });

      // find unique (label, (type), id, (min_hop)) tuples

      auto last = thrust::unique(
        handle.get_thrust_policy(),
        tmp_indices.begin() + h_edge_offsets[i],
        tmp_indices.begin() + h_edge_offsets[i + 1],
        [edgelist_label_offsets = *edgelist_label_offsets,
         edgelist_edge_types    = detail::to_thrust_optional(edgelist_edge_types),
         edgelist_edge_ids] __device__(size_t l_idx, size_t r_idx) {
          auto l_it = thrust::upper_bound(
            thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), l_idx);
          auto r_it = thrust::upper_bound(
            thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), r_idx);
          if (l_it != r_it) { return false; }

          if (edgelist_edge_types) {
            auto l_type = (*edgelist_edge_types)[l_idx];
            auto r_type = (*edgelist_edge_types)[r_idx];
            if (l_type != r_type) { return false; }
          }

          auto l_id = edgelist_edge_ids[l_idx];
          auto r_id = edgelist_edge_ids[r_idx];
          return l_id == r_id;
        });

      // sort by (label, (type), (min_hop), id)

      if (edgelist_hops) {
        thrust::sort(
          handle.get_thrust_policy(),
          tmp_indices.begin() + h_edge_offsets[i],
          last,
          [edgelist_label_offsets =
             raft::device_span<size_t const>((*edgelist_label_offsets).data() + h_label_offsets[i],
                                             (h_label_offsets[i + 1] - h_label_offsets[i]) + 1),
           edgelist_edge_types = detail::to_thrust_optional(edgelist_edge_types),
           edgelist_edge_ids,
           edgelist_hops = detail::to_thrust_optional(edgelist_hops)] __device__(size_t l_idx,
                                                                                 size_t r_idx) {
            auto l_it = thrust::upper_bound(
              thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), l_idx);
            auto r_it = thrust::upper_bound(
              thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), r_idx);
            if (l_it != r_it) { return l_it < r_it; }

            if (edgelist_edge_types) {
              auto l_type = (*edgelist_edge_types)[l_idx];
              auto r_type = (*edgelist_edge_types)[r_idx];
              if (l_type != r_type) { return l_type < r_type; }
            }

            if (edgelist_hops) {
              auto l_hop = (*edgelist_hops)[l_idx];
              auto r_hop = (*edgelist_hops)[r_idx];
              return l_hop < r_hop;
            }

            auto l_id = edgelist_edge_ids[l_idx];
            auto r_id = edgelist_edge_ids[r_idx];
            if (l_id != r_id) { return l_id < r_id; }

            return false;
          });
      }

      // mark invalid indices

      thrust::fill(handle.get_thrust_policy(),
                   last,
                   tmp_indices.begin() + h_edge_offsets[i + 1],
                   std::numeric_limits<size_t>::max());
    }

    tmp_indices.resize(cuda::std::distance(tmp_indices.begin(),
                                           thrust::remove(handle.get_thrust_policy(),
                                                          tmp_indices.begin(),
                                                          tmp_indices.end(),
                                                          std::numeric_limits<size_t>::max())),
                       handle.get_stream());

    renumber_map = rmm::device_uvector<edge_id_t>(tmp_indices.size(), handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   tmp_indices.begin(),
                   tmp_indices.end(),
                   edgelist_edge_ids.begin(),
                   renumber_map.begin());

    renumber_map_label_type_offsets =
      rmm::device_uvector<size_t>(num_labels * num_edge_types + 1, handle.get_stream());
    (*renumber_map_label_type_offsets).set_element_to_zero_async(0, handle.get_stream());
    if (edgelist_edge_types) {
      auto label_type_pair_first = thrust::make_transform_iterator(
        tmp_indices.begin(),
        cuda::proclaim_return_type<thrust::tuple<label_index_t, edge_type_t>>(
          [edgelist_label_offsets = *edgelist_label_offsets,
           edgelist_edge_types    = *edgelist_edge_types] __device__(size_t i) {
            auto label_idx = cuda::std::distance(
              edgelist_label_offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), i));
            return thrust::make_tuple(static_cast<label_index_t>(label_idx),
                                      edgelist_edge_types[i]);
          }));
      auto value_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<thrust::tuple<label_index_t, edge_type_t>>(
          [num_edge_types] __device__(size_t i) {
            return thrust::make_tuple(static_cast<label_index_t>(i / num_edge_types),
                                      static_cast<edge_type_t>(i % num_edge_types));
          }));
      thrust::upper_bound(handle.get_thrust_policy(),
                          label_type_pair_first,
                          label_type_pair_first + renumber_map.size(),
                          value_first,
                          value_first + (num_labels * num_edge_types),
                          (*renumber_map_label_type_offsets).begin() + 1);
    } else {
      auto label_first = thrust::make_transform_iterator(
        tmp_indices.begin(),
        cuda::proclaim_return_type<label_index_t>(
          [edgelist_label_offsets = *edgelist_label_offsets] __device__(size_t i) {
            auto label_idx = cuda::std::distance(
              edgelist_label_offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, edgelist_label_offsets.begin() + 1, edgelist_label_offsets.end(), i));
            return static_cast<label_index_t>(label_idx);
          }));
      auto value_first = thrust::make_counting_iterator(label_index_t{0});
      thrust::upper_bound(handle.get_thrust_policy(),
                          label_first,
                          label_first + renumber_map.size(),
                          value_first,
                          value_first + num_labels,
                          (*renumber_map_label_type_offsets).begin() + 1);
    }
  } else {
    // copy

    std::optional<rmm::device_uvector<edge_type_t>> tmp_types{std::nullopt};
    if (edgelist_edge_types) {
      tmp_types =
        rmm::device_uvector<edge_type_t>((*edgelist_edge_types).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_edge_types).begin(),
                   (*edgelist_edge_types).end(),
                   (*tmp_types).begin());
    }
    rmm::device_uvector<edge_id_t> tmp_ids(edgelist_edge_ids.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 edgelist_edge_ids.begin(),
                 edgelist_edge_ids.end(),
                 tmp_ids.begin());
    std::optional<rmm::device_uvector<int32_t>> tmp_hops{std::nullopt};
    if (edgelist_hops) {
      tmp_hops = rmm::device_uvector<int32_t>((*edgelist_hops).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*edgelist_hops).begin(),
                   (*edgelist_hops).end(),
                   (*tmp_hops).begin());
    }

    // sort by ((type), id, (hop))

    if (tmp_types) {
      if (tmp_hops) {
        auto triplet_first =
          thrust::make_zip_iterator((*tmp_types).begin(), tmp_ids.begin(), (*tmp_hops).begin());
        thrust::sort(handle.get_thrust_policy(), triplet_first, triplet_first + tmp_ids.size());
      } else {
        auto pair_first = thrust::make_zip_iterator((*tmp_types).begin(), tmp_ids.begin());
        thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + tmp_ids.size());
      }
    } else {
      if (tmp_hops) {
        auto pair_first = thrust::make_zip_iterator(tmp_ids.begin(), (*tmp_hops).begin());
        thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + tmp_ids.size());
      } else {
        thrust::sort(handle.get_thrust_policy(), tmp_ids.begin(), tmp_ids.end());
      }
    }

    // find unique ((type), id, (min_hop)) tuples

    if (tmp_types) {
      auto pair_first = thrust::make_zip_iterator((*tmp_types).begin(), tmp_ids.begin());
      if (tmp_hops) {
        tmp_ids.resize(
          cuda::std::distance(pair_first,
                              thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                                   pair_first,
                                                                   pair_first + tmp_ids.size(),
                                                                   (*tmp_hops).begin()))),
          handle.get_stream());
        (*tmp_hops).resize(tmp_ids.size(), handle.get_stream());
      } else {
        tmp_ids.resize(
          cuda::std::distance(
            pair_first,
            thrust::unique(handle.get_thrust_policy(), pair_first, pair_first + tmp_ids.size())),
          handle.get_stream());
      }
      (*tmp_types).resize(tmp_ids.size(), handle.get_stream());
    } else {
      if (tmp_hops) {
        tmp_ids.resize(
          cuda::std::distance(
            tmp_ids.begin(),
            thrust::get<0>(thrust::unique_by_key(
              handle.get_thrust_policy(), tmp_ids.begin(), tmp_ids.end(), (*tmp_hops).begin()))),
          handle.get_stream());
        (*tmp_hops).resize(tmp_ids.size(), handle.get_stream());
      } else {
        tmp_ids.resize(
          cuda::std::distance(
            tmp_ids.begin(),
            thrust::unique(handle.get_thrust_policy(), tmp_ids.begin(), tmp_ids.end())),
          handle.get_stream());
      }
    }

    // sort by ((type), (min_hop), id)

    if (tmp_hops) {
      if (tmp_types) {
        auto triplet_first =
          thrust::make_zip_iterator((*tmp_types).begin(), (*tmp_hops).begin(), tmp_ids.begin());
        thrust::sort(handle.get_thrust_policy(), triplet_first, triplet_first + tmp_ids.size());
      } else {
        auto pair_first = thrust::make_zip_iterator((*tmp_hops).begin(), tmp_ids.begin());
        thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + tmp_ids.size());
      }
    }

    renumber_map = std::move(tmp_ids);

    if (tmp_types) {
      renumber_map_label_type_offsets =
        rmm::device_uvector<size_t>(num_edge_types + 1, handle.get_stream());
      (*renumber_map_label_type_offsets).set_element_to_zero_async(0, handle.get_stream());
      thrust::upper_bound(handle.get_thrust_policy(),
                          (*tmp_types).begin(),
                          (*tmp_types).end(),
                          thrust::make_counting_iterator(edge_type_t{0}),
                          thrust::make_counting_iterator(static_cast<edge_type_t>(num_edge_types)),
                          (*renumber_map_label_type_offsets).begin() + 1);
    }
  }

  return std::make_tuple(std::move(renumber_map), std::move(renumber_map_label_type_offsets));
}

// this function does not reorder edges (the i'th returned edge is the renumbered output of the
// i'th input edge)
template <typename vertex_t, typename label_index_t>
std::tuple<rmm::device_uvector<vertex_t>,                 // edgelist_majors
           rmm::device_uvector<vertex_t>,                 // edgelist minors
           std::optional<rmm::device_uvector<vertex_t>>,  // seed_vertices,
           rmm::device_uvector<vertex_t>,                 // renumber_map
           std::optional<rmm::device_uvector<size_t>>>    // renumber_map_label_offsets
renumber_sampled_edgelist(raft::handle_t const& handle,
                          rmm::device_uvector<vertex_t>&& edgelist_majors,
                          rmm::device_uvector<vertex_t>&& edgelist_minors,
                          std::optional<raft::device_span<int32_t const>> edgelist_hops,
                          std::optional<rmm::device_uvector<vertex_t>>&& seed_vertices,
                          std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
                          std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
                          size_t num_labels,
                          bool do_expensive_check)
{
  using vertex_type_t = uint32_t;  // dummy

  // 1. compute renumber_map

  auto [renumber_map, renumber_map_label_offsets] =
    compute_vertex_renumber_map<vertex_t, vertex_type_t, label_index_t>(
      handle,
      raft::device_span<vertex_t const>(edgelist_majors.data(), edgelist_majors.size()),
      raft::device_span<vertex_t const>(edgelist_minors.data(), edgelist_minors.size()),
      edgelist_hops,
      seed_vertices ? std::make_optional<raft::device_span<vertex_t const>>((*seed_vertices).data(),
                                                                            (*seed_vertices).size())
                    : std::nullopt,
      seed_vertex_label_offsets,
      edgelist_label_offsets,
      std::nullopt,
      num_labels,
      size_t{1});

  // 2. renumber input edges

  if (edgelist_label_offsets) {
    rmm::device_uvector<vertex_t> new_vertices(renumber_map.size(), handle.get_stream());
    thrust::tabulate(handle.get_thrust_policy(),
                     new_vertices.begin(),
                     new_vertices.end(),
                     [renumber_map_label_offsets = raft::device_span<size_t const>(
                        (*renumber_map_label_offsets).data(),
                        (*renumber_map_label_offsets).size())] __device__(size_t i) {
                       auto label_index        = static_cast<label_index_t>(cuda::std::distance(
                         renumber_map_label_offsets.begin() + 1,
                         thrust::upper_bound(thrust::seq,
                                             renumber_map_label_offsets.begin() + 1,
                                             renumber_map_label_offsets.end(),
                                             i)));
                       auto label_start_offset = renumber_map_label_offsets[label_index];
                       return static_cast<vertex_t>(i - label_start_offset);
                     });

    rmm::device_uvector<vertex_t> segment_sorted_renumber_map(renumber_map.size(),
                                                              handle.get_stream());
    rmm::device_uvector<vertex_t> segment_sorted_new_vertices(new_vertices.size(),
                                                              handle.get_stream());

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

    auto approx_items_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 20) /* tuning parameter */;  // for segmented sort

    auto [h_label_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>{(*renumber_map_label_offsets).data(),
                                      (*renumber_map_label_offsets).size()},
      renumber_map.size(),
      approx_items_to_sort_per_iteration);
    auto num_chunks = h_label_offsets.size() - 1;

    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};

      auto offset_first =
        thrust::make_transform_iterator((*renumber_map_label_offsets).data() + h_label_offsets[i],
                                        detail::shift_left_t<size_t>{h_edge_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          renumber_map.begin() + h_edge_offsets[i],
                                          segment_sorted_renumber_map.begin() + h_edge_offsets[i],
                                          new_vertices.begin() + h_edge_offsets[i],
                                          segment_sorted_new_vertices.begin() + h_edge_offsets[i],
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_label_offsets[i + 1] - h_label_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());

      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }

      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          renumber_map.begin() + h_edge_offsets[i],
                                          segment_sorted_renumber_map.begin() + h_edge_offsets[i],
                                          new_vertices.begin() + h_edge_offsets[i],
                                          segment_sorted_new_vertices.begin() + h_edge_offsets[i],
                                          h_edge_offsets[i + 1] - h_edge_offsets[i],
                                          h_label_offsets[i + 1] - h_label_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
    }
    new_vertices.resize(0, handle.get_stream());
    d_tmp_storage.resize(0, handle.get_stream());
    new_vertices.shrink_to_fit(handle.get_stream());
    d_tmp_storage.shrink_to_fit(handle.get_stream());

    auto pair_first =
      thrust::make_zip_iterator(edgelist_majors.begin(), thrust::make_counting_iterator(size_t{0}));
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_majors.size(),
      edgelist_majors.begin(),
      [edgelist_label_offsets     = *edgelist_label_offsets,
       renumber_map_label_offsets = raft::device_span<size_t const>(
         (*renumber_map_label_offsets).data(), (*renumber_map_label_offsets).size()),
       old_vertices = raft::device_span<vertex_t const>(segment_sorted_renumber_map.data(),
                                                        segment_sorted_renumber_map.size()),
       new_vertices = raft::device_span<vertex_t const>(
         segment_sorted_new_vertices.data(),
         segment_sorted_new_vertices.size())] __device__(auto pair) {
        auto old_vertex = thrust::get<0>(pair);
        auto label_idx  = static_cast<label_index_t>(
          cuda::std::distance(edgelist_label_offsets.begin() + 1,
                              thrust::upper_bound(thrust::seq,
                                                  edgelist_label_offsets.begin() + 1,
                                                  edgelist_label_offsets.end(),
                                                  thrust::get<1>(pair))));
        auto label_start_offset = renumber_map_label_offsets[label_idx];
        auto label_end_offset   = renumber_map_label_offsets[label_idx + 1];
        auto it                 = thrust::lower_bound(thrust::seq,
                                      old_vertices.begin() + label_start_offset,
                                      old_vertices.begin() + label_end_offset,
                                      old_vertex);
        assert(*it == old_vertex);
        return *(new_vertices.begin() + cuda::std::distance(old_vertices.begin(), it));
      });

    pair_first =
      thrust::make_zip_iterator(edgelist_minors.begin(), thrust::make_counting_iterator(size_t{0}));
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_minors.size(),
      edgelist_minors.begin(),
      [edgelist_label_offsets     = *edgelist_label_offsets,
       renumber_map_label_offsets = raft::device_span<size_t const>(
         (*renumber_map_label_offsets).data(), (*renumber_map_label_offsets).size()),
       old_vertices = raft::device_span<vertex_t const>(segment_sorted_renumber_map.data(),
                                                        segment_sorted_renumber_map.size()),
       new_vertices = raft::device_span<vertex_t const>(
         segment_sorted_new_vertices.data(),
         segment_sorted_new_vertices.size())] __device__(auto pair) {
        auto old_vertex = thrust::get<0>(pair);
        auto label_idx  = static_cast<label_index_t>(
          cuda::std::distance(edgelist_label_offsets.begin() + 1,
                              thrust::upper_bound(thrust::seq,
                                                  edgelist_label_offsets.begin() + 1,
                                                  edgelist_label_offsets.end(),
                                                  thrust::get<1>(pair))));
        auto label_start_offset = renumber_map_label_offsets[label_idx];
        auto label_end_offset   = renumber_map_label_offsets[label_idx + 1];
        auto it                 = thrust::lower_bound(thrust::seq,
                                      old_vertices.begin() + label_start_offset,
                                      old_vertices.begin() + label_end_offset,
                                      old_vertex);
        assert(*it == old_vertex);
        return new_vertices[cuda::std::distance(old_vertices.begin(), it)];
      });

    if (seed_vertices) {
      pair_first = thrust::make_zip_iterator((*seed_vertices).begin(),
                                             thrust::make_counting_iterator(size_t{0}));
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + (*seed_vertices).size(),
        (*seed_vertices).begin(),
        [seed_vertex_label_offsets  = *seed_vertex_label_offsets,
         renumber_map_label_offsets = raft::device_span<size_t const>(
           (*renumber_map_label_offsets).data(), (*renumber_map_label_offsets).size()),
         old_vertices = raft::device_span<vertex_t const>(segment_sorted_renumber_map.data(),
                                                          segment_sorted_renumber_map.size()),
         new_vertices = raft::device_span<vertex_t const>(
           segment_sorted_new_vertices.data(),
           segment_sorted_new_vertices.size())] __device__(auto pair) {
          auto old_vertex = thrust::get<0>(pair);
          auto label_idx  = static_cast<label_index_t>(
            cuda::std::distance(seed_vertex_label_offsets.begin() + 1,
                                thrust::upper_bound(thrust::seq,
                                                    seed_vertex_label_offsets.begin() + 1,
                                                    seed_vertex_label_offsets.end(),
                                                    thrust::get<1>(pair))));
          auto label_start_offset = renumber_map_label_offsets[label_idx];
          auto label_end_offset   = renumber_map_label_offsets[label_idx + 1];
          auto it                 = thrust::lower_bound(thrust::seq,
                                        old_vertices.begin() + label_start_offset,
                                        old_vertices.begin() + label_end_offset,
                                        old_vertex);
          assert(*it == old_vertex);
          return new_vertices[cuda::std::distance(old_vertices.begin(), it)];
        });
    }
  } else {
    kv_store_t<vertex_t, vertex_t, false> kv_store(renumber_map.begin(),
                                                   renumber_map.end(),
                                                   thrust::make_counting_iterator(vertex_t{0}),
                                                   std::numeric_limits<vertex_t>::max(),
                                                   std::numeric_limits<vertex_t>::max(),
                                                   handle.get_stream());
    auto kv_store_view = kv_store.view();

    kv_store_view.find(
      edgelist_majors.begin(), edgelist_majors.end(), edgelist_majors.begin(), handle.get_stream());
    kv_store_view.find(
      edgelist_minors.begin(), edgelist_minors.end(), edgelist_minors.begin(), handle.get_stream());

    if (seed_vertices) {
      kv_store_view.find((*seed_vertices).begin(),
                         (*seed_vertices).end(),
                         (*seed_vertices).begin(),
                         handle.get_stream());
    }
  }

  return std::make_tuple(std::move(edgelist_majors),
                         std::move(edgelist_minors),
                         std::move(seed_vertices),
                         std::move(renumber_map),
                         std::move(renumber_map_label_offsets));
}

// this function does not reorder edges (the i'th returned edge is the renumbered output of the
// i'th input edge)
template <typename vertex_t,
          typename vertex_type_t,
          typename edge_id_t,
          typename edge_type_t,
          typename label_index_t>
std::tuple<
  rmm::device_uvector<vertex_t>,                  // edgelist_majors
  rmm::device_uvector<vertex_t>,                  // edgelist minors
  std::optional<rmm::device_uvector<edge_id_t>>,  // edgelist edge IDs
  std::optional<rmm::device_uvector<vertex_t>>,   // seed_vertices,
  rmm::device_uvector<vertex_t>,                  // vertex renumber_map
  rmm::device_uvector<size_t>,  // vertex renumber_map (label, vertex type) offsets
  std::optional<rmm::device_uvector<edge_id_t>>,  // edge ID renumber map
  std::optional<rmm::device_uvector<size_t>>>     // edge ID renumber map (label, edge type) offsets
heterogeneous_renumber_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_majors,
  rmm::device_uvector<vertex_t>&& edgelist_minors,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<raft::device_span<edge_type_t const>>&& edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<rmm::device_uvector<vertex_t>>&& seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  raft::device_span<vertex_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  bool do_expensive_check)
{
  // 1. compute vertex renumber map

  auto [vertex_renumber_map, vertex_renumber_map_label_type_offsets] =
    compute_vertex_renumber_map<vertex_t, vertex_type_t, label_index_t>(
      handle,
      raft::device_span<vertex_t const>(edgelist_majors.data(), edgelist_majors.size()),
      raft::device_span<vertex_t const>(edgelist_minors.data(), edgelist_minors.size()),
      edgelist_hops,
      seed_vertices ? std::make_optional<raft::device_span<vertex_t const>>((*seed_vertices).data(),
                                                                            (*seed_vertices).size())
                    : std::nullopt,
      seed_vertex_label_offsets,
      edgelist_label_offsets,
      std::make_optional(vertex_type_offsets),
      num_labels,
      num_vertex_types);
  assert(vertex_renumber_map_label_type_offsets.has_value());

  // 2. compute edge renumber map

  std::optional<rmm::device_uvector<edge_id_t>> edge_id_renumber_map{std::nullopt};
  std::optional<rmm::device_uvector<size_t>> edge_id_renumber_map_label_type_offsets{std::nullopt};
  if (edgelist_edge_ids) {
    std::tie(edge_id_renumber_map, edge_id_renumber_map_label_type_offsets) =
      compute_edge_id_renumber_map<vertex_t, edge_id_t, edge_type_t, label_index_t>(
        handle,
        raft::device_span<edge_id_t const>((*edgelist_edge_ids).data(),
                                           (*edgelist_edge_ids).size()),
        edgelist_edge_types,
        edgelist_hops,
        edgelist_label_offsets,
        num_labels,
        num_edge_types);
  }

  auto approx_items_to_sort_per_iteration =
    static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
    (1 << 20) /* tuning parameter */;  // for segmented sort

  // 3. renumber input edge source/destination vertices

  {
    rmm::device_uvector<vertex_t> new_vertices(vertex_renumber_map.size(), handle.get_stream());
    thrust::tabulate(handle.get_thrust_policy(),
                     new_vertices.begin(),
                     new_vertices.end(),
                     [renumber_map_label_type_offsets = raft::device_span<size_t const>(
                        (*vertex_renumber_map_label_type_offsets).data(),
                        (*vertex_renumber_map_label_type_offsets).size())] __device__(size_t i) {
                       auto idx          = static_cast<size_t>(cuda::std::distance(
                         renumber_map_label_type_offsets.begin() + 1,
                         thrust::upper_bound(thrust::seq,
                                             renumber_map_label_type_offsets.begin() + 1,
                                             renumber_map_label_type_offsets.end(),
                                             i)));
                       auto start_offset = renumber_map_label_type_offsets[idx];
                       return static_cast<vertex_t>(i - start_offset);
                     });

    rmm::device_uvector<vertex_t> segment_sorted_vertex_renumber_map(vertex_renumber_map.size(),
                                                                     handle.get_stream());
    rmm::device_uvector<vertex_t> segment_sorted_new_vertices(new_vertices.size(),
                                                              handle.get_stream());

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

    auto [h_label_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>{(*vertex_renumber_map_label_type_offsets).data(),
                                      (*vertex_renumber_map_label_type_offsets).size()},
      vertex_renumber_map.size(),
      approx_items_to_sort_per_iteration);
    auto num_chunks = h_label_offsets.size() - 1;

    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};

      auto offset_first = thrust::make_transform_iterator(
        (*vertex_renumber_map_label_type_offsets).data() + h_label_offsets[i],
        detail::shift_left_t<size_t>{h_edge_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(
        static_cast<void*>(nullptr),
        tmp_storage_bytes,
        vertex_renumber_map.begin() + h_edge_offsets[i],
        segment_sorted_vertex_renumber_map.begin() + h_edge_offsets[i],
        new_vertices.begin() + h_edge_offsets[i],
        segment_sorted_new_vertices.begin() + h_edge_offsets[i],
        h_edge_offsets[i + 1] - h_edge_offsets[i],
        h_label_offsets[i + 1] - h_label_offsets[i],
        offset_first,
        offset_first + 1,
        handle.get_stream());

      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }

      cub::DeviceSegmentedSort::SortPairs(
        d_tmp_storage.data(),
        tmp_storage_bytes,
        vertex_renumber_map.begin() + h_edge_offsets[i],
        segment_sorted_vertex_renumber_map.begin() + h_edge_offsets[i],
        new_vertices.begin() + h_edge_offsets[i],
        segment_sorted_new_vertices.begin() + h_edge_offsets[i],
        h_edge_offsets[i + 1] - h_edge_offsets[i],
        h_label_offsets[i + 1] - h_label_offsets[i],
        offset_first,
        offset_first + 1,
        handle.get_stream());
    }

    new_vertices.resize(0, handle.get_stream());
    new_vertices.shrink_to_fit(handle.get_stream());

    auto pair_first =
      thrust::make_zip_iterator(edgelist_majors.begin(), thrust::make_counting_iterator(size_t{0}));
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_majors.size(),
      edgelist_majors.begin(),
      [edgelist_label_offsets = detail::to_thrust_optional(edgelist_label_offsets),
       vertex_type_offsets,
       renumber_map_label_type_offsets =
         raft::device_span<size_t const>((*vertex_renumber_map_label_type_offsets).data(),
                                         (*vertex_renumber_map_label_type_offsets).size()),
       old_vertices = raft::device_span<vertex_t const>(segment_sorted_vertex_renumber_map.data(),
                                                        segment_sorted_vertex_renumber_map.size()),
       new_vertices = raft::device_span<vertex_t const>(segment_sorted_new_vertices.data(),
                                                        segment_sorted_new_vertices.size()),
       num_vertex_types] __device__(auto pair) {
        auto old_vertex = thrust::get<0>(pair);
        label_index_t label_idx{0};
        if (edgelist_label_offsets) {
          label_idx = static_cast<label_index_t>(
            cuda::std::distance((*edgelist_label_offsets).begin() + 1,
                                thrust::upper_bound(thrust::seq,
                                                    (*edgelist_label_offsets).begin() + 1,
                                                    (*edgelist_label_offsets).end(),
                                                    thrust::get<1>(pair))));
        }
        auto v_type       = static_cast<vertex_type_t>(cuda::std::distance(
          vertex_type_offsets.begin() + 1,
          thrust::upper_bound(
            thrust::seq, vertex_type_offsets.begin() + 1, vertex_type_offsets.end(), old_vertex)));
        auto start_offset = renumber_map_label_type_offsets[label_idx * num_vertex_types + v_type];
        auto end_offset =
          renumber_map_label_type_offsets[label_idx * num_vertex_types + v_type + 1];
        auto it = thrust::lower_bound(thrust::seq,
                                      old_vertices.begin() + start_offset,
                                      old_vertices.begin() + end_offset,
                                      old_vertex);
        assert(*it == old_vertex);
        return *(new_vertices.begin() + cuda::std::distance(old_vertices.begin(), it));
      });

    pair_first =
      thrust::make_zip_iterator(edgelist_minors.begin(), thrust::make_counting_iterator(size_t{0}));
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + edgelist_minors.size(),
      edgelist_minors.begin(),
      [edgelist_label_offsets = detail::to_thrust_optional(edgelist_label_offsets),
       vertex_type_offsets,
       renumber_map_label_type_offsets =
         raft::device_span<size_t const>((*vertex_renumber_map_label_type_offsets).data(),
                                         (*vertex_renumber_map_label_type_offsets).size()),
       old_vertices = raft::device_span<vertex_t const>(segment_sorted_vertex_renumber_map.data(),
                                                        segment_sorted_vertex_renumber_map.size()),
       new_vertices = raft::device_span<vertex_t const>(segment_sorted_new_vertices.data(),
                                                        segment_sorted_new_vertices.size()),
       num_vertex_types] __device__(auto pair) {
        auto old_vertex = thrust::get<0>(pair);
        label_index_t label_idx{0};
        if (edgelist_label_offsets) {
          label_idx = static_cast<label_index_t>(
            cuda::std::distance((*edgelist_label_offsets).begin() + 1,
                                thrust::upper_bound(thrust::seq,
                                                    (*edgelist_label_offsets).begin() + 1,
                                                    (*edgelist_label_offsets).end(),
                                                    thrust::get<1>(pair))));
        }
        auto v_type       = static_cast<vertex_type_t>(cuda::std::distance(
          vertex_type_offsets.begin() + 1,
          thrust::upper_bound(
            thrust::seq, vertex_type_offsets.begin() + 1, vertex_type_offsets.end(), old_vertex)));
        auto start_offset = renumber_map_label_type_offsets[label_idx * num_vertex_types + v_type];
        auto end_offset =
          renumber_map_label_type_offsets[label_idx * num_vertex_types + v_type + 1];
        auto it = thrust::lower_bound(thrust::seq,
                                      old_vertices.begin() + start_offset,
                                      old_vertices.begin() + end_offset,
                                      old_vertex);
        assert(*it == old_vertex);
        return *(new_vertices.begin() + cuda::std::distance(old_vertices.begin(), it));
      });

    if (seed_vertices) {
      pair_first = thrust::make_zip_iterator((*seed_vertices).begin(),
                                             thrust::make_counting_iterator(size_t{0}));
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + (*seed_vertices).size(),
        (*seed_vertices).begin(),
        [seed_vertex_label_offsets = detail::to_thrust_optional(seed_vertex_label_offsets),
         vertex_type_offsets,
         renumber_map_label_type_offsets =
           raft::device_span<size_t const>((*vertex_renumber_map_label_type_offsets).data(),
                                           (*vertex_renumber_map_label_type_offsets).size()),
         old_vertices = raft::device_span<vertex_t const>(
           segment_sorted_vertex_renumber_map.data(), segment_sorted_vertex_renumber_map.size()),
         new_vertices = raft::device_span<vertex_t const>(segment_sorted_new_vertices.data(),
                                                          segment_sorted_new_vertices.size()),
         num_vertex_types] __device__(auto pair) {
          auto old_vertex = thrust::get<0>(pair);
          label_index_t label_idx{0};
          if (seed_vertex_label_offsets) {
            label_idx = static_cast<label_index_t>(
              cuda::std::distance((*seed_vertex_label_offsets).begin() + 1,
                                  thrust::upper_bound(thrust::seq,
                                                      (*seed_vertex_label_offsets).begin() + 1,
                                                      (*seed_vertex_label_offsets).end(),
                                                      thrust::get<1>(pair))));
          }
          auto v_type = static_cast<vertex_type_t>(
            cuda::std::distance(vertex_type_offsets.begin() + 1,
                                thrust::upper_bound(thrust::seq,
                                                    vertex_type_offsets.begin() + 1,
                                                    vertex_type_offsets.end(),
                                                    old_vertex)));
          auto start_offset =
            renumber_map_label_type_offsets[label_idx * num_vertex_types + v_type];
          auto end_offset =
            renumber_map_label_type_offsets[label_idx * num_vertex_types + v_type + 1];
          auto it = thrust::lower_bound(thrust::seq,
                                        old_vertices.begin() + start_offset,
                                        old_vertices.begin() + end_offset,
                                        old_vertex);
          assert(*it == old_vertex);
          return new_vertices[cuda::std::distance(old_vertices.begin(), it)];
        });
    }
  }

  // 4. renumber input edge IDs

  if (edgelist_edge_ids) {
    rmm::device_uvector<edge_id_t> new_edge_ids((*edge_id_renumber_map).size(),
                                                handle.get_stream());
    if (edge_id_renumber_map_label_type_offsets) {
      thrust::tabulate(handle.get_thrust_policy(),
                       new_edge_ids.begin(),
                       new_edge_ids.end(),
                       [renumber_map_label_type_offsets = raft::device_span<size_t const>(
                          (*edge_id_renumber_map_label_type_offsets).data(),
                          (*edge_id_renumber_map_label_type_offsets).size())] __device__(size_t i) {
                         auto idx          = static_cast<size_t>(cuda::std::distance(
                           renumber_map_label_type_offsets.begin() + 1,
                           thrust::upper_bound(thrust::seq,
                                               renumber_map_label_type_offsets.begin() + 1,
                                               renumber_map_label_type_offsets.end(),
                                               i)));
                         auto start_offset = renumber_map_label_type_offsets[idx];
                         return static_cast<edge_id_t>(i - start_offset);
                       });
    } else {
      thrust::sequence(
        handle.get_thrust_policy(), new_edge_ids.begin(), new_edge_ids.end(), edge_id_t{0});
    }

    rmm::device_uvector<edge_id_t> segment_sorted_edge_id_renumber_map(
      (*edge_id_renumber_map).size(), handle.get_stream());
    rmm::device_uvector<edge_id_t> segment_sorted_new_edge_ids(new_edge_ids.size(),
                                                               handle.get_stream());

    if (edge_id_renumber_map_label_type_offsets) {
      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

      auto [h_label_offsets, h_edge_offsets] = detail::compute_offset_aligned_element_chunks(
        handle,
        raft::device_span<size_t const>{(*edge_id_renumber_map_label_type_offsets).data(),
                                        (*edge_id_renumber_map_label_type_offsets).size()},
        (*edge_id_renumber_map).size(),
        approx_items_to_sort_per_iteration);
      auto num_chunks = h_label_offsets.size() - 1;

      for (size_t i = 0; i < num_chunks; ++i) {
        size_t tmp_storage_bytes{0};

        auto offset_first = thrust::make_transform_iterator(
          (*edge_id_renumber_map_label_type_offsets).data() + h_label_offsets[i],
          detail::shift_left_t<size_t>{h_edge_offsets[i]});
        cub::DeviceSegmentedSort::SortPairs(
          static_cast<void*>(nullptr),
          tmp_storage_bytes,
          (*edge_id_renumber_map).begin() + h_edge_offsets[i],
          segment_sorted_edge_id_renumber_map.begin() + h_edge_offsets[i],
          new_edge_ids.begin() + h_edge_offsets[i],
          segment_sorted_new_edge_ids.begin() + h_edge_offsets[i],
          h_edge_offsets[i + 1] - h_edge_offsets[i],
          h_label_offsets[i + 1] - h_label_offsets[i],
          offset_first,
          offset_first + 1,
          handle.get_stream());

        if (tmp_storage_bytes > d_tmp_storage.size()) {
          d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
        }

        cub::DeviceSegmentedSort::SortPairs(
          d_tmp_storage.data(),
          tmp_storage_bytes,
          (*edge_id_renumber_map).begin() + h_edge_offsets[i],
          segment_sorted_edge_id_renumber_map.begin() + h_edge_offsets[i],
          new_edge_ids.begin() + h_edge_offsets[i],
          segment_sorted_new_edge_ids.begin() + h_edge_offsets[i],
          h_edge_offsets[i + 1] - h_edge_offsets[i],
          h_label_offsets[i + 1] - h_label_offsets[i],
          offset_first,
          offset_first + 1,
          handle.get_stream());
      }

      new_edge_ids.resize(0, handle.get_stream());
      new_edge_ids.shrink_to_fit(handle.get_stream());
    } else {
      thrust::copy(handle.get_thrust_policy(),
                   (*edge_id_renumber_map).begin(),
                   (*edge_id_renumber_map).end(),
                   segment_sorted_edge_id_renumber_map.begin());
      segment_sorted_new_edge_ids = std::move(new_edge_ids);
      thrust::sort_by_key(handle.get_thrust_policy(),
                          segment_sorted_edge_id_renumber_map.begin(),
                          segment_sorted_edge_id_renumber_map.end(),
                          segment_sorted_new_edge_ids.begin());
    }

    if (edge_id_renumber_map_label_type_offsets) {
      auto pair_first = thrust::make_zip_iterator((*edgelist_edge_ids).begin(),
                                                  thrust::make_counting_iterator(size_t{0}));
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + (*edgelist_edge_ids).size(),
        (*edgelist_edge_ids).begin(),
        cuda::proclaim_return_type<edge_id_t>(
          [edgelist_label_offsets = detail::to_thrust_optional(edgelist_label_offsets),
           edge_types             = edgelist_edge_types
                                      ? cuda::std::make_optional<raft::device_span<edge_type_t const>>(
                              (*edgelist_edge_types).data(), (*edgelist_edge_types).size())
                                      : cuda::std::nullopt,
           renumber_map =
             raft::device_span<edge_id_t const>(segment_sorted_edge_id_renumber_map.data(),
                                                segment_sorted_edge_id_renumber_map.size()),
           new_edge_ids = raft::device_span<edge_id_t const>(segment_sorted_new_edge_ids.data(),
                                                             segment_sorted_new_edge_ids.size()),
           renumber_map_label_type_offsets =
             raft::device_span<size_t const>((*edge_id_renumber_map_label_type_offsets).data(),
                                             (*edge_id_renumber_map_label_type_offsets).size()),
           num_edge_types] __device__(auto pair) {
            auto old_edge_id = thrust::get<0>(pair);
            auto edge_idx    = thrust::get<1>(pair);
            size_t label_idx{0};
            if (edgelist_label_offsets) {
              label_idx = static_cast<size_t>(
                cuda::std::distance((*edgelist_label_offsets).begin() + 1,
                                    thrust::upper_bound(thrust::seq,
                                                        (*edgelist_label_offsets).begin() + 1,
                                                        (*edgelist_label_offsets).end(),
                                                        edge_idx)));
            }
            edge_type_t edge_type{0};
            if (edge_types) { edge_type = (*edge_types)[edge_idx]; }
            auto renumber_map_start_offset =
              renumber_map_label_type_offsets[label_idx * num_edge_types + edge_type];
            auto renumber_map_end_offset =
              renumber_map_label_type_offsets[label_idx * num_edge_types + edge_type + 1];
            auto it = thrust::lower_bound(thrust::seq,
                                          renumber_map.begin() + renumber_map_start_offset,
                                          renumber_map.begin() + renumber_map_end_offset,
                                          old_edge_id);
            assert(*it == old_edge_id);
            return *(new_edge_ids.begin() + cuda::std::distance(renumber_map.begin(), it));
          }));
    } else {
      thrust::transform(
        handle.get_thrust_policy(),
        (*edgelist_edge_ids).begin(),
        (*edgelist_edge_ids).end(),
        (*edgelist_edge_ids).begin(),
        cuda::proclaim_return_type<edge_id_t>(
          [renumber_map =
             raft::device_span<edge_id_t const>(segment_sorted_edge_id_renumber_map.data(),
                                                segment_sorted_edge_id_renumber_map.size()),
           new_edge_ids = raft::device_span<edge_id_t const>(
             segment_sorted_new_edge_ids.data(),
             segment_sorted_new_edge_ids.size())] __device__(edge_id_t old_edge_id) {
            auto it = thrust::lower_bound(
              thrust::seq, renumber_map.begin(), renumber_map.end(), old_edge_id);
            assert(*it == old_edge_id);
            return *(new_edge_ids.begin() + cuda::std::distance(renumber_map.begin(), it));
          }));
    }
  }

  return std::make_tuple(std::move(edgelist_majors),
                         std::move(edgelist_minors),
                         std::move(edgelist_edge_ids),
                         std::move(seed_vertices),
                         std::move(vertex_renumber_map),
                         std::move(*vertex_renumber_map_label_type_offsets),
                         std::move(edge_id_renumber_map),
                         std::move(edge_id_renumber_map_label_type_offsets));
}

template <typename IndexIterator, typename ValueIterator>
void permute_array(raft::handle_t const& handle,
                   IndexIterator index_first,
                   IndexIterator index_last,
                   ValueIterator value_first /* [INOUT] */)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  auto tmp_buffer = allocate_dataframe_buffer<value_t>(cuda::std::distance(index_first, index_last),
                                                       handle.get_stream());
  thrust::gather(handle.get_thrust_policy(),
                 index_first,
                 index_last,
                 value_first,
                 get_dataframe_buffer_begin(tmp_buffer));
  thrust::copy(handle.get_thrust_policy(),
               get_dataframe_buffer_begin(tmp_buffer),
               get_dataframe_buffer_end(tmp_buffer),
               value_first);
}

// key:
// ((label), (edge type), (hop), major, minor) if use_edge_type_as_sort_key is true
// ((label), (hop), major, minor) if use_edge_type_as_sort_key is false
template <typename vertex_t, typename weight_t, typename edge_id_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_id_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>>
sort_sampled_edge_tuples(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>&& edgelist_majors,
                         rmm::device_uvector<vertex_t>&& edgelist_minors,
                         std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                         std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
                         std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                         std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
                         std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
                         bool use_edge_type_as_sort_key)
{
  std::vector<size_t> h_label_offsets{};
  std::vector<size_t> h_edge_offsets{};

  if (edgelist_label_offsets) {
    auto approx_edges_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 20) /* tuning parameter */;  // for sorts in chunks

    std::tie(h_label_offsets, h_edge_offsets) = detail::compute_offset_aligned_element_chunks(
      handle, *edgelist_label_offsets, edgelist_majors.size(), approx_edges_to_sort_per_iteration);
  } else {
    h_label_offsets = {0, 1};
    h_edge_offsets  = {0, edgelist_majors.size()};
  }

  auto num_chunks = h_label_offsets.size() - 1;
  for (size_t i = 0; i < num_chunks; ++i) {
    rmm::device_uvector<size_t> indices(h_edge_offsets[i + 1] - h_edge_offsets[i],
                                        handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
    edge_order_t<vertex_t, edge_type_t> edge_order_comp{
      edgelist_label_offsets ? cuda::std::make_optional<raft::device_span<size_t const>>(
                                 (*edgelist_label_offsets).data() + h_label_offsets[i],
                                 (h_label_offsets[i + 1] - h_label_offsets[i]) + 1)
                             : cuda::std::nullopt,
      edgelist_edge_types && use_edge_type_as_sort_key
        ? cuda::std::make_optional<raft::device_span<edge_type_t const>>(
            (*edgelist_edge_types).data() + h_edge_offsets[i], indices.size())
        : cuda::std::nullopt,
      edgelist_hops ? cuda::std::make_optional<raft::device_span<int32_t const>>(
                        (*edgelist_hops).data() + h_edge_offsets[i], indices.size())
                    : cuda::std::nullopt,
      raft::device_span<vertex_t const>(edgelist_majors.data() + h_edge_offsets[i], indices.size()),
      raft::device_span<vertex_t const>(edgelist_minors.data() + h_edge_offsets[i],
                                        indices.size())};
    thrust::sort(handle.get_thrust_policy(), indices.begin(), indices.end(), edge_order_comp);

    permute_array(handle,
                  indices.begin(),
                  indices.end(),
                  thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin()) +
                    h_edge_offsets[i]);

    if (edgelist_weights) {
      permute_array(
        handle, indices.begin(), indices.end(), (*edgelist_weights).begin() + h_edge_offsets[i]);
    }

    if (edgelist_edge_ids) {
      permute_array(
        handle, indices.begin(), indices.end(), (*edgelist_edge_ids).begin() + h_edge_offsets[i]);
    }

    if (edgelist_edge_types) {
      permute_array(
        handle, indices.begin(), indices.end(), (*edgelist_edge_types).begin() + h_edge_offsets[i]);
    }

    if (edgelist_hops) {
      permute_array(
        handle, indices.begin(), indices.end(), (*edgelist_hops).begin() + h_edge_offsets[i]);
    }
  }

  return std::make_tuple(std::move(edgelist_majors),
                         std::move(edgelist_minors),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_hops));
}

}  // namespace

template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>,     // dcsr/dcsc major vertices
           rmm::device_uvector<size_t>,                      // (d)csr/(d)csc offset values
           rmm::device_uvector<vertex_t>,                    // minor vertices
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,       // (label, hop) offsets to the
                                                             // (d)csr/(d)csc offset array
           rmm::device_uvector<vertex_t>,                    // renumber map
           std::optional<rmm::device_uvector<size_t>>>       // label offsets to the renumber map
renumber_and_compress_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  size_t num_labels,
  size_t num_hops,
  bool src_is_major,
  bool compress_per_hop,
  bool doubly_compress,
  bool do_expensive_check)
{
  using label_index_t = uint32_t;
  using vertex_type_t = uint32_t;  // dummy

  auto edgelist_majors = src_is_major ? std::move(edgelist_srcs) : std::move(edgelist_dsts);
  auto edgelist_minors = src_is_major ? std::move(edgelist_dsts) : std::move(edgelist_srcs);

  // 1. check input arguments

  check_input_edges<label_index_t, vertex_t, vertex_type_t>(handle,
                                                            edgelist_majors,
                                                            edgelist_minors,
                                                            edgelist_weights,
                                                            edgelist_edge_ids,
                                                            edgelist_edge_types,
                                                            edgelist_hops,
                                                            seed_vertices,
                                                            seed_vertex_label_offsets,
                                                            edgelist_label_offsets,
                                                            std::nullopt,
                                                            num_labels,
                                                            num_hops,
                                                            size_t{1},
                                                            std::optional<size_t>{std::nullopt},
                                                            do_expensive_check);

  CUGRAPH_EXPECTS(
    !doubly_compress || !compress_per_hop,
    "Invalid input arguments: compress_per_hop should be false if doubly_compress is true.");
  CUGRAPH_EXPECTS(!compress_per_hop || edgelist_hops,
                  "Invalid input arguments: edgelist_hops.has_value() should be true if "
                  "compress_per_hop is true.");

  // 2. renumber

  std::optional<rmm::device_uvector<vertex_t>> renumbered_seed_vertices{std::nullopt};
  if (seed_vertices) {
    renumbered_seed_vertices =
      rmm::device_uvector<vertex_t>((*seed_vertices).size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*seed_vertices).begin(),
                 (*seed_vertices).end(),
                 (*renumbered_seed_vertices).begin());
  }
  rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> renumber_map_label_offsets{std::nullopt};
  std::tie(edgelist_majors,
           edgelist_minors,
           renumbered_seed_vertices,
           renumber_map,
           renumber_map_label_offsets) =
    renumber_sampled_edgelist<vertex_t, label_index_t>(
      handle,
      std::move(edgelist_majors),
      std::move(edgelist_minors),
      edgelist_hops ? std::make_optional(raft::device_span<int32_t const>((*edgelist_hops).data(),
                                                                          (*edgelist_hops).size()))
                    : std::nullopt,
      std::move(renumbered_seed_vertices),
      seed_vertex_label_offsets,
      edgelist_label_offsets,
      num_labels,
      do_expensive_check);

  // 3. sort by ((l), (h), major, minor)

  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_weights,
           edgelist_edge_ids,
           edgelist_edge_types,
           edgelist_hops) = sort_sampled_edge_tuples(handle,
                                                     std::move(edgelist_majors),
                                                     std::move(edgelist_minors),
                                                     std::move(edgelist_weights),
                                                     std::move(edgelist_edge_ids),
                                                     std::move(edgelist_edge_types),
                                                     std::move(edgelist_hops),
                                                     edgelist_label_offsets,
                                                     false);

  if (renumbered_seed_vertices) {
    if (seed_vertex_label_offsets) {
      auto label_indices = detail::expand_sparse_offsets(
        *seed_vertex_label_offsets, label_index_t{0}, handle.get_stream());
      auto pair_first =
        thrust::make_zip_iterator(label_indices.begin(), (*renumbered_seed_vertices).begin());
      thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + label_indices.size());
    } else {
      thrust::sort(handle.get_thrust_policy(),
                   (*renumbered_seed_vertices).begin(),
                   (*renumbered_seed_vertices).end());
    }
  }

  if (do_expensive_check) {
    if (!compress_per_hop && edgelist_hops) {
      rmm::device_uvector<vertex_t> min_vertices(num_labels * num_hops, handle.get_stream());
      rmm::device_uvector<vertex_t> max_vertices(min_vertices.size(), handle.get_stream());

      auto label_index_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        optionally_compute_label_index_t<label_index_t>{
          edgelist_label_offsets ? cuda::std::make_optional(*edgelist_label_offsets)
                                 : cuda::std::nullopt});
      auto input_key_first = thrust::make_zip_iterator(label_index_first, (*edgelist_hops).begin());
      rmm::device_uvector<label_index_t> unique_key_label_indices(min_vertices.size(),
                                                                  handle.get_stream());
      rmm::device_uvector<int32_t> unique_key_hops(min_vertices.size(), handle.get_stream());
      auto output_key_first =
        thrust::make_zip_iterator(unique_key_label_indices.begin(), unique_key_hops.begin());

      auto output_it =
        thrust::reduce_by_key(handle.get_thrust_policy(),
                              input_key_first,
                              input_key_first + edgelist_majors.size(),
                              edgelist_majors.begin(),
                              output_key_first,
                              min_vertices.begin(),
                              thrust::equal_to<thrust::tuple<label_index_t, int32_t>>{},
                              thrust::minimum<vertex_t>{});
      auto num_unique_keys =
        static_cast<size_t>(cuda::std::distance(output_key_first, thrust::get<0>(output_it)));
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            edgelist_majors.begin(),
                            output_key_first,
                            max_vertices.begin(),
                            thrust::equal_to<thrust::tuple<label_index_t, int32_t>>{},
                            thrust::maximum<vertex_t>{});

      if (renumbered_seed_vertices) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(num_labels),
          [seed_vertices = raft::device_span<vertex_t const>((*renumbered_seed_vertices).data(),
                                                             (*renumbered_seed_vertices).size()),
           seed_vertex_label_offsets = detail::to_thrust_optional(seed_vertex_label_offsets),
           num_hops,
           min_vertices = raft::device_span<vertex_t>(min_vertices.data(), min_vertices.size()),
           max_vertices = raft::device_span<vertex_t>(
             max_vertices.data(), max_vertices.size())] __device__(size_t l_idx) {
            size_t label_start_offset{0};
            auto label_end_offset = seed_vertices.size();
            if (seed_vertex_label_offsets) {
              label_start_offset = (*seed_vertex_label_offsets)[l_idx];
              label_end_offset   = (*seed_vertex_label_offsets)[l_idx + 1];
            }
            if (label_start_offset < label_end_offset) {
              min_vertices[l_idx * num_hops] =
                cuda::std::min(min_vertices[l_idx * num_hops], seed_vertices[label_start_offset]);
              max_vertices[l_idx * num_hops] =
                cuda::std::max(max_vertices[l_idx * num_hops], seed_vertices[label_end_offset - 1]);
            }
          });
      }

      if (num_unique_keys > 1) {
        auto num_invalids = thrust::count_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{1}),
          thrust::make_counting_iterator(num_unique_keys),
          [output_key_first,
           min_vertices = raft::device_span<vertex_t const>(min_vertices.data(), num_unique_keys),
           max_vertices = raft::device_span<vertex_t const>(max_vertices.data(),
                                                            num_unique_keys)] __device__(size_t i) {
            auto prev_key = *(output_key_first + (i - 1));
            auto this_key = *(output_key_first + i);
            if (thrust::get<0>(prev_key) == thrust::get<0>(this_key)) {
              auto this_min = min_vertices[i];
              auto prev_max = max_vertices[i - 1];
              return prev_max >= this_min;
            } else {
              return false;
            }
          });
        CUGRAPH_EXPECTS(
          num_invalids == 0,
          "Invalid input arguments: if both compress_per_hop is false and "
          "edgelist_hops.has_value() is true, majors should be non-decreasing within each label "
          "after renumbering and sorting by (hop, major, minor). Also, majors in hop N should not "
          "appear in any of the previous hops. This condition is satisfied if majors in hop N + 1 "
          "does not have any vertices from the previous hops excluding the minors from hop N.");
      }
    }
  }

  // 4. compute offsets for ((l), (h), major) triplets with non zero neighbors (update
  // compressed_label_indices, compressed_hops, compressed_nzd_vertices, and compressed_offsets)

  auto num_uniques = thrust::count_if(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(edgelist_majors.size()),
    is_first_triplet_in_run_t<vertex_t>{
      detail::to_thrust_optional(edgelist_label_offsets),
      edgelist_hops ? cuda::std::make_optional<raft::device_span<int32_t const>>(
                        (*edgelist_hops).data(), (*edgelist_hops).size())
                    : cuda::std::nullopt,
      raft::device_span<vertex_t const>(
        edgelist_majors.data(),
        edgelist_majors.size())});  // number of unique ((label), (hop), major) triplets

  auto compressed_label_indices =
    edgelist_label_offsets
      ? std::make_optional<rmm::device_uvector<label_index_t>>(num_uniques, handle.get_stream())
      : std::nullopt;
  auto compressed_hops = edgelist_hops ? std::make_optional<rmm::device_uvector<int32_t>>(
                                           num_uniques, handle.get_stream())
                                       : std::nullopt;
  rmm::device_uvector<vertex_t> compressed_nzd_vertices(num_uniques, handle.get_stream());
  rmm::device_uvector<size_t> compressed_offsets(num_uniques + 1, handle.get_stream());
  compressed_offsets.set_element_to_zero_async(num_uniques, handle.get_stream());

  if (edgelist_label_offsets) {
    auto label_index_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      compute_label_index_t<label_index_t>{*edgelist_label_offsets});

    if (edgelist_hops) {
      auto input_key_first = thrust::make_zip_iterator(
        label_index_first, (*edgelist_hops).begin(), edgelist_majors.begin());
      auto output_key_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                        (*compressed_hops).begin(),
                                                        compressed_nzd_vertices.begin());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    } else {
      auto input_key_first  = thrust::make_zip_iterator(label_index_first, edgelist_majors.begin());
      auto output_key_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                        compressed_nzd_vertices.begin());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    }
  } else {
    if (edgelist_hops) {
      auto input_key_first =
        thrust::make_zip_iterator((*edgelist_hops).begin(), edgelist_majors.begin());
      auto output_key_first =
        thrust::make_zip_iterator((*compressed_hops).begin(), compressed_nzd_vertices.begin());
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    } else {
      auto input_key_first  = edgelist_majors.begin();
      auto output_key_first = compressed_nzd_vertices.begin();
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + edgelist_majors.size(),
                            thrust::make_constant_iterator(size_t{1}),
                            output_key_first,
                            compressed_offsets.begin());
    }
  }
  thrust::exclusive_scan(handle.get_thrust_policy(),
                         compressed_offsets.begin(),
                         compressed_offsets.end(),
                         compressed_offsets.begin());

  // 5. update compressed_offsets to include zero degree vertices (if doubly_compress is false)
  // and compressed_offset_label_hop_offsets (if edgelist_label_offsets.has_value() or
  // edgelist_hops.has_value() is true)

  std::optional<rmm::device_uvector<size_t>> compressed_offset_label_hop_offsets{std::nullopt};
  if (doubly_compress) {
    if (edgelist_label_offsets || edgelist_hops) {
      rmm::device_uvector<size_t> offset_array_offsets(num_labels * num_hops + 1,
                                                       handle.get_stream());
      offset_array_offsets.set_element_to_zero_async(0, handle.get_stream());

      if (edgelist_label_offsets) {
        if (edgelist_hops) {
          auto pair_first       = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                      (*compressed_hops).begin());
          auto value_pair_first = thrust::make_transform_iterator(
            thrust::make_counting_iterator(size_t{0}),
            cuda::proclaim_return_type<thrust::tuple<label_index_t, int32_t>>(
              [num_hops] __device__(size_t i) {
                return thrust::make_tuple(static_cast<label_index_t>(i / num_hops),
                                          static_cast<int32_t>(i % num_hops));
              }));
          thrust::upper_bound(handle.get_thrust_policy(),
                              pair_first,
                              pair_first + (*compressed_label_indices).size(),
                              value_pair_first,
                              value_pair_first + (num_labels * num_hops),
                              offset_array_offsets.begin() + 1);
        } else {
          thrust::upper_bound(
            handle.get_thrust_policy(),
            (*compressed_label_indices).begin(),
            (*compressed_label_indices).end(),
            thrust::make_counting_iterator(label_index_t{0}),
            thrust::make_counting_iterator(static_cast<label_index_t>(num_labels)),
            offset_array_offsets.begin() + 1);
        }
      } else {
        thrust::upper_bound(handle.get_thrust_policy(),
                            (*compressed_hops).begin(),
                            (*compressed_hops).end(),
                            thrust::make_counting_iterator(int32_t{0}),
                            thrust::make_counting_iterator(static_cast<int32_t>(num_hops)),
                            offset_array_offsets.begin() + 1);
      }

      compressed_offset_label_hop_offsets = std::move(offset_array_offsets);
    }
  } else {  // !doubly_compress
    rmm::device_uvector<vertex_t> major_vertex_counts(num_labels * num_hops, handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      major_vertex_counts.begin(),
      major_vertex_counts.end(),
      [edgelist_label_offsets = detail::to_thrust_optional(edgelist_label_offsets),
       edgelist_hops = edgelist_hops ? cuda::std::make_optional<raft::device_span<int32_t>>(
                                         (*edgelist_hops).data(), (*edgelist_hops).size())
                                     : cuda::std::nullopt,
       edgelist_majors =
         raft::device_span<vertex_t const>(edgelist_majors.data(), edgelist_majors.size()),
       seed_vertices             = renumbered_seed_vertices
                                     ? cuda::std::make_optional<raft::device_span<vertex_t const>>(
                             (*renumbered_seed_vertices).data(), (*renumbered_seed_vertices).size())
                                     : cuda::std::nullopt,
       seed_vertex_label_offsets = detail::to_thrust_optional(seed_vertex_label_offsets),
       num_hops,
       compress_per_hop] __device__(size_t i) {
        vertex_t num_vertices_from_edgelist{0};
        {
          size_t start_offset{0};
          auto end_offset         = edgelist_majors.size();
          auto label_start_offset = start_offset;
          auto label_end_offset   = end_offset;

          if (edgelist_label_offsets) {
            auto l_idx         = static_cast<label_index_t>(i / num_hops);
            start_offset       = (*edgelist_label_offsets)[l_idx];
            end_offset         = (*edgelist_label_offsets)[l_idx + 1];
            label_start_offset = start_offset;
            label_end_offset   = end_offset;
          }

          if (num_hops > 1) {
            auto h        = static_cast<int32_t>(i % num_hops);
            auto lower_it = thrust::lower_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            auto upper_it = thrust::upper_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            start_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), lower_it));
            end_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), upper_it));
          }
          if (compress_per_hop) {
            if (start_offset < end_offset)
              num_vertices_from_edgelist = edgelist_majors[end_offset - 1] + 1;
          } else {
            if (end_offset != label_end_offset) {
              num_vertices_from_edgelist = edgelist_majors[end_offset];
            } else if (label_start_offset < label_end_offset) {
              num_vertices_from_edgelist = edgelist_majors[end_offset - 1] + 1;
            }
          }
        }

        vertex_t num_vertices_from_seed_vertices{0};
        if (seed_vertices && (!compress_per_hop || (i % num_hops == 0))) {
          size_t label_start_offset{0};
          auto label_end_offset = (*seed_vertices).size();
          if (seed_vertex_label_offsets) {
            auto l_idx         = static_cast<label_index_t>(i / num_hops);
            label_start_offset = (*seed_vertex_label_offsets)[l_idx];
            label_end_offset   = (*seed_vertex_label_offsets)[l_idx + 1];
          }
          if (label_start_offset < label_end_offset) {
            num_vertices_from_seed_vertices = (*seed_vertices)[label_end_offset - 1] + 1;
          }
        }

        return cuda::std::max(num_vertices_from_edgelist, num_vertices_from_seed_vertices);
      });

    std::optional<rmm::device_uvector<vertex_t>> minor_vertex_counts{std::nullopt};
    if (compress_per_hop) {
      minor_vertex_counts =
        rmm::device_uvector<vertex_t>(major_vertex_counts.size(), handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   (*minor_vertex_counts).begin(),
                   (*minor_vertex_counts).end(),
                   vertex_t{0});
      if (edgelist_label_offsets) {
        auto triplet_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                       (*compressed_hops).begin(),
                                                       thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(handle.get_thrust_policy(),
                         triplet_first,
                         triplet_first + compressed_nzd_vertices.size(),
                         [edgelist_minors = raft::device_span<vertex_t const>(
                            edgelist_minors.data(), edgelist_minors.size()),
                          compressed_offsets = raft::device_span<size_t const>(
                            compressed_offsets.data(), compressed_offsets.size()),
                          minor_vertex_counts = raft::device_span<vertex_t>(
                            (*minor_vertex_counts).data(), (*minor_vertex_counts).size()),
                          num_hops] __device__(auto triplet) {
                           auto nzd_v_idx    = thrust::get<2>(triplet);
                           size_t end_offset = compressed_offsets[nzd_v_idx + 1];
                           auto l_idx        = thrust::get<0>(triplet);
                           auto h            = thrust::get<1>(triplet);
                           cuda::atomic_ref<vertex_t, cuda::thread_scope_device> minor_vertex_count(
                             minor_vertex_counts[l_idx * num_hops + h]);
                           minor_vertex_count.fetch_max(edgelist_minors[end_offset - 1] + 1,
                                                        cuda::std::memory_order_relaxed);
                         });
      } else {
        auto pair_first = thrust::make_zip_iterator((*compressed_hops).begin(),
                                                    thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(handle.get_thrust_policy(),
                         pair_first,
                         pair_first + compressed_nzd_vertices.size(),
                         [edgelist_minors = raft::device_span<vertex_t const>(
                            edgelist_minors.data(), edgelist_minors.size()),
                          compressed_offsets = raft::device_span<size_t const>(
                            compressed_offsets.data(), compressed_offsets.size()),
                          minor_vertex_counts = raft::device_span<vertex_t>(
                            (*minor_vertex_counts).data(), (*minor_vertex_counts).size()),
                          num_hops] __device__(auto pair) {
                           auto nzd_v_idx    = thrust::get<1>(pair);
                           size_t end_offset = compressed_offsets[nzd_v_idx + 1];
                           auto h            = thrust::get<0>(pair);
                           cuda::atomic_ref<vertex_t, cuda::thread_scope_device> minor_vertex_count(
                             minor_vertex_counts[h]);
                           minor_vertex_count.fetch_max(edgelist_minors[end_offset - 1] + 1,
                                                        cuda::std::memory_order_relaxed);
                         });
      }
    }

    rmm::device_uvector<size_t> offset_array_offsets(num_labels * num_hops + 1,
                                                     handle.get_stream());
    offset_array_offsets.set_element_to_zero_async(num_labels * num_hops, handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      offset_array_offsets.begin(),
      offset_array_offsets.begin() + (num_labels * num_hops),
      [major_vertex_counts =
         raft::device_span<vertex_t const>(major_vertex_counts.data(), major_vertex_counts.size()),
       minor_vertex_counts = minor_vertex_counts
                               ? cuda::std::make_optional<raft::device_span<vertex_t const>>(
                                   (*minor_vertex_counts).data(), (*minor_vertex_counts).size())
                               : cuda::std::nullopt,
       num_hops,
       compress_per_hop] __device__(size_t i) {
        auto vertex_count = major_vertex_counts[i];
        if (num_hops > 1) {
          if (compress_per_hop) {
            for (size_t j = (i - (i % num_hops)); j < i; ++j) {
              vertex_count = cuda::std::max(vertex_count, major_vertex_counts[j]);
              vertex_count = cuda::std::max(vertex_count, (*minor_vertex_counts)[j]);
            }
          } else {
            if (i % num_hops != 0) { vertex_count -= major_vertex_counts[i - 1]; }
          }
        }
        return vertex_count;
      });
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           offset_array_offsets.begin(),
                           offset_array_offsets.end(),
                           offset_array_offsets.begin());

    auto tmp_compressed_offsets = rmm::device_uvector<size_t>(
      offset_array_offsets.back_element(handle.get_stream()) + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 tmp_compressed_offsets.begin(),
                 tmp_compressed_offsets.end(),
                 size_t{0});

    if (edgelist_label_offsets) {
      if (edgelist_hops) {
        auto triplet_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                       (*compressed_hops).begin(),
                                                       thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(
          handle.get_thrust_policy(),
          triplet_first,
          triplet_first + compressed_nzd_vertices.size(),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           offset_array_offsets = raft::device_span<size_t const>(offset_array_offsets.data(),
                                                                  offset_array_offsets.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets = raft::device_span<size_t>(tmp_compressed_offsets.data(),
                                                              tmp_compressed_offsets.size()),
           compress_per_hop,
           num_hops] __device__(auto triplet) {
            auto nzd_v_idx      = thrust::get<2>(triplet);
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            auto l_idx          = thrust::get<0>(triplet);
            auto h              = thrust::get<1>(triplet);
            tmp_compressed_offsets[offset_array_offsets[l_idx * num_hops +
                                                        (compress_per_hop ? h : int32_t{0})] +
                                   compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      } else {
        auto pair_first = thrust::make_zip_iterator((*compressed_label_indices).begin(),
                                                    thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + compressed_nzd_vertices.size(),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           offset_array_offsets = raft::device_span<size_t const>(offset_array_offsets.data(),
                                                                  offset_array_offsets.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets = raft::device_span<size_t>(
             tmp_compressed_offsets.data(), tmp_compressed_offsets.size())] __device__(auto pair) {
            auto nzd_v_idx      = thrust::get<1>(pair);
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            auto l_idx          = thrust::get<0>(pair);
            tmp_compressed_offsets[offset_array_offsets[l_idx] +
                                   compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      }
    } else {
      if (edgelist_hops) {
        auto pair_first = thrust::make_zip_iterator((*compressed_hops).begin(),
                                                    thrust::make_counting_iterator(size_t{0}));
        thrust::for_each(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + compressed_nzd_vertices.size(),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           offset_array_offsets = raft::device_span<size_t const>(offset_array_offsets.data(),
                                                                  offset_array_offsets.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets = raft::device_span<size_t>(tmp_compressed_offsets.data(),
                                                              tmp_compressed_offsets.size()),
           compress_per_hop] __device__(auto pair) {
            auto nzd_v_idx      = thrust::get<1>(pair);
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            auto h              = thrust::get<0>(pair);
            tmp_compressed_offsets[offset_array_offsets[compress_per_hop ? h : int32_t{0}] +
                                   compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      } else {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(compressed_nzd_vertices.size()),
          [compressed_nzd_vertices = raft::device_span<vertex_t const>(
             compressed_nzd_vertices.data(), compressed_nzd_vertices.size()),
           compressed_offsets =
             raft::device_span<size_t>(compressed_offsets.data(), compressed_offsets.size()),
           tmp_compressed_offsets =
             raft::device_span<size_t>(tmp_compressed_offsets.data(),
                                       tmp_compressed_offsets.size())] __device__(auto nzd_v_idx) {
            size_t start_offset = compressed_offsets[nzd_v_idx];
            size_t end_offset   = compressed_offsets[nzd_v_idx + 1];
            tmp_compressed_offsets[compressed_nzd_vertices[nzd_v_idx]] = end_offset - start_offset;
          });
      }
    }

    thrust::exclusive_scan(handle.get_thrust_policy(),
                           tmp_compressed_offsets.begin(),
                           tmp_compressed_offsets.end(),
                           tmp_compressed_offsets.begin());

    compressed_offsets = std::move(tmp_compressed_offsets);

    if (edgelist_label_offsets || edgelist_hops) {
      compressed_offset_label_hop_offsets = std::move(offset_array_offsets);
    }
  }

  edgelist_hops = std::nullopt;

  return std::make_tuple(
    doubly_compress ? std::make_optional(std::move(compressed_nzd_vertices)) : std::nullopt,
    std::move(compressed_offsets),
    std::move(edgelist_minors),
    std::move(edgelist_weights),
    std::move(edgelist_edge_ids),
    std::move(edgelist_edge_types),
    std::move(compressed_offset_label_hop_offsets),
    std::move(renumber_map),
    std::move(renumber_map_label_offsets));
}

template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,       // (label, hop) offsets to the edges
           rmm::device_uvector<vertex_t>,                    // renumber map
           std::optional<rmm::device_uvector<size_t>>>       // label offsets to the renumber map
renumber_and_sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  size_t num_labels,
  size_t num_hops,
  bool src_is_major,
  bool do_expensive_check)
{
  using label_index_t = uint32_t;
  using vertex_type_t = uint32_t;  // dummy

  auto edgelist_majors = src_is_major ? std::move(edgelist_srcs) : std::move(edgelist_dsts);
  auto edgelist_minors = src_is_major ? std::move(edgelist_dsts) : std::move(edgelist_srcs);

  // 1. check input arguments

  check_input_edges<label_index_t, vertex_t, vertex_type_t>(handle,
                                                            edgelist_majors,
                                                            edgelist_minors,
                                                            edgelist_weights,
                                                            edgelist_edge_ids,
                                                            edgelist_edge_types,
                                                            edgelist_hops,
                                                            seed_vertices,
                                                            seed_vertex_label_offsets,
                                                            edgelist_label_offsets,
                                                            std::nullopt,
                                                            num_labels,
                                                            num_hops,
                                                            size_t{1},
                                                            std::optional<size_t>{std::nullopt},
                                                            do_expensive_check);

  // 2. renumber

  std::optional<rmm::device_uvector<vertex_t>> renumbered_seed_vertices{std::nullopt};
  if (seed_vertices) {
    renumbered_seed_vertices =
      rmm::device_uvector<vertex_t>((*seed_vertices).size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*seed_vertices).begin(),
                 (*seed_vertices).end(),
                 (*renumbered_seed_vertices).begin());
  }
  rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> renumber_map_label_offsets{std::nullopt};
  std::tie(
    edgelist_majors, edgelist_minors, std::ignore, renumber_map, renumber_map_label_offsets) =
    renumber_sampled_edgelist<vertex_t, label_index_t>(
      handle,
      std::move(edgelist_majors),
      std::move(edgelist_minors),
      edgelist_hops ? std::make_optional(raft::device_span<int32_t const>((*edgelist_hops).data(),
                                                                          (*edgelist_hops).size()))
                    : std::nullopt,
      std::move(renumbered_seed_vertices),
      seed_vertex_label_offsets,
      edgelist_label_offsets,
      num_labels,
      do_expensive_check);

  // 3. sort by ((l), (h), major, minor)

  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_weights,
           edgelist_edge_ids,
           edgelist_edge_types,
           edgelist_hops) = sort_sampled_edge_tuples(handle,
                                                     std::move(edgelist_majors),
                                                     std::move(edgelist_minors),
                                                     std::move(edgelist_weights),
                                                     std::move(edgelist_edge_ids),
                                                     std::move(edgelist_edge_types),
                                                     std::move(edgelist_hops),
                                                     edgelist_label_offsets,
                                                     false);

  // 4. compute edgelist_label_hop_offsets

  std::optional<rmm::device_uvector<size_t>> edgelist_label_hop_offsets{std::nullopt};
  if (edgelist_label_offsets || edgelist_hops) {
    edgelist_label_hop_offsets =
      rmm::device_uvector<size_t>(num_labels * num_hops + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 (*edgelist_label_hop_offsets).begin(),
                 (*edgelist_label_hop_offsets).end(),
                 size_t{0});
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_labels * num_hops),
      (*edgelist_label_hop_offsets).begin(),
      cuda::proclaim_return_type<size_t>(
        [edgelist_label_offsets = detail::to_thrust_optional(edgelist_label_offsets),
         edgelist_hops = edgelist_hops ? cuda::std::make_optional<raft::device_span<int32_t const>>(
                                           (*edgelist_hops).data(), (*edgelist_hops).size())
                                       : cuda::std::nullopt,
         num_hops,
         num_edges = edgelist_majors.size()] __device__(size_t i) {
          size_t start_offset{0};
          auto end_offset = num_edges;

          if (edgelist_label_offsets) {
            auto l_idx   = static_cast<label_index_t>(i / num_hops);
            start_offset = (*edgelist_label_offsets)[l_idx];
            end_offset   = (*edgelist_label_offsets)[l_idx + 1];
          }

          if (edgelist_hops) {
            auto h        = static_cast<int32_t>(i % num_hops);
            auto lower_it = thrust::lower_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            auto upper_it = thrust::upper_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            start_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), lower_it));
            end_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), upper_it));
          }

          return end_offset - start_offset;
        }));
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           (*edgelist_label_hop_offsets).begin(),
                           (*edgelist_label_hop_offsets).end(),
                           (*edgelist_label_hop_offsets).begin());
  }

  edgelist_hops = std::nullopt;

  return std::make_tuple(std::move(src_is_major ? edgelist_majors : edgelist_minors),
                         std::move(src_is_major ? edgelist_minors : edgelist_majors),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_label_hop_offsets),
                         std::move(renumber_map),
                         std::move(renumber_map_label_offsets));
}

template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                  // srcs
           rmm::device_uvector<vertex_t>,                  // dsts
           std::optional<rmm::device_uvector<weight_t>>,   // weights
           std::optional<rmm::device_uvector<edge_id_t>>,  // edge IDs
           std::optional<rmm::device_uvector<size_t>>,     // (label, edge type, hop) offsets to the
                                                           // edges
           rmm::device_uvector<vertex_t>,                  // vertex renumber map
           rmm::device_uvector<size_t>,  // (label, vertex type) offsets to the vertex renumber map
           std::optional<rmm::device_uvector<edge_id_t>>,  // edge ID renumber map
           std::optional<
             rmm::device_uvector<size_t>>>  // (label, edge type) offsets to the vertex renumber map
heterogeneous_renumber_and_sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  raft::device_span<vertex_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_hops,
  size_t num_vertex_types,
  size_t num_edge_types,
  bool src_is_major,
  bool do_expensive_check)
{
  using label_index_t = uint32_t;
  using vertex_type_t = uint32_t;

  auto edgelist_majors = src_is_major ? std::move(edgelist_srcs) : std::move(edgelist_dsts);
  auto edgelist_minors = src_is_major ? std::move(edgelist_dsts) : std::move(edgelist_srcs);

  // 1. check input arguments

  check_input_edges<label_index_t, vertex_t, vertex_type_t>(handle,
                                                            edgelist_majors,
                                                            edgelist_minors,
                                                            edgelist_weights,
                                                            edgelist_edge_ids,
                                                            edgelist_edge_types,
                                                            edgelist_hops,
                                                            seed_vertices,
                                                            seed_vertex_label_offsets,
                                                            edgelist_label_offsets,
                                                            vertex_type_offsets,
                                                            num_labels,
                                                            num_hops,
                                                            num_vertex_types,
                                                            std::optional<size_t>{num_edge_types},
                                                            do_expensive_check);

  // 2. renumber

  std::optional<rmm::device_uvector<vertex_t>> renumbered_seed_vertices{std::nullopt};
  if (seed_vertices) {
    renumbered_seed_vertices =
      rmm::device_uvector<vertex_t>((*seed_vertices).size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*seed_vertices).begin(),
                 (*seed_vertices).end(),
                 (*renumbered_seed_vertices).begin());
  }
  rmm::device_uvector<vertex_t> vertex_renumber_map(0, handle.get_stream());
  rmm::device_uvector<size_t> vertex_renumber_map_label_type_offsets(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_id_t>> edge_id_renumber_map{std::nullopt};
  std::optional<rmm::device_uvector<size_t>> edge_id_renumber_map_label_type_offsets{std::nullopt};
  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_edge_ids,
           std::ignore,
           vertex_renumber_map,
           vertex_renumber_map_label_type_offsets,
           edge_id_renumber_map,
           edge_id_renumber_map_label_type_offsets) =
    heterogeneous_renumber_sampled_edgelist<vertex_t,
                                            vertex_type_t,
                                            edge_id_t,
                                            edge_type_t,
                                            label_index_t>(
      handle,
      std::move(edgelist_majors),
      std::move(edgelist_minors),
      std::move(edgelist_edge_ids),
      edgelist_edge_types ? std::make_optional(raft::device_span<edge_type_t const>(
                              (*edgelist_edge_types).data(), (*edgelist_edge_types).size()))
                          : std::nullopt,
      edgelist_hops ? std::make_optional(raft::device_span<int32_t const>((*edgelist_hops).data(),
                                                                          (*edgelist_hops).size()))
                    : std::nullopt,
      std::move(renumbered_seed_vertices),
      seed_vertex_label_offsets,
      edgelist_label_offsets,
      vertex_type_offsets,
      num_labels,
      num_vertex_types,
      num_edge_types,
      do_expensive_check);

  // 3. sort by ((label), (edge type), (hop), major, minor)

  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_weights,
           edgelist_edge_ids,
           edgelist_edge_types,
           edgelist_hops) = sort_sampled_edge_tuples(handle,
                                                     std::move(edgelist_majors),
                                                     std::move(edgelist_minors),
                                                     std::move(edgelist_weights),
                                                     std::move(edgelist_edge_ids),
                                                     std::move(edgelist_edge_types),
                                                     std::move(edgelist_hops),
                                                     edgelist_label_offsets,
                                                     true);

  // 4. compute edgelist (label, edge type, hop) offsets

  std::optional<rmm::device_uvector<size_t>> edgelist_label_type_hop_offsets{std::nullopt};
  if (edgelist_label_offsets || edgelist_edge_types || edgelist_hops) {
    edgelist_label_type_hop_offsets =
      rmm::device_uvector<size_t>(num_labels * num_edge_types * num_hops + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 (*edgelist_label_type_hop_offsets).begin(),
                 (*edgelist_label_type_hop_offsets).end(),
                 size_t{0});
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_labels * num_edge_types * num_hops),
      (*edgelist_label_type_hop_offsets).begin(),
      cuda::proclaim_return_type<size_t>(
        [edgelist_label_offsets = detail::to_thrust_optional(edgelist_label_offsets),
         edgelist_edge_types    = edgelist_edge_types
                                    ? cuda::std::make_optional<raft::device_span<edge_type_t const>>(
                                     (*edgelist_edge_types).data(), (*edgelist_edge_types).size())
                                    : cuda::std::nullopt,
         edgelist_hops = edgelist_hops ? cuda::std::make_optional<raft::device_span<int32_t const>>(
                                           (*edgelist_hops).data(), (*edgelist_hops).size())
                                       : cuda::std::nullopt,
         num_edge_types,
         num_hops,
         num_edges = edgelist_majors.size()] __device__(size_t i) {
          size_t start_offset{0};
          auto end_offset = num_edges;

          if (edgelist_label_offsets) {
            auto l_idx   = static_cast<label_index_t>(i / (num_edge_types * num_hops));
            start_offset = (*edgelist_label_offsets)[l_idx];
            end_offset   = (*edgelist_label_offsets)[l_idx + 1];
          }

          if (edgelist_edge_types) {
            auto t        = static_cast<edge_type_t>((i % (num_edge_types * num_hops)) / num_hops);
            auto lower_it = thrust::lower_bound(thrust::seq,
                                                (*edgelist_edge_types).begin() + start_offset,
                                                (*edgelist_edge_types).begin() + end_offset,
                                                t);
            auto upper_it = thrust::upper_bound(thrust::seq,
                                                (*edgelist_edge_types).begin() + start_offset,
                                                (*edgelist_edge_types).begin() + end_offset,
                                                t);
            start_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_edge_types).begin(), lower_it));
            end_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_edge_types).begin(), upper_it));
          }

          if (edgelist_hops) {
            auto h        = static_cast<int32_t>(i % num_hops);
            auto lower_it = thrust::lower_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            auto upper_it = thrust::upper_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            start_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), lower_it));
            end_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), upper_it));
          }

          return end_offset - start_offset;
        }));
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           (*edgelist_label_type_hop_offsets).begin(),
                           (*edgelist_label_type_hop_offsets).end(),
                           (*edgelist_label_type_hop_offsets).begin());
  }

  edgelist_edge_types = std::nullopt;
  edgelist_hops       = std::nullopt;

  return std::make_tuple(std::move(src_is_major ? edgelist_majors : edgelist_minors),
                         std::move(src_is_major ? edgelist_minors : edgelist_majors),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_label_type_hop_offsets),
                         std::move(vertex_renumber_map),
                         std::move(vertex_renumber_map_label_type_offsets),
                         std::move(edge_id_renumber_map),
                         std::move(edge_id_renumber_map_label_type_offsets));
}

template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>>       // (label, hop) offsets to the edges
sort_sampled_edgelist(raft::handle_t const& handle,
                      rmm::device_uvector<vertex_t>&& edgelist_srcs,
                      rmm::device_uvector<vertex_t>&& edgelist_dsts,
                      std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                      std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
                      std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                      std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
                      std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
                      size_t num_labels,
                      size_t num_hops,
                      bool src_is_major,
                      bool do_expensive_check)
{
  using label_index_t = uint32_t;
  using vertex_type_t = uint32_t;  // dummy

  auto edgelist_majors = src_is_major ? std::move(edgelist_srcs) : std::move(edgelist_dsts);
  auto edgelist_minors = src_is_major ? std::move(edgelist_dsts) : std::move(edgelist_srcs);

  // 1. check input arguments

  check_input_edges<label_index_t, vertex_t, vertex_type_t>(handle,
                                                            edgelist_majors,
                                                            edgelist_minors,
                                                            edgelist_weights,
                                                            edgelist_edge_ids,
                                                            edgelist_edge_types,
                                                            edgelist_hops,
                                                            std::nullopt,
                                                            std::nullopt,
                                                            edgelist_label_offsets,
                                                            std::nullopt,
                                                            num_labels,
                                                            num_hops,
                                                            size_t{1},
                                                            std::optional<size_t>{std::nullopt},
                                                            do_expensive_check);

  // 2. sort by ((l), (h), major, minor)

  std::tie(edgelist_majors,
           edgelist_minors,
           edgelist_weights,
           edgelist_edge_ids,
           edgelist_edge_types,
           edgelist_hops) = sort_sampled_edge_tuples(handle,
                                                     std::move(edgelist_majors),
                                                     std::move(edgelist_minors),
                                                     std::move(edgelist_weights),
                                                     std::move(edgelist_edge_ids),
                                                     std::move(edgelist_edge_types),
                                                     std::move(edgelist_hops),
                                                     edgelist_label_offsets,
                                                     false);

  // 3. compute edgelist_label_hop_offsets

  std::optional<rmm::device_uvector<size_t>> edgelist_label_hop_offsets{std::nullopt};
  if (edgelist_label_offsets || edgelist_hops) {
    edgelist_label_hop_offsets =
      rmm::device_uvector<size_t>(num_labels * num_hops + 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 (*edgelist_label_hop_offsets).begin(),
                 (*edgelist_label_hop_offsets).end(),
                 size_t{0});
    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_labels * num_hops),
      (*edgelist_label_hop_offsets).begin(),
      cuda::proclaim_return_type<size_t>(
        [edgelist_label_offsets = detail::to_thrust_optional(edgelist_label_offsets),
         edgelist_hops = edgelist_hops ? cuda::std::make_optional<raft::device_span<int32_t const>>(
                                           (*edgelist_hops).data(), (*edgelist_hops).size())
                                       : cuda::std::nullopt,
         num_hops,
         num_edges = edgelist_majors.size()] __device__(size_t i) {
          size_t start_offset{0};
          auto end_offset = num_edges;

          if (edgelist_label_offsets) {
            auto l_idx   = static_cast<label_index_t>(i / num_hops);
            start_offset = (*edgelist_label_offsets)[l_idx];
            end_offset   = (*edgelist_label_offsets)[l_idx + 1];
          }

          if (edgelist_hops) {
            auto h        = static_cast<int32_t>(i % num_hops);
            auto lower_it = thrust::lower_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            auto upper_it = thrust::upper_bound(thrust::seq,
                                                (*edgelist_hops).begin() + start_offset,
                                                (*edgelist_hops).begin() + end_offset,
                                                h);
            start_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), lower_it));
            end_offset =
              static_cast<size_t>(cuda::std::distance((*edgelist_hops).begin(), upper_it));
          }

          return end_offset - start_offset;
        }));
    thrust::exclusive_scan(handle.get_thrust_policy(),
                           (*edgelist_label_hop_offsets).begin(),
                           (*edgelist_label_hop_offsets).end(),
                           (*edgelist_label_hop_offsets).begin());
  }

  edgelist_hops = std::nullopt;

  return std::make_tuple(std::move(src_is_major ? edgelist_majors : edgelist_minors),
                         std::move(src_is_major ? edgelist_minors : edgelist_majors),
                         std::move(edgelist_weights),
                         std::move(edgelist_edge_ids),
                         std::move(edgelist_edge_types),
                         std::move(edgelist_label_hop_offsets));
}

}  // namespace cugraph

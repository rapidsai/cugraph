/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

template <typename index_t>
bool check_offsets(raft::handle_t const& handle,
                   raft::device_span<index_t const> offsets,
                   index_t num_segments,
                   index_t num_elements)
{
  if (offsets.size() != num_segments + 1) { return false; }

  if (!thrust::is_sorted(handle.get_thrust_policy(), offsets.begin(), offsets.end())) {
    return false;
  }

  index_t front_element{};
  index_t back_element{};
  raft::update_host(&front_element, offsets.data(), index_t{1}, handle.get_stream());
  raft::update_host(
    &back_element, offsets.data() + offsets.size() - 1, index_t{1}, handle.get_stream());
  handle.sync_stream();

  if (front_element != index_t{0}) { return false; }

  if (back_element != num_elements) { return false; }

  return true;
}

template bool check_offsets(raft::handle_t const& handle,
                            raft::device_span<size_t const> offsets,
                            size_t num_segments,
                            size_t num_elements);

template <typename vertex_t>
bool check_edgelist_is_sorted(raft::handle_t const& handle,
                              raft::device_span<vertex_t const> edgelist_majors,
                              raft::device_span<vertex_t const> edgelist_minors)
{
  auto edge_first = thrust::make_zip_iterator(edgelist_majors.begin(), edgelist_minors.begin());
  return thrust::is_sorted(
    handle.get_thrust_policy(), edge_first, edge_first + edgelist_majors.size());
}

template bool check_edgelist_is_sorted(raft::handle_t const& handle,
                                       raft::device_span<int32_t const> edgelist_majors,
                                       raft::device_span<int32_t const> edgelist_minors);

template bool check_edgelist_is_sorted(raft::handle_t const& handle,
                                       raft::device_span<int64_t const> edgelist_majors,
                                       raft::device_span<int64_t const> edgelist_minors);

// unrenumber the renumbered edge list and check whether the original & unrenumbered edge lists are
// identical
template <typename vertex_t, typename weight_t>
bool compare_edgelist(raft::handle_t const& handle,
                      raft::device_span<vertex_t const> org_edgelist_srcs,
                      raft::device_span<vertex_t const> org_edgelist_dsts,
                      std::optional<raft::device_span<weight_t const>> org_edgelist_weights,
                      raft::device_span<vertex_t const> renumbered_edgelist_srcs,
                      raft::device_span<vertex_t const> renumbered_edgelist_dsts,
                      std::optional<raft::device_span<weight_t const>> renumbered_edgelist_weights,
                      std::optional<raft::device_span<vertex_t const>> renumber_map)
{
  if (org_edgelist_srcs.size() != renumbered_edgelist_srcs.size()) { return false; }

  rmm::device_uvector<vertex_t> sorted_org_edgelist_srcs(org_edgelist_srcs.size(),
                                                         handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               org_edgelist_srcs.begin(),
               org_edgelist_srcs.end(),
               sorted_org_edgelist_srcs.begin());
  rmm::device_uvector<vertex_t> sorted_org_edgelist_dsts(org_edgelist_dsts.size(),
                                                         handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               org_edgelist_dsts.begin(),
               org_edgelist_dsts.end(),
               sorted_org_edgelist_dsts.begin());
  auto sorted_org_edgelist_weights = org_edgelist_weights
                                       ? std::make_optional<rmm::device_uvector<weight_t>>(
                                           (*org_edgelist_weights).size(), handle.get_stream())
                                       : std::nullopt;
  if (sorted_org_edgelist_weights) {
    thrust::copy(handle.get_thrust_policy(),
                 (*org_edgelist_weights).begin(),
                 (*org_edgelist_weights).end(),
                 (*sorted_org_edgelist_weights).begin());
  }

  if (sorted_org_edgelist_weights) {
    auto sorted_org_edge_first = thrust::make_zip_iterator(sorted_org_edgelist_srcs.begin(),
                                                           sorted_org_edgelist_dsts.begin(),
                                                           (*sorted_org_edgelist_weights).begin());
    thrust::sort(handle.get_thrust_policy(),
                 sorted_org_edge_first,
                 sorted_org_edge_first + sorted_org_edgelist_srcs.size());
  } else {
    auto sorted_org_edge_first =
      thrust::make_zip_iterator(sorted_org_edgelist_srcs.begin(), sorted_org_edgelist_dsts.begin());
    thrust::sort(handle.get_thrust_policy(),
                 sorted_org_edge_first,
                 sorted_org_edge_first + sorted_org_edgelist_srcs.size());
  }

  rmm::device_uvector<vertex_t> sorted_unrenumbered_edgelist_srcs(renumbered_edgelist_srcs.size(),
                                                                  handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               renumbered_edgelist_srcs.begin(),
               renumbered_edgelist_srcs.end(),
               sorted_unrenumbered_edgelist_srcs.begin());
  rmm::device_uvector<vertex_t> sorted_unrenumbered_edgelist_dsts(renumbered_edgelist_dsts.size(),
                                                                  handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               renumbered_edgelist_dsts.begin(),
               renumbered_edgelist_dsts.end(),
               sorted_unrenumbered_edgelist_dsts.begin());
  auto sorted_unrenumbered_edgelist_weights =
    renumbered_edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                    (*renumbered_edgelist_weights).size(), handle.get_stream())
                                : std::nullopt;
  if (sorted_unrenumbered_edgelist_weights) {
    thrust::copy(handle.get_thrust_policy(),
                 (*renumbered_edgelist_weights).begin(),
                 (*renumbered_edgelist_weights).end(),
                 (*sorted_unrenumbered_edgelist_weights).begin());
  }

  if (renumber_map) {
    cugraph::unrenumber_int_vertices<vertex_t, false>(
      handle,
      sorted_unrenumbered_edgelist_srcs.data(),
      sorted_unrenumbered_edgelist_srcs.size(),
      (*renumber_map).data(),
      std::vector<vertex_t>{static_cast<vertex_t>((*renumber_map).size())});
    cugraph::unrenumber_int_vertices<vertex_t, false>(
      handle,
      sorted_unrenumbered_edgelist_dsts.data(),
      sorted_unrenumbered_edgelist_dsts.size(),
      (*renumber_map).data(),
      std::vector<vertex_t>{static_cast<vertex_t>((*renumber_map).size())});
  }

  if (sorted_unrenumbered_edgelist_weights) {
    auto sorted_unrenumbered_edge_first =
      thrust::make_zip_iterator(sorted_unrenumbered_edgelist_srcs.begin(),
                                sorted_unrenumbered_edgelist_dsts.begin(),
                                (*sorted_unrenumbered_edgelist_weights).begin());
    thrust::sort(handle.get_thrust_policy(),
                 sorted_unrenumbered_edge_first,
                 sorted_unrenumbered_edge_first + sorted_unrenumbered_edgelist_srcs.size());

    auto sorted_org_edge_first = thrust::make_zip_iterator(sorted_org_edgelist_srcs.begin(),
                                                           sorted_org_edgelist_dsts.begin(),
                                                           (*sorted_org_edgelist_weights).begin());
    return thrust::equal(handle.get_thrust_policy(),
                         sorted_org_edge_first,
                         sorted_org_edge_first + sorted_org_edgelist_srcs.size(),
                         sorted_unrenumbered_edge_first);
  } else {
    auto sorted_unrenumbered_edge_first = thrust::make_zip_iterator(
      sorted_unrenumbered_edgelist_srcs.begin(), sorted_unrenumbered_edgelist_dsts.begin());
    thrust::sort(handle.get_thrust_policy(),
                 sorted_unrenumbered_edge_first,
                 sorted_unrenumbered_edge_first + sorted_unrenumbered_edgelist_srcs.size());

    auto sorted_org_edge_first =
      thrust::make_zip_iterator(sorted_org_edgelist_srcs.begin(), sorted_org_edgelist_dsts.begin());
    return thrust::equal(handle.get_thrust_policy(),
                         sorted_org_edge_first,
                         sorted_org_edge_first + sorted_org_edgelist_srcs.size(),
                         sorted_unrenumbered_edge_first);
  }
}

template bool compare_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> org_edgelist_srcs,
  raft::device_span<int32_t const> org_edgelist_dsts,
  std::optional<raft::device_span<float const>> org_edgelist_weights,
  raft::device_span<int32_t const> renumbered_edgelist_srcs,
  raft::device_span<int32_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<float const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template bool compare_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> org_edgelist_srcs,
  raft::device_span<int32_t const> org_edgelist_dsts,
  std::optional<raft::device_span<double const>> org_edgelist_weights,
  raft::device_span<int32_t const> renumbered_edgelist_srcs,
  raft::device_span<int32_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<double const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template bool compare_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> org_edgelist_srcs,
  raft::device_span<int64_t const> org_edgelist_dsts,
  std::optional<raft::device_span<float const>> org_edgelist_weights,
  raft::device_span<int64_t const> renumbered_edgelist_srcs,
  raft::device_span<int64_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<float const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template bool compare_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> org_edgelist_srcs,
  raft::device_span<int64_t const> org_edgelist_dsts,
  std::optional<raft::device_span<double const>> org_edgelist_weights,
  raft::device_span<int64_t const> renumbered_edgelist_srcs,
  raft::device_span<int64_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<double const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> renumber_map);

// unrenumber the renumbered edge list and check whether the original & unrenumbered edge lists are
// identical
template <typename vertex_t, typename weight_t, typename edge_id_t, typename edge_type_t>
bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> org_edgelist_srcs,
  raft::device_span<vertex_t const> org_edgelist_dsts,
  std::optional<raft::device_span<weight_t const>> org_edgelist_weights,
  std::optional<raft::device_span<edge_id_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<edge_type_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<vertex_t const> renumbered_edgelist_srcs,
  raft::device_span<vertex_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<weight_t const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<edge_id_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<vertex_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<edge_id_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<vertex_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops)
{
  if (org_edgelist_srcs.size() != renumbered_edgelist_srcs.size()) { return false; }

  for (size_t i = 0; i < num_labels; ++i) {
    size_t label_start_offset{0};
    size_t label_end_offset = org_edgelist_srcs.size();
    if (org_edgelist_label_offsets) {
      raft::update_host(&label_start_offset,
                        (*org_edgelist_label_offsets).data() + i,
                        size_t{1},
                        handle.get_stream());
      raft::update_host(&label_end_offset,
                        (*org_edgelist_label_offsets).data() + i + 1,
                        size_t{1},
                        handle.get_stream());
      handle.sync_stream();
    }

    if (label_start_offset == label_end_offset) { continue; }

    if (renumbered_edgelist_label_edge_type_hop_offsets) {
      size_t renumbered_label_start_offset{0};
      size_t renumbered_label_end_offset{0};
      raft::update_host(
        &renumbered_label_start_offset,
        (*renumbered_edgelist_label_edge_type_hop_offsets).data() + i * num_edge_types * num_hops,
        size_t{1},
        handle.get_stream());
      raft::update_host(&renumbered_label_end_offset,
                        (*renumbered_edgelist_label_edge_type_hop_offsets).data() +
                          (i + 1) * num_edge_types * num_hops,
                        size_t{1},
                        handle.get_stream());
      handle.sync_stream();
      if (renumbered_label_start_offset != label_start_offset) { return false; }
      if (renumbered_label_end_offset != label_end_offset) { return false; }
    }

    // sort org edgelist by ((edge_type), (hop), src, dst, (weight), (edge ID))

    rmm::device_uvector<size_t> this_label_org_sorted_indices(label_end_offset - label_start_offset,
                                                              handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(),
                     this_label_org_sorted_indices.begin(),
                     this_label_org_sorted_indices.end(),
                     size_t{0});

    thrust::sort(
      handle.get_thrust_policy(),
      this_label_org_sorted_indices.begin(),
      this_label_org_sorted_indices.end(),
      [edge_types = org_edgelist_edge_types
                      ? thrust::make_optional<raft::device_span<edge_type_t const>>(
                          (*org_edgelist_edge_types).data() + label_start_offset,
                          label_end_offset - label_start_offset)
                      : thrust::nullopt,
       hops       = org_edgelist_hops ? thrust::make_optional<raft::device_span<int32_t const>>(
                                    (*org_edgelist_hops).data() + label_start_offset,
                                    label_end_offset - label_start_offset)
                                      : thrust::nullopt,
       srcs       = raft::device_span<vertex_t const>(org_edgelist_srcs.data() + label_start_offset,
                                                label_end_offset - label_start_offset),
       dsts       = raft::device_span<vertex_t const>(org_edgelist_dsts.data() + label_start_offset,
                                                label_end_offset - label_start_offset),
       weights    = org_edgelist_weights ? thrust::make_optional<raft::device_span<weight_t const>>(
                                          (*org_edgelist_weights).data() + label_start_offset,
                                          label_end_offset - label_start_offset)
                                         : thrust::nullopt,
       edge_ids = org_edgelist_edge_ids ? thrust::make_optional<raft::device_span<edge_id_t const>>(
                                            (*org_edgelist_edge_ids).data() + label_start_offset,
                                            label_end_offset - label_start_offset)
                                        : thrust::nullopt] __device__(size_t l_idx, size_t r_idx) {
        edge_type_t l_edge_type{0};
        edge_type_t r_edge_type{0};
        if (edge_types) {
          l_edge_type = (*edge_types)[l_idx];
          r_edge_type = (*edge_types)[r_idx];
        }

        int32_t l_hop{0};
        int32_t r_hop{0};
        if (hops) {
          l_hop = (*hops)[l_idx];
          r_hop = (*hops)[r_idx];
        }

        vertex_t l_src = srcs[l_idx];
        vertex_t r_src = srcs[r_idx];

        vertex_t l_dst = dsts[l_idx];
        vertex_t r_dst = dsts[r_idx];

        weight_t l_weight{0.0};
        weight_t r_weight{0.0};
        if (weights) {
          l_weight = (*weights)[l_idx];
          r_weight = (*weights)[r_idx];
        }

        edge_id_t l_edge_id{0};
        edge_id_t r_edge_id{0};
        if (edge_ids) {
          l_edge_id = (*edge_ids)[l_idx];
          r_edge_id = (*edge_ids)[r_idx];
        }

        return thrust::make_tuple(l_edge_type, l_hop, l_src, l_dst, l_weight, l_edge_id) <
               thrust::make_tuple(r_edge_type, r_hop, r_src, r_dst, r_weight, r_edge_id);
      });

    for (size_t j = 0; j < num_edge_types; ++j) {
      auto edge_type_start_offset = label_start_offset;
      auto edge_type_end_offset   = label_end_offset;
      if (renumbered_edgelist_label_edge_type_hop_offsets) {
        raft::update_host(&edge_type_start_offset,
                          (*renumbered_edgelist_label_edge_type_hop_offsets).data() +
                            i * num_edge_types * num_hops + j * num_hops,
                          size_t{1},
                          handle.get_stream());
        raft::update_host(&edge_type_end_offset,
                          (*renumbered_edgelist_label_edge_type_hop_offsets).data() +
                            i * num_edge_types * num_hops + (j + 1) * num_hops,
                          size_t{1},
                          handle.get_stream());
        handle.sync_stream();
      }

      if (edge_type_start_offset == edge_type_end_offset) { continue; }

      if (org_edgelist_edge_types) {
        if (static_cast<size_t>(thrust::count_if(
              handle.get_thrust_policy(),
              this_label_org_sorted_indices.begin() + (edge_type_start_offset - label_start_offset),
              this_label_org_sorted_indices.begin() + (edge_type_end_offset - label_start_offset),
              [edge_types = raft::device_span<edge_type_t const>(
                 (*org_edgelist_edge_types).data() + label_start_offset,
                 label_end_offset - label_start_offset),
               edge_type = static_cast<edge_type_t>(j)] __device__(auto i) {
                return edge_types[i] == edge_type;
              })) != edge_type_end_offset - edge_type_start_offset) {
          return false;
        }
      }

      if (org_edgelist_hops) {
        for (size_t k = 0; k < num_hops; ++k) {
          auto hop_start_offset = edge_type_start_offset;
          auto hop_end_offset   = edge_type_end_offset;
          if (renumbered_edgelist_label_edge_type_hop_offsets) {
            raft::update_host(&hop_start_offset,
                              (*renumbered_edgelist_label_edge_type_hop_offsets).data() +
                                i * num_edge_types * num_hops + j * num_hops + k,
                              size_t{1},
                              handle.get_stream());
            raft::update_host(&hop_end_offset,
                              (*renumbered_edgelist_label_edge_type_hop_offsets).data() +
                                i * num_edge_types * num_hops + j * num_hops + k + 1,
                              size_t{1},
                              handle.get_stream());
            handle.sync_stream();
          }

          if (hop_start_offset == hop_end_offset) { continue; }

          if (static_cast<size_t>(thrust::count_if(
                handle.get_thrust_policy(),
                this_label_org_sorted_indices.begin() + (hop_start_offset - label_start_offset),
                this_label_org_sorted_indices.begin() + (hop_end_offset - label_start_offset),
                [hops = raft::device_span<int32_t const>(
                   (*org_edgelist_hops).data() + label_start_offset,
                   label_end_offset - label_start_offset),
                 hop = static_cast<int32_t>(k)] __device__(auto i) { return hops[i] == hop; })) !=
              hop_end_offset - hop_start_offset) {
            return false;
          }
        }
      }

      // unrenumber source vertex IDs

      rmm::device_uvector<vertex_t> this_edge_type_unrenumbered_edgelist_srcs(
        edge_type_end_offset - edge_type_start_offset, handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   renumbered_edgelist_srcs.begin() + edge_type_start_offset,
                   renumbered_edgelist_srcs.begin() + edge_type_end_offset,
                   this_edge_type_unrenumbered_edgelist_srcs.begin());
      {
        vertex_t org_src{};
        raft::update_host(&org_src,
                          org_edgelist_srcs.data() + label_start_offset +
                            this_label_org_sorted_indices.element(
                              edge_type_start_offset - label_start_offset, handle.get_stream()),
                          size_t{1},
                          handle.get_stream());
        handle.sync_stream();
        auto vertex_type = thrust::distance(vertex_type_offsets.begin() + 1,
                                            thrust::upper_bound(handle.get_thrust_policy(),
                                                                vertex_type_offsets.begin() + 1,
                                                                vertex_type_offsets.end(),
                                                                org_src));
        size_t renumber_map_label_start_offset{};
        size_t renumber_map_label_end_offset{};
        raft::update_host(
          &renumber_map_label_start_offset,
          vertex_renumber_map_label_type_offsets.data() + i * num_vertex_types + vertex_type,
          size_t{1},
          handle.get_stream());
        raft::update_host(
          &renumber_map_label_end_offset,
          vertex_renumber_map_label_type_offsets.data() + i * num_vertex_types + vertex_type + 1,
          size_t{1},
          handle.get_stream());
        handle.sync_stream();
        auto renumber_map = raft::device_span<vertex_t const>(
          vertex_renumber_map.data() + renumber_map_label_start_offset,
          renumber_map_label_end_offset - renumber_map_label_start_offset);
        cugraph::unrenumber_int_vertices<vertex_t, false>(
          handle,
          this_edge_type_unrenumbered_edgelist_srcs.data(),
          edge_type_end_offset - edge_type_start_offset,
          renumber_map.data(),
          std::vector<vertex_t>{static_cast<vertex_t>(renumber_map.size())});
      }

      // unrenumber destination vertex IDs

      rmm::device_uvector<vertex_t> this_edge_type_unrenumbered_edgelist_dsts(
        edge_type_end_offset - edge_type_start_offset, handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   renumbered_edgelist_dsts.begin() + edge_type_start_offset,
                   renumbered_edgelist_dsts.begin() + edge_type_end_offset,
                   this_edge_type_unrenumbered_edgelist_dsts.begin());
      {
        vertex_t org_dst{};
        raft::update_host(&org_dst,
                          org_edgelist_dsts.data() + label_start_offset +
                            this_label_org_sorted_indices.element(
                              edge_type_start_offset - label_start_offset, handle.get_stream()),
                          size_t{1},
                          handle.get_stream());
        handle.sync_stream();
        auto vertex_type = thrust::distance(vertex_type_offsets.begin() + 1,
                                            thrust::upper_bound(handle.get_thrust_policy(),
                                                                vertex_type_offsets.begin() + 1,
                                                                vertex_type_offsets.end(),
                                                                org_dst));
        size_t renumber_map_label_start_offset{0};
        size_t renumber_map_label_end_offset{};
        raft::update_host(
          &renumber_map_label_start_offset,
          vertex_renumber_map_label_type_offsets.data() + i * num_vertex_types + vertex_type,
          size_t{1},
          handle.get_stream());
        raft::update_host(
          &renumber_map_label_end_offset,
          vertex_renumber_map_label_type_offsets.data() + i * num_vertex_types + vertex_type + 1,
          size_t{1},
          handle.get_stream());
        handle.sync_stream();
        auto renumber_map = raft::device_span<vertex_t const>(
          vertex_renumber_map.data() + renumber_map_label_start_offset,
          renumber_map_label_end_offset - renumber_map_label_start_offset);
        cugraph::unrenumber_int_vertices<vertex_t, false>(
          handle,
          this_edge_type_unrenumbered_edgelist_dsts.data(),
          edge_type_end_offset - edge_type_start_offset,
          renumber_map.data(),
          std::vector<vertex_t>{static_cast<vertex_t>(renumber_map.size())});
      }

      // unrenumber edge IDs

      std::optional<rmm::device_uvector<edge_id_t>> unrenumbered_edgelist_edge_ids{std::nullopt};
      if (renumbered_edgelist_edge_ids) {
        unrenumbered_edgelist_edge_ids = rmm::device_uvector<edge_id_t>(
          edge_type_end_offset - edge_type_start_offset, handle.get_stream());
        size_t renumber_map_type_start_offset{0};
        size_t renumber_map_type_end_offset = (*edge_id_renumber_map).size();
        if (edge_id_renumber_map_label_type_offsets) {
          raft::update_host(&renumber_map_type_start_offset,
                            (*edge_id_renumber_map_label_type_offsets).data() + i * num_edge_types +
                              static_cast<edge_type_t>(j),
                            size_t{1},
                            handle.get_stream());
          raft::update_host(&renumber_map_type_end_offset,
                            (*edge_id_renumber_map_label_type_offsets).data() + i * num_edge_types +
                              static_cast<edge_type_t>(j) + 1,
                            size_t{1},
                            handle.get_stream());
          handle.sync_stream();
        }
        auto renumber_map = raft::device_span<edge_id_t const>(
          (*edge_id_renumber_map).data() + renumber_map_type_start_offset,
          renumber_map_type_end_offset - renumber_map_type_start_offset);
        thrust::gather(handle.get_thrust_policy(),
                       (*renumbered_edgelist_edge_ids).begin() + edge_type_start_offset,
                       (*renumbered_edgelist_edge_ids).begin() + edge_type_end_offset,
                       renumber_map.begin(),
                       (*unrenumbered_edgelist_edge_ids).begin());
      }

      // sort sorted & renumbered edgelist by (src, dst, (weight), (edge ID))

      rmm::device_uvector<size_t> this_edge_type_unrenumbered_sorted_indices(
        edge_type_end_offset - edge_type_start_offset, handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(),
                       this_edge_type_unrenumbered_sorted_indices.begin(),
                       this_edge_type_unrenumbered_sorted_indices.end(),
                       size_t{0});

      for (size_t k = 0; k < num_hops; ++k) {
        auto hop_start_offset = edge_type_start_offset;
        auto hop_end_offset   = edge_type_end_offset;
        if (renumbered_edgelist_label_edge_type_hop_offsets) {
          raft::update_host(&hop_start_offset,
                            (*renumbered_edgelist_label_edge_type_hop_offsets).data() +
                              i * num_edge_types * num_hops + j * num_hops + k,
                            size_t{1},
                            handle.get_stream());
          raft::update_host(&hop_end_offset,
                            (*renumbered_edgelist_label_edge_type_hop_offsets).data() +
                              i * num_edge_types * num_hops + j * num_hops + k + 1,
                            size_t{1},
                            handle.get_stream());
          handle.sync_stream();
        }

        if (hop_start_offset == hop_end_offset) { continue; }

        thrust::sort(
          handle.get_thrust_policy(),
          this_edge_type_unrenumbered_sorted_indices.begin() +
            (hop_start_offset - edge_type_start_offset),
          this_edge_type_unrenumbered_sorted_indices.begin() +
            (hop_end_offset - edge_type_start_offset),
          [srcs =
             raft::device_span<vertex_t const>(this_edge_type_unrenumbered_edgelist_srcs.data(),
                                               this_edge_type_unrenumbered_edgelist_srcs.size()),
           dsts =
             raft::device_span<vertex_t const>(this_edge_type_unrenumbered_edgelist_dsts.data(),
                                               this_edge_type_unrenumbered_edgelist_dsts.size()),
           weights  = renumbered_edgelist_weights
                        ? thrust::make_optional<raft::device_span<weight_t const>>(
                           (*renumbered_edgelist_weights).data() + edge_type_start_offset,
                           edge_type_end_offset - edge_type_start_offset)
                        : thrust::nullopt,
           edge_ids = renumbered_edgelist_edge_ids
                        ? thrust::make_optional<raft::device_span<edge_id_t const>>(
                            (*renumbered_edgelist_edge_ids).data() + edge_type_start_offset,
                            edge_type_end_offset - edge_type_start_offset)
                        : thrust::nullopt] __device__(size_t l_idx, size_t r_idx) {
            vertex_t l_src = srcs[l_idx];
            vertex_t r_src = srcs[r_idx];

            vertex_t l_dst = dsts[l_idx];
            vertex_t r_dst = dsts[r_idx];

            weight_t l_weight{0.0};
            weight_t r_weight{0.0};
            if (weights) {
              l_weight = (*weights)[l_idx];
              r_weight = (*weights)[r_idx];
            }

            edge_id_t l_edge_id{0};
            edge_id_t r_edge_id{0};
            if (edge_ids) {
              l_edge_id = (*edge_ids)[l_idx];
              r_edge_id = (*edge_ids)[r_idx];
            }

            return thrust::make_tuple(l_src, l_dst, l_weight, l_edge_id) <
                   thrust::make_tuple(r_src, r_dst, r_weight, r_edge_id);
          });
      }

      // compare

      if (!thrust::equal(
            handle.get_thrust_policy(),
            this_label_org_sorted_indices.begin() + (edge_type_start_offset - label_start_offset),
            this_label_org_sorted_indices.begin() + (edge_type_end_offset - label_start_offset),
            this_edge_type_unrenumbered_sorted_indices.begin(),
            [org_srcs =
               raft::device_span<vertex_t const>(org_edgelist_srcs.data() + label_start_offset,
                                                 label_end_offset - label_start_offset),
             org_dsts =
               raft::device_span<vertex_t const>(org_edgelist_dsts.data() + label_start_offset,
                                                 label_end_offset - label_start_offset),
             org_weights  = org_edgelist_weights
                              ? thrust::make_optional<raft::device_span<weight_t const>>(
                                 (*org_edgelist_weights).data() + label_start_offset,
                                 label_end_offset - label_start_offset)
                              : thrust::nullopt,
             org_edge_ids = org_edgelist_edge_ids
                              ? thrust::make_optional<raft::device_span<edge_id_t const>>(
                                  (*org_edgelist_edge_ids).data() + label_start_offset,
                                  label_end_offset - label_start_offset)
                              : thrust::nullopt,
             unrenumbered_srcs =
               raft::device_span<vertex_t const>(this_edge_type_unrenumbered_edgelist_srcs.data(),
                                                 this_edge_type_unrenumbered_edgelist_srcs.size()),
             unrenumbered_dsts =
               raft::device_span<vertex_t const>(this_edge_type_unrenumbered_edgelist_dsts.data(),
                                                 this_edge_type_unrenumbered_edgelist_dsts.size()),
             unrenumbered_weights =
               renumbered_edgelist_weights
                 ? thrust::make_optional<raft::device_span<weight_t const>>(
                     (*renumbered_edgelist_weights).data() + edge_type_start_offset,
                     edge_type_end_offset - edge_type_start_offset)
                 : thrust::nullopt,
             unrenumbered_edge_ids =
               unrenumbered_edgelist_edge_ids
                 ? thrust::make_optional<raft::device_span<edge_id_t const>>(
                     (*unrenumbered_edgelist_edge_ids).data(),
                     (*unrenumbered_edgelist_edge_ids).size())
                 : thrust::
                     nullopt] __device__(size_t org_idx /* from label_start_offset */,
                                         size_t
                                           unrenumbered_idx /* from edge_type_start_offset */) {
              auto org_src          = org_srcs[org_idx];
              auto unrenumbered_src = unrenumbered_srcs[unrenumbered_idx];
              if (org_src != unrenumbered_src) { return false; }

              auto org_dst          = org_dsts[org_idx];
              auto unrenumbered_dst = unrenumbered_dsts[unrenumbered_idx];
              if (org_dst != unrenumbered_dst) { return false; }

              weight_t org_weight{0.0};
              if (org_weights) { org_weight = (*org_weights)[org_idx]; }
              weight_t unrenumbered_weight{0.0};
              if (unrenumbered_weights) {
                unrenumbered_weight = (*unrenumbered_weights)[unrenumbered_idx];
              }
              if (org_weight != unrenumbered_weight) { return false; }

              edge_id_t org_edge_id{0};
              if (org_edge_ids) { org_edge_id = (*org_edge_ids)[org_idx]; }
              edge_id_t unrenumbered_edge_id{0};
              if (unrenumbered_edge_ids) {
                unrenumbered_edge_id = (*unrenumbered_edge_ids)[unrenumbered_idx];
              }

              if (org_edge_id != unrenumbered_edge_id) {
                     (int)org_edge_id,
                     (int)unrenumbered_edge_id);
              }
              return org_edge_id == unrenumbered_edge_id;
            })) {
        return false;
      }
    }
  }

  return true;
}

template bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> org_edgelist_srcs,
  raft::device_span<int32_t const> org_edgelist_dsts,
  std::optional<raft::device_span<float const>> org_edgelist_weights,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<int32_t const> renumbered_edgelist_srcs,
  raft::device_span<int32_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<float const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int32_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<int32_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<int32_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<int32_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops);

template bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> org_edgelist_srcs,
  raft::device_span<int32_t const> org_edgelist_dsts,
  std::optional<raft::device_span<double const>> org_edgelist_weights,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<int32_t const> renumbered_edgelist_srcs,
  raft::device_span<int32_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<double const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int32_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<int32_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<int32_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<int32_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops);

template bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> org_edgelist_srcs,
  raft::device_span<int32_t const> org_edgelist_dsts,
  std::optional<raft::device_span<float const>> org_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<int32_t const> renumbered_edgelist_srcs,
  raft::device_span<int32_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<float const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<int32_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<int64_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<int32_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops);

template bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> org_edgelist_srcs,
  raft::device_span<int32_t const> org_edgelist_dsts,
  std::optional<raft::device_span<double const>> org_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<int32_t const> renumbered_edgelist_srcs,
  raft::device_span<int32_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<double const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<int32_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<int64_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<int32_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops);

template bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> org_edgelist_srcs,
  raft::device_span<int64_t const> org_edgelist_dsts,
  std::optional<raft::device_span<float const>> org_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<int64_t const> renumbered_edgelist_srcs,
  raft::device_span<int64_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<float const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<int64_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<int64_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<int64_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops);

template bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> org_edgelist_srcs,
  raft::device_span<int64_t const> org_edgelist_dsts,
  std::optional<raft::device_span<double const>> org_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<int32_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<int64_t const> renumbered_edgelist_srcs,
  raft::device_span<int64_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<double const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<int64_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<int64_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<int64_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<int64_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops);

template <typename vertex_t>
bool check_vertex_renumber_map_invariants(
  raft::handle_t const& handle,
  std::optional<raft::device_span<vertex_t const>> starting_vertices,
  raft::device_span<vertex_t const> org_edgelist_srcs,
  raft::device_span<vertex_t const> org_edgelist_dsts,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  raft::device_span<vertex_t const> renumber_map,
  bool src_is_major)
{
  // Check the invariants in renumber_map
  // Say we found the minimum (primary key:hop, secondary key:flag) pairs for every unique vertices,
  // where flag is 0 for majors and 1 for minors. Then, vertices with smaller (hop, flag)
  // pairs should be renumbered to smaller numbers than vertices with larger (hop, flag) pairs.
  auto org_edgelist_majors = src_is_major ? org_edgelist_srcs : org_edgelist_dsts;
  auto org_edgelist_minors = src_is_major ? org_edgelist_dsts : org_edgelist_srcs;

  rmm::device_uvector<vertex_t> unique_majors(org_edgelist_majors.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               org_edgelist_majors.begin(),
               org_edgelist_majors.end(),
               unique_majors.begin());
  if (starting_vertices) {
    auto old_size = unique_majors.size();
    unique_majors.resize(old_size + (*starting_vertices).size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 (*starting_vertices).begin(),
                 (*starting_vertices).end(),
                 unique_majors.begin() + old_size);
  }

  std::optional<rmm::device_uvector<int32_t>> unique_major_hops =
    org_edgelist_hops ? std::make_optional<rmm::device_uvector<int32_t>>(
                          (*org_edgelist_hops).size(), handle.get_stream())
                      : std::nullopt;
  if (org_edgelist_hops) {
    thrust::copy(handle.get_thrust_policy(),
                 (*org_edgelist_hops).begin(),
                 (*org_edgelist_hops).end(),
                 (*unique_major_hops).begin());
    if (starting_vertices) {
      auto old_size = (*unique_major_hops).size();
      (*unique_major_hops).resize(old_size + (*starting_vertices).size(), handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   (*unique_major_hops).begin() + old_size,
                   (*unique_major_hops).end(),
                   int32_t{0});
    }

    auto pair_first =
      thrust::make_zip_iterator(unique_majors.begin(), (*unique_major_hops).begin());
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + unique_majors.size());
    unique_majors.resize(
      thrust::distance(unique_majors.begin(),
                       thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                            unique_majors.begin(),
                                                            unique_majors.end(),
                                                            (*unique_major_hops).begin()))),
      handle.get_stream());
    (*unique_major_hops).resize(unique_majors.size(), handle.get_stream());
  } else {
    thrust::sort(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end());
    unique_majors.resize(
      thrust::distance(
        unique_majors.begin(),
        thrust::unique(handle.get_thrust_policy(), unique_majors.begin(), unique_majors.end())),
      handle.get_stream());
  }

  rmm::device_uvector<vertex_t> unique_minors(org_edgelist_minors.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               org_edgelist_minors.begin(),
               org_edgelist_minors.end(),
               unique_minors.begin());
  std::optional<rmm::device_uvector<int32_t>> unique_minor_hops =
    org_edgelist_hops ? std::make_optional<rmm::device_uvector<int32_t>>(
                          (*org_edgelist_hops).size(), handle.get_stream())
                      : std::nullopt;
  if (org_edgelist_hops) {
    thrust::copy(handle.get_thrust_policy(),
                 (*org_edgelist_hops).begin(),
                 (*org_edgelist_hops).end(),
                 (*unique_minor_hops).begin());

    auto pair_first =
      thrust::make_zip_iterator(unique_minors.begin(), (*unique_minor_hops).begin());
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + unique_minors.size());
    unique_minors.resize(
      thrust::distance(unique_minors.begin(),
                       thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                            unique_minors.begin(),
                                                            unique_minors.end(),
                                                            (*unique_minor_hops).begin()))),
      handle.get_stream());
    (*unique_minor_hops).resize(unique_minors.size(), handle.get_stream());
  } else {
    thrust::sort(handle.get_thrust_policy(), unique_minors.begin(), unique_minors.end());
    unique_minors.resize(
      thrust::distance(
        unique_minors.begin(),
        thrust::unique(handle.get_thrust_policy(), unique_minors.begin(), unique_minors.end())),
      handle.get_stream());
  }

  rmm::device_uvector<vertex_t> sorted_org_vertices(renumber_map.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> matching_renumbered_vertices(sorted_org_vertices.size(),
                                                             handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               renumber_map.begin(),
               renumber_map.end(),
               sorted_org_vertices.begin());
  thrust::sequence(handle.get_thrust_policy(),
                   matching_renumbered_vertices.begin(),
                   matching_renumbered_vertices.end(),
                   vertex_t{0});
  thrust::sort_by_key(handle.get_thrust_policy(),
                      sorted_org_vertices.begin(),
                      sorted_org_vertices.end(),
                      matching_renumbered_vertices.begin());

  if (org_edgelist_hops) {
    rmm::device_uvector<vertex_t> merged_vertices(unique_majors.size() + unique_minors.size(),
                                                  handle.get_stream());
    rmm::device_uvector<int32_t> merged_hops(merged_vertices.size(), handle.get_stream());
    rmm::device_uvector<int8_t> merged_flags(merged_vertices.size(), handle.get_stream());

    auto major_triplet_first = thrust::make_zip_iterator(unique_majors.begin(),
                                                         (*unique_major_hops).begin(),
                                                         thrust::make_constant_iterator(int8_t{0}));
    auto minor_triplet_first = thrust::make_zip_iterator(unique_minors.begin(),
                                                         (*unique_minor_hops).begin(),
                                                         thrust::make_constant_iterator(int8_t{1}));
    thrust::merge(handle.get_thrust_policy(),
                  major_triplet_first,
                  major_triplet_first + unique_majors.size(),
                  minor_triplet_first,
                  minor_triplet_first + unique_minors.size(),
                  thrust::make_zip_iterator(
                    merged_vertices.begin(), merged_hops.begin(), merged_flags.begin()));
    merged_vertices.resize(
      thrust::distance(merged_vertices.begin(),
                       thrust::get<0>(thrust::unique_by_key(
                         handle.get_thrust_policy(),
                         merged_vertices.begin(),
                         merged_vertices.end(),
                         thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin())))),
      handle.get_stream());
    merged_hops.resize(merged_vertices.size(), handle.get_stream());
    merged_flags.resize(merged_vertices.size(), handle.get_stream());

    auto sort_key_first = thrust::make_zip_iterator(merged_hops.begin(), merged_flags.begin());
    thrust::sort_by_key(handle.get_thrust_policy(),
                        sort_key_first,
                        sort_key_first + merged_hops.size(),
                        merged_vertices.begin());

    auto num_unique_keys = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(merged_hops.size()),
      cugraph::detail::is_first_in_run_t<decltype(sort_key_first)>{sort_key_first});
    rmm::device_uvector<vertex_t> min_vertices(num_unique_keys, handle.get_stream());
    rmm::device_uvector<vertex_t> max_vertices(num_unique_keys, handle.get_stream());

    auto renumbered_merged_vertex_first = thrust::make_transform_iterator(
      merged_vertices.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [sorted_org_vertices = raft::device_span<vertex_t const>(sorted_org_vertices.data(),
                                                                 sorted_org_vertices.size()),
         matching_renumbered_vertices = raft::device_span<vertex_t const>(
           matching_renumbered_vertices.data(),
           matching_renumbered_vertices.size())] __device__(vertex_t major) {
          auto it = thrust::lower_bound(
            thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), major);
          return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(), it)];
        }));

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          sort_key_first,
                          sort_key_first + merged_hops.size(),
                          renumbered_merged_vertex_first,
                          thrust::make_discard_iterator(),
                          min_vertices.begin(),
                          thrust::equal_to<thrust::tuple<int32_t, int8_t>>{},
                          thrust::minimum<vertex_t>{});
    thrust::reduce_by_key(handle.get_thrust_policy(),
                          sort_key_first,
                          sort_key_first + merged_hops.size(),
                          renumbered_merged_vertex_first,
                          thrust::make_discard_iterator(),
                          max_vertices.begin(),
                          thrust::equal_to<thrust::tuple<int32_t, int8_t>>{},
                          thrust::maximum<vertex_t>{});

    auto num_violations = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{1}),
      thrust::make_counting_iterator(min_vertices.size()),
      [min_vertices = raft::device_span<vertex_t const>(min_vertices.data(), min_vertices.size()),
       max_vertices = raft::device_span<vertex_t const>(max_vertices.data(),
                                                        max_vertices.size())] __device__(size_t i) {
        return min_vertices[i] <= max_vertices[i - 1];
      });

    return (num_violations == 0);
  } else {
    unique_minors.resize(
      thrust::distance(
        unique_minors.begin(),
        thrust::remove_if(handle.get_thrust_policy(),
                          unique_minors.begin(),
                          unique_minors.end(),
                          [sorted_unique_majors = raft::device_span<vertex_t const>(
                             unique_majors.data(), unique_majors.size())] __device__(auto minor) {
                            return thrust::binary_search(thrust::seq,
                                                         sorted_unique_majors.begin(),
                                                         sorted_unique_majors.end(),
                                                         minor);
                          })),
      handle.get_stream());

    auto max_major_renumbered_vertex = thrust::transform_reduce(
      handle.get_thrust_policy(),
      unique_majors.begin(),
      unique_majors.end(),
      cuda::proclaim_return_type<vertex_t>(
        [sorted_org_vertices = raft::device_span<vertex_t const>(sorted_org_vertices.data(),
                                                                 sorted_org_vertices.size()),
         matching_renumbered_vertices = raft::device_span<vertex_t const>(
           matching_renumbered_vertices.data(),
           matching_renumbered_vertices.size())] __device__(vertex_t major) -> vertex_t {
          auto it = thrust::lower_bound(
            thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), major);
          return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(), it)];
        }),
      std::numeric_limits<vertex_t>::lowest(),
      thrust::maximum<vertex_t>{});

    auto min_minor_renumbered_vertex = thrust::transform_reduce(
      handle.get_thrust_policy(),
      unique_minors.begin(),
      unique_minors.end(),
      cuda::proclaim_return_type<vertex_t>(
        [sorted_org_vertices = raft::device_span<vertex_t const>(sorted_org_vertices.data(),
                                                                 sorted_org_vertices.size()),
         matching_renumbered_vertices = raft::device_span<vertex_t const>(
           matching_renumbered_vertices.data(),
           matching_renumbered_vertices.size())] __device__(vertex_t minor) -> vertex_t {
          auto it = thrust::lower_bound(
            thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), minor);
          return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(), it)];
        }),
      std::numeric_limits<vertex_t>::max(),
      thrust::minimum<vertex_t>{});

    return (max_major_renumbered_vertex < min_minor_renumbered_vertex);
  }
}

template bool check_vertex_renumber_map_invariants(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int32_t const>> starting_vertices,
  raft::device_span<int32_t const> org_edgelist_srcs,
  raft::device_span<int32_t const> org_edgelist_dsts,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  raft::device_span<int32_t const> renumber_map,
  bool src_is_major);

template bool check_vertex_renumber_map_invariants(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int64_t const>> starting_vertices,
  raft::device_span<int64_t const> org_edgelist_srcs,
  raft::device_span<int64_t const> org_edgelist_dsts,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  raft::device_span<int64_t const> renumber_map,
  bool src_is_major);

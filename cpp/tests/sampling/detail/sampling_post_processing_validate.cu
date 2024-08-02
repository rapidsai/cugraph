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

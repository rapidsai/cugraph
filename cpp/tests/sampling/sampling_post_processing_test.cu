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

#include "utilities/base_fixture.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>

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

#include <gtest/gtest.h>

struct SamplingPostProcessing_Usecase {
  size_t num_labels{};
  size_t num_seeds_per_label{};
  std::vector<int32_t> fanouts{{-1}};
  bool sample_with_replacement{false};

  bool src_is_major{true};
  bool renumber_with_seeds{false};
  bool compress_per_hop{false};
  bool doubly_compress{false};
  bool check_correctness{true};
};

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

template <typename vertex_t>
bool check_renumber_map_invariants(
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
  // where flag is 0 for sources and 1 for destinations. Then, vertices with smaller (hop, flag)
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
      [sorted_org_vertices =
         raft::device_span<vertex_t const>(sorted_org_vertices.data(), sorted_org_vertices.size()),
       matching_renumbered_vertices = raft::device_span<vertex_t const>(
         matching_renumbered_vertices.data(),
         matching_renumbered_vertices.size())] __device__(vertex_t major) -> vertex_t {
        auto it = thrust::lower_bound(
          thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), major);
        return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(), it)];
      },
      std::numeric_limits<vertex_t>::lowest(),
      thrust::maximum<vertex_t>{});

    auto min_minor_renumbered_vertex = thrust::transform_reduce(
      handle.get_thrust_policy(),
      unique_minors.begin(),
      unique_minors.end(),
      [sorted_org_vertices =
         raft::device_span<vertex_t const>(sorted_org_vertices.data(), sorted_org_vertices.size()),
       matching_renumbered_vertices = raft::device_span<vertex_t const>(
         matching_renumbered_vertices.data(),
         matching_renumbered_vertices.size())] __device__(vertex_t minor) -> vertex_t {
        auto it = thrust::lower_bound(
          thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), minor);
        return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(), it)];
      },
      std::numeric_limits<vertex_t>::max(),
      thrust::minimum<vertex_t>{});

    return (max_major_renumbered_vertex < min_minor_renumbered_vertex);
  }
}

template <typename input_usecase_t>
class Tests_SamplingPostProcessing
  : public ::testing::TestWithParam<std::tuple<SamplingPostProcessing_Usecase, input_usecase_t>> {
 public:
  Tests_SamplingPostProcessing() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(
    std::tuple<SamplingPostProcessing_Usecase const&, input_usecase_t const&> const& param)
  {
    using label_t     = int32_t;
    using weight_t    = float;
    using edge_id_t   = vertex_t;
    using edge_type_t = int32_t;

    bool constexpr store_transposed = false;
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = true;

    auto [sampling_post_processing_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    raft::random::RngState rng_state(0);

    rmm::device_uvector<vertex_t> starting_vertices(
      sampling_post_processing_usecase.num_labels *
        sampling_post_processing_usecase.num_seeds_per_label,
      handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         starting_vertices.data(),
                                         starting_vertices.size(),
                                         vertex_t{0},
                                         graph_view.number_of_vertices(),
                                         rng_state);
    auto starting_vertex_labels = (sampling_post_processing_usecase.num_labels > 1)
                                    ? std::make_optional<rmm::device_uvector<label_t>>(
                                        starting_vertices.size(), handle.get_stream())
                                    : std::nullopt;
    auto starting_vertex_label_offsets =
      (sampling_post_processing_usecase.num_labels > 1)
        ? std::make_optional<rmm::device_uvector<size_t>>(
            sampling_post_processing_usecase.num_labels + 1, handle.get_stream())
        : std::nullopt;
    if (starting_vertex_labels) {
      thrust::tabulate(
        handle.get_thrust_policy(),
        (*starting_vertex_labels).begin(),
        (*starting_vertex_labels).end(),
        [num_seeds_per_label = sampling_post_processing_usecase.num_seeds_per_label] __device__(
          size_t i) { return static_cast<label_t>(i / num_seeds_per_label); });
      thrust::tabulate(
        handle.get_thrust_policy(),
        (*starting_vertex_label_offsets).begin(),
        (*starting_vertex_label_offsets).end(),
        [num_seeds_per_label = sampling_post_processing_usecase.num_seeds_per_label] __device__(
          size_t i) { return num_seeds_per_label * i; });
    }

    rmm::device_uvector<vertex_t> org_edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> org_edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> org_edgelist_weights{std::nullopt};
    std::optional<rmm::device_uvector<int32_t>> org_edgelist_hops{std::nullopt};
    std::optional<rmm::device_uvector<label_t>> org_labels{std::nullopt};
    std::optional<rmm::device_uvector<size_t>> org_edgelist_label_offsets{std::nullopt};
    std::tie(org_edgelist_srcs,
             org_edgelist_dsts,
             org_edgelist_weights,
             std::ignore,
             std::ignore,
             org_edgelist_hops,
             org_labels,
             org_edgelist_label_offsets) = cugraph::uniform_neighbor_sample<vertex_t,
                                                                            edge_t,
                                                                            weight_t,
                                                                            edge_type_t,
                                                                            label_t,
                                                                            store_transposed,
                                                                            false>(
      handle,
      graph_view,
      edge_weight_view,
      std::nullopt,
      std::nullopt,
      raft::device_span<vertex_t const>(starting_vertices.data(), starting_vertices.size()),
      starting_vertex_labels ? std::make_optional<raft::device_span<label_t const>>(
                                 (*starting_vertex_labels).data(), (*starting_vertex_labels).size())
                             : std::nullopt,
      std::nullopt,
      raft::host_span<int32_t const>(sampling_post_processing_usecase.fanouts.data(),
                                     sampling_post_processing_usecase.fanouts.size()),
      rng_state,
      sampling_post_processing_usecase.fanouts.size() > 1,
      sampling_post_processing_usecase.sample_with_replacement,
      (!sampling_post_processing_usecase.compress_per_hop &&
       (sampling_post_processing_usecase.fanouts.size() > 1))
        ? cugraph::prior_sources_behavior_t::EXCLUDE
        : cugraph::prior_sources_behavior_t::DEFAULT,
      false);

    if (!sampling_post_processing_usecase.src_is_major) {
      std::swap(org_edgelist_srcs, org_edgelist_dsts);
    }

    {
      rmm::device_uvector<vertex_t> renumbered_and_sorted_edgelist_srcs(org_edgelist_srcs.size(),
                                                                        handle.get_stream());
      rmm::device_uvector<vertex_t> renumbered_and_sorted_edgelist_dsts(org_edgelist_dsts.size(),
                                                                        handle.get_stream());
      auto renumbered_and_sorted_edgelist_weights =
        org_edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                 (*org_edgelist_weights).size(), handle.get_stream())
                             : std::nullopt;
      std::optional<rmm::device_uvector<edge_id_t>> renumbered_and_sorted_edgelist_edge_ids{
        std::nullopt};
      std::optional<rmm::device_uvector<edge_type_t>> renumbered_and_sorted_edgelist_edge_types{
        std::nullopt};
      auto renumbered_and_sorted_edgelist_hops =
        org_edgelist_hops ? std::make_optional(rmm::device_uvector<int32_t>(
                              (*org_edgelist_hops).size(), handle.get_stream()))
                          : std::nullopt;

      raft::copy(renumbered_and_sorted_edgelist_srcs.data(),
                 org_edgelist_srcs.data(),
                 org_edgelist_srcs.size(),
                 handle.get_stream());
      raft::copy(renumbered_and_sorted_edgelist_dsts.data(),
                 org_edgelist_dsts.data(),
                 org_edgelist_dsts.size(),
                 handle.get_stream());
      if (renumbered_and_sorted_edgelist_weights) {
        raft::copy((*renumbered_and_sorted_edgelist_weights).data(),
                   (*org_edgelist_weights).data(),
                   (*org_edgelist_weights).size(),
                   handle.get_stream());
      }
      if (renumbered_and_sorted_edgelist_hops) {
        raft::copy((*renumbered_and_sorted_edgelist_hops).data(),
                   (*org_edgelist_hops).data(),
                   (*org_edgelist_hops).size(),
                   handle.get_stream());
      }

      std::optional<rmm::device_uvector<size_t>> renumbered_and_sorted_edgelist_label_hop_offsets{
        std::nullopt};
      rmm::device_uvector<vertex_t> renumbered_and_sorted_renumber_map(0, handle.get_stream());
      std::optional<rmm::device_uvector<size_t>> renumbered_and_sorted_renumber_map_label_offsets{
        std::nullopt};

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.start("Renumber and sort sampled edgelist");
      }

      std::tie(renumbered_and_sorted_edgelist_srcs,
               renumbered_and_sorted_edgelist_dsts,
               renumbered_and_sorted_edgelist_weights,
               renumbered_and_sorted_edgelist_edge_ids,
               renumbered_and_sorted_edgelist_edge_types,
               renumbered_and_sorted_edgelist_label_hop_offsets,
               renumbered_and_sorted_renumber_map,
               renumbered_and_sorted_renumber_map_label_offsets) =
        cugraph::renumber_and_sort_sampled_edgelist<vertex_t, weight_t, edge_id_t, edge_type_t>(
          handle,
          std::move(renumbered_and_sorted_edgelist_srcs),
          std::move(renumbered_and_sorted_edgelist_dsts),
          std::move(renumbered_and_sorted_edgelist_weights),
          std::move(renumbered_and_sorted_edgelist_edge_ids),
          std::move(renumbered_and_sorted_edgelist_edge_types),
          std::move(renumbered_and_sorted_edgelist_hops),
          sampling_post_processing_usecase.renumber_with_seeds
            ? std::make_optional<raft::device_span<vertex_t const>>(starting_vertices.data(),
                                                                    starting_vertices.size())
            : std::nullopt,
          (sampling_post_processing_usecase.renumber_with_seeds && starting_vertex_label_offsets)
            ? std::make_optional<raft::device_span<size_t const>>(
                (*starting_vertex_label_offsets).data(), (*starting_vertex_label_offsets).size())
            : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional(raft::device_span<size_t const>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size()))
            : std::nullopt,
          sampling_post_processing_usecase.num_labels,
          sampling_post_processing_usecase.fanouts.size(),
          sampling_post_processing_usecase.src_is_major);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      if (sampling_post_processing_usecase.check_correctness) {
        if (renumbered_and_sorted_edgelist_label_hop_offsets) {
          ASSERT_TRUE((*renumbered_and_sorted_edgelist_label_hop_offsets).size() ==
                      sampling_post_processing_usecase.num_labels *
                          sampling_post_processing_usecase.fanouts.size() +
                        1)
            << "Renumbered and sorted edge list (label,hop) offset array size should coincide with "
               "the number of labels * the number of hops + 1.";

          ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                        (*renumbered_and_sorted_edgelist_label_hop_offsets).begin(),
                                        (*renumbered_and_sorted_edgelist_label_hop_offsets).end()))
            << "Renumbered and sorted edge list (label,hop) offset array values should be "
               "non-decreasing.";

          ASSERT_TRUE(
            (*renumbered_and_sorted_edgelist_label_hop_offsets).back_element(handle.get_stream()) ==
            renumbered_and_sorted_edgelist_srcs.size())
            << "Renumbered and sorted edge list (label,hop) offset array's last element should "
               "coincide with the number of edges.";
        }

        if (renumbered_and_sorted_renumber_map_label_offsets) {
          ASSERT_TRUE((*renumbered_and_sorted_renumber_map_label_offsets).size() ==
                      sampling_post_processing_usecase.num_labels + 1)
            << "Renumbered and sorted offset (label, hop) offset array size should coincide with "
               "the number of labels + 1.";

          ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                        (*renumbered_and_sorted_renumber_map_label_offsets).begin(),
                                        (*renumbered_and_sorted_renumber_map_label_offsets).end()))
            << "Renumbered and sorted renumber map label offset array values should be "
               "non-decreasing.";

          ASSERT_TRUE(
            (*renumbered_and_sorted_renumber_map_label_offsets).back_element(handle.get_stream()) ==
            renumbered_and_sorted_renumber_map.size())
            << "Renumbered and sorted renumber map label offset array's last value should coincide "
               "with the renumber map size.";
        }

        for (size_t i = 0; i < sampling_post_processing_usecase.num_labels; ++i) {
          size_t starting_vertex_start_offset =
            starting_vertex_label_offsets
              ? (*starting_vertex_label_offsets).element(i, handle.get_stream())
              : size_t{0};
          size_t starting_vertex_end_offset =
            starting_vertex_label_offsets
              ? (*starting_vertex_label_offsets).element(i + 1, handle.get_stream())
              : starting_vertices.size();

          size_t edgelist_start_offset =
            org_edgelist_label_offsets
              ? (*org_edgelist_label_offsets).element(i, handle.get_stream())
              : size_t{0};
          size_t edgelist_end_offset =
            org_edgelist_label_offsets
              ? (*org_edgelist_label_offsets).element(i + 1, handle.get_stream())
              : org_edgelist_srcs.size();
          if (edgelist_start_offset == edgelist_end_offset) continue;

          auto this_label_starting_vertices = raft::device_span<vertex_t const>(
            starting_vertices.data() + starting_vertex_start_offset,
            starting_vertex_end_offset - starting_vertex_start_offset);

          auto this_label_org_edgelist_srcs =
            raft::device_span<vertex_t const>(org_edgelist_srcs.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_org_edgelist_dsts =
            raft::device_span<vertex_t const>(org_edgelist_dsts.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_org_edgelist_hops =
            org_edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                                  (*org_edgelist_hops).data() + edgelist_start_offset,
                                  edgelist_end_offset - edgelist_start_offset)
                              : std::nullopt;
          auto this_label_org_edgelist_weights =
            org_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                     (*org_edgelist_weights).data() + edgelist_start_offset,
                                     edgelist_end_offset - edgelist_start_offset)
                                 : std::nullopt;

          auto this_label_output_edgelist_srcs = raft::device_span<vertex_t const>(
            renumbered_and_sorted_edgelist_srcs.data() + edgelist_start_offset,
            edgelist_end_offset - edgelist_start_offset);
          auto this_label_output_edgelist_dsts = raft::device_span<vertex_t const>(
            renumbered_and_sorted_edgelist_dsts.data() + edgelist_start_offset,
            edgelist_end_offset - edgelist_start_offset);
          auto this_label_output_edgelist_weights =
            renumbered_and_sorted_edgelist_weights
              ? std::make_optional<raft::device_span<weight_t const>>(
                  (*renumbered_and_sorted_edgelist_weights).data() + edgelist_start_offset,
                  edgelist_end_offset - edgelist_start_offset)
              : std::nullopt;

          size_t renumber_map_start_offset =
            renumbered_and_sorted_renumber_map_label_offsets
              ? (*renumbered_and_sorted_renumber_map_label_offsets).element(i, handle.get_stream())
              : size_t{0};
          size_t renumber_map_end_offset      = renumbered_and_sorted_renumber_map_label_offsets
                                                  ? (*renumbered_and_sorted_renumber_map_label_offsets)
                                                 .element(i + 1, handle.get_stream())
                                                  : renumbered_and_sorted_renumber_map.size();
          auto this_label_output_renumber_map = raft::device_span<vertex_t const>(
            renumbered_and_sorted_renumber_map.data() + renumber_map_start_offset,
            renumber_map_end_offset - renumber_map_start_offset);

          // check whether the edges are properly sorted

          auto this_label_output_edgelist_majors = sampling_post_processing_usecase.src_is_major
                                                     ? this_label_output_edgelist_srcs
                                                     : this_label_output_edgelist_dsts;
          auto this_label_output_edgelist_minors = sampling_post_processing_usecase.src_is_major
                                                     ? this_label_output_edgelist_dsts
                                                     : this_label_output_edgelist_srcs;

          if (this_label_org_edgelist_hops) {
            auto num_hops   = sampling_post_processing_usecase.fanouts.size();
            auto edge_first = thrust::make_zip_iterator(this_label_output_edgelist_majors.begin(),
                                                        this_label_output_edgelist_minors.begin());
            for (size_t j = 0; j < num_hops; ++j) {
              auto hop_start_offset = (*renumbered_and_sorted_edgelist_label_hop_offsets)
                                        .element(i * num_hops + j, handle.get_stream()) -
                                      (*renumbered_and_sorted_edgelist_label_hop_offsets)
                                        .element(i * num_hops, handle.get_stream());
              auto hop_end_offset = (*renumbered_and_sorted_edgelist_label_hop_offsets)
                                      .element(i * num_hops + j + 1, handle.get_stream()) -
                                    (*renumbered_and_sorted_edgelist_label_hop_offsets)
                                      .element(i * num_hops, handle.get_stream());
              ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                            edge_first + hop_start_offset,
                                            edge_first + hop_end_offset))
                << "Renumbered and sorted output edges are not properly sorted.";
            }
          } else {
            auto edge_first = thrust::make_zip_iterator(this_label_output_edgelist_majors.begin(),
                                                        this_label_output_edgelist_minors.begin());
            ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                          edge_first,
                                          edge_first + this_label_output_edgelist_majors.size()))
              << "Renumbered and sorted output edges are not properly sorted.";
          }

          // check whether renumbering recovers the original edge list

          ASSERT_TRUE(compare_edgelist(handle,
                                       this_label_org_edgelist_srcs,
                                       this_label_org_edgelist_dsts,
                                       this_label_org_edgelist_weights,
                                       this_label_output_edgelist_srcs,
                                       this_label_output_edgelist_dsts,
                                       this_label_output_edgelist_weights,
                                       std::make_optional(this_label_output_renumber_map)))
            << "Unrenumbering the renumbered and sorted edge list does not recover the original "
               "edgelist.";

          // Check the invariants in renumber_map

          ASSERT_TRUE(check_renumber_map_invariants(
            handle,
            sampling_post_processing_usecase.renumber_with_seeds
              ? std::make_optional<raft::device_span<vertex_t const>>(
                  this_label_starting_vertices.data(), this_label_starting_vertices.size())
              : std::nullopt,
            this_label_org_edgelist_srcs,
            this_label_org_edgelist_dsts,
            this_label_org_edgelist_hops,
            this_label_output_renumber_map,
            sampling_post_processing_usecase.src_is_major))
            << "Renumbered and sorted output renumber map violates invariants.";
        }
      }
    }

    {
      rmm::device_uvector<vertex_t> renumbered_and_compressed_edgelist_srcs(
        org_edgelist_srcs.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> renumbered_and_compressed_edgelist_dsts(
        org_edgelist_dsts.size(), handle.get_stream());
      auto renumbered_and_compressed_edgelist_weights =
        org_edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                 (*org_edgelist_weights).size(), handle.get_stream())
                             : std::nullopt;
      std::optional<rmm::device_uvector<edge_id_t>> renumbered_and_compressed_edgelist_edge_ids{
        std::nullopt};
      std::optional<rmm::device_uvector<edge_type_t>> renumbered_and_compressed_edgelist_edge_types{
        std::nullopt};
      auto renumbered_and_compressed_edgelist_hops =
        org_edgelist_hops ? std::make_optional(rmm::device_uvector<int32_t>(
                              (*org_edgelist_hops).size(), handle.get_stream()))
                          : std::nullopt;

      raft::copy(renumbered_and_compressed_edgelist_srcs.data(),
                 org_edgelist_srcs.data(),
                 org_edgelist_srcs.size(),
                 handle.get_stream());
      raft::copy(renumbered_and_compressed_edgelist_dsts.data(),
                 org_edgelist_dsts.data(),
                 org_edgelist_dsts.size(),
                 handle.get_stream());
      if (renumbered_and_compressed_edgelist_weights) {
        raft::copy((*renumbered_and_compressed_edgelist_weights).data(),
                   (*org_edgelist_weights).data(),
                   (*org_edgelist_weights).size(),
                   handle.get_stream());
      }
      if (renumbered_and_compressed_edgelist_hops) {
        raft::copy((*renumbered_and_compressed_edgelist_hops).data(),
                   (*org_edgelist_hops).data(),
                   (*org_edgelist_hops).size(),
                   handle.get_stream());
      }

      std::optional<rmm::device_uvector<vertex_t>> renumbered_and_compressed_nzd_vertices{
        std::nullopt};
      rmm::device_uvector<size_t> renumbered_and_compressed_offsets(0, handle.get_stream());
      rmm::device_uvector<vertex_t> renumbered_and_compressed_edgelist_minors(0,
                                                                              handle.get_stream());
      std::optional<rmm::device_uvector<size_t>> renumbered_and_compressed_offset_label_hop_offsets{
        std::nullopt};
      rmm::device_uvector<vertex_t> renumbered_and_compressed_renumber_map(0, handle.get_stream());
      std::optional<rmm::device_uvector<size_t>>
        renumbered_and_compressed_renumber_map_label_offsets{std::nullopt};

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.start("Renumber and compressed sampled edgelist");
      }

      std::tie(renumbered_and_compressed_nzd_vertices,
               renumbered_and_compressed_offsets,
               renumbered_and_compressed_edgelist_minors,
               renumbered_and_compressed_edgelist_weights,
               renumbered_and_compressed_edgelist_edge_ids,
               renumbered_and_compressed_edgelist_edge_types,
               renumbered_and_compressed_offset_label_hop_offsets,
               renumbered_and_compressed_renumber_map,
               renumbered_and_compressed_renumber_map_label_offsets) =
        cugraph::renumber_and_compress_sampled_edgelist<vertex_t, weight_t, edge_id_t, edge_type_t>(
          handle,
          std::move(renumbered_and_compressed_edgelist_srcs),
          std::move(renumbered_and_compressed_edgelist_dsts),
          std::move(renumbered_and_compressed_edgelist_weights),
          std::move(renumbered_and_compressed_edgelist_edge_ids),
          std::move(renumbered_and_compressed_edgelist_edge_types),
          std::move(renumbered_and_compressed_edgelist_hops),
          sampling_post_processing_usecase.renumber_with_seeds
            ? std::make_optional<raft::device_span<vertex_t const>>(starting_vertices.data(),
                                                                    starting_vertices.size())
            : std::nullopt,
          (sampling_post_processing_usecase.renumber_with_seeds && starting_vertex_label_offsets)
            ? std::make_optional<raft::device_span<size_t const>>(
                (*starting_vertex_label_offsets).data(), (*starting_vertex_label_offsets).size())
            : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional(raft::device_span<size_t const>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size()))
            : std::nullopt,
          sampling_post_processing_usecase.num_labels,
          sampling_post_processing_usecase.fanouts.size(),
          sampling_post_processing_usecase.src_is_major,
          sampling_post_processing_usecase.compress_per_hop,
          sampling_post_processing_usecase.doubly_compress);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      if (sampling_post_processing_usecase.check_correctness) {
        if (renumbered_and_compressed_nzd_vertices) {
          ASSERT_TRUE(renumbered_and_compressed_offsets.size() ==
                      (*renumbered_and_compressed_nzd_vertices).size() + 1)
            << "Renumbered and compressed offset array size should coincide with the number of "
               "non-zero-degree vertices + 1.";
        }

        ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                      renumbered_and_compressed_offsets.begin(),
                                      renumbered_and_compressed_offsets.end()))
          << "Renumbered and compressed offset array values should be non-decreasing.";

        ASSERT_TRUE(renumbered_and_compressed_offsets.back_element(handle.get_stream()) ==
                    renumbered_and_compressed_edgelist_minors.size())
          << "Renumbered and compressed offset array's last value should coincide with the number "
             "of "
             "edges.";

        if (renumbered_and_compressed_offset_label_hop_offsets) {
          ASSERT_TRUE((*renumbered_and_compressed_offset_label_hop_offsets).size() ==
                      sampling_post_processing_usecase.num_labels *
                          sampling_post_processing_usecase.fanouts.size() +
                        1)
            << "Renumbered and compressed offset (label,hop) offset array size should coincide "
               "with "
               "the number of labels * the number of hops + 1.";

          ASSERT_TRUE(
            thrust::is_sorted(handle.get_thrust_policy(),
                              (*renumbered_and_compressed_offset_label_hop_offsets).begin(),
                              (*renumbered_and_compressed_offset_label_hop_offsets).end()))
            << "Renumbered and compressed offset (label,hop) offset array values should be "
               "non-decreasing.";

          ASSERT_TRUE((*renumbered_and_compressed_offset_label_hop_offsets)
                        .back_element(handle.get_stream()) ==
                      renumbered_and_compressed_offsets.size() - 1)
            << "Renumbered and compressed offset (label,hop) offset array's last value should "
               "coincide with the offset array size - 1.";
        }

        if (renumbered_and_compressed_renumber_map_label_offsets) {
          ASSERT_TRUE((*renumbered_and_compressed_renumber_map_label_offsets).size() ==
                      sampling_post_processing_usecase.num_labels + 1)
            << "Renumbered and compressed offset (label, hop) offset array size should coincide "
               "with "
               "the number of labels + 1.";

          ASSERT_TRUE(
            thrust::is_sorted(handle.get_thrust_policy(),
                              (*renumbered_and_compressed_renumber_map_label_offsets).begin(),
                              (*renumbered_and_compressed_renumber_map_label_offsets).end()))
            << "Renumbered and compressed renumber map label offset array values should be "
               "non-decreasing.";

          ASSERT_TRUE((*renumbered_and_compressed_renumber_map_label_offsets)
                        .back_element(handle.get_stream()) ==
                      renumbered_and_compressed_renumber_map.size())
            << "Renumbered and compressed renumber map label offset array's last value should "
               "coincide with the renumber map size.";
        }

        for (size_t i = 0; i < sampling_post_processing_usecase.num_labels; ++i) {
          size_t starting_vertex_start_offset =
            starting_vertex_label_offsets
              ? (*starting_vertex_label_offsets).element(i, handle.get_stream())
              : size_t{0};
          size_t starting_vertex_end_offset =
            starting_vertex_label_offsets
              ? (*starting_vertex_label_offsets).element(i + 1, handle.get_stream())
              : starting_vertices.size();

          size_t edgelist_start_offset =
            org_edgelist_label_offsets
              ? (*org_edgelist_label_offsets).element(i, handle.get_stream())
              : size_t{0};
          size_t edgelist_end_offset =
            org_edgelist_label_offsets
              ? (*org_edgelist_label_offsets).element(i + 1, handle.get_stream())
              : org_edgelist_srcs.size();
          if (edgelist_start_offset == edgelist_end_offset) continue;

          auto this_label_starting_vertices = raft::device_span<vertex_t const>(
            starting_vertices.data() + starting_vertex_start_offset,
            starting_vertex_end_offset - starting_vertex_start_offset);

          auto this_label_org_edgelist_srcs =
            raft::device_span<vertex_t const>(org_edgelist_srcs.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_org_edgelist_dsts =
            raft::device_span<vertex_t const>(org_edgelist_dsts.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_org_edgelist_hops =
            org_edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                                  (*org_edgelist_hops).data() + edgelist_start_offset,
                                  edgelist_end_offset - edgelist_start_offset)
                              : std::nullopt;
          auto this_label_org_edgelist_weights =
            org_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                     (*org_edgelist_weights).data() + edgelist_start_offset,
                                     edgelist_end_offset - edgelist_start_offset)
                                 : std::nullopt;

          rmm::device_uvector<vertex_t> this_label_output_edgelist_srcs(0, handle.get_stream());
          rmm::device_uvector<vertex_t> this_label_output_edgelist_dsts(0, handle.get_stream());
          auto this_label_output_edgelist_weights =
            renumbered_and_compressed_edgelist_weights
              ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
              : std::nullopt;
          this_label_output_edgelist_srcs.reserve(edgelist_end_offset - edgelist_start_offset,
                                                  handle.get_stream());
          this_label_output_edgelist_dsts.reserve(edgelist_end_offset - edgelist_start_offset,
                                                  handle.get_stream());
          if (this_label_output_edgelist_weights) {
            (*this_label_output_edgelist_weights)
              .reserve(edgelist_end_offset - edgelist_start_offset, handle.get_stream());
          }

          // decompress

          auto num_hops = sampling_post_processing_usecase.fanouts.size();
          for (size_t j = 0; j < num_hops; ++j) {
            auto offset_start_offset = renumbered_and_compressed_offset_label_hop_offsets
                                         ? (*renumbered_and_compressed_offset_label_hop_offsets)
                                             .element(i * num_hops + j, handle.get_stream())
                                         : size_t{0};
            auto offset_end_offset   = renumbered_and_compressed_offset_label_hop_offsets
                                         ? ((*renumbered_and_compressed_offset_label_hop_offsets)
                                            .element(i * num_hops + j + 1, handle.get_stream()) +
                                          1)
                                         : renumbered_and_compressed_offsets.size();

            auto base_v =
              (!sampling_post_processing_usecase.doubly_compress &&
               !sampling_post_processing_usecase.compress_per_hop && (j > 0))
                ? static_cast<vertex_t>(offset_start_offset -
                                        (*renumbered_and_compressed_offset_label_hop_offsets)
                                          .element(i * num_hops, handle.get_stream()))
                : vertex_t{0};

            raft::device_span<size_t const> d_offsets(
              renumbered_and_compressed_offsets.data() + offset_start_offset,
              offset_end_offset - offset_start_offset);
            std::vector<size_t> h_offsets(d_offsets.size());
            raft::update_host(
              h_offsets.data(), d_offsets.data(), h_offsets.size(), handle.get_stream());
            handle.sync_stream();

            auto old_size = this_label_output_edgelist_srcs.size();
            this_label_output_edgelist_srcs.resize(old_size + (h_offsets.back() - h_offsets[0]),
                                                   handle.get_stream());
            this_label_output_edgelist_dsts.resize(this_label_output_edgelist_srcs.size(),
                                                   handle.get_stream());
            if (this_label_output_edgelist_weights) {
              (*this_label_output_edgelist_weights)
                .resize(this_label_output_edgelist_srcs.size(), handle.get_stream());
            }
            thrust::transform(
              handle.get_thrust_policy(),
              thrust::make_counting_iterator(h_offsets[0]),
              thrust::make_counting_iterator(h_offsets.back()),
              (sampling_post_processing_usecase.src_is_major
                 ? this_label_output_edgelist_srcs.begin()
                 : this_label_output_edgelist_dsts.begin()) +
                old_size,
              cuda::proclaim_return_type<vertex_t>(
                [offsets = raft::device_span<size_t const>(d_offsets.data(), d_offsets.size()),
                 nzd_vertices =
                   renumbered_and_compressed_nzd_vertices
                     ? thrust::make_optional<raft::device_span<vertex_t const>>(
                         (*renumbered_and_compressed_nzd_vertices).data() + offset_start_offset,
                         (offset_end_offset - offset_start_offset) - 1)
                     : thrust::nullopt,
                 base_v] __device__(size_t i) {
                  auto idx = static_cast<size_t>(thrust::distance(
                    offsets.begin() + 1,
                    thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i)));
                  if (nzd_vertices) {
                    return (*nzd_vertices)[idx];
                  } else {
                    return base_v + static_cast<vertex_t>(idx);
                  }
                }));
            thrust::copy(handle.get_thrust_policy(),
                         renumbered_and_compressed_edgelist_minors.begin() + h_offsets[0],
                         renumbered_and_compressed_edgelist_minors.begin() + h_offsets.back(),
                         (sampling_post_processing_usecase.src_is_major
                            ? this_label_output_edgelist_dsts.begin()
                            : this_label_output_edgelist_srcs.begin()) +
                           old_size);
            if (this_label_output_edgelist_weights) {
              thrust::copy(handle.get_thrust_policy(),
                           (*renumbered_and_compressed_edgelist_weights).begin() + h_offsets[0],
                           (*renumbered_and_compressed_edgelist_weights).begin() + h_offsets.back(),
                           (*this_label_output_edgelist_weights).begin() + old_size);
            }
          }

          size_t renumber_map_start_offset =
            renumbered_and_compressed_renumber_map_label_offsets
              ? (*renumbered_and_compressed_renumber_map_label_offsets)
                  .element(i, handle.get_stream())
              : size_t{0};
          size_t renumber_map_end_offset =
            renumbered_and_compressed_renumber_map_label_offsets
              ? (*renumbered_and_compressed_renumber_map_label_offsets)
                  .element(i + 1, handle.get_stream())
              : renumbered_and_compressed_renumber_map.size();
          auto this_label_output_renumber_map = raft::device_span<vertex_t const>(
            renumbered_and_compressed_renumber_map.data() + renumber_map_start_offset,
            renumber_map_end_offset - renumber_map_start_offset);

          // check whether renumbering recovers the original edge list

          ASSERT_TRUE(compare_edgelist(
            handle,
            this_label_org_edgelist_srcs,
            this_label_org_edgelist_dsts,
            this_label_org_edgelist_weights,
            raft::device_span<vertex_t const>(this_label_output_edgelist_srcs.data(),
                                              this_label_output_edgelist_srcs.size()),
            raft::device_span<vertex_t const>(this_label_output_edgelist_dsts.data(),
                                              this_label_output_edgelist_dsts.size()),
            this_label_output_edgelist_weights
              ? std::make_optional<raft::device_span<weight_t const>>(
                  (*this_label_output_edgelist_weights).data(),
                  (*this_label_output_edgelist_weights).size())
              : std::nullopt,
            std::make_optional(this_label_output_renumber_map)))
            << "Unrenumbering the renumbered and sorted edge list does not recover the original "
               "edgelist.";

          // Check the invariants in renumber_map

          ASSERT_TRUE(check_renumber_map_invariants(
            handle,
            sampling_post_processing_usecase.renumber_with_seeds
              ? std::make_optional<raft::device_span<vertex_t const>>(
                  this_label_starting_vertices.data(), this_label_starting_vertices.size())
              : std::nullopt,
            this_label_org_edgelist_srcs,
            this_label_org_edgelist_dsts,
            this_label_org_edgelist_hops,
            this_label_output_renumber_map,
            sampling_post_processing_usecase.src_is_major))
            << "Renumbered and sorted output renumber map violates invariants.";
        }
      }
    }

    {
      rmm::device_uvector<vertex_t> sorted_edgelist_srcs(org_edgelist_srcs.size(),
                                                         handle.get_stream());
      rmm::device_uvector<vertex_t> sorted_edgelist_dsts(org_edgelist_dsts.size(),
                                                         handle.get_stream());
      auto sorted_edgelist_weights = org_edgelist_weights
                                       ? std::make_optional<rmm::device_uvector<weight_t>>(
                                           (*org_edgelist_weights).size(), handle.get_stream())
                                       : std::nullopt;
      std::optional<rmm::device_uvector<edge_id_t>> sorted_edgelist_edge_ids{std::nullopt};
      std::optional<rmm::device_uvector<edge_type_t>> sorted_edgelist_edge_types{std::nullopt};
      auto sorted_edgelist_hops = org_edgelist_hops
                                    ? std::make_optional(rmm::device_uvector<int32_t>(
                                        (*org_edgelist_hops).size(), handle.get_stream()))
                                    : std::nullopt;

      raft::copy(sorted_edgelist_srcs.data(),
                 org_edgelist_srcs.data(),
                 org_edgelist_srcs.size(),
                 handle.get_stream());
      raft::copy(sorted_edgelist_dsts.data(),
                 org_edgelist_dsts.data(),
                 org_edgelist_dsts.size(),
                 handle.get_stream());
      if (sorted_edgelist_weights) {
        raft::copy((*sorted_edgelist_weights).data(),
                   (*org_edgelist_weights).data(),
                   (*org_edgelist_weights).size(),
                   handle.get_stream());
      }
      if (sorted_edgelist_hops) {
        raft::copy((*sorted_edgelist_hops).data(),
                   (*org_edgelist_hops).data(),
                   (*org_edgelist_hops).size(),
                   handle.get_stream());
      }

      std::optional<rmm::device_uvector<size_t>> sorted_edgelist_label_hop_offsets{std::nullopt};

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.start("Sort sampled edgelist");
      }

      std::tie(sorted_edgelist_srcs,
               sorted_edgelist_dsts,
               sorted_edgelist_weights,
               sorted_edgelist_edge_ids,
               sorted_edgelist_edge_types,
               sorted_edgelist_label_hop_offsets) =
        cugraph::sort_sampled_edgelist<vertex_t, weight_t, edge_id_t, edge_type_t>(
          handle,
          std::move(sorted_edgelist_srcs),
          std::move(sorted_edgelist_dsts),
          std::move(sorted_edgelist_weights),
          std::move(sorted_edgelist_edge_ids),
          std::move(sorted_edgelist_edge_types),
          std::move(sorted_edgelist_hops),
          org_edgelist_label_offsets
            ? std::make_optional(raft::device_span<size_t const>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size()))
            : std::nullopt,
          sampling_post_processing_usecase.num_labels,
          sampling_post_processing_usecase.fanouts.size(),
          sampling_post_processing_usecase.src_is_major);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      if (sampling_post_processing_usecase.check_correctness) {
        if (sorted_edgelist_label_hop_offsets) {
          ASSERT_TRUE((*sorted_edgelist_label_hop_offsets).size() ==
                      sampling_post_processing_usecase.num_labels *
                          sampling_post_processing_usecase.fanouts.size() +
                        1)
            << "Sorted edge list (label,hop) offset array size should coincide with "
               "the number of labels * the number of hops + 1.";

          ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                        (*sorted_edgelist_label_hop_offsets).begin(),
                                        (*sorted_edgelist_label_hop_offsets).end()))
            << "Sorted edge list (label,hop) offset array values should be "
               "non-decreasing.";

          ASSERT_TRUE((*sorted_edgelist_label_hop_offsets).back_element(handle.get_stream()) ==
                      sorted_edgelist_srcs.size())
            << "Sorted edge list (label,hop) offset array's last element should coincide with the "
               "number of edges.";
        }

        for (size_t i = 0; i < sampling_post_processing_usecase.num_labels; ++i) {
          size_t edgelist_start_offset =
            org_edgelist_label_offsets
              ? (*org_edgelist_label_offsets).element(i, handle.get_stream())
              : size_t{0};
          size_t edgelist_end_offset =
            org_edgelist_label_offsets
              ? (*org_edgelist_label_offsets).element(i + 1, handle.get_stream())
              : org_edgelist_srcs.size();
          if (edgelist_start_offset == edgelist_end_offset) continue;

          auto this_label_org_edgelist_srcs =
            raft::device_span<vertex_t const>(org_edgelist_srcs.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_org_edgelist_dsts =
            raft::device_span<vertex_t const>(org_edgelist_dsts.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_org_edgelist_hops =
            org_edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                                  (*org_edgelist_hops).data() + edgelist_start_offset,
                                  edgelist_end_offset - edgelist_start_offset)
                              : std::nullopt;
          auto this_label_org_edgelist_weights =
            org_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                     (*org_edgelist_weights).data() + edgelist_start_offset,
                                     edgelist_end_offset - edgelist_start_offset)
                                 : std::nullopt;

          auto this_label_output_edgelist_srcs =
            raft::device_span<vertex_t const>(sorted_edgelist_srcs.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_output_edgelist_dsts =
            raft::device_span<vertex_t const>(sorted_edgelist_dsts.data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset);
          auto this_label_output_edgelist_weights =
            sorted_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                        (*sorted_edgelist_weights).data() + edgelist_start_offset,
                                        edgelist_end_offset - edgelist_start_offset)
                                    : std::nullopt;

          // check whether the edges are properly sorted

          auto this_label_output_edgelist_majors = sampling_post_processing_usecase.src_is_major
                                                     ? this_label_output_edgelist_srcs
                                                     : this_label_output_edgelist_dsts;
          auto this_label_output_edgelist_minors = sampling_post_processing_usecase.src_is_major
                                                     ? this_label_output_edgelist_dsts
                                                     : this_label_output_edgelist_srcs;

          if (this_label_org_edgelist_hops) {
            auto num_hops   = sampling_post_processing_usecase.fanouts.size();
            auto edge_first = thrust::make_zip_iterator(this_label_output_edgelist_majors.begin(),
                                                        this_label_output_edgelist_minors.begin());
            for (size_t j = 0; j < num_hops; ++j) {
              auto hop_start_offset =
                (*sorted_edgelist_label_hop_offsets)
                  .element(i * num_hops + j, handle.get_stream()) -
                (*sorted_edgelist_label_hop_offsets).element(i * num_hops, handle.get_stream());
              auto hop_end_offset =
                (*sorted_edgelist_label_hop_offsets)
                  .element(i * num_hops + j + 1, handle.get_stream()) -
                (*sorted_edgelist_label_hop_offsets).element(i * num_hops, handle.get_stream());
              ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                            edge_first + hop_start_offset,
                                            edge_first + hop_end_offset))
                << "Renumbered and sorted output edges are not properly sorted.";
            }
          } else {
            auto edge_first = thrust::make_zip_iterator(this_label_output_edgelist_majors.begin(),
                                                        this_label_output_edgelist_minors.begin());
            ASSERT_TRUE(thrust::is_sorted(handle.get_thrust_policy(),
                                          edge_first,
                                          edge_first + this_label_output_edgelist_majors.size()))
              << "Renumbered and sorted output edges are not properly sorted.";
          }

          // check whether renumbering recovers the original edge list

          ASSERT_TRUE(
            compare_edgelist(handle,
                             this_label_org_edgelist_srcs,
                             this_label_org_edgelist_dsts,
                             this_label_org_edgelist_weights,
                             this_label_output_edgelist_srcs,
                             this_label_output_edgelist_dsts,
                             this_label_output_edgelist_weights,
                             std::optional<raft::device_span<vertex_t const>>{std::nullopt}))
            << "Sorted edge list does not coincide with the original edgelist.";
        }
      }
    }
  }
};

using Tests_SamplingPostProcessing_File = Tests_SamplingPostProcessing<cugraph::test::File_Usecase>;
using Tests_SamplingPostProcessing_Rmat = Tests_SamplingPostProcessing<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_SamplingPostProcessing_File, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SamplingPostProcessing_Rmat, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SamplingPostProcessing_Rmat, CheckInt32Int64)
{
  run_current_test<int32_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SamplingPostProcessing_Rmat, CheckInt64Int64)
{
  run_current_test<int64_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SamplingPostProcessing_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, true, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, true, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, true, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, true, true, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SamplingPostProcessing_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {10}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, false, true, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, false, true, true, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, false, true, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{1, 16, {5, 10, 25}, true, true, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {10}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, false, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, false, true, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, false, true, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, false, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, false, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, false, true, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, true, false, false},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, true, false, true},
      SamplingPostProcessing_Usecase{32, 16, {5, 10, 25}, true, true, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_SamplingPostProcessing_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      SamplingPostProcessing_Usecase{1, 64, {10}, false, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, false, false, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, false, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, false, false, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, false, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, false, true, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, false, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, false, true, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, false, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, false, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, true, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {10}, true, true, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, false, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, false, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, false, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, false, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, true, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, true, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, true, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, false, true, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, false, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, false, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, false, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, false, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, false, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, false, true, true, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, true, false, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, true, false, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, true, false, true, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, true, true, false, false, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, true, true, false, true, false},
      SamplingPostProcessing_Usecase{1, 64, {5, 10, 15}, true, true, true, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, false, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, false, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, false, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, false, true, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, true, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, true, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, true, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, false, true, true, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, false, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, false, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, false, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, false, true, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, true, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, true, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, true, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {10}, true, true, true, false, true, false},
      SamplingPostProcessing_Usecase{
        256, 64, {5, 10, 15}, false, false, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, false, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, false, false, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, false, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, false, true, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, false, true, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, true, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, true, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, true, false, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, true, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, true, true, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, false, true, true, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, false, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, false, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, false, false, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, false, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, false, true, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, false, true, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, true, false, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, true, false, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, true, false, true, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, true, true, false, false, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, true, true, false, true, false},
      SamplingPostProcessing_Usecase{256, 64, {5, 10, 15}, true, true, true, true, false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

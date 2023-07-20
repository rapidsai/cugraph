/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

struct RenumberSampledEdgelist_Usecase {
  size_t num_vertices{};
  size_t num_sampled_edges{};
  size_t num_hops{1};    // enabled if larger than 1
  size_t num_labels{1};  // enabled if larger than 1
  bool check_correctness{true};
};

class Tests_RenumberSampledEdgelist
  : public ::testing::TestWithParam<RenumberSampledEdgelist_Usecase> {
 public:
  Tests_RenumberSampledEdgelist() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t>
  void run_current_test(RenumberSampledEdgelist_Usecase const& usecase)
  {
    using label_t = int32_t;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    raft::random::RngState rng_state(0);

    rmm::device_uvector<vertex_t> org_edgelist_srcs(usecase.num_sampled_edges, handle.get_stream());
    rmm::device_uvector<vertex_t> org_edgelist_dsts(usecase.num_sampled_edges, handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         org_edgelist_srcs.data(),
                                         org_edgelist_srcs.size(),
                                         vertex_t{0},
                                         static_cast<vertex_t>(usecase.num_vertices),
                                         rng_state);
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         org_edgelist_dsts.data(),
                                         org_edgelist_dsts.size(),
                                         vertex_t{0},
                                         static_cast<vertex_t>(usecase.num_vertices),
                                         rng_state);

    std::optional<rmm::device_uvector<int32_t>> edgelist_hops{std::nullopt};
    if (usecase.num_hops > 1) {
      edgelist_hops = rmm::device_uvector<int32_t>(usecase.num_sampled_edges, handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           (*edgelist_hops).data(),
                                           (*edgelist_hops).size(),
                                           int32_t{0},
                                           static_cast<int32_t>(usecase.num_hops),
                                           rng_state);
    }

    std::optional<std::tuple<rmm::device_uvector<label_t>, rmm::device_uvector<size_t>>>
      label_offsets{std::nullopt};
    if (usecase.num_labels > 1) {
      rmm::device_uvector<label_t> labels(usecase.num_labels, handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(), labels.begin(), labels.end(), label_t{0});

      rmm::device_uvector<label_t> edgelist_labels(usecase.num_sampled_edges, handle.get_stream());
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           edgelist_labels.data(),
                                           edgelist_labels.size(),
                                           label_t{0},
                                           static_cast<label_t>(usecase.num_labels),
                                           rng_state);

      rmm::device_uvector<size_t> offsets(usecase.num_labels + 1, handle.get_stream());
      thrust::fill(handle.get_thrust_policy(), offsets.begin(), offsets.end(), size_t{0});

      thrust::for_each(
        handle.get_thrust_policy(),
        edgelist_labels.begin(),
        edgelist_labels.end(),
        [offsets =
           raft::device_span<size_t>(offsets.data(), offsets.size())] __device__(label_t label) {
          cuda::atomic_ref<size_t, cuda::thread_scope_device> atomic_counter(offsets[label]);
          atomic_counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
        });

      thrust::exclusive_scan(
        handle.get_thrust_policy(), offsets.begin(), offsets.end(), offsets.begin());

      label_offsets = std::make_tuple(std::move(labels), std::move(offsets));
    }

    rmm::device_uvector<vertex_t> renumbered_edgelist_srcs(org_edgelist_srcs.size(),
                                                           handle.get_stream());
    rmm::device_uvector<vertex_t> renumbered_edgelist_dsts(org_edgelist_dsts.size(),
                                                           handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 org_edgelist_srcs.begin(),
                 org_edgelist_srcs.end(),
                 renumbered_edgelist_srcs.begin());
    thrust::copy(handle.get_thrust_policy(),
                 org_edgelist_dsts.begin(),
                 org_edgelist_dsts.end(),
                 renumbered_edgelist_dsts.begin());

    rmm::device_uvector<vertex_t> renumber_map(0, handle.get_stream());
    std::optional<rmm::device_uvector<size_t>> renumber_map_label_offsets{std::nullopt};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Renumber sampled edgelist");
    }

    std::tie(renumbered_edgelist_srcs,
             renumbered_edgelist_dsts,
             renumber_map,
             renumber_map_label_offsets) =
      cugraph::renumber_sampled_edgelist(
        handle,
        std::move(renumbered_edgelist_srcs),
        edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                          (*edgelist_hops).data(), (*edgelist_hops).size())
                      : std::nullopt,
        std::move(renumbered_edgelist_dsts),
        label_offsets
          ? std::make_optional<
              std::tuple<raft::device_span<label_t const>, raft::device_span<size_t const>>>(
              std::make_tuple(raft::device_span<label_t const>(std::get<0>(*label_offsets).data(),
                                                               std::get<0>(*label_offsets).size()),
                              raft::device_span<size_t const>(std::get<1>(*label_offsets).data(),
                                                              std::get<1>(*label_offsets).size())))
          : std::nullopt);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (usecase.check_correctness) {
      for (size_t i = 0; i < usecase.num_labels; ++i) {
        size_t edgelist_start_offset =
          label_offsets ? std::get<1>(*label_offsets).element(i, handle.get_stream()) : size_t{0};
        size_t edgelist_end_offset =
          label_offsets ? std::get<1>(*label_offsets).element(i + 1, handle.get_stream())
                        : usecase.num_sampled_edges;
        auto this_label_org_edgelist_srcs =
          raft::device_span<vertex_t const>(org_edgelist_srcs.data() + edgelist_start_offset,
                                            edgelist_end_offset - edgelist_start_offset);
        auto this_label_org_edgelist_dsts =
          raft::device_span<vertex_t const>(org_edgelist_dsts.data() + edgelist_start_offset,
                                            edgelist_end_offset - edgelist_start_offset);
        auto this_label_edgelist_hops = edgelist_hops
                                          ? std::make_optional<raft::device_span<int32_t const>>(
                                              (*edgelist_hops).data() + edgelist_start_offset,
                                              edgelist_end_offset - edgelist_start_offset)
                                          : std::nullopt;
        auto this_label_renumbered_edgelist_srcs =
          raft::device_span<vertex_t const>(renumbered_edgelist_srcs.data() + edgelist_start_offset,
                                            edgelist_end_offset - edgelist_start_offset);
        auto this_label_renumbered_edgelist_dsts =
          raft::device_span<vertex_t const>(renumbered_edgelist_dsts.data() + edgelist_start_offset,
                                            edgelist_end_offset - edgelist_start_offset);

        size_t renumber_map_start_offset =
          renumber_map_label_offsets ? (*renumber_map_label_offsets).element(i, handle.get_stream())
                                     : size_t{0};
        size_t renumber_map_end_offset =
          renumber_map_label_offsets
            ? (*renumber_map_label_offsets).element(i + 1, handle.get_stream())
            : renumber_map.size();
        auto this_label_renumber_map =
          raft::device_span<vertex_t const>(renumber_map.data() + renumber_map_start_offset,
                                            renumber_map_end_offset - renumber_map_start_offset);

        // check un-renumbering recovers the original edge list

        auto pair_first = thrust::make_zip_iterator(this_label_org_edgelist_srcs.begin(),
                                                    this_label_renumbered_edgelist_srcs.begin());
        auto num_renumber_errors =
          thrust::count_if(handle.get_thrust_policy(),
                           pair_first,
                           pair_first + this_label_org_edgelist_srcs.size(),
                           [this_label_renumber_map] __device__(auto pair) {
                             auto org        = thrust::get<0>(pair);
                             auto renumbered = thrust::get<1>(pair);
                             return this_label_renumber_map[renumbered] != org;
                           });
        ASSERT_TRUE(num_renumber_errors == 0) << "Renumber error in edge list sources.";

        pair_first          = thrust::make_zip_iterator(this_label_org_edgelist_dsts.begin(),
                                               this_label_renumbered_edgelist_dsts.begin());
        num_renumber_errors = thrust::count_if(handle.get_thrust_policy(),
                                               pair_first,
                                               pair_first + this_label_org_edgelist_dsts.size(),
                                               [this_label_renumber_map] __device__(auto pair) {
                                                 auto org        = thrust::get<0>(pair);
                                                 auto renumbered = thrust::get<1>(pair);
                                                 return this_label_renumber_map[renumbered] != org;
                                               });
        ASSERT_TRUE(num_renumber_errors == 0) << "Renumber error in edge list destinations.";

        // check the invariants in renumber_map (1. vertices appeared in edge list sources should
        // have a smaller renumbered vertex ID than the vertices appear only in edge list
        // destinations, 2. edge list source vertices with a smaller minimum hop number should have
        // a smaller renumbered vertex ID than the edge list source vertices with a larger hop
        // number)

        rmm::device_uvector<vertex_t> unique_srcs(this_label_org_edgelist_srcs.size(),
                                                  handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     this_label_org_edgelist_srcs.begin(),
                     this_label_org_edgelist_srcs.end(),
                     unique_srcs.begin());
        std::optional<rmm::device_uvector<int32_t>> unique_src_hops =
          this_label_edgelist_hops ? std::make_optional<rmm::device_uvector<int32_t>>(
                                       (*this_label_edgelist_hops).size(), handle.get_stream())
                                   : std::nullopt;
        if (this_label_edgelist_hops) {
          thrust::copy(handle.get_thrust_policy(),
                       (*this_label_edgelist_hops).begin(),
                       (*this_label_edgelist_hops).end(),
                       (*unique_src_hops).begin());

          auto pair_first =
            thrust::make_zip_iterator(unique_srcs.begin(), (*unique_src_hops).begin());
          thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + unique_srcs.size());
          unique_srcs.resize(
            thrust::distance(unique_srcs.begin(),
                             thrust::get<0>(thrust::unique_by_key(handle.get_thrust_policy(),
                                                                  unique_srcs.begin(),
                                                                  unique_srcs.end(),
                                                                  (*unique_src_hops).begin()))),
            handle.get_stream());
          (*unique_src_hops).resize(unique_srcs.size(), handle.get_stream());
        } else {
          thrust::sort(handle.get_thrust_policy(), unique_srcs.begin(), unique_srcs.end());
          unique_srcs.resize(
            thrust::distance(
              unique_srcs.begin(),
              thrust::unique(handle.get_thrust_policy(), unique_srcs.begin(), unique_srcs.end())),
            handle.get_stream());
        }

        rmm::device_uvector<vertex_t> unique_dsts(this_label_org_edgelist_dsts.size(),
                                                  handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     this_label_org_edgelist_dsts.begin(),
                     this_label_org_edgelist_dsts.end(),
                     unique_dsts.begin());
        thrust::sort(handle.get_thrust_policy(), unique_dsts.begin(), unique_dsts.end());
        unique_dsts.resize(
          thrust::distance(
            unique_dsts.begin(),
            thrust::unique(handle.get_thrust_policy(), unique_dsts.begin(), unique_dsts.end())),
          handle.get_stream());

        unique_dsts.resize(
          thrust::distance(
            unique_dsts.begin(),
            thrust::remove_if(handle.get_thrust_policy(),
                              unique_dsts.begin(),
                              unique_dsts.end(),
                              [sorted_unique_srcs = raft::device_span<vertex_t const>(
                                 unique_srcs.data(), unique_srcs.size())] __device__(auto dst) {
                                return thrust::binary_search(thrust::seq,
                                                             sorted_unique_srcs.begin(),
                                                             sorted_unique_srcs.end(),
                                                             dst);
                              })),
          handle.get_stream());

        rmm::device_uvector<vertex_t> sorted_org_vertices(this_label_renumber_map.size(),
                                                          handle.get_stream());
        rmm::device_uvector<vertex_t> matching_renumbered_vertices(sorted_org_vertices.size(),
                                                                   handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     this_label_renumber_map.begin(),
                     this_label_renumber_map.end(),
                     sorted_org_vertices.begin());
        thrust::sequence(handle.get_thrust_policy(),
                         matching_renumbered_vertices.begin(),
                         matching_renumbered_vertices.end(),
                         vertex_t{0});
        thrust::sort_by_key(handle.get_thrust_policy(),
                            sorted_org_vertices.begin(),
                            sorted_org_vertices.end(),
                            matching_renumbered_vertices.begin());

        auto max_src_renumbered_vertex = thrust::transform_reduce(
          handle.get_thrust_policy(),
          unique_srcs.begin(),
          unique_srcs.end(),
          [sorted_org_vertices = raft::device_span<vertex_t const>(sorted_org_vertices.data(),
                                                                   sorted_org_vertices.size()),
           matching_renumbered_vertices = raft::device_span<vertex_t const>(
             matching_renumbered_vertices.data(),
             matching_renumbered_vertices.size())] __device__(vertex_t src) {
            auto it = thrust::lower_bound(
              thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), src);
            return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(), it)];
          },
          std::numeric_limits<vertex_t>::lowest(),
          thrust::maximum<vertex_t>{});

        auto min_dst_renumbered_vertex = thrust::transform_reduce(
          handle.get_thrust_policy(),
          unique_dsts.begin(),
          unique_dsts.end(),
          [sorted_org_vertices = raft::device_span<vertex_t const>(sorted_org_vertices.data(),
                                                                   sorted_org_vertices.size()),
           matching_renumbered_vertices = raft::device_span<vertex_t const>(
             matching_renumbered_vertices.data(),
             matching_renumbered_vertices.size())] __device__(vertex_t dst) {
            auto it = thrust::lower_bound(
              thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), dst);
            return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(), it)];
          },
          std::numeric_limits<vertex_t>::max(),
          thrust::minimum<vertex_t>{});

        ASSERT_TRUE(max_src_renumbered_vertex < min_dst_renumbered_vertex)
          << "Invariants violated, a source vertex is renumbered to a non-smaller value than a "
             "vertex that appear only in the edge list destinations.";

        if (this_label_edgelist_hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              (*unique_src_hops).begin(),
                              (*unique_src_hops).end(),
                              unique_srcs.begin());
          rmm::device_uvector<vertex_t> min_vertices(usecase.num_hops, handle.get_stream());
          rmm::device_uvector<vertex_t> max_vertices(usecase.num_hops, handle.get_stream());
          auto unique_renumbered_src_first = thrust::make_transform_iterator(
            unique_srcs.begin(),
            [sorted_org_vertices = raft::device_span<vertex_t const>(sorted_org_vertices.data(),
                                                                     sorted_org_vertices.size()),
             matching_renumbered_vertices = raft::device_span<vertex_t const>(
               matching_renumbered_vertices.data(),
               matching_renumbered_vertices.size())] __device__(vertex_t src) {
              auto it = thrust::lower_bound(
                thrust::seq, sorted_org_vertices.begin(), sorted_org_vertices.end(), src);
              return matching_renumbered_vertices[thrust::distance(sorted_org_vertices.begin(),
                                                                   it)];
            });

          auto this_label_num_unique_hops = static_cast<size_t>(
            thrust::distance(min_vertices.begin(),
                             thrust::get<1>(thrust::reduce_by_key(handle.get_thrust_policy(),
                                                                  (*unique_src_hops).begin(),
                                                                  (*unique_src_hops).end(),
                                                                  unique_renumbered_src_first,
                                                                  thrust::make_discard_iterator(),
                                                                  min_vertices.begin(),
                                                                  thrust::equal_to<vertex_t>{},
                                                                  thrust::minimum<vertex_t>{}))));
          min_vertices.resize(this_label_num_unique_hops, handle.get_stream());

          thrust::reduce_by_key(handle.get_thrust_policy(),
                                (*unique_src_hops).begin(),
                                (*unique_src_hops).end(),
                                unique_renumbered_src_first,
                                thrust::make_discard_iterator(),
                                max_vertices.begin(),
                                thrust::equal_to<vertex_t>{},
                                thrust::maximum<vertex_t>{});
          max_vertices.resize(this_label_num_unique_hops, handle.get_stream());

          auto num_violations =
            thrust::count_if(handle.get_thrust_policy(),
                             thrust::make_counting_iterator(size_t{1}),
                             thrust::make_counting_iterator(this_label_num_unique_hops),
                             [min_vertices = raft::device_span<vertex_t const>(min_vertices.data(),
                                                                               min_vertices.size()),
                              max_vertices = raft::device_span<vertex_t const>(
                                max_vertices.data(), max_vertices.size())] __device__(size_t i) {
                               return min_vertices[i] <= max_vertices[i - 1];
                             });

          ASSERT_TRUE(num_violations == 0)
            << "Invariant violated, a vertex with a smaller hop is renumbered to a non-smaller "
               "value than a vertex with a larger hop.";
        }
      }
    }
  }
};

TEST_P(Tests_RenumberSampledEdgelist, CheckInt32)
{
  auto param = GetParam();
  run_current_test<int32_t>(param);
}

TEST_P(Tests_RenumberSampledEdgelist, CheckInt64)
{
  auto param = GetParam();
  run_current_test<int64_t>(param);
}

INSTANTIATE_TEST_SUITE_P(
  small_test,
  Tests_RenumberSampledEdgelist,
  ::testing::Values(RenumberSampledEdgelist_Usecase{1024, 4096, 1, 1, true},
                    RenumberSampledEdgelist_Usecase{1024, 4096, 3, 1, true},
                    RenumberSampledEdgelist_Usecase{1024, 32768, 1, 256, true},
                    RenumberSampledEdgelist_Usecase{1024, 32768, 3, 256, true}));

INSTANTIATE_TEST_SUITE_P(
  benchmark_test,
  Tests_RenumberSampledEdgelist,
  ::testing::Values(RenumberSampledEdgelist_Usecase{1 << 20, 1 << 20, 1, 1, false},
                    RenumberSampledEdgelist_Usecase{1 << 20, 1 << 20, 5, 1, false},
                    RenumberSampledEdgelist_Usecase{1 << 20, 1 << 24, 1, 1 << 20, false},
                    RenumberSampledEdgelist_Usecase{1 << 20, 1 << 24, 5, 1 << 20, false}));

CUGRAPH_TEST_PROGRAM_MAIN()

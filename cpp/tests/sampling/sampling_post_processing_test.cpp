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

#include "detail/sampling_post_processing_validate.hpp"
#include "utilities/base_fixture.hpp"

#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

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
    using edge_id_t   = edge_t;
    using edge_type_t = int32_t;

    bool constexpr store_transposed = false;
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = true;

    auto [sampling_post_processing_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // 1. create a graph

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

    // 2. seed vertices (& labels)

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
      auto num_seeds_per_label = sampling_post_processing_usecase.num_seeds_per_label;
      for (size_t i = 0; i < sampling_post_processing_usecase.num_labels; ++i) {
        cugraph::detail::scalar_fill(handle.get_stream(),
                                     (*starting_vertex_labels).data() + i * num_seeds_per_label,
                                     num_seeds_per_label,
                                     static_cast<label_t>(i));
      }
      cugraph::detail::stride_fill(handle.get_stream(),
                                   (*starting_vertex_label_offsets).data(),
                                   (*starting_vertex_label_offsets).size(),
                                   size_t{0},
                                   num_seeds_per_label);
    }

    // 3. sampling

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

    // 4. post processing: renumber & sort

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
          ASSERT_TRUE(check_offsets(handle,
                                    raft::device_span<size_t const>(
                                      (*renumbered_and_sorted_edgelist_label_hop_offsets).data(),
                                      (*renumbered_and_sorted_edgelist_label_hop_offsets).size()),
                                    sampling_post_processing_usecase.num_labels *
                                      sampling_post_processing_usecase.fanouts.size(),
                                    renumbered_and_sorted_edgelist_srcs.size()))
            << "Renumbered and sorted edge (label, hop) offset array is invalid.";
        }

        if (renumbered_and_sorted_renumber_map_label_offsets) {
          ASSERT_TRUE(check_offsets(handle,
                                    raft::device_span<size_t const>(
                                      (*renumbered_and_sorted_renumber_map_label_offsets).data(),
                                      (*renumbered_and_sorted_renumber_map_label_offsets).size()),
                                    sampling_post_processing_usecase.num_labels,
                                    renumbered_and_sorted_renumber_map.size()))
            << "Renumbered and sorted renumber map label offset array is invalid.";
        }

        // check whether the edges are properly sorted

        auto renumbered_and_sorted_edgelist_majors =
          sampling_post_processing_usecase.src_is_major
            ? raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_srcs.data(),
                                                renumbered_and_sorted_edgelist_srcs.size())
            : raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_dsts.data(),
                                                renumbered_and_sorted_edgelist_dsts.size());
        auto renumbered_and_sorted_edgelist_minors =
          sampling_post_processing_usecase.src_is_major
            ? raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_dsts.data(),
                                                renumbered_and_sorted_edgelist_dsts.size())
            : raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_srcs.data(),
                                                renumbered_and_sorted_edgelist_srcs.size());

        if (renumbered_and_sorted_edgelist_label_hop_offsets) {
          for (size_t i = 0; i < sampling_post_processing_usecase.num_labels *
                                   sampling_post_processing_usecase.fanouts.size();
               ++i) {
            auto hop_start_offset =
              (*renumbered_and_sorted_edgelist_label_hop_offsets).element(i, handle.get_stream());
            auto hop_end_offset = (*renumbered_and_sorted_edgelist_label_hop_offsets)
                                    .element(i + 1, handle.get_stream());
            ASSERT_TRUE(check_edgelist_is_sorted(
              handle,
              raft::device_span<vertex_t const>(
                renumbered_and_sorted_edgelist_majors.data() + hop_start_offset,
                hop_end_offset - hop_start_offset),
              raft::device_span<vertex_t const>(
                renumbered_and_sorted_edgelist_minors.data() + hop_start_offset,
                hop_end_offset - hop_start_offset)))
              << "Renumbered and sorted edge list is not properly sorted.";
          }
        } else {
          ASSERT_TRUE(check_edgelist_is_sorted(
            handle,
            raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_majors.data(),
                                              renumbered_and_sorted_edgelist_majors.size()),
            raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_minors.data(),
                                              renumbered_and_sorted_edgelist_minors.size())))
            << "Renumbered and sorted edge list is not properly sorted.";
        }

        ASSERT_TRUE(compare_edgelist(
          handle,
          raft::device_span<vertex_t const>(org_edgelist_srcs.data(), org_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(org_edgelist_dsts.data(), org_edgelist_dsts.size()),
          org_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                   (*org_edgelist_weights).data(), (*org_edgelist_weights).size())
                               : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size())
            : std::nullopt,
          raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_srcs.data(),
                                            renumbered_and_sorted_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_dsts.data(),
                                            renumbered_and_sorted_edgelist_dsts.size()),
          renumbered_and_sorted_edgelist_weights
            ? std::make_optional<raft::device_span<weight_t const>>(
                (*renumbered_and_sorted_edgelist_weights).data(),
                (*renumbered_and_sorted_edgelist_weights).size())
            : std::nullopt,
          std::make_optional<raft::device_span<vertex_t const>>(
            renumbered_and_sorted_renumber_map.data(), renumbered_and_sorted_renumber_map.size()),
          renumbered_and_sorted_renumber_map_label_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*renumbered_and_sorted_renumber_map_label_offsets).data(),
                (*renumbered_and_sorted_renumber_map_label_offsets).size())
            : std::nullopt,
          sampling_post_processing_usecase.num_labels))
          << "Unrenumbering the renumbered and sorted edge list does not recover the original "
             "edgelist.";

        // Check the invariants in renumber_map

        ASSERT_TRUE(check_vertex_renumber_map_invariants<vertex_t>(
          handle,
          sampling_post_processing_usecase.renumber_with_seeds
            ? std::make_optional<raft::device_span<vertex_t const>>(starting_vertices.data(),
                                                                    starting_vertices.size())
            : std::nullopt,
          (sampling_post_processing_usecase.renumber_with_seeds && starting_vertex_label_offsets)
            ? std::make_optional<raft::device_span<size_t const>>(
                (*starting_vertex_label_offsets).data(), (*starting_vertex_label_offsets).size())
            : std::nullopt,
          raft::device_span<vertex_t const>(org_edgelist_srcs.data(), org_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(org_edgelist_dsts.data(), org_edgelist_dsts.size()),
          org_edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                                (*org_edgelist_hops).data(), (*org_edgelist_hops).size())
                            : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size())
            : std::nullopt,
          raft::device_span<vertex_t const>(renumbered_and_sorted_renumber_map.data(),
                                            renumbered_and_sorted_renumber_map.size()),
          renumbered_and_sorted_renumber_map_label_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*renumbered_and_sorted_renumber_map_label_offsets).data(),
                (*renumbered_and_sorted_renumber_map_label_offsets).size())
            : std::nullopt,
          std::nullopt,
          sampling_post_processing_usecase.num_labels,
          1,
          sampling_post_processing_usecase.src_is_major))
          << "Renumbered and sorted output renumber map violates invariants.";
      }
    }

    // 5. post processing: renumber & compress

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
        ASSERT_TRUE(check_offsets(
          handle,
          raft::device_span<size_t const>(renumbered_and_compressed_offsets.data(),
                                          renumbered_and_compressed_offsets.size()),
          renumbered_and_compressed_nzd_vertices ? (*renumbered_and_compressed_nzd_vertices).size()
                                                 : renumbered_and_compressed_offsets.size() - 1,
          renumbered_and_compressed_edgelist_minors.size()))
          << "Renumbered and compressed offset array is invalid";

        if (renumbered_and_compressed_offset_label_hop_offsets) {
          ASSERT_TRUE(check_offsets(handle,
                                    raft::device_span<size_t const>(
                                      (*renumbered_and_compressed_offset_label_hop_offsets).data(),
                                      (*renumbered_and_compressed_offset_label_hop_offsets).size()),
                                    sampling_post_processing_usecase.num_labels *
                                      sampling_post_processing_usecase.fanouts.size(),
                                    renumbered_and_compressed_offsets.size() - 1))
            << "Renumbered and compressed offset (label, hop) offset array is invalid";
        }

        if (renumbered_and_compressed_renumber_map_label_offsets) {
          ASSERT_TRUE(
            check_offsets(handle,
                          raft::device_span<size_t const>(
                            (*renumbered_and_compressed_renumber_map_label_offsets).data(),
                            (*renumbered_and_compressed_renumber_map_label_offsets).size()),
                          sampling_post_processing_usecase.num_labels,
                          renumbered_and_compressed_renumber_map.size()))
            << "Renumbered and compressed renumber map label offset array is invalid";
        }

        // check whether renumbering recovers the original edge list

        rmm::device_uvector<vertex_t> output_edgelist_srcs(0, handle.get_stream());
        rmm::device_uvector<vertex_t> output_edgelist_dsts(0, handle.get_stream());
        auto output_edgelist_weights =
          renumbered_and_compressed_edgelist_weights
            ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
            : std::nullopt;
        output_edgelist_srcs.reserve(org_edgelist_srcs.size(), handle.get_stream());
        output_edgelist_dsts.reserve(org_edgelist_srcs.capacity(), handle.get_stream());
        if (output_edgelist_weights) {
          (*output_edgelist_weights).reserve(org_edgelist_srcs.capacity(), handle.get_stream());
        }

        for (size_t i = 0; i < sampling_post_processing_usecase.num_labels; ++i) {
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

            auto old_size = output_edgelist_srcs.size();
            output_edgelist_srcs.resize(old_size + (h_offsets.back() - h_offsets[0]),
                                        handle.get_stream());
            output_edgelist_dsts.resize(output_edgelist_srcs.size(), handle.get_stream());
            if (output_edgelist_weights) {
              (*output_edgelist_weights).resize(output_edgelist_srcs.size(), handle.get_stream());
            }
            if (renumbered_and_compressed_nzd_vertices) {
              cugraph::test::expand_hypersparse_offsets(
                handle,
                raft::device_span<size_t const>(d_offsets.data(), d_offsets.size()),
                raft::device_span<vertex_t const>(
                  (*renumbered_and_compressed_nzd_vertices).data() + offset_start_offset,
                  (offset_end_offset - offset_start_offset) - 1),
                raft::device_span<vertex_t>(
                  (sampling_post_processing_usecase.src_is_major ? output_edgelist_srcs.data()
                                                                 : output_edgelist_dsts.data()) +
                    old_size,
                  h_offsets.back() - h_offsets[0]),
                h_offsets[0]);
            } else {
              cugraph::test::expand_sparse_offsets(
                handle,
                raft::device_span<size_t const>(d_offsets.data(), d_offsets.size()),
                raft::device_span<vertex_t>(
                  (sampling_post_processing_usecase.src_is_major ? output_edgelist_srcs.data()
                                                                 : output_edgelist_dsts.data()) +
                    old_size,
                  h_offsets.back() - h_offsets[0]),
                h_offsets[0],
                base_v);
            }
            raft::copy(
              (sampling_post_processing_usecase.src_is_major ? output_edgelist_dsts.begin()
                                                             : output_edgelist_srcs.begin()) +
                old_size,
              renumbered_and_compressed_edgelist_minors.begin() + h_offsets[0],
              h_offsets.back() - h_offsets[0],
              handle.get_stream());
            if (output_edgelist_weights) {
              raft::copy((*output_edgelist_weights).begin() + old_size,
                         (*renumbered_and_compressed_edgelist_weights).begin() + h_offsets[0],
                         h_offsets.back() - h_offsets[0],
                         handle.get_stream());
            }
          }
        }

        ASSERT_TRUE(compare_edgelist(
          handle,
          raft::device_span<vertex_t const>(org_edgelist_srcs.data(), org_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(org_edgelist_dsts.data(), org_edgelist_dsts.size()),
          org_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                   (*org_edgelist_weights).data(), (*org_edgelist_weights).size())
                               : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional(raft::device_span<size_t const>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size()))
            : std::nullopt,
          raft::device_span<vertex_t const>(output_edgelist_srcs.data(),
                                            output_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(output_edgelist_dsts.data(),
                                            output_edgelist_dsts.size()),
          output_edgelist_weights
            ? std::make_optional<raft::device_span<weight_t const>>(
                (*output_edgelist_weights).data(), (*output_edgelist_weights).size())
            : std::nullopt,
          std::make_optional<raft::device_span<vertex_t const>>(
            renumbered_and_compressed_renumber_map.data(),
            renumbered_and_compressed_renumber_map.size()),
          renumbered_and_compressed_renumber_map_label_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*renumbered_and_compressed_renumber_map_label_offsets).data(),
                (*renumbered_and_compressed_renumber_map_label_offsets).size())
            : std::nullopt,
          sampling_post_processing_usecase.num_labels))
          << "Unrenumbering the renumbered and sorted edge list does not recover the original "
             "edgelist.";

        // Check the invariants in renumber_map

        ASSERT_TRUE(check_vertex_renumber_map_invariants<vertex_t>(
          handle,
          sampling_post_processing_usecase.renumber_with_seeds
            ? std::make_optional<raft::device_span<vertex_t const>>(starting_vertices.data(),
                                                                    starting_vertices.size())
            : std::nullopt,
          (sampling_post_processing_usecase.renumber_with_seeds && starting_vertex_label_offsets)
            ? std::make_optional<raft::device_span<size_t const>>(
                (*starting_vertex_label_offsets).data(), (*starting_vertex_label_offsets).size())
            : std::nullopt,
          raft::device_span<vertex_t const>(org_edgelist_srcs.data(), org_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(org_edgelist_dsts.data(), org_edgelist_dsts.size()),
          org_edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                                (*org_edgelist_hops).data(), (*org_edgelist_hops).size())
                            : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional(raft::device_span<size_t const>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size()))
            : std::nullopt,
          raft::device_span<vertex_t const>(renumbered_and_compressed_renumber_map.data(),
                                            renumbered_and_compressed_renumber_map.size()),
          renumbered_and_compressed_renumber_map_label_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*renumbered_and_compressed_renumber_map_label_offsets).data(),
                (*renumbered_and_compressed_renumber_map_label_offsets).size())
            : std::nullopt,
          std::nullopt,
          sampling_post_processing_usecase.num_labels,
          1,
          sampling_post_processing_usecase.src_is_major))
          << "Renumbered and sorted output renumber map violates invariants.";
      }
    }

    // 6. post processing: sort only

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
          ASSERT_TRUE(check_offsets(
            handle,
            raft::device_span<size_t const>((*sorted_edgelist_label_hop_offsets).data(),
                                            (*sorted_edgelist_label_hop_offsets).size()),
            sampling_post_processing_usecase.num_labels *
              sampling_post_processing_usecase.fanouts.size(),
            sorted_edgelist_srcs.size()))
            << "Sorted edge list (label, hop) offset array is invalid.";
        }

        // check whether renumbering recovers the original edge list

        ASSERT_TRUE(compare_edgelist(
          handle,
          raft::device_span<vertex_t const>(org_edgelist_srcs.data(), org_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(org_edgelist_dsts.data(), org_edgelist_dsts.size()),
          org_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                   (*org_edgelist_weights).data(), (*org_edgelist_weights).size())
                               : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional(raft::device_span<size_t const>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size()))
            : std::nullopt,
          raft::device_span<vertex_t const>(sorted_edgelist_srcs.data(),
                                            sorted_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(sorted_edgelist_dsts.data(),
                                            sorted_edgelist_dsts.size()),
          sorted_edgelist_weights
            ? std::make_optional<raft::device_span<weight_t const>>(
                (*sorted_edgelist_weights).data(), (*sorted_edgelist_weights).size())
            : std::nullopt,
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<size_t const>>{std::nullopt},
          sampling_post_processing_usecase.num_labels))
          << "Sorted edge list does not coincide with the original edgelist.";

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
            auto num_hops = sampling_post_processing_usecase.fanouts.size();
            for (size_t j = 0; j < num_hops; ++j) {
              auto hop_start_offset =
                (*sorted_edgelist_label_hop_offsets)
                  .element(i * num_hops + j, handle.get_stream()) -
                (*sorted_edgelist_label_hop_offsets).element(i * num_hops, handle.get_stream());
              auto hop_end_offset =
                (*sorted_edgelist_label_hop_offsets)
                  .element(i * num_hops + j + 1, handle.get_stream()) -
                (*sorted_edgelist_label_hop_offsets).element(i * num_hops, handle.get_stream());
              ASSERT_TRUE(check_edgelist_is_sorted(
                handle,
                raft::device_span<vertex_t const>(
                  this_label_output_edgelist_majors.data() + hop_start_offset,
                  hop_end_offset - hop_start_offset),
                raft::device_span<vertex_t const>(
                  this_label_output_edgelist_minors.data() + hop_start_offset,
                  hop_end_offset - hop_start_offset)))
                << "Sorted edge list is not properly sorted.";
            }
          } else {
            ASSERT_TRUE(check_edgelist_is_sorted(
              handle,
              raft::device_span<vertex_t const>(this_label_output_edgelist_majors.data(),
                                                this_label_output_edgelist_majors.size()),
              raft::device_span<vertex_t const>(this_label_output_edgelist_minors.data(),
                                                this_label_output_edgelist_minors.size())))
              << "Sorted edge list is not properly sorted.";
          }
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
      SamplingPostProcessing_Usecase{128, 64, {10}, false, false, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, false, false, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, false, false, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, false, false, true, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, false, true, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, false, true, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, false, true, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, false, true, true, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, false, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, false, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, false, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, false, true, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, true, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, true, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, true, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {10}, true, true, true, false, true, false},
      SamplingPostProcessing_Usecase{
        128, 64, {5, 10, 15}, false, false, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, false, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, false, false, true, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, false, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, false, true, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, false, true, true, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, true, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, true, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, true, false, true, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, true, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, true, true, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, false, true, true, true, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, false, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, false, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, false, false, true, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, false, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, false, true, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, false, true, true, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, true, false, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, true, false, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, true, false, true, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, true, true, false, false, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, true, true, false, true, false},
      SamplingPostProcessing_Usecase{128, 64, {5, 10, 15}, true, true, true, true, false, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

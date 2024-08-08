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
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"

#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

struct SamplingHeterogeneousPostProcessing_Usecase {
  size_t num_labels{};
  size_t num_seeds_per_label{};
  size_t num_vertex_types{};
  std::vector<int32_t> fanouts{{-1}};
  bool sample_with_replacement{false};

  bool src_is_major{true};
  bool renumber_with_seeds{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_SamplingHeterogeneousPostProcessing
  : public ::testing::TestWithParam<
      std::tuple<SamplingHeterogeneousPostProcessing_Usecase, input_usecase_t>> {
 public:
  Tests_SamplingHeterogeneousPostProcessing() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(std::tuple<SamplingHeterogeneousPostProcessing_Usecase const&,
                                   input_usecase_t const&> const& param)
  {
    using label_t     = int32_t;
    using weight_t    = float;
    using edge_id_t   = edge_t;
    using edge_type_t = int32_t;

    bool constexpr store_transposed = false;
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = true;

    auto [sampling_heterogeneous_post_processing_usecase, input_usecase] = param;

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

    // 2. vertex type offsets

    raft::random::RngState rng_state(0);

    rmm::device_uvector<vertex_t> vertex_type_offsets(
      sampling_heterogeneous_post_processing_usecase.num_vertex_types + 1, handle.get_stream());
    {
      auto num_vertices = graph_view.number_of_vertices();
      vertex_type_offsets.set_element_to_zero_async(0, handle.get_stream());
      vertex_type_offsets.set_element_async(
        vertex_type_offsets.size() - 1, num_vertices, handle.get_stream());
      auto tmp = cugraph::select_random_vertices<vertex_t, edge_t, store_transposed, false>(
        handle,
        graph_view,
        std::nullopt,
        rng_state,
        sampling_heterogeneous_post_processing_usecase.num_vertex_types - 1,
        false /* with_replacement */,
        true /* sort_vertices */);
      raft::copy(vertex_type_offsets.data() + 1, tmp.data(), tmp.size(), handle.get_stream());
    }

    // 3. seed vertices (& labels)

    rmm::device_uvector<vertex_t> starting_vertices(
      sampling_heterogeneous_post_processing_usecase.num_labels *
        sampling_heterogeneous_post_processing_usecase.num_seeds_per_label,
      handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         starting_vertices.data(),
                                         starting_vertices.size(),
                                         vertex_t{0},
                                         graph_view.number_of_vertices(),
                                         rng_state);
    auto starting_vertex_labels = (sampling_heterogeneous_post_processing_usecase.num_labels > 1)
                                    ? std::make_optional<rmm::device_uvector<label_t>>(
                                        starting_vertices.size(), handle.get_stream())
                                    : std::nullopt;
    auto starting_vertex_label_offsets =
      (sampling_heterogeneous_post_processing_usecase.num_labels > 1)
        ? std::make_optional<rmm::device_uvector<size_t>>(
            sampling_heterogeneous_post_processing_usecase.num_labels + 1, handle.get_stream())
        : std::nullopt;
    if (starting_vertex_labels) {
      auto num_seeds_per_label = sampling_heterogeneous_post_processing_usecase.num_seeds_per_label;
      for (size_t i = 0; i < sampling_heterogeneous_post_processing_usecase.num_labels; ++i) {
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

    // 4. generate edge IDs and types

    auto num_edge_types =
      sampling_heterogeneous_post_processing_usecase.num_vertex_types *
      sampling_heterogeneous_post_processing_usecase
        .num_vertex_types;  // necessary to enforce that edge type dictates edge source vertex type
                            // and edge destination vertex type.

    std::optional<cugraph::edge_property_t<decltype(graph_view), edge_type_t>> edge_types{
      std::nullopt};
    if (num_edge_types > 1) {
      edge_types =
        cugraph::test::generate<decltype(graph_view), edge_type_t>::edge_property_by_src_dst_types(
          handle,
          graph_view,
          raft::device_span<vertex_t const>(vertex_type_offsets.data(), vertex_type_offsets.size()),
          num_edge_types);
    }

    cugraph::edge_property_t<decltype(graph_view), edge_id_t> edge_ids(handle);
    if (edge_types) {
      static_assert(std::is_same_v<edge_type_t, int32_t>);
      edge_ids =
        cugraph::test::generate<decltype(graph_view), edge_id_t>::unique_edge_property_per_type(
          handle, graph_view, (*edge_types).view(), static_cast<int32_t>(num_edge_types));
    } else {
      edge_ids = cugraph::test::generate<decltype(graph_view), edge_id_t>::unique_edge_property(
        handle, graph_view);
    }

    // 5. sampling

    rmm::device_uvector<vertex_t> org_edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> org_edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> org_edgelist_weights{std::nullopt};
    std::optional<rmm::device_uvector<edge_id_t>> org_edgelist_edge_ids{std::nullopt};
    std::optional<rmm::device_uvector<edge_type_t>> org_edgelist_edge_types{std::nullopt};
    std::optional<rmm::device_uvector<int32_t>> org_edgelist_hops{std::nullopt};
    std::optional<rmm::device_uvector<label_t>> org_labels{std::nullopt};
    std::optional<rmm::device_uvector<size_t>> org_edgelist_label_offsets{std::nullopt};
    std::tie(org_edgelist_srcs,
             org_edgelist_dsts,
             org_edgelist_weights,
             org_edgelist_edge_ids,
             org_edgelist_edge_types,
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
      std::optional<cugraph::edge_property_view_t<edge_t, edge_id_t const*>>{edge_ids.view()},
      edge_types
        ? std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{(*edge_types)
                                                                                     .view()}
        : std::nullopt,
      raft::device_span<vertex_t const>(starting_vertices.data(), starting_vertices.size()),
      starting_vertex_labels ? std::make_optional<raft::device_span<label_t const>>(
                                 (*starting_vertex_labels).data(), (*starting_vertex_labels).size())
                             : std::nullopt,
      std::nullopt,
      raft::host_span<int32_t const>(sampling_heterogeneous_post_processing_usecase.fanouts.data(),
                                     sampling_heterogeneous_post_processing_usecase.fanouts.size()),
      rng_state,
      sampling_heterogeneous_post_processing_usecase.fanouts.size() > 1,
      sampling_heterogeneous_post_processing_usecase.sample_with_replacement,
      cugraph::prior_sources_behavior_t::EXCLUDE,
      false);

    if (!sampling_heterogeneous_post_processing_usecase.src_is_major) {
      std::swap(org_edgelist_srcs, org_edgelist_dsts);
    }

    // 6. post processing: renumber & sort

    {
      rmm::device_uvector<vertex_t> renumbered_and_sorted_edgelist_srcs(org_edgelist_srcs.size(),
                                                                        handle.get_stream());
      rmm::device_uvector<vertex_t> renumbered_and_sorted_edgelist_dsts(org_edgelist_dsts.size(),
                                                                        handle.get_stream());
      auto renumbered_and_sorted_edgelist_weights =
        org_edgelist_weights ? std::make_optional<rmm::device_uvector<weight_t>>(
                                 (*org_edgelist_weights).size(), handle.get_stream())
                             : std::nullopt;
      auto renumbered_and_sorted_edgelist_edge_ids =
        org_edgelist_edge_ids ? std::make_optional<rmm::device_uvector<edge_id_t>>(
                                  (*org_edgelist_edge_ids).size(), handle.get_stream())
                              : std::nullopt;
      auto renumbered_and_sorted_edgelist_edge_types =
        org_edgelist_edge_types ? std::make_optional<rmm::device_uvector<edge_type_t>>(
                                    (*org_edgelist_edge_types).size(), handle.get_stream())
                                : std::nullopt;
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
      if (renumbered_and_sorted_edgelist_edge_ids) {
        raft::copy((*renumbered_and_sorted_edgelist_edge_ids).data(),
                   (*org_edgelist_edge_ids).data(),
                   (*org_edgelist_edge_ids).size(),
                   handle.get_stream());
      }
      if (renumbered_and_sorted_edgelist_edge_types) {
        raft::copy((*renumbered_and_sorted_edgelist_edge_types).data(),
                   (*org_edgelist_edge_types).data(),
                   (*org_edgelist_edge_types).size(),
                   handle.get_stream());
      }
      if (renumbered_and_sorted_edgelist_hops) {
        raft::copy((*renumbered_and_sorted_edgelist_hops).data(),
                   (*org_edgelist_hops).data(),
                   (*org_edgelist_hops).size(),
                   handle.get_stream());
      }

      std::optional<rmm::device_uvector<size_t>>
        renumbered_and_sorted_edgelist_label_type_hop_offsets{std::nullopt};
      rmm::device_uvector<vertex_t> renumbered_and_sorted_vertex_renumber_map(0,
                                                                              handle.get_stream());
      rmm::device_uvector<size_t> renumbered_and_sorted_vertex_renumber_map_label_type_offsets(
        0, handle.get_stream());
      std::optional<rmm::device_uvector<edge_id_t>> renumbered_and_sorted_edge_id_renumber_map{
        std::nullopt};
      std::optional<rmm::device_uvector<size_t>>
        renumbered_and_sorted_edge_id_renumber_map_label_type_offsets{std::nullopt};

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.start("Renumber and sort sampled edgelist");
      }

      std::tie(renumbered_and_sorted_edgelist_srcs,
               renumbered_and_sorted_edgelist_dsts,
               renumbered_and_sorted_edgelist_weights,
               renumbered_and_sorted_edgelist_edge_ids,
               renumbered_and_sorted_edgelist_label_type_hop_offsets,
               renumbered_and_sorted_vertex_renumber_map,
               renumbered_and_sorted_vertex_renumber_map_label_type_offsets,
               renumbered_and_sorted_edge_id_renumber_map,
               renumbered_and_sorted_edge_id_renumber_map_label_type_offsets) =
        cugraph::heterogeneous_renumber_and_sort_sampled_edgelist<vertex_t,
                                                                  weight_t,
                                                                  edge_id_t,
                                                                  edge_type_t>(
          handle,
          std::move(renumbered_and_sorted_edgelist_srcs),
          std::move(renumbered_and_sorted_edgelist_dsts),
          std::move(renumbered_and_sorted_edgelist_weights),
          std::move(renumbered_and_sorted_edgelist_edge_ids),
          std::move(renumbered_and_sorted_edgelist_edge_types),
          std::move(renumbered_and_sorted_edgelist_hops),
          sampling_heterogeneous_post_processing_usecase.renumber_with_seeds
            ? std::make_optional<raft::device_span<vertex_t const>>(starting_vertices.data(),
                                                                    starting_vertices.size())
            : std::nullopt,
          (sampling_heterogeneous_post_processing_usecase.renumber_with_seeds &&
           starting_vertex_label_offsets)
            ? std::make_optional<raft::device_span<size_t const>>(
                (*starting_vertex_label_offsets).data(), (*starting_vertex_label_offsets).size())
            : std::nullopt,
          org_edgelist_label_offsets
            ? std::make_optional(raft::device_span<size_t const>(
                (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size()))
            : std::nullopt,
          raft::device_span<vertex_t const>(vertex_type_offsets.data(), vertex_type_offsets.size()),
          sampling_heterogeneous_post_processing_usecase.num_labels,
          sampling_heterogeneous_post_processing_usecase.fanouts.size(),
          sampling_heterogeneous_post_processing_usecase.num_vertex_types,
          num_edge_types,
          sampling_heterogeneous_post_processing_usecase.src_is_major,
          true);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      if (sampling_heterogeneous_post_processing_usecase.check_correctness) {
        if (renumbered_and_sorted_edgelist_label_type_hop_offsets) {
          ASSERT_TRUE(check_offsets(
            handle,
            raft::device_span<size_t const>(
              (*renumbered_and_sorted_edgelist_label_type_hop_offsets).data(),
              (*renumbered_and_sorted_edgelist_label_type_hop_offsets).size()),
            sampling_heterogeneous_post_processing_usecase.num_labels * num_edge_types *
              sampling_heterogeneous_post_processing_usecase.fanouts.size(),
            renumbered_and_sorted_edgelist_srcs.size()))
            << "Renumbered and sorted edge (label, edge type, hop) offset array is invalid.";
        }

        ASSERT_TRUE(
          check_offsets(handle,
                        raft::device_span<size_t const>(
                          renumbered_and_sorted_vertex_renumber_map_label_type_offsets.data(),
                          renumbered_and_sorted_vertex_renumber_map_label_type_offsets.size()),
                        sampling_heterogeneous_post_processing_usecase.num_labels *
                          sampling_heterogeneous_post_processing_usecase.num_vertex_types,
                        renumbered_and_sorted_vertex_renumber_map.size()))
          << "Renumbered and sorted vertex renumber map (label, vertex type) offset array is "
             "invalid.";

        if (renumbered_and_sorted_edge_id_renumber_map_label_type_offsets) {
          ASSERT_TRUE(check_offsets(
            handle,
            raft::device_span<size_t const>(
              (*renumbered_and_sorted_edge_id_renumber_map_label_type_offsets).data(),
              (*renumbered_and_sorted_edge_id_renumber_map_label_type_offsets).size()),
            sampling_heterogeneous_post_processing_usecase.num_labels * num_edge_types,
            (*renumbered_and_sorted_edge_id_renumber_map).size()))
            << "Renumbered and sorted edge renumber map (label, edge type) offset array is "
               "invalid.";
        }

        // check whether the edges are properly sorted

        auto renumbered_and_sorted_edgelist_majors =
          sampling_heterogeneous_post_processing_usecase.src_is_major
            ? raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_srcs.data(),
                                                renumbered_and_sorted_edgelist_srcs.size())
            : raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_dsts.data(),
                                                renumbered_and_sorted_edgelist_dsts.size());
        auto renumbered_and_sorted_edgelist_minors =
          sampling_heterogeneous_post_processing_usecase.src_is_major
            ? raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_dsts.data(),
                                                renumbered_and_sorted_edgelist_dsts.size())
            : raft::device_span<vertex_t const>(renumbered_and_sorted_edgelist_srcs.data(),
                                                renumbered_and_sorted_edgelist_srcs.size());

        if (renumbered_and_sorted_edgelist_label_type_hop_offsets) {
          for (size_t i = 0;
               i < sampling_heterogeneous_post_processing_usecase.num_labels * num_edge_types *
                     sampling_heterogeneous_post_processing_usecase.fanouts.size();
               ++i) {
            auto hop_start_offset = (*renumbered_and_sorted_edgelist_label_type_hop_offsets)
                                      .element(i, handle.get_stream());
            auto hop_end_offset = (*renumbered_and_sorted_edgelist_label_type_hop_offsets)
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

        // check whether renumbering recovers the original edge list

        ASSERT_TRUE(compare_heterogeneous_edgelist(
          handle,
          raft::device_span<vertex_t const>(org_edgelist_srcs.data(), org_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(org_edgelist_dsts.data(), org_edgelist_dsts.size()),
          org_edgelist_weights ? std::make_optional<raft::device_span<weight_t const>>(
                                   (*org_edgelist_weights).data(), (*org_edgelist_weights).size())
                               : std::nullopt,
          org_edgelist_edge_ids
            ? std::make_optional<raft::device_span<edge_id_t const>>(
                (*org_edgelist_edge_ids).data(), (*org_edgelist_edge_ids).size())
            : std::nullopt,
          org_edgelist_edge_types
            ? std::make_optional<raft::device_span<edge_type_t const>>(
                (*org_edgelist_edge_types).data(), (*org_edgelist_edge_types).size())
            : std::nullopt,
          org_edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                                (*org_edgelist_hops).data(), (*org_edgelist_hops).size())
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
          renumbered_and_sorted_edgelist_edge_ids
            ? std::make_optional<raft::device_span<edge_id_t const>>(
                (*renumbered_and_sorted_edgelist_edge_ids).data(),
                (*renumbered_and_sorted_edgelist_edge_ids).size())
            : std::nullopt,
          renumbered_and_sorted_edgelist_label_type_hop_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*renumbered_and_sorted_edgelist_label_type_hop_offsets).data(),
                (*renumbered_and_sorted_edgelist_label_type_hop_offsets).size())
            : std::nullopt,
          raft::device_span<vertex_t const>(renumbered_and_sorted_vertex_renumber_map.data(),
                                            renumbered_and_sorted_vertex_renumber_map.size()),
          raft::device_span<size_t const>(
            renumbered_and_sorted_vertex_renumber_map_label_type_offsets.data(),
            renumbered_and_sorted_vertex_renumber_map_label_type_offsets.size()),
          renumbered_and_sorted_edge_id_renumber_map
            ? std::make_optional<raft::device_span<edge_id_t const>>(
                (*renumbered_and_sorted_edge_id_renumber_map).data(),
                (*renumbered_and_sorted_edge_id_renumber_map).size())
            : std::nullopt,
          renumbered_and_sorted_edge_id_renumber_map_label_type_offsets
            ? std::make_optional<raft::device_span<size_t const>>(
                (*renumbered_and_sorted_edge_id_renumber_map_label_type_offsets).data(),
                (*renumbered_and_sorted_edge_id_renumber_map_label_type_offsets).size())
            : std::nullopt,
          raft::device_span<vertex_t const>(vertex_type_offsets.data(), vertex_type_offsets.size()),
          sampling_heterogeneous_post_processing_usecase.num_labels,
          sampling_heterogeneous_post_processing_usecase.num_vertex_types,
          num_edge_types,
          sampling_heterogeneous_post_processing_usecase.fanouts.size()))
          << "Unrenumbering the renumbered and sorted edge list does not recover the original "
             "edgelist.";

        // Check the invariants in vertex renumber_map

        ASSERT_TRUE(check_vertex_renumber_map_invariants<vertex_t>(
          handle,
          sampling_heterogeneous_post_processing_usecase.renumber_with_seeds
            ? std::make_optional<raft::device_span<vertex_t const>>(starting_vertices.data(),
                                                                    starting_vertices.size())
            : std::nullopt,
          (sampling_heterogeneous_post_processing_usecase.renumber_with_seeds &&
           starting_vertex_label_offsets)
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
          raft::device_span<vertex_t const>(renumbered_and_sorted_vertex_renumber_map.data(),
                                            renumbered_and_sorted_vertex_renumber_map.size()),
          std::make_optional<raft::device_span<size_t const>>(
            renumbered_and_sorted_vertex_renumber_map_label_type_offsets.data(),
            renumbered_and_sorted_vertex_renumber_map_label_type_offsets.size()),
          raft::device_span<vertex_t const>(vertex_type_offsets.data(), vertex_type_offsets.size()),
          sampling_heterogeneous_post_processing_usecase.num_labels,
          sampling_heterogeneous_post_processing_usecase.num_vertex_types,
          sampling_heterogeneous_post_processing_usecase.src_is_major))
          << "Renumbered and sorted output vertex renumber map violates invariants.";

        // Check the invariants in edge renumber_map

        if (org_edgelist_edge_ids) {
          ASSERT_TRUE(check_edge_id_renumber_map_invariants(
            handle,
            raft::device_span<edge_id_t const>((*org_edgelist_edge_ids).data(),
                                               (*org_edgelist_edge_ids).size()),
            org_edgelist_edge_types
              ? std::make_optional<raft::device_span<edge_type_t const>>(
                  (*org_edgelist_edge_types).data(), (*org_edgelist_edge_types).size())
              : std::nullopt,
            org_edgelist_hops ? std::make_optional<raft::device_span<int32_t const>>(
                                  (*org_edgelist_hops).data(), (*org_edgelist_hops).size())
                              : std::nullopt,
            org_edgelist_label_offsets
              ? std::make_optional<raft::device_span<size_t const>>(
                  (*org_edgelist_label_offsets).data(), (*org_edgelist_label_offsets).size())
              : std::nullopt,
            raft::device_span<edge_id_t const>(
              (*renumbered_and_sorted_edge_id_renumber_map).data(),
              (*renumbered_and_sorted_edge_id_renumber_map).size()),
            renumbered_and_sorted_edge_id_renumber_map_label_type_offsets
              ? std::make_optional<raft::device_span<size_t const>>(
                  (*renumbered_and_sorted_edge_id_renumber_map_label_type_offsets).data(),
                  (*renumbered_and_sorted_edge_id_renumber_map_label_type_offsets).size())
              : std::nullopt,
            sampling_heterogeneous_post_processing_usecase.num_labels,
            num_edge_types))
            << "Renumbered and sorted output edge ID renumber map violates invariants.";
        }
      }
    }
  }
};

using Tests_SamplingHeterogeneousPostProcessing_File =
  Tests_SamplingHeterogeneousPostProcessing<cugraph::test::File_Usecase>;
using Tests_SamplingHeterogeneousPostProcessing_Rmat =
  Tests_SamplingHeterogeneousPostProcessing<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_SamplingHeterogeneousPostProcessing_File, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SamplingHeterogeneousPostProcessing_Rmat, CheckInt32Int32)
{
  run_current_test<int32_t, int32_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SamplingHeterogeneousPostProcessing_Rmat, CheckInt32Int64)
{
  run_current_test<int32_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_SamplingHeterogeneousPostProcessing_Rmat, CheckInt64Int64)
{
  run_current_test<int64_t, int64_t>(override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SamplingHeterogeneousPostProcessing_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 15}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SamplingHeterogeneousPostProcessing_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 1, {5, 10, 25}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{1, 16, 4, {5, 10, 25}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {10}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, false, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, false, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 1, {5, 10, 25}, true, true, true},
      SamplingHeterogeneousPostProcessing_Usecase{32, 16, 4, {5, 10, 25}, true, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_SamplingHeterogeneousPostProcessing_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {10}, true, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {10}, true, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        1, 64, 1, {5, 10, 15}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        1, 64, 16, {5, 10, 15}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {5, 10, 15}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        1, 64, 16, {5, 10, 15}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {5, 10, 15}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        1, 64, 16, {5, 10, 15}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {5, 10, 15}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {5, 10, 15}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {5, 10, 15}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        1, 64, 16, {5, 10, 15}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {5, 10, 15}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {5, 10, 15}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {5, 10, 15}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {5, 10, 15}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 1, {5, 10, 15}, true, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{1, 64, 16, {5, 10, 15}, true, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {10}, true, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 16, {10}, true, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 1, {5, 10, 15}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, false, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 1, {5, 10, 15}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, false, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 1, {5, 10, 15}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, false, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 1, {5, 10, 15}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, false, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 1, {5, 10, 15}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, true, false, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 1, {5, 10, 15}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, true, false, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 1, {5, 10, 15}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, true, true, false, false},
      SamplingHeterogeneousPostProcessing_Usecase{128, 64, 1, {5, 10, 15}, true, true, true, false},
      SamplingHeterogeneousPostProcessing_Usecase{
        128, 64, 16, {5, 10, 15}, true, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

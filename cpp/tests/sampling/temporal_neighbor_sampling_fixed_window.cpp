/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"

#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace {

using vertex_t     = int32_t;
using edge_t       = int32_t;
using weight_t     = float;
using edge_type_t  = int32_t;
using time_stamp_t = int32_t;

struct expected_edge_t {
  vertex_t src{};
  vertex_t dst{};
  time_stamp_t edge_start_time{};
  int32_t hop{};
};

bool operator<(expected_edge_t const& a, expected_edge_t const& b)
{
  return std::tie(a.src, a.dst, a.edge_start_time, a.hop) <
         std::tie(b.src, b.dst, b.edge_start_time, b.hop);
}

bool operator==(expected_edge_t const& a, expected_edge_t const& b)
{
  return std::tie(a.src, a.dst, a.edge_start_time, a.hop) ==
         std::tie(b.src, b.dst, b.edge_start_time, b.hop);
}

std::tuple<cugraph::graph_t<vertex_t, edge_t, false, false>,
           cugraph::edge_property_t<edge_t, time_stamp_t>,
           std::optional<cugraph::edge_property_t<edge_t, time_stamp_t>>>
make_temporal_graph(raft::handle_t const& handle,
                    std::vector<vertex_t> const& h_srcs,
                    std::vector<vertex_t> const& h_dsts,
                    std::vector<time_stamp_t> const& h_edge_start_times,
                    std::optional<std::vector<time_stamp_t>> const& h_edge_end_times)
{
  auto d_srcs             = cugraph::test::to_device(handle, h_srcs);
  auto d_dsts             = cugraph::test::to_device(handle, h_dsts);
  auto d_edge_start_times = cugraph::test::to_device(handle, h_edge_start_times);
  auto d_edge_end_times   = cugraph::test::to_device(handle, h_edge_end_times);

  std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};
  edge_properties.push_back(std::move(d_edge_start_times));
  if (d_edge_end_times) { edge_properties.push_back(std::move(*d_edge_end_times)); }

  cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
  std::vector<cugraph::edge_arithmetic_property_t<edge_t>> stored_properties{};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, stored_properties, renumber_map) =
    cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
      handle,
      std::nullopt,
      std::move(d_srcs),
      std::move(d_dsts),
      std::move(edge_properties),
      cugraph::graph_properties_t{false, true},
      false /* renumber */);

  size_t pos{0};
  auto edge_start_times =
    std::move(std::get<cugraph::edge_property_t<edge_t, time_stamp_t>>(stored_properties[pos++]));
  std::optional<cugraph::edge_property_t<edge_t, time_stamp_t>> edge_end_times{std::nullopt};
  if (h_edge_end_times) {
    edge_end_times =
      std::move(std::get<cugraph::edge_property_t<edge_t, time_stamp_t>>(stored_properties[pos++]));
  }

  return {std::move(graph), std::move(edge_start_times), std::move(edge_end_times)};
}

std::vector<expected_edge_t> run_and_collect(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  cugraph::edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<cugraph::edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  std::vector<vertex_t> const& h_starts,
  std::optional<std::vector<time_stamp_t>> const& h_start_times,
  std::optional<std::vector<time_stamp_t>> const& h_end_times,
  std::vector<int32_t> const& fan_out,
  cugraph::neighbor_selection_t neighbor_selection,
  cugraph::temporal_sampling_comparison_t temporal_sampling_comparison)
{
  raft::random::RngState rng_state{0};

  auto d_starts      = cugraph::test::to_device(handle, h_starts);
  auto d_start_times = cugraph::test::to_device(handle, h_start_times);
  auto d_end_times   = cugraph::test::to_device(handle, h_end_times);

  auto [srcs,
        dsts,
        weights,
        edge_ids,
        edge_types,
        edge_start_times,
        edge_end_times,
        hops,
        offsets] =
    cugraph::neighbor_sample<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t, false, false>(
      handle,
      rng_state,
      graph_view,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::make_optional(edge_start_time_view),
      edge_end_time_view,
      std::nullopt,
      raft::device_span<vertex_t const>{d_starts.data(), d_starts.size()},
      d_start_times ? std::make_optional(raft::device_span<time_stamp_t const>{
                        d_start_times->data(), d_start_times->size()})
                    : std::nullopt,
      d_end_times ? std::make_optional(raft::device_span<time_stamp_t const>{d_end_times->data(),
                                                                             d_end_times->size()})
                  : std::nullopt,
      std::nullopt,
      std::nullopt,
      raft::host_span<int32_t const>{fan_out.data(), fan_out.size()},
      std::nullopt,
      neighbor_selection,
      temporal_sampling_comparison,
      cugraph::sampling_flags_t{cugraph::prior_sources_behavior_t::DEFAULT,
                                true,   // return_hops
                                false,  // dedupe_sources
                                false,  // with_replacement
                                temporal_sampling_comparison,
                                true,  // disjoint_sampling
                                neighbor_selection});

  EXPECT_TRUE(edge_start_times.has_value());
  EXPECT_TRUE(hops.has_value());
  EXPECT_EQ(srcs.size(), dsts.size());
  EXPECT_EQ(srcs.size(), edge_start_times->size());
  EXPECT_EQ(srcs.size(), hops->size());

  auto h_srcs  = cugraph::test::to_host(handle, srcs);
  auto h_dsts  = cugraph::test::to_host(handle, dsts);
  auto h_times = cugraph::test::to_host(handle, *edge_start_times);
  auto h_hops  = cugraph::test::to_host(handle, *hops);

  std::vector<expected_edge_t> result(h_srcs.size());
  for (size_t i = 0; i < h_srcs.size(); ++i) {
    result[i] = expected_edge_t{h_srcs[i], h_dsts[i], h_times[i], h_hops[i]};
  }
  std::sort(result.begin(), result.end());
  return result;
}

void expect_edges(std::vector<expected_edge_t> actual, std::vector<expected_edge_t> expected)
{
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "mismatch at sorted index " << i;
  }
}

}  // namespace

TEST(TemporalNeighborSamplingFixedWindow, Multihop)
{
  raft::handle_t handle{};

  // Chain/fork: 0->1 (t=50), then from 1: 1->2 (t=30) and 1->3 (t=80).
  // Seed window [10, 100]. FIXED_WINDOW keeps that window at hop 1, so both edges remain eligible.
  auto [graph, edge_start_times, edge_end_times] =
    make_temporal_graph(handle,
                        /*srcs*/ {0, 1, 1},
                        /*dsts*/ {1, 2, 3},
                        /*start times*/ {50, 30, 80},
                        std::vector<time_stamp_t>{51, 31, 81});

  auto actual =
    run_and_collect(handle,
                    graph.view(),
                    edge_start_times.view(),
                    edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
                    /*starts*/ {0},
                    std::vector<time_stamp_t>{10},
                    std::vector<time_stamp_t>{100},
                    /*fan_out*/ {-1, -1},
                    cugraph::neighbor_selection_t::RANDOM,
                    cugraph::temporal_sampling_comparison_t::FIXED_WINDOW);

  expect_edges(std::move(actual), {{0, 1, 50, 0}, {1, 2, 30, 1}, {1, 3, 80, 1}});
}

CUGRAPH_TEST_PROGRAM_MAIN()

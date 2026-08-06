/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"

#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/shuffle_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
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

std::tuple<cugraph::graph_t<vertex_t, edge_t, false, true>,
           cugraph::edge_property_t<edge_t, time_stamp_t>,
           std::optional<cugraph::edge_property_t<edge_t, time_stamp_t>>,
           rmm::device_uvector<vertex_t>>
make_mg_temporal_graph(raft::handle_t const& handle,
                       std::vector<vertex_t> const& h_srcs,
                       std::vector<vertex_t> const& h_dsts,
                       std::vector<time_stamp_t> const& h_edge_start_times,
                       std::optional<std::vector<time_stamp_t>> const& h_edge_end_times)
{
  auto const rank = handle.get_comms().get_rank();

  auto d_srcs             = (rank == 0) ? cugraph::test::to_device(handle, h_srcs)
                                        : rmm::device_uvector<vertex_t>(0, handle.get_stream());
  auto d_dsts             = (rank == 0) ? cugraph::test::to_device(handle, h_dsts)
                                        : rmm::device_uvector<vertex_t>(0, handle.get_stream());
  auto d_edge_start_times = (rank == 0) ? cugraph::test::to_device(handle, h_edge_start_times)
                                        : rmm::device_uvector<time_stamp_t>(0, handle.get_stream());
  std::optional<rmm::device_uvector<time_stamp_t>> d_edge_end_times{std::nullopt};
  if (h_edge_end_times) {
    d_edge_end_times =
      (rank == 0) ? std::make_optional(cugraph::test::to_device(handle, *h_edge_end_times))
                  : std::make_optional(rmm::device_uvector<time_stamp_t>(0, handle.get_stream()));
  }

  std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};
  edge_properties.push_back(std::move(d_edge_start_times));
  if (d_edge_end_times) { edge_properties.push_back(std::move(*d_edge_end_times)); }

  std::tie(d_srcs, d_dsts, edge_properties) = cugraph::shuffle_ext_edges(
    handle, std::move(d_srcs), std::move(d_dsts), std::move(edge_properties), false);

  cugraph::graph_t<vertex_t, edge_t, false, true> graph(handle);
  std::vector<cugraph::edge_arithmetic_property_t<edge_t>> stored_properties{};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, stored_properties, renumber_map) =
    cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, true>(
      handle,
      std::nullopt,
      std::move(d_srcs),
      std::move(d_dsts),
      std::move(edge_properties),
      cugraph::graph_properties_t{false, true},
      true /* renumber */);

  size_t pos{0};
  auto edge_start_times =
    std::move(std::get<cugraph::edge_property_t<edge_t, time_stamp_t>>(stored_properties[pos++]));
  std::optional<cugraph::edge_property_t<edge_t, time_stamp_t>> edge_end_times{std::nullopt};
  if (h_edge_end_times) {
    edge_end_times =
      std::move(std::get<cugraph::edge_property_t<edge_t, time_stamp_t>>(stored_properties[pos++]));
  }

  return {std::move(graph),
          std::move(edge_start_times),
          std::move(edge_end_times),
          std::move(*renumber_map)};
}

std::vector<expected_edge_t> run_and_collect(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& graph_view,
  rmm::device_uvector<vertex_t> const& renumber_map,
  cugraph::edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<cugraph::edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  std::vector<vertex_t> const& h_starts,
  std::optional<std::vector<time_stamp_t>> const& h_start_times,
  std::optional<std::vector<time_stamp_t>> const& h_end_times,
  std::vector<int32_t> const& fan_out,
  cugraph::neighbor_selection_t neighbor_selection,
  cugraph::temporal_sampling_comparison_t temporal_sampling_comparison)
{
  raft::random::RngState rng_state{static_cast<uint64_t>(handle.get_comms().get_rank())};
  auto const rank = handle.get_comms().get_rank();

  auto d_starts = (rank == 0) ? cugraph::test::to_device(handle, h_starts)
                              : rmm::device_uvector<vertex_t>(0, handle.get_stream());
  std::optional<rmm::device_uvector<time_stamp_t>> d_start_times{std::nullopt};
  if (h_start_times) {
    d_start_times =
      (rank == 0) ? std::make_optional(cugraph::test::to_device(handle, *h_start_times))
                  : std::make_optional(rmm::device_uvector<time_stamp_t>(0, handle.get_stream()));
  }
  std::optional<rmm::device_uvector<time_stamp_t>> d_end_times{std::nullopt};
  if (h_end_times) {
    d_end_times = (rank == 0)
                    ? std::make_optional(cugraph::test::to_device(handle, *h_end_times))
                    : std::make_optional(rmm::device_uvector<time_stamp_t>(0, handle.get_stream()));
  }

  std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
  if (d_start_times) { vertex_properties.push_back(std::move(*d_start_times)); }
  if (d_end_times) { vertex_properties.push_back(std::move(*d_end_times)); }

  std::tie(d_starts, vertex_properties) =
    cugraph::shuffle_ext_vertices(handle, std::move(d_starts), std::move(vertex_properties));

  size_t pos{0};
  if (h_start_times) {
    d_start_times =
      std::move(std::get<rmm::device_uvector<time_stamp_t>>(vertex_properties[pos++]));
  } else {
    d_start_times = std::nullopt;
  }
  if (h_end_times) {
    d_end_times = std::move(std::get<rmm::device_uvector<time_stamp_t>>(vertex_properties[pos++]));
  } else {
    d_end_times = std::nullopt;
  }

  cugraph::renumber_local_ext_vertices<vertex_t, true>(
    handle,
    d_starts.data(),
    d_starts.size(),
    renumber_map.data(),
    graph_view.local_vertex_partition_range_first(),
    graph_view.local_vertex_partition_range_last());

  auto [srcs,
        dsts,
        weights,
        edge_ids,
        edge_types,
        edge_start_times,
        edge_end_times,
        hops,
        offsets] =
    cugraph::neighbor_sample<vertex_t, edge_t, weight_t, edge_type_t, time_stamp_t, false, true>(
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

  cugraph::unrenumber_int_vertices<vertex_t, true>(handle,
                                                   srcs.data(),
                                                   srcs.size(),
                                                   renumber_map.data(),
                                                   graph_view.vertex_partition_range_lasts());
  cugraph::unrenumber_int_vertices<vertex_t, true>(handle,
                                                   dsts.data(),
                                                   dsts.size(),
                                                   renumber_map.data(),
                                                   graph_view.vertex_partition_range_lasts());

  auto gathered_srcs = cugraph::test::device_gatherv(handle, srcs.data(), srcs.size());
  auto gathered_dsts = cugraph::test::device_gatherv(handle, dsts.data(), dsts.size());
  auto gathered_times =
    cugraph::test::device_gatherv(handle, edge_start_times->data(), edge_start_times->size());
  auto gathered_hops = cugraph::test::device_gatherv(handle, hops->data(), hops->size());

  std::vector<expected_edge_t> result{};
  if (rank == 0) {
    auto h_srcs  = cugraph::test::to_host(handle, gathered_srcs);
    auto h_dsts  = cugraph::test::to_host(handle, gathered_dsts);
    auto h_times = cugraph::test::to_host(handle, gathered_times);
    auto h_hops  = cugraph::test::to_host(handle, gathered_hops);

    result.resize(h_srcs.size());
    for (size_t i = 0; i < h_srcs.size(); ++i) {
      result[i] = expected_edge_t{h_srcs[i], h_dsts[i], h_times[i], h_hops[i]};
    }
    std::sort(result.begin(), result.end());
  }
  return result;
}

void expect_edges(raft::handle_t const& handle,
                  std::vector<expected_edge_t> actual,
                  std::vector<expected_edge_t> expected)
{
  if (handle.get_comms().get_rank() != 0) { return; }

  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "mismatch at sorted index " << i;
  }
}

class Tests_MGTemporalNeighborSamplingDeterministic : public ::testing::Test {
 public:
  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

 protected:
  static std::unique_ptr<raft::handle_t> handle_;
};

std::unique_ptr<raft::handle_t> Tests_MGTemporalNeighborSamplingDeterministic::handle_{nullptr};

}  // namespace

TEST_F(Tests_MGTemporalNeighborSamplingDeterministic, FirstSingleHop)
{
  auto [graph, edge_start_times, edge_end_times, renumber_map] =
    make_mg_temporal_graph(*handle_,
                           /*srcs*/ {0, 0, 0},
                           /*dsts*/ {1, 2, 3},
                           /*start times*/ {10, 20, 30},
                           std::vector<time_stamp_t>{11, 21, 31});

  auto actual =
    run_and_collect(*handle_,
                    graph.view(),
                    renumber_map,
                    edge_start_times.view(),
                    edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
                    /*starts*/ {0},
                    std::vector<time_stamp_t>{0},
                    std::vector<time_stamp_t>{100},
                    /*fan_out*/ {1},
                    cugraph::neighbor_selection_t::FIRST,
                    cugraph::temporal_sampling_comparison_t::MONOTONICALLY_INCREASING);

  expect_edges(*handle_, std::move(actual), {{0, 1, 10, 0}});
}

TEST_F(Tests_MGTemporalNeighborSamplingDeterministic, LastSingleHop)
{
  auto [graph, edge_start_times, edge_end_times, renumber_map] =
    make_mg_temporal_graph(*handle_,
                           /*srcs*/ {0, 0, 0},
                           /*dsts*/ {1, 2, 3},
                           /*start times*/ {10, 20, 30},
                           std::vector<time_stamp_t>{11, 21, 31});

  auto actual =
    run_and_collect(*handle_,
                    graph.view(),
                    renumber_map,
                    edge_start_times.view(),
                    edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
                    /*starts*/ {0},
                    std::vector<time_stamp_t>{0},
                    std::vector<time_stamp_t>{100},
                    /*fan_out*/ {1},
                    cugraph::neighbor_selection_t::LAST,
                    cugraph::temporal_sampling_comparison_t::MONOTONICALLY_INCREASING);

  expect_edges(*handle_, std::move(actual), {{0, 3, 30, 0}});
}

TEST_F(Tests_MGTemporalNeighborSamplingDeterministic, FixedWindowMultihop)
{
  auto [graph, edge_start_times, edge_end_times, renumber_map] =
    make_mg_temporal_graph(*handle_,
                           /*srcs*/ {0, 1, 1},
                           /*dsts*/ {1, 2, 3},
                           /*start times*/ {50, 30, 80},
                           std::vector<time_stamp_t>{51, 31, 81});

  auto actual =
    run_and_collect(*handle_,
                    graph.view(),
                    renumber_map,
                    edge_start_times.view(),
                    edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
                    /*starts*/ {0},
                    std::vector<time_stamp_t>{10},
                    std::vector<time_stamp_t>{100},
                    /*fan_out*/ {-1, -1},
                    cugraph::neighbor_selection_t::RANDOM,
                    cugraph::temporal_sampling_comparison_t::FIXED_WINDOW);

  expect_edges(*handle_, std::move(actual), {{0, 1, 50, 0}, {1, 2, 30, 1}, {1, 3, 80, 1}});
}

TEST_F(Tests_MGTemporalNeighborSamplingDeterministic, FixedWindowFirst)
{
  auto [graph, edge_start_times, edge_end_times, renumber_map] =
    make_mg_temporal_graph(*handle_,
                           /*srcs*/ {0, 1, 1},
                           /*dsts*/ {1, 2, 3},
                           /*start times*/ {50, 30, 80},
                           std::vector<time_stamp_t>{51, 31, 81});

  auto actual =
    run_and_collect(*handle_,
                    graph.view(),
                    renumber_map,
                    edge_start_times.view(),
                    edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
                    /*starts*/ {0},
                    std::vector<time_stamp_t>{10},
                    std::vector<time_stamp_t>{100},
                    /*fan_out*/ {-1, 1},
                    cugraph::neighbor_selection_t::FIRST,
                    cugraph::temporal_sampling_comparison_t::FIXED_WINDOW);

  expect_edges(*handle_, std::move(actual), {{0, 1, 50, 0}, {1, 2, 30, 1}});
}

TEST_F(Tests_MGTemporalNeighborSamplingDeterministic, FixedWindowLast)
{
  auto [graph, edge_start_times, edge_end_times, renumber_map] =
    make_mg_temporal_graph(*handle_,
                           /*srcs*/ {0, 1, 1},
                           /*dsts*/ {1, 2, 3},
                           /*start times*/ {50, 30, 80},
                           std::vector<time_stamp_t>{51, 31, 81});

  auto actual =
    run_and_collect(*handle_,
                    graph.view(),
                    renumber_map,
                    edge_start_times.view(),
                    edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
                    /*starts*/ {0},
                    std::vector<time_stamp_t>{10},
                    std::vector<time_stamp_t>{100},
                    /*fan_out*/ {-1, 1},
                    cugraph::neighbor_selection_t::LAST,
                    cugraph::temporal_sampling_comparison_t::FIXED_WINDOW);

  expect_edges(*handle_, std::move(actual), {{0, 1, 50, 0}, {1, 3, 80, 1}});
}

CUGRAPH_MG_TEST_PROGRAM_MAIN()

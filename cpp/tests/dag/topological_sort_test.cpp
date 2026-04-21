/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dag/dag_test_utilities.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

// Kahn's algorithm for topological sort, computing BFS levels.
// Each vertex is assigned the BFS depth at which its in-degree first reaches zero.
// (https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
template <typename vertex_t, typename edge_t>
void topological_sort_reference(edge_t const* offsets,
                                vertex_t const* indices,
                                vertex_t* levels,
                                vertex_t num_vertices)
{
  std::vector<edge_t> in_degree(num_vertices, edge_t{0});
  for (vertex_t v = 0; v < num_vertices; ++v) {
    for (edge_t e = offsets[v]; e < offsets[v + 1]; ++e) {
      ++in_degree[indices[e]];
    }
  }

  std::fill(levels, levels + num_vertices, vertex_t{0});

  std::vector<vertex_t> frontier{};
  for (vertex_t v = 0; v < num_vertices; ++v) {
    if (in_degree[v] == 0) { frontier.push_back(v); }
  }

  vertex_t level{0};
  while (!frontier.empty()) {
    std::vector<vertex_t> next_frontier{};
    for (auto v : frontier) {
      levels[v] = level;
      for (edge_t e = offsets[v]; e < offsets[v + 1]; ++e) {
        vertex_t w = indices[e];
        --in_degree[w];
        if (in_degree[w] == 0) { next_frontier.push_back(w); }
      }
    }
    frontier = std::move(next_frontier);
    ++level;
  }
}

struct TopologicalSort_Usecase {
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_TopologicalSort
  : public ::testing::TestWithParam<std::tuple<TopologicalSort_Usecase, input_usecase_t>> {
 public:
  Tests_TopologicalSort() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(TopologicalSort_Usecase const& topological_sort_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    using weight_t = float;  // dummy

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    ASSERT_FALSE(graph_view.is_symmetric())
      << "Topological sort works only on directed (asymmetric) graphs.";

    std::optional<cugraph::edge_property_t<edge_t, bool>> random_mask{std::nullopt};
    if (topological_sort_usecase.edge_masking) {
      random_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask(random_mask->view());
    }

    // Mask out every edge that lives inside a non-trivial SCC (and every self-loop) so the graph
    // handed to topological_sort is acyclic. If a random mask is already attached, the call below
    // composes with it (edges already masked stay masked).
    auto acyclic_mask = cugraph::test::build_acyclic_edge_mask(handle, graph_view);
    graph_view.attach_edge_mask(acyclic_mask.view());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("topological_sort");
    }

    auto d_levels = cugraph::topological_sort(handle, graph_view);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (topological_sort_usecase.check_correctness) {
      std::vector<edge_t> h_offsets{};
      std::vector<vertex_t> h_indices{};
      std::tie(h_offsets, h_indices, std::ignore) =
        cugraph::test::graph_to_host_csr<vertex_t, edge_t, weight_t, false, false>(
          handle, graph_view, std::nullopt, std::nullopt);

      std::vector<vertex_t> h_reference_levels(graph_view.number_of_vertices());
      topological_sort_reference(h_offsets.data(),
                                 h_indices.data(),
                                 h_reference_levels.data(),
                                 graph_view.number_of_vertices());

      auto h_cugraph_levels = cugraph::test::to_host(handle, d_levels);

      ASSERT_TRUE(std::equal(
        h_reference_levels.begin(), h_reference_levels.end(), h_cugraph_levels.begin()))
        << "topological levels do not match with the reference values.";
    }
  }
};

using Tests_TopologicalSort_File = Tests_TopologicalSort<cugraph::test::File_Usecase>;
using Tests_TopologicalSort_Rmat = Tests_TopologicalSort<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_TopologicalSort_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_TopologicalSort_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_TopologicalSort_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_TopologicalSort_File,
  ::testing::Values(
    std::make_tuple(TopologicalSort_Usecase{false},
                    cugraph::test::File_Usecase("karate-asymmetric.csv")),  // TODO: replace with real DAG dataset
    std::make_tuple(TopologicalSort_Usecase{true},
                    cugraph::test::File_Usecase("karate-asymmetric.csv")),
    std::make_tuple(TopologicalSort_Usecase{false},
                    cugraph::test::File_Usecase("test/datasets/cage6.mtx")),
    std::make_tuple(TopologicalSort_Usecase{true},
                    cugraph::test::File_Usecase("test/datasets/cage6.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_TopologicalSort_Rmat,
  ::testing::Values(
    std::make_tuple(TopologicalSort_Usecase{false},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(TopologicalSort_Usecase{true},
                    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_TopologicalSort_Rmat,
  ::testing::Values(
    std::make_tuple(TopologicalSort_Usecase{false, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(TopologicalSort_Usecase{true, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

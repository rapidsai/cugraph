/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

template <typename vertex_t, typename edge_t>
std::optional<std::tuple<edge_t, edge_t>> count_self_loops_and_multi_edges_reference(
  edge_t const* offsets, vertex_t const* indices, vertex_t num_vertices)
{
  edge_t num_self_loops{0};
  edge_t num_multi_edges{0};
  for (vertex_t i = 0; i < num_vertices; ++i) {
    if (!std::is_sorted(indices + offsets[i], indices + offsets[i + 1])) { return std::nullopt; }
    for (edge_t j = offsets[i]; j < offsets[i + 1]; ++j) {
      if (i == indices[j]) { ++num_self_loops; }
      if ((j > offsets[i]) && (indices[j - 1] == indices[j])) {  // assumes neighbors are sorted
        ++num_multi_edges;
      }
    }
  }

  return std::make_tuple(num_self_loops, num_multi_edges);
}

struct CountSelfLoopsAndMultiEdges_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_CountSelfLoopsAndMultiEdges
  : public ::testing::TestWithParam<
      std::tuple<CountSelfLoopsAndMultiEdges_Usecase, input_usecase_t>> {
 public:
  Tests_CountSelfLoopsAndMultiEdges() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(
    CountSelfLoopsAndMultiEdges_Usecase const& count_self_loops_and_multi_edges_usecase,
    input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Count self-loops");
    }

    auto num_self_loops = graph_view.count_self_loops(handle);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Count multi-edges");
    }

    auto num_multi_edges = graph_view.count_multi_edges(handle);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (count_self_loops_and_multi_edges_usecase.check_correctness) {
      cugraph::graph_t<vertex_t, edge_t, store_transposed, false> unrenumbered_graph(handle);
      if (renumber) {
        std::tie(unrenumbered_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle, input_usecase, false, false);
      }
      auto unrenumbered_graph_view = renumber ? unrenumbered_graph.view() : graph_view;

      std::vector<edge_t> h_offsets = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().offsets());
      std::vector<vertex_t> h_indices = cugraph::test::to_host(
        handle, unrenumbered_graph_view.local_edge_partition_view().indices());

      auto self_loop_and_multi_edge_counts = count_self_loops_and_multi_edges_reference(
        h_offsets.data(), h_indices.data(), unrenumbered_graph_view.number_of_vertices());

      ASSERT_TRUE(self_loop_and_multi_edge_counts.has_value())
        << "Invalid input graph to the reference code, neighbor lists must be sorted.";

      ASSERT_TRUE(num_self_loops == std::get<0>(*self_loop_and_multi_edge_counts))
        << "# self-loops does not match with the reference value.";
      ASSERT_TRUE(num_multi_edges == std::get<1>(*self_loop_and_multi_edge_counts))
        << "# multi-edges does not match with the reference value.";
    }
  }
};

using Tests_CountSelfLoopsAndMultiEdges_File =
  Tests_CountSelfLoopsAndMultiEdges<cugraph::test::File_Usecase>;
using Tests_CountSelfLoopsAndMultiEdges_Rmat =
  Tests_CountSelfLoopsAndMultiEdges<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_CountSelfLoopsAndMultiEdges_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_CountSelfLoopsAndMultiEdges_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_CountSelfLoopsAndMultiEdges_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_CountSelfLoopsAndMultiEdges_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_CountSelfLoopsAndMultiEdges_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_CountSelfLoopsAndMultiEdges_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_CountSelfLoopsAndMultiEdges_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_CountSelfLoopsAndMultiEdges_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_CountSelfLoopsAndMultiEdges_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(CountSelfLoopsAndMultiEdges_Usecase{false},
                      CountSelfLoopsAndMultiEdges_Usecase{false},
                      CountSelfLoopsAndMultiEdges_Usecase{false},
                      CountSelfLoopsAndMultiEdges_Usecase{false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

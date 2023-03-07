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
#include <utilities/test_utilities.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

typedef struct Transpose_Usecase_t {
  bool test_weighted{false};
  bool check_correctness{true};
} Transpose_Usecase;

template <typename input_usecase_t>
class Tests_Transpose
  : public ::testing::TestWithParam<std::tuple<Transpose_Usecase, input_usecase_t>> {
 public:
  Tests_Transpose() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(Transpose_Usecase const& transpose_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, transpose_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    rmm::device_uvector<vertex_t> d_org_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_org_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> d_org_weights{std::nullopt};
    if (transpose_usecase.check_correctness) {
      std::tie(d_org_srcs, d_org_dsts, d_org_weights) = cugraph::decompress_to_edgelist(
        handle,
        graph.view(),
        edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt,
        d_renumber_map_labels ? std::make_optional<raft::device_span<vertex_t const>>(
                                  (*d_renumber_map_labels).data(), (*d_renumber_map_labels).size())
                              : std::nullopt);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Transpose");
    }

    std::tie(graph, edge_weights, d_renumber_map_labels) = transpose_graph(
      handle, std::move(graph), std::move(edge_weights), std::move(d_renumber_map_labels));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (transpose_usecase.check_correctness) {
      auto [d_transposed_srcs, d_transposed_dsts, d_transposed_weights] =
        cugraph::decompress_to_edgelist(
          handle,
          graph.view(),
          edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt,
          d_renumber_map_labels
            ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
                                                                    (*d_renumber_map_labels).size())
            : std::nullopt);

      auto h_org_srcs    = cugraph::test::to_host(handle, d_org_srcs);
      auto h_org_dsts    = cugraph::test::to_host(handle, d_org_dsts);
      auto h_org_weights = cugraph::test::to_host(handle, d_org_weights);

      auto h_transposed_srcs    = cugraph::test::to_host(handle, d_transposed_srcs);
      auto h_transposed_dsts    = cugraph::test::to_host(handle, d_transposed_dsts);
      auto h_transposed_weights = cugraph::test::to_host(handle, d_transposed_weights);

      if (transpose_usecase.test_weighted) {
        std::vector<std::tuple<vertex_t, vertex_t, weight_t>> org_edges(h_org_srcs.size());
        for (size_t i = 0; i < org_edges.size(); ++i) {
          org_edges[i] = std::make_tuple(h_org_srcs[i], h_org_dsts[i], (*h_org_weights)[i]);
        }
        std::sort(org_edges.begin(), org_edges.end());

        std::vector<std::tuple<vertex_t, vertex_t, weight_t>> transposed_edges(
          h_transposed_srcs.size());
        for (size_t i = 0; i < transposed_edges.size(); ++i) {
          transposed_edges[i] =
            std::make_tuple(h_transposed_dsts[i],
                            h_transposed_srcs[i],
                            (*h_transposed_weights)[i]);  // flip sources and destinations
        }
        std::sort(transposed_edges.begin(), transposed_edges.end());

        ASSERT_TRUE(std::equal(org_edges.begin(), org_edges.end(), transposed_edges.begin()));
      } else {
        std::vector<std::tuple<vertex_t, vertex_t>> org_edges(h_org_srcs.size());
        for (size_t i = 0; i < org_edges.size(); ++i) {
          org_edges[i] = std::make_tuple(h_org_srcs[i], h_org_dsts[i]);
        }
        std::sort(org_edges.begin(), org_edges.end());

        std::vector<std::tuple<vertex_t, vertex_t>> transposed_edges(h_transposed_srcs.size());
        for (size_t i = 0; i < transposed_edges.size(); ++i) {
          transposed_edges[i] = std::make_tuple(
            h_transposed_dsts[i], h_transposed_srcs[i]);  // flip sources and destinations
        }
        std::sort(transposed_edges.begin(), transposed_edges.end());

        ASSERT_TRUE(std::equal(org_edges.begin(), org_edges.end(), transposed_edges.begin()));
      }
    }
  }
};

using Tests_Transpose_File = Tests_Transpose<cugraph::test::File_Usecase>;
using Tests_Transpose_Rmat = Tests_Transpose<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Transpose_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Transpose_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_Transpose_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Transpose_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Transpose_Rmat, CheckInt32Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Transpose_Rmat, CheckInt32Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Transpose_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_Transpose_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Transpose_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Transpose_Usecase{false}, Transpose_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Transpose_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Transpose_Usecase{false}, Transpose_Usecase{true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Transpose_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Transpose_Usecase{false, false}, Transpose_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

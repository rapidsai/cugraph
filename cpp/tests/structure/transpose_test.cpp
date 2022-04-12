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
#include <utilities/high_res_clock.h>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
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
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(Transpose_Usecase const& transpose_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, input_usecase, transpose_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    rmm::device_uvector<vertex_t> d_org_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_org_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> d_org_weights{std::nullopt};
    if (transpose_usecase.check_correctness) {
      std::tie(d_org_srcs, d_org_dsts, d_org_weights) =
        graph.decompress_to_edgelist(handle, d_renumber_map_labels, false);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    d_renumber_map_labels = graph.transpose(handle, std::move(d_renumber_map_labels));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "Transpose took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (transpose_usecase.check_correctness) {
      auto [d_transposed_srcs, d_transposed_dsts, d_transposed_weights] =
        graph.decompress_to_edgelist(handle, d_renumber_map_labels, false);

      std::vector<vertex_t> h_org_srcs(d_org_srcs.size());
      std::vector<vertex_t> h_org_dsts(h_org_srcs.size());
      auto h_org_weights =
        d_org_weights ? std::make_optional<std::vector<weight_t>>(h_org_srcs.size()) : std::nullopt;

      std::vector<vertex_t> h_transposed_srcs(d_transposed_srcs.size());
      std::vector<vertex_t> h_transposed_dsts(h_transposed_srcs.size());
      auto h_transposed_weights =
        d_transposed_weights ? std::make_optional<std::vector<weight_t>>(h_transposed_srcs.size())
                             : std::nullopt;

      raft::update_host(
        h_org_srcs.data(), d_org_srcs.data(), d_org_srcs.size(), handle.get_stream());
      raft::update_host(
        h_org_dsts.data(), d_org_dsts.data(), d_org_dsts.size(), handle.get_stream());
      if (h_org_weights) {
        raft::update_host((*h_org_weights).data(),
                          (*d_org_weights).data(),
                          (*d_org_weights).size(),
                          handle.get_stream());
      }

      raft::update_host(h_transposed_srcs.data(),
                        d_transposed_srcs.data(),
                        d_transposed_srcs.size(),
                        handle.get_stream());
      raft::update_host(h_transposed_dsts.data(),
                        d_transposed_dsts.data(),
                        d_transposed_dsts.size(),
                        handle.get_stream());
      if (h_transposed_weights) {
        raft::update_host((*h_transposed_weights).data(),
                          (*d_transposed_weights).data(),
                          (*d_transposed_weights).size(),
                          handle.get_stream());
      }

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

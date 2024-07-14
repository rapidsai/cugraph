/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
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
#include <vector>

struct Louvain_Usecase {
  std::optional<size_t> max_level_{std::nullopt};
  std::optional<double> threshold_{std::nullopt};
  std::optional<double> resolution_{std::nullopt};
  bool check_correctness_{false};
  int expected_level_{0};
  float expected_modularity_{0};
};

template <typename input_usecase_t>
class Tests_Louvain
  : public ::testing::TestWithParam<std::tuple<Louvain_Usecase, input_usecase_t>> {
 public:
  Tests_Louvain() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<Louvain_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [louvain_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // Can't currently check correctness if we renumber
    bool renumber = true;
    if (louvain_usecase.check_correctness_) renumber = false;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, true, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Louvain");
    }

    louvain(graph_view,
            edge_weight_view,
            graph_view.local_vertex_partition_range_size(),
            louvain_usecase.max_level_,
            louvain_usecase.threshold_,
            louvain_usecase.resolution_,
            louvain_usecase.check_correctness_,
            louvain_usecase.expected_level_,
            louvain_usecase.expected_modularity_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void louvain(
    cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    vertex_t num_vertices,
    std::optional<size_t> max_level,
    std::optional<double> threshold,
    std::optional<double> resolution,
    bool check_correctness,
    int expected_level,
    float expected_modularity)
  {
    raft::handle_t handle{};

    rmm::device_uvector<vertex_t> clustering_v(num_vertices, handle.get_stream());
    size_t level;
    weight_t modularity;

    if (resolution) {
      std::tie(level, modularity) = cugraph::louvain(
        handle,
        std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
        graph_view,
        edge_weight_view,
        clustering_v.data(),
        max_level ? *max_level : size_t{100},
        threshold ? static_cast<weight_t>(*threshold) : weight_t{1e-7},
        static_cast<weight_t>(*resolution));
    } else if (threshold) {
      std::tie(level, modularity) = cugraph::louvain(
        handle,
        std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
        graph_view,
        edge_weight_view,
        clustering_v.data(),
        max_level ? *max_level : size_t{100},
        static_cast<weight_t>(*threshold));
    } else if (max_level) {
      std::tie(level, modularity) = cugraph::louvain(
        handle,
        std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
        graph_view,
        edge_weight_view,
        clustering_v.data(),
        *max_level);
    } else {
      std::tie(level, modularity) = cugraph::louvain(
        handle,
        std::optional<std::reference_wrapper<raft::random::RngState>>{std::nullopt},
        graph_view,
        edge_weight_view,
        clustering_v.data());
    }

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    float compare_modularity = static_cast<float>(modularity);

    if (check_correctness) {
      ASSERT_FLOAT_EQ(compare_modularity, expected_modularity);
      ASSERT_EQ(level, expected_level);
    }
  }
};

// FIXME: add tests for type combinations

using Tests_Louvain_File   = Tests_Louvain<cugraph::test::File_Usecase>;
using Tests_Louvain_File32 = Tests_Louvain<cugraph::test::File_Usecase>;
using Tests_Louvain_File64 = Tests_Louvain<cugraph::test::File_Usecase>;
using Tests_Louvain_Rmat   = Tests_Louvain<cugraph::test::Rmat_Usecase>;
using Tests_Louvain_Rmat32 = Tests_Louvain<cugraph::test::Rmat_Usecase>;
using Tests_Louvain_Rmat64 = Tests_Louvain<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Louvain_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Louvain_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Louvain_File32, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Louvain_File64, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

#if 0
// FIXME:  We should use these tests, gtest-1.11.0 makes it a runtime error
//         to define and not instantiate these.

TEST_P(Tests_Louvain_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Louvain_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}
#endif

TEST_P(Tests_Louvain_Rmat32, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Louvain_Rmat64, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

// FIXME: Expand testing once we evaluate RMM memory use
INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_Louvain_File,
  ::testing::Combine(::testing::Values(
                       Louvain_Usecase{
                         std::nullopt, std::nullopt, std::nullopt, true, 3, 0.39907956},
                       Louvain_Usecase{20, double{1e-3}, std::nullopt, true, 3, 0.39907956},
                       Louvain_Usecase{100, double{1e-3}, double{0.8}, true, 3, 0.47547662}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Louvain_File32,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file64_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Louvain_File64,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Louvain_Rmat32,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat64_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Louvain_Rmat64,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Louvain_Usecase{}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include <cugraph/utilities/high_res_timer.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>

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

struct Leiden_Usecase {
  size_t max_level_{100};
  double resolution_{1.0};
  bool check_correctness_{false};
  int expected_level_{0};
  float expected_modularity_{0};
};

template <typename input_usecase_t>
class Tests_Leiden : public ::testing::TestWithParam<std::tuple<Leiden_Usecase, input_usecase_t>> {
 public:
  Tests_Leiden() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<Leiden_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [leiden_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // Can't currently check correctness if we renumber
    bool renumber = true;
    if (leiden_usecase.check_correctness_) renumber = false;

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

    // "FIXME": remove this check once we drop support for Pascal
    //
    // Calling leiden on Pascal will throw an exception, we'll check that
    // this is the behavior while we still support Pascal (device_prop.major < 7)
    //
    cudaDeviceProp device_prop;
    RAFT_CUDA_TRY(cudaGetDeviceProperties(&device_prop, 0));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Leiden");
    }

    if (device_prop.major < 7) {
      EXPECT_THROW(leiden(graph_view,
                          edge_weight_view,
                          graph_view.local_vertex_partition_range_size(),
                          leiden_usecase.max_level_,
                          leiden_usecase.resolution_,
                          leiden_usecase.check_correctness_,
                          leiden_usecase.expected_level_,
                          leiden_usecase.expected_modularity_),
                   cugraph::logic_error);
    } else {
      leiden(graph_view,
             edge_weight_view,
             graph_view.local_vertex_partition_range_size(),
             leiden_usecase.max_level_,
             leiden_usecase.resolution_,
             leiden_usecase.check_correctness_,
             leiden_usecase.expected_level_,
             leiden_usecase.expected_modularity_);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void leiden(
    cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    vertex_t num_vertices,
    size_t max_level,
    float resolution,
    bool check_correctness,
    int expected_level,
    float expected_modularity)
  {
    raft::handle_t handle{};

    rmm::device_uvector<vertex_t> clustering_v(num_vertices, handle.get_stream());
    size_t level;
    weight_t modularity;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    raft::random::RngState rng_state(seed);

    std::tie(level, modularity) = cugraph::leiden(
      handle, rng_state, graph_view, edge_weight_view, clustering_v.data(), max_level, resolution);

    float compare_modularity = static_cast<float>(modularity);

    if (check_correctness) {
      ASSERT_FLOAT_EQ(compare_modularity, expected_modularity);
      ASSERT_EQ(level, expected_level);
    }
  }
};

using Tests_Leiden_File   = Tests_Leiden<cugraph::test::File_Usecase>;
using Tests_Leiden_File32 = Tests_Leiden<cugraph::test::File_Usecase>;
using Tests_Leiden_File64 = Tests_Leiden<cugraph::test::File_Usecase>;
using Tests_Leiden_Rmat   = Tests_Leiden<cugraph::test::Rmat_Usecase>;
using Tests_Leiden_Rmat32 = Tests_Leiden<cugraph::test::Rmat_Usecase>;
using Tests_Leiden_Rmat64 = Tests_Leiden<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Leiden_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_File, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_File32, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_File64, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

#if 0
// FIXME:  We should use these tests, gtest-1.11.0 makes it a runtime error
//         to define and not instantiate these.
TEST_P(Tests_Leiden_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_Rmat32, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_Rmat64, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}
#endif

// FIXME: Expand testing once we evaluate RMM memory use
INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_Leiden_File,
  ::testing::Combine(::testing::Values(Leiden_Usecase{100, 1, false, 3, 0.408695}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Leiden_File32,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Leiden_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file64_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Leiden_File64,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Leiden_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

CUGRAPH_TEST_PROGRAM_MAIN()

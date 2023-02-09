/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

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
  size_t max_level_{100};
  double resolution_{1};
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
  void run_legacy_test(std::tuple<Louvain_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [louvain_usecase, input_usecase] = param;

    raft::handle_t handle{};

    bool directed{false};

    auto graph = cugraph::test::legacy::construct_graph_csr<vertex_t, edge_t, weight_t>(
      handle, input_usecase, true);
    auto graph_view = graph->view();

    // "FIXME": remove this check once we drop support for Pascal
    //
    // Calling louvain on Pascal will throw an exception, we'll check that
    // this is the behavior while we still support Pascal (device_prop.major < 7)
    //
    cudaDeviceProp device_prop;
    RAFT_CUDA_TRY(cudaGetDeviceProperties(&device_prop, 0));

    if (device_prop.major < 7) {
      EXPECT_THROW(louvain_legacy(graph_view,
                                  graph_view.get_number_of_vertices(),
                                  louvain_usecase.check_correctness_,
                                  louvain_usecase.expected_level_,
                                  louvain_usecase.expected_modularity_),
                   cugraph::logic_error);
    } else {
      louvain_legacy(graph_view,
                     graph_view.get_number_of_vertices(),
                     louvain_usecase.check_correctness_,
                     louvain_usecase.expected_level_,
                     louvain_usecase.expected_modularity_);
    }
  }

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

    // "FIXME": remove this check once we drop support for Pascal
    //
    // Calling louvain on Pascal will throw an exception, we'll check that
    // this is the behavior while we still support Pascal (device_prop.major < 7)
    //
    cudaDeviceProp device_prop;
    RAFT_CUDA_TRY(cudaGetDeviceProperties(&device_prop, 0));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Louvain");
    }

    if (device_prop.major < 7) {
      EXPECT_THROW(louvain(graph_view,
                           edge_weight_view,
                           graph_view.local_vertex_partition_range_size(),
                           louvain_usecase.check_correctness_,
                           louvain_usecase.expected_level_,
                           louvain_usecase.expected_modularity_),
                   cugraph::logic_error);
    } else {
      louvain(graph_view,
              edge_weight_view,
              graph_view.local_vertex_partition_range_size(),
              louvain_usecase.check_correctness_,
              louvain_usecase.expected_level_,
              louvain_usecase.expected_modularity_);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void louvain_legacy(cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph_view,
                      vertex_t num_vertices,
                      bool check_correctness,
                      int expected_level,
                      float expected_modularity)
  {
    raft::handle_t handle{};

    rmm::device_uvector<vertex_t> clustering_v(num_vertices, handle.get_stream());
    size_t level;
    weight_t modularity;

    std::tie(level, modularity) =
      cugraph::louvain(handle, graph_view, clustering_v.data(), size_t{100}, weight_t{1});

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    float compare_modularity = static_cast<float>(modularity);

    if (check_correctness) {
      ASSERT_FLOAT_EQ(compare_modularity, expected_modularity);
      ASSERT_EQ(level, expected_level);
    }
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void louvain(
    cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
    vertex_t num_vertices,
    bool check_correctness,
    int expected_level,
    float expected_modularity)
  {
    raft::handle_t handle{};

    rmm::device_uvector<vertex_t> clustering_v(num_vertices, handle.get_stream());
    size_t level;
    weight_t modularity;

    std::tie(level, modularity) = cugraph::louvain(
      handle, graph_view, edge_weight_view, clustering_v.data(), size_t{100}, weight_t{1});

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    float compare_modularity = static_cast<float>(modularity);

    if (check_correctness) {
      ASSERT_FLOAT_EQ(compare_modularity, expected_modularity);
      ASSERT_EQ(level, expected_level);
    }
  }
};

// FIXME: add tests for type combinations

TEST(louvain_legacy, success)
{
  raft::handle_t handle;

  auto stream = handle.get_stream();

  std::vector<int> off_h = {0,  16,  25,  35,  41,  44,  48,  52,  56,  61,  63, 66,
                            67, 69,  74,  76,  78,  80,  82,  84,  87,  89,  91, 93,
                            98, 101, 104, 106, 110, 113, 117, 121, 127, 139, 156};
  std::vector<int> ind_h = {
    1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 0,  2,  3,  7,  13, 17, 19,
    21, 30, 0,  1,  3,  7,  8,  9,  13, 27, 28, 32, 0,  1,  2,  7,  12, 13, 0,  6,  10, 0,  6,
    10, 16, 0,  4,  5,  16, 0,  1,  2,  3,  0,  2,  30, 32, 33, 2,  33, 0,  4,  5,  0,  0,  3,
    0,  1,  2,  3,  33, 32, 33, 32, 33, 5,  6,  0,  1,  32, 33, 0,  1,  33, 32, 33, 0,  1,  32,
    33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31, 29, 33, 2,  23, 24, 33, 2,  31, 33, 23, 26,
    32, 33, 1,  8,  32, 33, 0,  24, 25, 28, 32, 33, 2,  8,  14, 15, 18, 20, 22, 23, 29, 30, 31,
    33, 8,  9,  13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32};
  std::vector<float> w_h = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  std::vector<int> result_h = {1, 1, 1, 1, 0, 0, 0, 1, 3, 1, 0, 1, 1, 1, 3, 3, 0,
                               1, 3, 1, 3, 1, 3, 2, 2, 2, 3, 2, 1, 3, 3, 2, 3, 3};

  int num_verts = off_h.size() - 1;
  int num_edges = ind_h.size();

  rmm::device_uvector<int> offsets_v(num_verts + 1, stream);
  rmm::device_uvector<int> indices_v(num_edges, stream);
  rmm::device_uvector<float> weights_v(num_edges, stream);
  rmm::device_uvector<int> result_v(num_verts, stream);

  raft::update_device(offsets_v.data(), off_h.data(), off_h.size(), stream);
  raft::update_device(indices_v.data(), ind_h.data(), ind_h.size(), stream);
  raft::update_device(weights_v.data(), w_h.data(), w_h.size(), stream);

  cugraph::legacy::GraphCSRView<int, int, float> G(
    offsets_v.data(), indices_v.data(), weights_v.data(), num_verts, num_edges);

  float modularity{0.0};
  size_t num_level = 40;

  // "FIXME": remove this check once we drop support for Pascal
  //
  // Calling louvain on Pascal will throw an exception, we'll check that
  // this is the behavior while we still support Pascal (device_prop.major < 7)
  //
  if (handle.get_device_properties().major < 7) {
    EXPECT_THROW(cugraph::louvain(handle, G, result_v.data()), cugraph::logic_error);
  } else {
    std::tie(num_level, modularity) = cugraph::louvain(handle, G, result_v.data());

    auto cluster_id = cugraph::test::to_host(handle, result_v);

    int min = *min_element(cluster_id.begin(), cluster_id.end());

    ASSERT_GE(min, 0);
    ASSERT_FLOAT_EQ(modularity, 0.408695);
    ASSERT_EQ(cluster_id, result_h);
  }
}

TEST(louvain_legacy_renumbered, success)
{
  raft::handle_t handle;

  auto stream = handle.get_stream();

  std::vector<int> off_h = {0,   16,  25,  30,  34,  38,  42,  44,  46,  48,  50,  52,
                            54,  56,  73,  85,  95,  101, 107, 112, 117, 121, 125, 129,
                            132, 135, 138, 141, 144, 147, 149, 151, 153, 155, 156};
  std::vector<int> ind_h = {
    1,  3,  7,  11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 30, 33, 0,  5,  11, 15, 16, 19, 21,
    25, 30, 4,  13, 14, 22, 27, 0,  9,  20, 24, 2,  13, 15, 26, 1,  13, 14, 18, 13, 15, 0,  16,
    13, 14, 3,  20, 13, 14, 0,  1,  13, 22, 2,  4,  5,  6,  8,  10, 12, 14, 17, 18, 19, 22, 25,
    28, 29, 31, 32, 2,  5,  8,  10, 13, 15, 17, 18, 22, 29, 31, 32, 0,  1,  4,  6,  14, 16, 18,
    19, 21, 28, 0,  1,  7,  15, 19, 21, 0,  13, 14, 26, 27, 28, 0,  5,  13, 14, 15, 0,  1,  13,
    16, 16, 0,  3,  9,  23, 0,  1,  15, 16, 2,  12, 13, 14, 0,  20, 24, 0,  3,  23, 0,  1,  13,
    4,  17, 27, 2,  17, 26, 13, 15, 17, 13, 14, 0,  1,  13, 14, 13, 14, 0};

  std::vector<float> w_h = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  int num_verts = off_h.size() - 1;
  int num_edges = ind_h.size();

  rmm::device_uvector<int> offsets_v(num_verts + 1, stream);
  rmm::device_uvector<int> indices_v(num_edges, stream);
  rmm::device_uvector<float> weights_v(num_edges, stream);
  rmm::device_uvector<int> result_v(num_verts, stream);

  raft::update_device(offsets_v.data(), off_h.data(), off_h.size(), stream);
  raft::update_device(indices_v.data(), ind_h.data(), ind_h.size(), stream);
  raft::update_device(weights_v.data(), w_h.data(), w_h.size(), stream);

  cugraph::legacy::GraphCSRView<int, int, float> G(
    offsets_v.data(), indices_v.data(), weights_v.data(), num_verts, num_edges);

  float modularity{0.0};
  size_t num_level = 40;

  // "FIXME": remove this check once we drop support for Pascal
  //
  // Calling louvain on Pascal will throw an exception, we'll check that
  // this is the behavior while we still support Pascal (device_prop.major < 7)
  //
  if (handle.get_device_properties().major < 7) {
    EXPECT_THROW(cugraph::louvain(handle, G, result_v.data()), cugraph::logic_error);
  } else {
    std::tie(num_level, modularity) = cugraph::louvain(handle, G, result_v.data());

    auto cluster_id = cugraph::test::to_host(handle, result_v);

    int min = *min_element(cluster_id.begin(), cluster_id.end());

    ASSERT_GE(min, 0);
    ASSERT_FLOAT_EQ(modularity, 0.41880345);
  }
}

using Tests_Louvain_File   = Tests_Louvain<cugraph::test::File_Usecase>;
using Tests_Louvain_File32 = Tests_Louvain<cugraph::test::File_Usecase>;
using Tests_Louvain_File64 = Tests_Louvain<cugraph::test::File_Usecase>;
using Tests_Louvain_Rmat   = Tests_Louvain<cugraph::test::Rmat_Usecase>;
using Tests_Louvain_Rmat32 = Tests_Louvain<cugraph::test::Rmat_Usecase>;
using Tests_Louvain_Rmat64 = Tests_Louvain<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Louvain_File, CheckInt32Int32FloatFloatLegacy)
{
  run_legacy_test<int32_t, int32_t, float, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

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

TEST_P(Tests_Louvain_Rmat, CheckInt32Int32FloatFloatLegacy)
{
  run_legacy_test<int32_t, int32_t, float, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

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
  ::testing::Combine(::testing::Values(Louvain_Usecase{100, 1, true, 3, 0.408695}),
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

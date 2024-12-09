/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"

#include <cugraph_c/types.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/device_vector.hpp>
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
#include <map>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

struct TemporalGraph_Usecase {
  std::optional<cugraph_data_type_id_t> weight_type{std::nullopt};
  bool use_end_time{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_TemporalGraph
  : public ::testing::TestWithParam<std::tuple<TemporalGraph_Usecase, input_usecase_t>> {
 public:
  Tests_TemporalGraph() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(
    std::tuple<TemporalGraph_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber                      = true;
    auto [temporal_graph_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    std::vector<std::optional<std::function<cugraph::device_vector_t(size_t)>>> other_types;
    constexpr uint64_t seed{0};
    raft::random::RngState rng_state(seed);

    other_types.push_back(
      std::make_optional([stream_view = handle.get_stream(), &rng_state](size_t size) {
        rmm::device_uvector<int32_t> result(size, stream_view);

        cugraph::detail::uniform_random_fill(
          stream_view, result.data(), result.size(), int32_t{0}, int32_t{20000}, rng_state);

        return cugraph::device_vector_t(std::move(result));
      }));

    if (temporal_graph_usecase.use_end_time) {
      other_types.push_back(
        std::make_optional([stream_view = handle.get_stream(), &rng_state](size_t size) {
          rmm::device_uvector<int32_t> result(size, stream_view);

          cugraph::detail::uniform_random_fill(
            stream_view, result.data(), result.size(), int32_t{20000}, int32_t{40000}, rng_state);

          return cugraph::device_vector_t(std::move(result));
        }));
    }

    auto [graph, edge_properties, d_renumber_map_labels] =
      cugraph::test::construct_graph_with_properties<vertex_t, edge_t, store_transposed, false>(
        handle, input_usecase, temporal_graph_usecase.weight_type, other_types, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    if (temporal_graph_usecase.check_correctness) {}
  }
};

using Tests_TemporalGraph_File = Tests_TemporalGraph<cugraph::test::File_Usecase>;
using Tests_TemporalGraph_Rmat = Tests_TemporalGraph<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_TemporalGraph_File, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_TemporalGraph_File, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_TemporalGraph_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_TemporalGraph_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_TemporalGraph_Rmat, CheckInt64Int64FloatTransposeFalse)
{
  run_current_test<int64_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_TemporalGraph_Rmat, CheckInt64Int64FloatTransposeTrue)
{
  run_current_test<int64_t, int64_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_TemporalGraph_File,
  ::testing::Combine(
    // enable correctness check
    ::testing::Values(
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = true, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = true, .check_correctness = true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_TemporalGraph_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = true, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = true, .check_correctness = true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false),
                      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_TemporalGraph_File,
  ::testing::Combine(
    ::testing::Values(
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = true, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = true, .check_correctness = true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_TemporalGraph_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = std::nullopt, .use_end_time = true, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = false, .check_correctness = true},
      TemporalGraph_Usecase{
        .weight_type = FLOAT32, .use_end_time = true, .check_correctness = true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false),
                      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

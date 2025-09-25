/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "utilities/validation_utilities.hpp"

#include <cugraph_c/types.h>

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

using edge_type_t = int32_t;
using edge_time_t = int32_t;

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

    auto [edge_src_chunks, edge_dst_chunks, edge_weight_chunks, d_vertices_v, is_symmetric] =
      input_usecase.template construct_edgelist<vertex_t, weight_t>(
        handle, temporal_graph_usecase.weight_type.has_value(), store_transposed, false);

    size_t num_edges{0};
    for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
      num_edges += edge_src_chunks[i].size();
    }

    CUGRAPH_EXPECTS(num_edges <= static_cast<size_t>(std::numeric_limits<edge_t>::max()),
                    "Invalid template parameter: edge_t overflow.");

    auto edge_start_time_chunks =
      std::make_optional<std::vector<rmm::device_uvector<edge_time_t>>>();
    auto edge_end_time_chunks =
      temporal_graph_usecase.use_end_time
        ? std::make_optional<std::vector<rmm::device_uvector<edge_time_t>>>()
        : std::nullopt;
    constexpr uint64_t seed{0};
    raft::random::RngState rng_state(seed);

    for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
      edge_start_time_chunks->push_back(
        rmm::device_uvector<edge_time_t>(edge_src_chunks[i].size(), handle.get_stream()));
      cugraph::detail::uniform_random_fill(handle.get_stream(),
                                           edge_start_time_chunks->back().data(),
                                           edge_start_time_chunks->back().size(),
                                           edge_time_t{0},
                                           edge_time_t{20000},
                                           rng_state);

      if (temporal_graph_usecase.use_end_time) {
        edge_end_time_chunks->push_back(
          rmm::device_uvector<edge_time_t>(edge_src_chunks[i].size(), handle.get_stream()));
        cugraph::detail::uniform_random_fill(handle.get_stream(),
                                             edge_start_time_chunks->back().data(),
                                             edge_start_time_chunks->back().size(),
                                             edge_time_t{20000},
                                             edge_time_t{40000},
                                             rng_state);
      }
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

    size_t size = std::transform_reduce(edge_src_chunks.begin(),
                                        edge_src_chunks.end(),
                                        size_t{0},
                                        std::plus<size_t>(),
                                        [](auto const& vector) { return vector.size(); });

    rmm::device_uvector<vertex_t> original_srcs(size, handle.get_stream());
    rmm::device_uvector<vertex_t> original_dsts(size, handle.get_stream());
    auto original_wgts =
      temporal_graph_usecase.weight_type.has_value()
        ? std::make_optional<rmm::device_uvector<weight_t>>(size, handle.get_stream())
        : std::nullopt;
    auto original_start_times =
      std::make_optional<rmm::device_uvector<edge_time_t>>(size, handle.get_stream());
    auto original_end_times =
      temporal_graph_usecase.use_end_time
        ? std::make_optional<rmm::device_uvector<edge_time_t>>(size, handle.get_stream())
        : std::nullopt;

    size_t last_pos = 0;
    for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
      raft::copy(original_srcs.data() + last_pos,
                 edge_src_chunks[i].data(),
                 edge_src_chunks[i].size(),
                 handle.get_stream());
      raft::copy(original_dsts.data() + last_pos,
                 edge_dst_chunks[i].data(),
                 edge_dst_chunks[i].size(),
                 handle.get_stream());
      if (original_wgts)
        raft::copy(original_wgts->data() + last_pos,
                   (*edge_weight_chunks)[i].data(),
                   (*edge_weight_chunks)[i].size(),
                   handle.get_stream());
      if (original_start_times)
        raft::copy(original_start_times->data() + last_pos,
                   (*edge_start_time_chunks)[i].data(),
                   (*edge_start_time_chunks)[i].size(),
                   handle.get_stream());
      if (original_end_times)
        raft::copy(original_end_times->data() + last_pos,
                   (*edge_end_time_chunks)[i].data(),
                   (*edge_end_time_chunks)[i].size(),
                   handle.get_stream());

      last_pos += edge_src_chunks[i].size();
    }

    std::vector<std::vector<cugraph::arithmetic_device_uvector_t>> edgelist_edge_properties{};
    if (edge_weight_chunks) {
      std::vector<cugraph::arithmetic_device_uvector_t> edge_weight_properties{};
      for (size_t i = 0; i < edge_weight_chunks->size(); ++i) {
        edge_weight_properties.push_back(std::move((*edge_weight_chunks)[i]));
      }
      edgelist_edge_properties.push_back(std::move(edge_weight_properties));
    }
    if (edge_start_time_chunks) {
      std::vector<cugraph::arithmetic_device_uvector_t> edge_start_time_properties{};
      for (size_t i = 0; i < edge_start_time_chunks->size(); ++i) {
        edge_start_time_properties.push_back(std::move((*edge_start_time_chunks)[i]));
      }
      edgelist_edge_properties.push_back(std::move(edge_start_time_properties));
    }
    if (edge_end_time_chunks) {
      std::vector<cugraph::arithmetic_device_uvector_t> edge_end_time_properties{};
      for (size_t i = 0; i < edge_end_time_chunks->size(); ++i) {
        edge_end_time_properties.push_back(std::move((*edge_end_time_chunks)[i]));
      }
      edgelist_edge_properties.push_back(std::move(edge_end_time_properties));
    }

    std::vector<cugraph::edge_arithmetic_property_t<edge_t>> edge_properties{};

    std::tie(graph, edge_properties, renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, store_transposed, false>(
        handle,
        std::move(d_vertices_v),
        std::move(edge_src_chunks),
        std::move(edge_dst_chunks),
        std::move(edgelist_edge_properties),
        cugraph::graph_properties_t{is_symmetric, true},
        renumber);

    size_t pos{0};
    auto edge_weights =
      edge_weight_chunks
        ? std::make_optional(
            std::move(std::get<cugraph::edge_property_t<edge_t, weight_t>>(edge_properties[pos++])))
        : std::nullopt;
    auto edge_start_times =
      edge_start_time_chunks
        ? std::make_optional(std::move(
            std::get<cugraph::edge_property_t<edge_t, edge_time_t>>(edge_properties[pos++])))
        : std::nullopt;
    auto edge_end_times =
      edge_end_time_chunks
        ? std::make_optional(std::move(
            std::get<cugraph::edge_property_t<edge_t, edge_time_t>>(edge_properties[pos++])))
        : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();

    if (temporal_graph_usecase.check_correctness) {
      // FIXME:  decompress_to_edgelist should support all properties
      //  This hack only works if edge_t == edge_time_t
      auto [result_srcs, result_dsts, result_wgts, result_start_times, result_end_times] = cugraph::
        decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_time_t, store_transposed, false>(
          handle,
          graph.view(),
          edge_weights ? std::make_optional(edge_weights->view()) : std::nullopt,
          edge_start_times ? std::make_optional(edge_start_times->view()) : std::nullopt,
          edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
          renumber ? std::make_optional<raft::device_span<vertex_t const>>(renumber_map->data(),
                                                                           renumber_map->size())
                   : std::nullopt);

      cugraph::test::sort<vertex_t, edge_t, weight_t, int32_t, edge_time_t>(
        handle,
        raft::device_span<vertex_t>{result_srcs.data(), result_srcs.size()},
        raft::device_span<vertex_t>{result_dsts.data(), result_dsts.size()},
        result_wgts ? std::make_optional<raft::device_span<weight_t>>(result_wgts->data(),
                                                                      result_wgts->size())
                    : std::nullopt,
        std::nullopt,
        std::nullopt,
        result_start_times ? std::make_optional<raft::device_span<edge_time_t>>(
                               result_start_times->data(), result_start_times->size())
                           : std::nullopt,
        result_end_times ? std::make_optional<raft::device_span<edge_time_t>>(
                             result_end_times->data(), result_end_times->size())
                         : std::nullopt);

      cugraph::test::sort<vertex_t, edge_t, weight_t, int32_t, edge_time_t>(
        handle,
        raft::device_span<vertex_t>{original_srcs.data(), original_srcs.size()},
        raft::device_span<vertex_t>{original_dsts.data(), original_dsts.size()},
        original_wgts ? std::make_optional<raft::device_span<weight_t>>(original_wgts->data(),
                                                                        original_wgts->size())
                      : std::nullopt,
        std::nullopt,
        std::nullopt,
        original_start_times ? std::make_optional<raft::device_span<edge_time_t>>(
                                 original_start_times->data(), original_start_times->size())
                             : std::nullopt,
        original_end_times ? std::make_optional<raft::device_span<edge_time_t>>(
                               original_end_times->data(), original_end_times->size())
                           : std::nullopt);

      ASSERT_TRUE(cugraph::test::device_spans_equal(
        handle,
        raft::device_span<vertex_t const>{result_srcs.data(), result_srcs.size()},
        raft::device_span<vertex_t const>{original_srcs.data(), original_srcs.size()}));
      ASSERT_TRUE(cugraph::test::device_spans_equal(
        handle,
        raft::device_span<vertex_t const>{result_srcs.data(), result_srcs.size()},
        raft::device_span<vertex_t const>{original_srcs.data(), original_srcs.size()}));
      if (result_wgts)
        ASSERT_TRUE(
          result_wgts.has_value() == original_wgts.has_value() &&
          cugraph::test::device_spans_equal(
            handle,
            raft::device_span<weight_t const>{result_wgts->data(), result_wgts->size()},
            raft::device_span<weight_t const>{original_wgts->data(), original_wgts->size()}));
      if (result_start_times)
        ASSERT_TRUE(result_start_times.has_value() == original_start_times.has_value() &&
                    cugraph::test::device_spans_equal(
                      handle,
                      raft::device_span<edge_time_t const>{result_start_times->data(),
                                                           result_start_times->size()},
                      raft::device_span<edge_time_t const>{original_start_times->data(),
                                                           original_start_times->size()}));
      if (result_end_times)
        ASSERT_TRUE(result_end_times.has_value() == original_end_times.has_value() &&
                    cugraph::test::device_spans_equal(
                      handle,
                      raft::device_span<edge_time_t const>{result_end_times->data(),
                                                           result_end_times->size()},
                      raft::device_span<edge_time_t const>{original_end_times->data(),
                                                           original_end_times->size()}));
    }
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

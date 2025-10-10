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
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"
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
class Tests_MGTemporalGraph
  : public ::testing::TestWithParam<std::tuple<TemporalGraph_Usecase, input_usecase_t>> {
 public:
  Tests_MGTemporalGraph() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(
    std::tuple<TemporalGraph_Usecase const&, input_usecase_t const&> const& param)
  {
    constexpr bool renumber                      = true;
    auto [temporal_graph_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [edge_src_chunks, edge_dst_chunks, edge_weight_chunks, d_vertices_v, is_symmetric] =
      input_usecase.template construct_edgelist<vertex_t, weight_t>(
        *handle_, temporal_graph_usecase.weight_type.has_value(), store_transposed, true);

    size_t num_edges{0};
    for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
      num_edges += edge_src_chunks[i].size();
    }

    CUGRAPH_EXPECTS(num_edges <= static_cast<size_t>(std::numeric_limits<edge_t>::max()),
                    "Invalid template parameter: edge_t overflow.");

    size_t base_offset{0};
    auto base_offsets =
      cugraph::host_scalar_allgather(handle_->get_comms(), num_edges, handle_->get_stream());
    handle_->sync_stream();
    std::exclusive_scan(base_offsets.begin(), base_offsets.end(), base_offsets.begin(), size_t{0});
    base_offset = base_offsets[handle_->get_comms().get_rank()];

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
        rmm::device_uvector<edge_time_t>(edge_src_chunks[i].size(), handle_->get_stream()));
      cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                           edge_start_time_chunks->back().data(),
                                           edge_start_time_chunks->back().size(),
                                           edge_time_t{0},
                                           edge_time_t{20000},
                                           rng_state);

      if (temporal_graph_usecase.use_end_time) {
        edge_end_time_chunks->push_back(
          rmm::device_uvector<edge_time_t>(edge_src_chunks[i].size(), handle_->get_stream()));
        cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                             edge_end_time_chunks->back().data(),
                                             edge_end_time_chunks->back().size(),
                                             edge_time_t{20000},
                                             edge_time_t{40000},
                                             rng_state);
      }
    }

    cugraph::graph_t<vertex_t, edge_t, store_transposed, true> graph(*handle_);
    std::optional<cugraph::edge_property_t<edge_t, weight_t>> edge_weights{std::nullopt};
    std::optional<cugraph::edge_property_t<edge_t, edge_time_t>> edge_start_times{std::nullopt};
    std::optional<cugraph::edge_property_t<edge_t, edge_time_t>> edge_end_times{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

    size_t size = std::transform_reduce(edge_src_chunks.begin(),
                                        edge_src_chunks.end(),
                                        size_t{0},
                                        std::plus<size_t>(),
                                        [](auto const& vector) { return vector.size(); });

    rmm::device_uvector<vertex_t> local_original_srcs(size, handle_->get_stream());
    rmm::device_uvector<vertex_t> local_original_dsts(size, handle_->get_stream());
    auto local_original_wgts =
      temporal_graph_usecase.weight_type.has_value()
        ? std::make_optional<rmm::device_uvector<weight_t>>(size, handle_->get_stream())
        : std::nullopt;
    auto local_original_start_times =
      std::make_optional<rmm::device_uvector<edge_time_t>>(size, handle_->get_stream());
    auto local_original_end_times =
      temporal_graph_usecase.use_end_time
        ? std::make_optional<rmm::device_uvector<edge_time_t>>(size, handle_->get_stream())
        : std::nullopt;

    size_t last_pos = 0;
    for (size_t i = 0; i < edge_src_chunks.size(); ++i) {
      raft::copy(local_original_srcs.data() + last_pos,
                 edge_src_chunks[i].data(),
                 edge_src_chunks[i].size(),
                 handle_->get_stream());
      raft::copy(local_original_dsts.data() + last_pos,
                 edge_dst_chunks[i].data(),
                 edge_dst_chunks[i].size(),
                 handle_->get_stream());
      if (local_original_wgts)
        raft::copy(local_original_wgts->data() + last_pos,
                   (*edge_weight_chunks)[i].data(),
                   (*edge_weight_chunks)[i].size(),
                   handle_->get_stream());
      if (local_original_start_times)
        raft::copy(local_original_start_times->data() + last_pos,
                   (*edge_start_time_chunks)[i].data(),
                   (*edge_start_time_chunks)[i].size(),
                   handle_->get_stream());
      if (local_original_end_times)
        raft::copy(local_original_end_times->data() + last_pos,
                   (*edge_end_time_chunks)[i].data(),
                   (*edge_end_time_chunks)[i].size(),
                   handle_->get_stream());

      last_pos += edge_src_chunks[i].size();
    }

    std::vector<std::vector<cugraph::arithmetic_device_uvector_t>> edgelist_edge_properties{};
    if (local_original_wgts) {
      std::vector<cugraph::arithmetic_device_uvector_t> properties{};
      std::for_each(edge_weight_chunks->begin(),
                    edge_weight_chunks->end(),
                    [&properties](auto& vector) { properties.push_back(std::move(vector)); });
      edgelist_edge_properties.push_back(std::move(properties));
    }
    if (local_original_start_times) {
      std::vector<cugraph::arithmetic_device_uvector_t> properties{};
      std::for_each(edge_start_time_chunks->begin(),
                    edge_start_time_chunks->end(),
                    [&properties](auto& vector) { properties.push_back(std::move(vector)); });
      edgelist_edge_properties.push_back(std::move(properties));
    }
    if (local_original_end_times) {
      std::vector<cugraph::arithmetic_device_uvector_t> properties{};
      std::for_each(edge_end_time_chunks->begin(),
                    edge_end_time_chunks->end(),
                    [&properties](auto& vector) { properties.push_back(std::move(vector)); });
      edgelist_edge_properties.push_back(std::move(properties));
    }

    std::vector<cugraph::edge_arithmetic_property_t<edge_t>> edge_properties{};

    std::tie(graph, edge_properties, renumber_map) =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, store_transposed, true>(
        *handle_,
        std::move(d_vertices_v),
        std::move(edge_src_chunks),
        std::move(edge_dst_chunks),
        std::move(edgelist_edge_properties),
        cugraph::graph_properties_t{is_symmetric, true},
        renumber);

    size_t pos{0};
    auto edge_weights_property =
      local_original_wgts
        ? std::make_optional(
            std::move(std::get<cugraph::edge_property_t<edge_t, weight_t>>(edge_properties[pos++])))
        : std::nullopt;
    auto edge_start_times_property =
      local_original_start_times
        ? std::make_optional(std::move(
            std::get<cugraph::edge_property_t<edge_t, edge_time_t>>(edge_properties[pos++])))
        : std::nullopt;
    auto edge_end_times_property =
      local_original_end_times
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
      auto [local_result_srcs,
            local_result_dsts,
            local_result_wgts,
            local_result_start_times,
            local_result_end_times] = cugraph::
        decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_time_t, store_transposed, true>(
          *handle_,
          graph.view(),
          edge_weights ? std::make_optional(edge_weights->view()) : std::nullopt,
          edge_start_times ? std::make_optional(edge_start_times->view()) : std::nullopt,
          edge_end_times ? std::make_optional(edge_end_times->view()) : std::nullopt,
          renumber ? std::make_optional<raft::device_span<vertex_t const>>(renumber_map->data(),
                                                                           renumber_map->size())
                   : std::nullopt);

      auto result_srcs =
        cugraph::test::device_gatherv(*handle_, local_result_srcs.data(), local_result_srcs.size());
      auto result_dsts =
        cugraph::test::device_gatherv(*handle_, local_result_dsts.data(), local_result_dsts.size());

      std::optional<rmm::device_uvector<weight_t>> result_wgts{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> result_start_times{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> result_end_times{std::nullopt};

      if (local_result_wgts)
        result_wgts = cugraph::test::device_gatherv(
          *handle_, local_result_wgts->data(), local_result_wgts->size());
      if (local_result_start_times)
        result_start_times = cugraph::test::device_gatherv(
          *handle_, local_result_start_times->data(), local_result_start_times->size());
      if (local_result_end_times)
        result_end_times = cugraph::test::device_gatherv(
          *handle_, local_result_end_times->data(), local_result_end_times->size());

      auto original_srcs = cugraph::test::device_gatherv(
        *handle_, local_original_srcs.data(), local_original_srcs.size());
      auto original_dsts = cugraph::test::device_gatherv(
        *handle_, local_original_dsts.data(), local_original_dsts.size());

      std::optional<rmm::device_uvector<weight_t>> original_wgts{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> original_start_times{std::nullopt};
      std::optional<rmm::device_uvector<edge_time_t>> original_end_times{std::nullopt};

      if (local_original_wgts)
        original_wgts = cugraph::test::device_gatherv(
          *handle_, local_original_wgts->data(), local_original_wgts->size());
      if (local_original_start_times)
        original_start_times = cugraph::test::device_gatherv(
          *handle_, local_original_start_times->data(), local_original_start_times->size());
      if (local_original_end_times)
        original_end_times = cugraph::test::device_gatherv(
          *handle_, local_original_end_times->data(), local_original_end_times->size());

      if (handle_->get_comms().get_rank() == 0) {
        cugraph::test::sort<vertex_t, edge_t, weight_t, int32_t, edge_time_t>(
          *handle_,
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
          *handle_,
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
          *handle_,
          raft::device_span<vertex_t const>{result_srcs.data(), result_srcs.size()},
          raft::device_span<vertex_t const>{original_srcs.data(), original_srcs.size()}));
        ASSERT_TRUE(cugraph::test::device_spans_equal(
          *handle_,
          raft::device_span<vertex_t const>{result_srcs.data(), result_srcs.size()},
          raft::device_span<vertex_t const>{original_srcs.data(), original_srcs.size()}));
        if (result_wgts)
          ASSERT_TRUE(
            result_wgts.has_value() == original_wgts.has_value() &&
            cugraph::test::device_spans_equal(
              *handle_,
              raft::device_span<weight_t const>{result_wgts->data(), result_wgts->size()},
              raft::device_span<weight_t const>{original_wgts->data(), original_wgts->size()}));
        if (result_start_times)
          ASSERT_TRUE(result_start_times.has_value() == original_start_times.has_value() &&
                      cugraph::test::device_spans_equal(
                        *handle_,
                        raft::device_span<edge_time_t const>{result_start_times->data(),
                                                             result_start_times->size()},
                        raft::device_span<edge_time_t const>{original_start_times->data(),
                                                             original_start_times->size()}));
        if (result_end_times)
          ASSERT_TRUE(result_end_times.has_value() == original_end_times.has_value() &&
                      cugraph::test::device_spans_equal(
                        *handle_,
                        raft::device_span<edge_time_t const>{result_end_times->data(),
                                                             result_end_times->size()},
                        raft::device_span<edge_time_t const>{original_end_times->data(),
                                                             original_end_times->size()}));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTemporalGraph<input_usecase_t>::handle_ = nullptr;

using Tests_MGTemporalGraph_File = Tests_MGTemporalGraph<cugraph::test::File_Usecase>;
using Tests_MGTemporalGraph_Rmat = Tests_MGTemporalGraph<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTemporalGraph_File, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGTemporalGraph_File, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGTemporalGraph_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGTemporalGraph_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTemporalGraph_File,
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
  Tests_MGTemporalGraph_Rmat,
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
  Tests_MGTemporalGraph_File,
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
  Tests_MGTemporalGraph_Rmat,
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

CUGRAPH_MG_TEST_PROGRAM_MAIN()

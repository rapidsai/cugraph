/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <centrality/betweenness_centrality_validate.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
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

struct EdgeBetweennessCentrality_Usecase {
  size_t num_seeds{std::numeric_limits<size_t>::max()};
  bool normalized{false};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGEdgeBetweennessCentrality
  : public ::testing::TestWithParam<
      std::tuple<EdgeBetweennessCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_MGEdgeBetweennessCentrality() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<EdgeBetweennessCentrality_Usecase, input_usecase_t> const& param)
  {
    constexpr bool do_expensive_check = false;

    auto [betweenness_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, betweenness_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    raft::random::RngState rng_state(handle_->get_comms().get_rank());
    auto d_mg_seeds = cugraph::select_random_vertices(
      *handle_,
      mg_graph_view,
      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      rng_state,
      betweenness_usecase.num_seeds,
      false,
      true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG edge betweenness centrality");
    }

    auto d_centralities = cugraph::edge_betweenness_centrality(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::make_optional<raft::device_span<vertex_t const>>(
        raft::device_span<vertex_t const>{d_mg_seeds.data(), d_mg_seeds.size()}),
      betweenness_usecase.normalized);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (betweenness_usecase.check_correctness) {
      // Extract MG results
      auto [d_cugraph_src_vertex_ids, d_cugraph_dst_vertex_ids, d_cugraph_results] =
        cugraph::test::graph_to_device_coo(
          *handle_, mg_graph_view, std::make_optional(d_centralities.view()));

      // Create SG graph so we can generate SG results
      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        false);

      auto d_mg_aggregate_seeds = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{d_mg_seeds.data(), d_mg_seeds.size()});

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_edge_weights_view =
          sg_edge_weights ? std::make_optional(sg_edge_weights->view()) : std::nullopt;

        // Generate SG results and compare
        auto d_sg_centralities = cugraph::edge_betweenness_centrality(
          *handle_,
          sg_graph.view(),
          sg_edge_weights_view,
          std::make_optional<raft::device_span<vertex_t const>>(raft::device_span<vertex_t const>{
            d_mg_aggregate_seeds.data(), d_mg_aggregate_seeds.size()}),
          betweenness_usecase.normalized,
          do_expensive_check);

        auto [d_sg_src_vertex_ids, d_sg_dst_vertex_ids, d_sg_reference_centralities] =
          cugraph::test::graph_to_device_coo(
            *handle_, sg_graph.view(), std::make_optional(d_sg_centralities.view()));

        cugraph::test::edge_betweenness_centrality_validate(*handle_,
                                                            d_cugraph_src_vertex_ids,
                                                            d_cugraph_dst_vertex_ids,
                                                            *d_cugraph_results,
                                                            d_sg_src_vertex_ids,
                                                            d_sg_dst_vertex_ids,
                                                            *d_sg_reference_centralities);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGEdgeBetweennessCentrality<input_usecase_t>::handle_ =
  nullptr;

using Tests_MGEdgeBetweennessCentrality_File =
  Tests_MGEdgeBetweennessCentrality<cugraph::test::File_Usecase>;
using Tests_MGEdgeBetweennessCentrality_Rmat =
  Tests_MGEdgeBetweennessCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_MGEdgeBetweennessCentrality_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEdgeBetweennessCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEdgeBetweennessCentrality_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEdgeBetweennessCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_MGEdgeBetweennessCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EdgeBetweennessCentrality_Usecase{20, false, false, true},
                      EdgeBetweennessCentrality_Usecase{20, false, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGEdgeBetweennessCentrality_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(EdgeBetweennessCentrality_Usecase{50, false, false, true},
                      EdgeBetweennessCentrality_Usecase{50, false, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGEdgeBetweennessCentrality_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(EdgeBetweennessCentrality_Usecase{500, false, false, false},
                      EdgeBetweennessCentrality_Usecase{500, false, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

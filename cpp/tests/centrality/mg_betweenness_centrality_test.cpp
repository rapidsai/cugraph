/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include "centrality/betweenness_centrality_validate.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

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

struct BetweennessCentrality_Usecase {
  size_t num_seeds{std::numeric_limits<size_t>::max()};
  bool normalized{false};
  bool include_endpoints{false};
  bool test_weighted{false};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGBetweennessCentrality
  : public ::testing::TestWithParam<std::tuple<BetweennessCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_MGBetweennessCentrality() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<BetweennessCentrality_Usecase, input_usecase_t> const& param)
  {
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

    std::optional<cugraph::edge_property_t<decltype(mg_graph_view), bool>> edge_mask{std::nullopt};
    if (betweenness_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

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
      hr_timer.start("MG betweenness centrality");
    }

    auto d_mg_centralities = cugraph::betweenness_centrality(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::make_optional<raft::device_span<vertex_t const>>(
        raft::device_span<vertex_t const>{d_mg_seeds.data(), d_mg_seeds.size()}),
      betweenness_usecase.normalized,
      betweenness_usecase.include_endpoints);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (betweenness_usecase.check_correctness) {
      if (mg_renumber_map) {
        cugraph::unrenumber_local_int_vertices(*handle_,
                                               d_mg_seeds.data(),
                                               d_mg_seeds.size(),
                                               (*mg_renumber_map).data(),
                                               mg_graph_view.local_vertex_partition_range_first(),
                                               mg_graph_view.local_vertex_partition_range_last());
      }

      auto d_mg_aggregate_seeds = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{d_mg_seeds.data(), d_mg_seeds.size()});

      rmm::device_uvector<weight_t> d_mg_aggregate_centralities(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_centralities) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<weight_t const>(d_mg_centralities.data(), d_mg_centralities.size()));

      cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(*handle_,
                                            mg_graph_view,
                                            mg_edge_weight_view,
                                            std::make_optional<raft::device_span<vertex_t const>>(
                                              (*mg_renumber_map).data(), (*mg_renumber_map).size()),
                                            false);

      if (handle_->get_comms().get_rank() == 0) {
        auto sg_graph_view = sg_graph.view();

        auto d_sg_centralities = cugraph::betweenness_centrality(
          *handle_,
          sg_graph_view,
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt,
          std::make_optional<raft::device_span<vertex_t const>>(raft::device_span<vertex_t const>{
            d_mg_aggregate_seeds.data(), d_mg_aggregate_seeds.size()}),
          betweenness_usecase.normalized,
          betweenness_usecase.include_endpoints);

        cugraph::test::betweenness_centrality_validate(
          *handle_, d_mg_aggregate_centralities, d_sg_centralities);
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGBetweennessCentrality<input_usecase_t>::handle_ = nullptr;

using Tests_MGBetweennessCentrality_File =
  Tests_MGBetweennessCentrality<cugraph::test::File_Usecase>;
using Tests_MGBetweennessCentrality_Rmat =
  Tests_MGBetweennessCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_MGBetweennessCentrality_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGBetweennessCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGBetweennessCentrality_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGBetweennessCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_MGBetweennessCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(BetweennessCentrality_Usecase{20, false, false, false, false},
                      BetweennessCentrality_Usecase{20, false, false, false, true},
                      BetweennessCentrality_Usecase{20, false, false, true, false},
                      BetweennessCentrality_Usecase{20, false, false, true, true},
                      BetweennessCentrality_Usecase{20, false, true, false, false},
                      BetweennessCentrality_Usecase{20, false, true, false, true},
                      BetweennessCentrality_Usecase{20, false, true, true, false},
                      BetweennessCentrality_Usecase{20, false, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGBetweennessCentrality_Rmat,
  // disable correctness checks, running out of memory
  ::testing::Combine(
    ::testing::Values(BetweennessCentrality_Usecase{50, false, false, false, false},
                      BetweennessCentrality_Usecase{50, false, false, false, true},
                      BetweennessCentrality_Usecase{50, false, false, true, false},
                      BetweennessCentrality_Usecase{50, false, false, true, true},
                      BetweennessCentrality_Usecase{50, false, true, false, false},
                      BetweennessCentrality_Usecase{50, false, true, false, true},
                      BetweennessCentrality_Usecase{50, false, true, true, false},
                      BetweennessCentrality_Usecase{50, false, true, true, true},
                      BetweennessCentrality_Usecase{50, true, false, false, false},
                      BetweennessCentrality_Usecase{50, true, false, false, true},
                      BetweennessCentrality_Usecase{50, true, false, true, false},
                      BetweennessCentrality_Usecase{50, true, false, true, true},
                      BetweennessCentrality_Usecase{50, true, true, false, false},
                      BetweennessCentrality_Usecase{50, true, true, false, true},
                      BetweennessCentrality_Usecase{50, true, true, true, false},
                      BetweennessCentrality_Usecase{50, true, true, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGBetweennessCentrality_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(BetweennessCentrality_Usecase{500, false, false, false, false, false},
                      BetweennessCentrality_Usecase{500, false, false, false, true, false},
                      BetweennessCentrality_Usecase{500, false, false, true, false, false},
                      BetweennessCentrality_Usecase{500, false, false, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

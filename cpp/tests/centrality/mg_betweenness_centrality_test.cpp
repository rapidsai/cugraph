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

struct BetweennessCentrality_Usecase {
  size_t num_seeds{std::numeric_limits<size_t>::max()};
  bool normalized{false};
  bool include_endpoints{false};
  bool test_weighted{false};
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
    constexpr bool renumber           = true;
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
        *handle_, input_usecase, betweenness_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    rmm::device_uvector<vertex_t> d_seeds(0, handle_->get_stream());

    if (handle_->get_comms().get_rank() == 0) {
      rmm::device_uvector<vertex_t> d_seeds(mg_graph_view.number_of_vertices(),
                                            handle_->get_stream());
      cugraph::detail::sequence_fill(
        handle_->get_stream(), d_seeds.data(), d_seeds.size(), vertex_t{0});

      d_seeds = cugraph::test::randomly_select(*handle_, d_seeds, betweenness_usecase.num_seeds);
    }

    d_seeds = cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
      *handle_, std::move(d_seeds));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG betweenness centrality");
    }

#if 0
    auto d_centralities = cugraph::betweenness_centrality(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::make_optional<std::variant<vertex_t, raft::device_span<vertex_t const>>>(
        raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()}),
      betweenness_usecase.normalized,
      betweenness_usecase.include_endpoints,
      do_expensive_check);
#else
    EXPECT_THROW(cugraph::betweenness_centrality(
                   *handle_,
                   mg_graph_view,
                   mg_edge_weight_view,
                   std::make_optional<std::variant<vertex_t, raft::device_span<vertex_t const>>>(
                     raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()}),
                   betweenness_usecase.normalized,
                   betweenness_usecase.include_endpoints,
                   do_expensive_check),
                 cugraph::logic_error);
#endif

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (betweenness_usecase.check_correctness) {
#if 0
      d_centralities = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{d_centralities.data(), d_centralities.size()});
      d_seeds = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()});

      auto [sg_graph, sg_edge_weights, sg_renumber_map] =
        cugraph::test::mg_graph_to_sg_graph(*handle_, mg_graph_view, mg_renumber_map, true);

      auto d_reference_centralities = cugraph::betweenness_centrality(
        *handle_,
        sg_graph.view(),
        sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt,
        std::optional<vertex_t>{std::nullopt},
        std::make_optional<raft::device_span<vertex_t const>>(d_seeds.data(), d_seeds.size()),
        betweenness_usecase.normalized,
        betweenness_usecase.include_endpoints,
        do_expensive_check);

      if (d_seeds.size() > 0) {
        cugraph::test::betweenness_centrality_validate(handle,
                                                       mg_renumber_map,
                                                       d_centralities,
                                                       sg_renumber_map,
                                                       d_reference_centralities);
      }
#endif
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
    ::testing::Values(BetweennessCentrality_Usecase{20, false, false, false, true},
                      BetweennessCentrality_Usecase{20, false, false, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGBetweennessCentrality_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(BetweennessCentrality_Usecase{50, false, false, false, true},
                      BetweennessCentrality_Usecase{50, false, false, true, true}),
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
    ::testing::Values(BetweennessCentrality_Usecase{500, false, false, false, false},
                      BetweennessCentrality_Usecase{500, false, false, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

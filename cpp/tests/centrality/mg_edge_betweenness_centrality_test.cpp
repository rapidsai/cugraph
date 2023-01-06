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
    constexpr bool renumber           = true;
    constexpr bool do_expensive_check = false;

    auto [betweenness_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_renumber_map_labels] =
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
      hr_timer.start("MG edge betweenness centrality");
    }

#if 0
    auto d_centralities = cugraph::edge_betweenness_centrality(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::make_optional<std::variant<vertex_t, raft::device_span<vertex_t const>>>(
        raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()}),
      betweenness_usecase.normalized,
      do_expensive_check);
#else
    EXPECT_THROW(cugraph::edge_betweenness_centrality(
                   *handle_,
                   mg_graph_view,
                   mg_edge_weight_view,
                   std::make_optional<std::variant<vertex_t, raft::device_span<vertex_t const>>>(
                     raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()}),
                   betweenness_usecase.normalized,
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
        *handle_, raft::device_span<weight_t const>(d_centralities.data(), d_centralities.size()));
      d_seeds = cugraph::test::device_gatherv(
        *handle_, raft::device_span<vertex_t const>(d_seeds.data(), d_seeds.size()));

      auto [h_src, h_dst, h_wgt] = cugraph::test::graph_to_host_coo(*handle_, graph_view);

      if (h_src.size() > 0) {
        auto h_centralities = cugraph::test::to_host(*handle_, d_centralities);
        auto h_seeds        = cugraph::test::to_host(*handle_, d_seeds);

        cugraph::test::edge_betweenness_centrality_validate(
          h_src, h_dst, h_wgt, h_centralities, h_seeds);
      }
#endif
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

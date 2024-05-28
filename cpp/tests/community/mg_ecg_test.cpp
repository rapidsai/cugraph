/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

#include <chrono>

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_SUITE_P()
//
struct Ecg_Usecase {
  double min_weight_{0.1};
  size_t ensemble_size_{10};
  size_t max_level_{100};
  double threshold_{1e-7};
  double resolution_{1.0};
  bool check_correctness_{true};
};

////////////////////////////////////////////////////////////////////////////////
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
template <typename input_usecase_t>
class Tests_MGEcg : public ::testing::TestWithParam<std::tuple<Ecg_Usecase, input_usecase_t>> {
 public:
  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  // Run once for each test instance
  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Ecg_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [ecg_usecase, input_usecase] = param;

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, true, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG ECG");
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    raft::random::RngState rng_state(seed);

    cugraph::ecg<vertex_t, edge_t, weight_t, true>(*handle_,
                                                   rng_state,
                                                   mg_graph_view,
                                                   mg_edge_weight_view,
                                                   ecg_usecase.min_weight_,
                                                   ecg_usecase.ensemble_size_,
                                                   ecg_usecase.max_level_,
                                                   ecg_usecase.threshold_,
                                                   ecg_usecase.resolution_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
    // Louvain and detail::permute_range are both tested, here we only make
    // sure that SG and MG ECG calls work expected.

    cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
    std::optional<
      cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
      sg_edge_weights{std::nullopt};
    std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore) =
      cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        false);  // crate a SG graph with MG graph vertex IDs

    auto const comm_rank = handle_->get_comms().get_rank();
    if (comm_rank == 0) {
      auto sg_graph_view = sg_graph.view();
      auto sg_edge_weight_view =
        sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

      cugraph::ecg<vertex_t, edge_t, weight_t, false>(*handle_,
                                                      rng_state,
                                                      sg_graph_view,
                                                      sg_edge_weight_view,
                                                      ecg_usecase.min_weight_,
                                                      ecg_usecase.ensemble_size_,
                                                      ecg_usecase.max_level_,
                                                      ecg_usecase.threshold_,
                                                      ecg_usecase.resolution_);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGEcg<input_usecase_t>::handle_ = nullptr;

using Tests_MGEcg_File = Tests_MGEcg<cugraph::test::File_Usecase>;
using Tests_MGEcg_Rmat = Tests_MGEcg<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGEcg_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEcg_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEcg_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEcg_Rmat, CheckInt32Int64Float)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGEcg_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGEcg_File,
  ::testing::Combine(
    // enable correctness checks for small graphs
    ::testing::Values(Ecg_Usecase{0.1, 10, 100, 1e-7, 1.0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_MGEcg_Rmat,
  ::testing::Combine(
    ::testing::Values(Ecg_Usecase{0.1, 10, 100, 1e-7, 1.0, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MGEcg_File,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Ecg_Usecase{0.1, 10, 100, 1e-7, 1.0, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGEcg_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Ecg_Usecase{0.1, 10, 100, 1e-7, 1.0, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(12, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

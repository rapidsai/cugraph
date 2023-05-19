/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/mg_utilities.hpp>
#include <utilities/test_utilities.hpp>

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

#include <chrono>
#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////
// Test param object. This defines the input and expected output for a test, and
// will be instantiated as the parameter to the tests defined below using
// INSTANTIATE_TEST_SUITE_P()
//
struct Leiden_Usecase {
  size_t max_level_{100};
  double resolution_{0.5};
  double theta_{0.7};
  bool check_correctness_{false};
};

////////////////////////////////////////////////////////////////////////////////
// Parameterized test fixture, to be used with TEST_P().  This defines common
// setup and teardown steps as well as common utilities used by each E2E MG
// test.  In this case, each test is identical except for the inputs and
// expected outputs, so the entire test is defined in the run_test() method.
//
template <typename input_usecase_t>
class Tests_MGLeiden
  : public ::testing::TestWithParam<std::tuple<Leiden_Usecase, input_usecase_t>> {
 public:
  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  // Run once for each test instance
  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of MNMG Leiden with the results of running
  // each step of SG Leiden, renumbering the coarsened graphs based
  // on the MNMG renumbering.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void compare_sg_results(
    raft::handle_t const& handle,
    raft::random::RngState& rng_state,
    cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> mg_edge_weight_view,
    cugraph::Dendrogram<vertex_t> const& mg_dendrogram,
    weight_t resolution,
    weight_t theta,
    weight_t mg_modularity)
  {
    auto& comm           = handle.get_comms();
    auto const comm_rank = comm.get_rank();

    cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(handle);
    std::optional<
      cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>>
      sg_edge_weights{std::nullopt};
    std::tie(sg_graph, sg_edge_weights, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      false);  // crate an SG graph with MG graph vertex IDs

    // FIXME: We need to figure out how to test each iteration of
    // SG vs MG Leiden, possibly by passing results of refinement phase

    weight_t sg_modularity{-1.0};

    auto sg_graph_view = sg_graph.view();
    auto sg_edge_weight_view =
      sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

    if (comm_rank == 0) {
      std::tie(std::ignore, sg_modularity) = cugraph::leiden(
        handle, rng_state, sg_graph_view, sg_edge_weight_view, 100, resolution, theta);
    }
    if (comm_rank == 0) {
      EXPECT_NEAR(mg_modularity, sg_modularity, std::max(mg_modularity, sg_modularity) * 1e-3);
    }
  }

  // Compare the results of running Leiden on multiple GPUs to that of a
  // single-GPU run for the configuration in param.  Note that MNMG Leiden
  // and single GPU Leiden are ONLY deterministic through a single
  // iteration of the outer loop.  Renumbering of the partitions when coarsening
  // the graph is a function of the number of GPUs in the GPU cluster.
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Leiden_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [leiden_usecase, input_usecase] = param;

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
      hr_timer.start("MG Leiden");
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    raft::random::RngState rng_state(seed);

    auto [dendrogram, mg_modularity] =
      cugraph::leiden<vertex_t, edge_t, weight_t, true>(*handle_,
                                                        rng_state,
                                                        mg_graph_view,
                                                        mg_edge_weight_view,
                                                        leiden_usecase.max_level_,
                                                        leiden_usecase.resolution_,
                                                        leiden_usecase.theta_);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (leiden_usecase.check_correctness_) {
      SCOPED_TRACE("compare modularity input");

      compare_sg_results<vertex_t, edge_t, weight_t>(*handle_,
                                                     rng_state,
                                                     mg_graph_view,
                                                     mg_edge_weight_view,
                                                     *dendrogram,
                                                     leiden_usecase.resolution_,
                                                     leiden_usecase.theta_,
                                                     mg_modularity);
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGLeiden<input_usecase_t>::handle_ = nullptr;

using Tests_MGLeiden_File = Tests_MGLeiden<cugraph::test::File_Usecase>;
using Tests_MGLeiden_Rmat = Tests_MGLeiden<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGLeiden_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLeiden_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLeiden_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLeiden_Rmat, CheckInt32Int64Float)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGLeiden_Rmat, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGLeiden_File,
  ::testing::Combine(
    // enable correctness checks for small graphs
    ::testing::Values(Leiden_Usecase{100, 1, 1, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_tests,
                         Tests_MGLeiden_Rmat,
                         ::testing::Combine(::testing::Values(Leiden_Usecase{100, 1, false}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_MGLeiden_File,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Leiden_Usecase{100, 1, 1, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGLeiden_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Leiden_Usecase{100, 1, 1, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(12, 32, 0.57, 0.19, 0.19, 0, true, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

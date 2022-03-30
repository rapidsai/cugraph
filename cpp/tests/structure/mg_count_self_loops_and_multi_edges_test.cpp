/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct CountSelfLoopsAndMultiEdges_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGCountSelfLoopsAndMultiEdges
  : public ::testing::TestWithParam<
      std::tuple<CountSelfLoopsAndMultiEdges_Usecase, input_usecase_t>> {
 public:
  Tests_MGCountSelfLoopsAndMultiEdges() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running count_self_loops & count_multi_edges on multiple GPUs to that of
  // a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(
    CountSelfLoopsAndMultiEdges_Usecase const& count_self_loops_and_multi_edges_usecase,
    input_usecase_t const& input_usecase)
  {
    // 1. initialize handle

    raft::handle_t handle{};
    HighResClock hr_clock{};

    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();
    auto const comm_rank = comm.get_rank();

    auto row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
    while (comm_size % row_comm_size != 0) {
      --row_comm_size;
    }
    cugraph::partition_2d::subcomm_factory_t<cugraph::partition_2d::key_naming_t, vertex_t>
      subcomm_factory(handle, row_comm_size);

    // 2. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        handle, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. run MG count_self_loops & count_multi_edges

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto num_self_loops = mg_graph_view.count_self_loops(handle);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG Counting self-loops took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto num_multi_edges = mg_graph_view.count_multi_edges(handle);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG Counting multi-edges took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 4. copmare SG & MG results

    if (count_self_loops_and_multi_edges_usecase.check_correctness) {
      // 4-1. create SG graph

      cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> sg_graph(handle);
      std::tie(sg_graph, std::ignore) =
        cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
          handle, input_usecase, false, false);

      auto sg_graph_view = sg_graph.view();

      ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

      // 4-2. run SG count_self_loops & count_multi_edges

      auto sg_num_self_loops  = sg_graph_view.count_self_loops(handle);
      auto sg_num_multi_edges = sg_graph_view.count_multi_edges(handle);

      // 4-3. compare

      ASSERT_EQ(num_self_loops, sg_num_self_loops);
      ASSERT_EQ(num_multi_edges, sg_num_multi_edges);
    }
  }
};

using Tests_MGCountSelfLoopsAndMultiEdges_File =
  Tests_MGCountSelfLoopsAndMultiEdges<cugraph::test::File_Usecase>;
using Tests_MGCountSelfLoopsAndMultiEdges_Rmat =
  Tests_MGCountSelfLoopsAndMultiEdges<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGCountSelfLoopsAndMultiEdges_File, CheckInt32Int32FloatTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGCountSelfLoopsAndMultiEdges_Rmat, CheckInt32Int32FloaTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountSelfLoopsAndMultiEdges_Rmat, CheckInt32Int64FloaTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountSelfLoopsAndMultiEdges_Rmat, CheckInt64Int64FloaTransposeFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGCountSelfLoopsAndMultiEdges_File, CheckInt32Int32FloatTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGCountSelfLoopsAndMultiEdges_Rmat, CheckInt32Int32FloaTransposeTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGCountSelfLoopsAndMultiEdges_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{},
                      CountSelfLoopsAndMultiEdges_Usecase{}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_MGCountSelfLoopsAndMultiEdges_Rmat,
  ::testing::Combine(::testing::Values(CountSelfLoopsAndMultiEdges_Usecase{},
                                       CountSelfLoopsAndMultiEdges_Usecase{},
                                       CountSelfLoopsAndMultiEdges_Usecase{},
                                       CountSelfLoopsAndMultiEdges_Usecase{}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGCountSelfLoopsAndMultiEdges_Rmat,
  ::testing::Combine(::testing::Values(CountSelfLoopsAndMultiEdges_Usecase{},
                                       CountSelfLoopsAndMultiEdges_Usecase{},
                                       CountSelfLoopsAndMultiEdges_Usecase{},
                                       CountSelfLoopsAndMultiEdges_Usecase{}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

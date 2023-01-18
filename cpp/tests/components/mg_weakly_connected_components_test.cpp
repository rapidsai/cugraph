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
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

struct WeaklyConnectedComponents_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGWeaklyConnectedComponents
  : public ::testing::TestWithParam<
      std::tuple<WeaklyConnectedComponents_Usecase, input_usecase_t>> {
 public:
  Tests_MGWeaklyConnectedComponents() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running weakly connected components on multiple GPUs to that of a
  // single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(
    WeaklyConnectedComponents_Usecase const& weakly_connected_components_usecase,
    input_usecase_t const& input_usecase)
  {
    using weight_t = float;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, true> mg_graph(*handle_);
    std::optional<rmm::device_uvector<vertex_t>> d_mg_renumber_map_labels{std::nullopt};
    std::tie(mg_graph, std::ignore, d_mg_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. run MG weakly connected components

    rmm::device_uvector<vertex_t> d_mg_components(mg_graph_view.local_vertex_partition_range_size(),
                                                  handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG weakly_connected_components");
    }

    cugraph::weakly_connected_components(*handle_, mg_graph_view, d_mg_components.data());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (weakly_connected_components_usecase.check_correctness) {
      // 3-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_components =
        cugraph::test::device_gatherv(*handle_, d_mg_components.data(), d_mg_components.size());

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. unrenumbr MG results

        std::tie(std::ignore, d_mg_aggregate_components) = cugraph::test::sort_by_key(
          *handle_, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_components);

        // 3-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
        std::tie(sg_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            *handle_, input_usecase, false, false);

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        // 3-4. run SG weakly connected components

        rmm::device_uvector<vertex_t> d_sg_components(sg_graph_view.number_of_vertices(),
                                                      handle_->get_stream());

        cugraph::weakly_connected_components(*handle_, sg_graph_view, d_sg_components.data());

        // 3-5. compare

        auto h_mg_aggregate_components =
          cugraph::test::to_host(*handle_, d_mg_aggregate_components);
        auto h_sg_components = cugraph::test::to_host(*handle_, d_sg_components);

        std::unordered_map<vertex_t, vertex_t> mg_to_sg_map{};
        for (size_t i = 0; i < h_sg_components.size(); ++i) {
          mg_to_sg_map.insert({h_mg_aggregate_components[i], h_sg_components[i]});
        }
        std::transform(h_mg_aggregate_components.begin(),
                       h_mg_aggregate_components.end(),
                       h_mg_aggregate_components.begin(),
                       [&mg_to_sg_map](auto mg_c) { return mg_to_sg_map[mg_c]; });

        ASSERT_TRUE(std::equal(
          h_sg_components.begin(), h_sg_components.end(), h_mg_aggregate_components.begin()))
          << "components do not match with the SG values.";
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGWeaklyConnectedComponents<input_usecase_t>::handle_ =
  nullptr;

using Tests_MGWeaklyConnectedComponents_File =
  Tests_MGWeaklyConnectedComponents<cugraph::test::File_Usecase>;
using Tests_MGWeaklyConnectedComponents_Rmat =
  Tests_MGWeaklyConnectedComponents<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGWeaklyConnectedComponents_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGWeaklyConnectedComponents_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGWeaklyConnectedComponents_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGWeaklyConnectedComponents_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGWeaklyConnectedComponents_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(WeaklyConnectedComponents_Usecase{0}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/polbooks.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGWeaklyConnectedComponents_Rmat,
                         ::testing::Values(
                           // enable correctness checks
                           std::make_tuple(WeaklyConnectedComponents_Usecase{},
                                           cugraph::test::Rmat_Usecase(
                                             10, 16, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGWeaklyConnectedComponents_Rmat,
  ::testing::Values(
    // disable correctness checks
    std::make_tuple(
      WeaklyConnectedComponents_Usecase{false},
      cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

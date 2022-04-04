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

struct WeaklyConnectedComponents_Usecase {
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGWeaklyConnectedComponents
  : public ::testing::TestWithParam<
      std::tuple<WeaklyConnectedComponents_Usecase, input_usecase_t>> {
 public:
  Tests_MGWeaklyConnectedComponents() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

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
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        handle, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. run MG weakly connected components

    rmm::device_uvector<vertex_t> d_mg_components(mg_graph_view.local_vertex_partition_range_size(),
                                                  handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    cugraph::weakly_connected_components(handle, mg_graph_view, d_mg_components.data());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG weakly_connected_components took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 4. compare SG & MG results

    if (weakly_connected_components_usecase.check_correctness) {
      // 4-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        handle, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_components =
        cugraph::test::device_gatherv(handle, d_mg_components.data(), d_mg_components.size());

      if (handle.get_comms().get_rank() == int{0}) {
        // 4-2. unrenumbr MG results

        std::tie(std::ignore, d_mg_aggregate_components) = cugraph::test::sort_by_key(
          handle, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_components);

        // 4-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> sg_graph(handle);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false);

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        // 4-4. run SG weakly connected components

        rmm::device_uvector<vertex_t> d_sg_components(sg_graph_view.number_of_vertices(),
                                                      handle.get_stream());

        cugraph::weakly_connected_components(handle, sg_graph_view, d_sg_components.data());

        // 4-5. compare

        std::vector<vertex_t> h_mg_aggregate_components(mg_graph_view.number_of_vertices());
        raft::update_host(h_mg_aggregate_components.data(),
                          d_mg_aggregate_components.data(),
                          d_mg_aggregate_components.size(),
                          handle.get_stream());

        std::vector<vertex_t> h_sg_components(sg_graph_view.number_of_vertices());
        raft::update_host(h_sg_components.data(),
                          d_sg_components.data(),
                          d_sg_components.size(),
                          handle.get_stream());

        handle.sync_stream();

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
};

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

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
#include <cugraph/partition_manager.hpp>

#include <raft/comms/comms.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

struct Hits_Usecase {
  bool check_correctness{true};
  bool check_initial_input{false};
};

template <typename input_usecase_t>
class Tests_MGHits : public ::testing::TestWithParam<std::tuple<Hits_Usecase, input_usecase_t>> {
 public:
  Tests_MGHits() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running HITS on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(Hits_Usecase const& hits_usecase, input_usecase_t const& input_usecase)
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
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, true>(
        handle, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 3. run hits

    auto mg_graph_view      = mg_graph.view();
    auto maximum_iterations = 200;
    weight_t tolerance      = 1e-8;
    rmm::device_uvector<weight_t> d_mg_hubs(mg_graph_view.local_vertex_partition_range_size(),
                                            handle.get_stream());

    rmm::device_uvector<weight_t> d_mg_authorities(
      mg_graph_view.local_vertex_partition_range_size(), handle.get_stream());

    std::vector<weight_t> initial_random_hubs =
      (hits_usecase.check_initial_input)
        ? cugraph::test::random_vector<weight_t>(d_mg_hubs.size(), comm_rank)
        : std::vector<weight_t>(0);

    if (hits_usecase.check_initial_input) {
      raft::update_device(d_mg_hubs.data(),
                          initial_random_hubs.data(),
                          initial_random_hubs.size(),
                          handle.get_stream());
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    auto result = cugraph::hits(handle,
                                mg_graph_view,
                                d_mg_hubs.data(),
                                d_mg_authorities.data(),
                                tolerance,
                                maximum_iterations,
                                hits_usecase.check_initial_input,
                                true,
                                hits_usecase.check_initial_input);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG Hits took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 4. compare SG & MG results

    if (hits_usecase.check_correctness) {
      // 4-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        handle, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_hubs =
        cugraph::test::device_gatherv(handle, d_mg_hubs.data(), d_mg_hubs.size());
      auto d_mg_aggregate_authorities =
        cugraph::test::device_gatherv(handle, d_mg_authorities.data(), d_mg_authorities.size());
      rmm::device_uvector<weight_t> d_initial_hubs(0, handle.get_stream());

      if (hits_usecase.check_initial_input) {
        d_initial_hubs.resize(initial_random_hubs.size(), handle.get_stream());
        raft::update_device(d_initial_hubs.data(),
                            initial_random_hubs.data(),
                            initial_random_hubs.size(),
                            handle.get_stream());
        d_initial_hubs =
          cugraph::test::device_gatherv(handle, d_initial_hubs.data(), d_initial_hubs.size());
        std::tie(std::ignore, d_initial_hubs) =
          cugraph::test::sort_by_key(handle, d_mg_aggregate_renumber_map_labels, d_initial_hubs);
      }

      if (handle.get_comms().get_rank() == int{0}) {
        // 4-2. unrenumber MG results

        std::tie(std::ignore, d_mg_aggregate_hubs) = cugraph::test::sort_by_key(
          handle, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_hubs);
        std::tie(std::ignore, d_mg_aggregate_authorities) = cugraph::test::sort_by_key(
          handle, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_authorities);

        // 4-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, true, false> sg_graph(handle);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
            handle, input_usecase, false, false);

        auto sg_graph_view = sg_graph.view();

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        // 4-4. run SG Hits

        rmm::device_uvector<weight_t> d_sg_hubs(sg_graph_view.number_of_vertices(),
                                                handle.get_stream());
        rmm::device_uvector<weight_t> d_sg_authorities(sg_graph_view.number_of_vertices(),
                                                       handle.get_stream());
        if (hits_usecase.check_initial_input) {
          raft::update_device(
            d_sg_hubs.begin(), d_initial_hubs.begin(), d_initial_hubs.size(), handle.get_stream());
        }

        auto result = cugraph::hits(handle,
                                    sg_graph_view,
                                    d_sg_hubs.data(),
                                    d_sg_authorities.data(),
                                    tolerance,
                                    maximum_iterations,
                                    hits_usecase.check_initial_input,
                                    true,
                                    hits_usecase.check_initial_input);

        // 4-5. compare

        std::vector<result_t> h_mg_aggregate_hubs(mg_graph_view.number_of_vertices());
        raft::update_host(h_mg_aggregate_hubs.data(),
                          d_mg_aggregate_hubs.data(),
                          d_mg_aggregate_hubs.size(),
                          handle.get_stream());

        std::vector<result_t> h_sg_hubs(sg_graph_view.number_of_vertices());
        raft::update_host(
          h_sg_hubs.data(), d_sg_hubs.data(), d_sg_hubs.size(), handle.get_stream());

        handle.sync_stream();

        auto threshold_ratio = 1e-3;
        auto threshold_magnitude =
          (1.0 / static_cast<result_t>(mg_graph_view.number_of_vertices())) *
          threshold_ratio;  // skip comparison for low Hits verties (lowly ranked
                            // vertices)
        auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
          return std::abs(lhs - rhs) <
                 std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
        };

        ASSERT_TRUE(std::equal(
          h_mg_aggregate_hubs.begin(), h_mg_aggregate_hubs.end(), h_sg_hubs.begin(), nearly_equal));
      }
    }
  }
};

using Tests_MGHits_File = Tests_MGHits<cugraph::test::File_Usecase>;
using Tests_MGHits_Rmat = Tests_MGHits<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGHits_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGHits_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGHits_Rmat, CheckInt32Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGHits_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGHits_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Hits_Usecase{true, false}, Hits_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGHits_Rmat,
                         ::testing::Combine(
                           // enable correctness checks
                           ::testing::Values(Hits_Usecase{true, false}, Hits_Usecase{true, true}),
                           ::testing::Values(cugraph::test::Rmat_Usecase(
                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGHits_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Hits_Usecase{false, false}, Hits_Usecase{true, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

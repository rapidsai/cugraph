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

struct PageRank_Usecase {
  double personalization_ratio{0.0};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGPageRank
  : public ::testing::TestWithParam<std::tuple<PageRank_Usecase, input_usecase_t>> {
 public:
  Tests_MGPageRank() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running PageRank on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(PageRank_Usecase const& pagerank_usecase,
                        input_usecase_t const& input_usecase)
  {
    // 1. initialize handle

    auto constexpr pool_size = 64;  // FIXME: tuning parameter
    raft::handle_t handle(rmm::cuda_stream_per_thread,
                          std::make_shared<rmm::cuda_stream_pool>(pool_size));
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
        handle, input_usecase, pagerank_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. generate personalization vertex/value pairs

    std::optional<std::vector<vertex_t>> h_mg_personalization_vertices{std::nullopt};
    std::optional<std::vector<result_t>> h_mg_personalization_values{std::nullopt};
    if (pagerank_usecase.personalization_ratio > 0.0) {
      std::default_random_engine generator{
        static_cast<long unsigned int>(comm.get_rank()) /* seed */};
      std::uniform_real_distribution<double> distribution{0.0, 1.0};
      h_mg_personalization_vertices =
        std::vector<vertex_t>(mg_graph_view.local_vertex_partition_range_size());
      std::iota((*h_mg_personalization_vertices).begin(),
                (*h_mg_personalization_vertices).end(),
                mg_graph_view.local_vertex_partition_range_first());
      (*h_mg_personalization_vertices)
        .erase(std::remove_if((*h_mg_personalization_vertices).begin(),
                              (*h_mg_personalization_vertices).end(),
                              [&generator, &distribution, pagerank_usecase](auto v) {
                                return distribution(generator) >=
                                       pagerank_usecase.personalization_ratio;
                              }),
               (*h_mg_personalization_vertices).end());
      h_mg_personalization_values = std::vector<result_t>((*h_mg_personalization_vertices).size());
      std::for_each((*h_mg_personalization_values).begin(),
                    (*h_mg_personalization_values).end(),
                    [&distribution, &generator](auto& val) { val = distribution(generator); });
    }

    auto d_mg_personalization_vertices =
      h_mg_personalization_vertices
        ? std::make_optional<rmm::device_uvector<vertex_t>>((*h_mg_personalization_vertices).size(),
                                                            handle.get_stream())
        : std::nullopt;
    auto d_mg_personalization_values =
      h_mg_personalization_values ? std::make_optional<rmm::device_uvector<result_t>>(
                                      (*d_mg_personalization_vertices).size(), handle.get_stream())
                                  : std::nullopt;
    if (d_mg_personalization_vertices) {
      raft::update_device((*d_mg_personalization_vertices).data(),
                          (*h_mg_personalization_vertices).data(),
                          (*h_mg_personalization_vertices).size(),
                          handle.get_stream());
      raft::update_device((*d_mg_personalization_values).data(),
                          (*h_mg_personalization_values).data(),
                          (*h_mg_personalization_values).size(),
                          handle.get_stream());
    }

    // 4. run MG PageRank

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    rmm::device_uvector<result_t> d_mg_pageranks(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    cugraph::pagerank<vertex_t, edge_t, weight_t>(
      handle,
      mg_graph_view,
      std::nullopt,
      d_mg_personalization_vertices
        ? std::optional<vertex_t const*>{(*d_mg_personalization_vertices).data()}
        : std::nullopt,
      d_mg_personalization_values
        ? std::optional<result_t const*>{(*d_mg_personalization_values).data()}
        : std::nullopt,
      d_mg_personalization_vertices
        ? std::optional{static_cast<vertex_t>((*d_mg_personalization_vertices).size())}
        : std::nullopt,
      d_mg_pageranks.data(),
      alpha,
      epsilon,
      std::numeric_limits<size_t>::max(),
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG PageRank took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 5. copmare SG & MG results

    if (pagerank_usecase.check_correctness) {
      // 5-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        handle, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_personalization_vertices =
        d_mg_personalization_vertices
          ? std::optional<rmm::device_uvector<vertex_t>>{cugraph::test::device_gatherv(
              handle,
              (*d_mg_personalization_vertices).data(),
              (*d_mg_personalization_vertices).size())}
          : std::nullopt;
      auto d_mg_aggregate_personalization_values =
        d_mg_personalization_values
          ? std::optional<rmm::device_uvector<result_t>>{cugraph::test::device_gatherv(
              handle, (*d_mg_personalization_values).data(), (*d_mg_personalization_values).size())}
          : std::nullopt;
      auto d_mg_aggregate_pageranks =
        cugraph::test::device_gatherv(handle, d_mg_pageranks.data(), d_mg_pageranks.size());

      if (handle.get_comms().get_rank() == int{0}) {
        // 5-2. unrenumbr MG results

        if (d_mg_aggregate_personalization_vertices) {
          cugraph::unrenumber_int_vertices<vertex_t, false>(
            handle,
            (*d_mg_aggregate_personalization_vertices).data(),
            (*d_mg_aggregate_personalization_vertices).size(),
            d_mg_aggregate_renumber_map_labels.data(),
            std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
          std::tie(d_mg_aggregate_personalization_vertices, d_mg_aggregate_personalization_values) =
            cugraph::test::sort_by_key(handle,
                                       *d_mg_aggregate_personalization_vertices,
                                       *d_mg_aggregate_personalization_values);
        }
        std::tie(std::ignore, d_mg_aggregate_pageranks) = cugraph::test::sort_by_key(
          handle, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_pageranks);

        // 5-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, true, false> sg_graph(handle);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
            handle, input_usecase, pagerank_usecase.test_weighted, false);

        auto sg_graph_view = sg_graph.view();

        ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

        // 5-4. run SG PageRank

        rmm::device_uvector<result_t> d_sg_pageranks(sg_graph_view.number_of_vertices(),
                                                     handle.get_stream());

        cugraph::pagerank<vertex_t, edge_t, weight_t>(
          handle,
          sg_graph_view,
          std::nullopt,
          d_mg_aggregate_personalization_vertices
            ? std::optional<vertex_t const*>{(*d_mg_aggregate_personalization_vertices).data()}
            : std::nullopt,
          d_mg_aggregate_personalization_values
            ? std::optional<result_t const*>{(*d_mg_aggregate_personalization_values).data()}
            : std::nullopt,
          d_mg_aggregate_personalization_vertices
            ? std::optional<vertex_t>{static_cast<vertex_t>(
                (*d_mg_aggregate_personalization_vertices).size())}
            : std::nullopt,
          d_sg_pageranks.data(),
          alpha,
          epsilon,
          std::numeric_limits<size_t>::max(),  // max_iterations
          false);

        // 5-4. compare

        std::vector<result_t> h_mg_aggregate_pageranks(mg_graph_view.number_of_vertices());
        raft::update_host(h_mg_aggregate_pageranks.data(),
                          d_mg_aggregate_pageranks.data(),
                          d_mg_aggregate_pageranks.size(),
                          handle.get_stream());

        std::vector<result_t> h_sg_pageranks(sg_graph_view.number_of_vertices());
        raft::update_host(
          h_sg_pageranks.data(), d_sg_pageranks.data(), d_sg_pageranks.size(), handle.get_stream());

        handle.sync_stream();

        auto threshold_ratio = 1e-3;
        auto threshold_magnitude =
          (1.0 / static_cast<result_t>(mg_graph_view.number_of_vertices())) *
          threshold_ratio;  // skip comparison for low PageRank verties (lowly ranked vertices)
        auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
          return std::abs(lhs - rhs) <
                 std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
        };

        ASSERT_TRUE(std::equal(h_mg_aggregate_pageranks.begin(),
                               h_mg_aggregate_pageranks.end(),
                               h_sg_pageranks.begin(),
                               nearly_equal));
      }
    }
  }
};

using Tests_MGPageRank_File = Tests_MGPageRank<cugraph::test::File_Usecase>;
using Tests_MGPageRank_Rmat = Tests_MGPageRank<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGPageRank_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGPageRank_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPageRank_Rmat, CheckInt32Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGPageRank_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGPageRank_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(PageRank_Usecase{0.0, false},
                      PageRank_Usecase{0.5, false},
                      PageRank_Usecase{0.0, true},
                      PageRank_Usecase{0.5, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_tests,
  Tests_MGPageRank_Rmat,
  ::testing::Combine(::testing::Values(PageRank_Usecase{0.0, false},
                                       PageRank_Usecase{0.5, false},
                                       PageRank_Usecase{0.0, true},
                                       PageRank_Usecase{0.5, true}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGPageRank_Rmat,
  ::testing::Combine(::testing::Values(PageRank_Usecase{0.0, false, false},
                                       PageRank_Usecase{0.5, false, false},
                                       PageRank_Usecase{0.0, true, false},
                                       PageRank_Usecase{0.5, true, false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

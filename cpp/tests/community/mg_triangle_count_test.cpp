/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

struct TriangleCount_Usecase {
  double vertex_subset_ratio{0.0};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTriangleCount
  : public ::testing::TestWithParam<std::tuple<TriangleCount_Usecase, input_usecase_t>> {
 public:
  Tests_MGTriangleCount() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running TriangleCount on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(TriangleCount_Usecase const& triangle_count_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t = float;

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
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        handle, input_usecase, false, true, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto mg_graph_view = mg_graph.view();

    // 3. generate a vertex subset to compute triangle counts

    std::optional<std::vector<vertex_t>> h_mg_vertices{std::nullopt};
    if (triangle_count_usecase.vertex_subset_ratio < 1.0) {
      std::default_random_engine generator{
        static_cast<long unsigned int>(comm.get_rank()) /* seed */};
      std::uniform_real_distribution<double> distribution{0.0, 1.0};
      h_mg_vertices = std::vector<vertex_t>(mg_graph_view.local_vertex_partition_range_size());
      std::iota((*h_mg_vertices).begin(),
                (*h_mg_vertices).end(),
                mg_graph_view.local_vertex_partition_range_first());
      (*h_mg_vertices)
        .erase(std::remove_if((*h_mg_vertices).begin(),
                              (*h_mg_vertices).end(),
                              [&generator, &distribution, triangle_count_usecase](auto v) {
                                return distribution(generator) >=
                                       triangle_count_usecase.vertex_subset_ratio;
                              }),
               (*h_mg_vertices).end());
    }

    auto d_mg_vertices = h_mg_vertices ? std::make_optional<rmm::device_uvector<vertex_t>>(
                                           (*h_mg_vertices).size(), handle.get_stream())
                                       : std::nullopt;
    if (d_mg_vertices) {
      raft::update_device((*d_mg_vertices).data(),
                          (*h_mg_vertices).data(),
                          (*h_mg_vertices).size(),
                          handle.get_stream());
    }

    // 4. run MG TriangleCount

    rmm::device_uvector<edge_t> d_mg_triangle_counts(
      d_mg_vertices ? (*d_mg_vertices).size() : mg_graph_view.local_vertex_partition_range_size(),
      handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      hr_clock.start();
    }

    cugraph::triangle_count<vertex_t, edge_t, weight_t, true>(
      handle,
      mg_graph_view,
      d_mg_vertices ? std::make_optional<raft::device_span<vertex_t const>>(
                        (*d_mg_vertices).begin(), (*d_mg_vertices).end())
                    : std::nullopt,
      raft::device_span<edge_t>(d_mg_triangle_counts.begin(), d_mg_triangle_counts.end()),
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle.get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG TriangleCount took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 5. copmare SG & MG results

    if (triangle_count_usecase.check_correctness) {
      // 5-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        handle, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_vertices =
        d_mg_vertices ? std::optional<rmm::device_uvector<vertex_t>>{cugraph::test::device_gatherv(
                          handle, (*d_mg_vertices).data(), (*d_mg_vertices).size())}
                      : std::nullopt;
      auto d_mg_aggregate_triangle_counts = cugraph::test::device_gatherv(
        handle, d_mg_triangle_counts.data(), d_mg_triangle_counts.size());

      if (handle.get_comms().get_rank() == int{0}) {
        // 5-2. unrenumbr MG results

        if (d_mg_aggregate_vertices) {
          cugraph::unrenumber_int_vertices<vertex_t, false>(
            handle,
            (*d_mg_aggregate_vertices).data(),
            (*d_mg_aggregate_vertices).size(),
            d_mg_aggregate_renumber_map_labels.data(),
            std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
          std::tie(d_mg_aggregate_vertices, d_mg_aggregate_triangle_counts) =
            cugraph::test::sort_by_key(
              handle, *d_mg_aggregate_vertices, d_mg_aggregate_triangle_counts);
        } else {
          std::tie(std::ignore, d_mg_aggregate_triangle_counts) = cugraph::test::sort_by_key(
            handle, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_triangle_counts);
        }

        // 5-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> sg_graph(handle);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            handle, input_usecase, false, false, false, true);

        auto sg_graph_view = sg_graph.view();

        ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

        // 5-4. run SG TriangleCount

        rmm::device_uvector<edge_t> d_sg_triangle_counts(d_mg_aggregate_vertices
                                                           ? (*d_mg_aggregate_vertices).size()
                                                           : sg_graph_view.number_of_vertices(),
                                                         handle.get_stream());

        cugraph::triangle_count<vertex_t, edge_t, weight_t>(
          handle,
          sg_graph_view,
          d_mg_aggregate_vertices
            ? std::make_optional<raft::device_span<vertex_t const>>(
                (*d_mg_aggregate_vertices).begin(), (*d_mg_aggregate_vertices).end())
            : std::nullopt,
          raft::device_span<edge_t>(d_sg_triangle_counts.begin(), d_sg_triangle_counts.end()),
          false);

        // 5-4. compare

        std::vector<edge_t> h_mg_aggregate_triangle_counts(d_mg_aggregate_triangle_counts.size());
        raft::update_host(h_mg_aggregate_triangle_counts.data(),
                          d_mg_aggregate_triangle_counts.data(),
                          d_mg_aggregate_triangle_counts.size(),
                          handle.get_stream());

        std::vector<edge_t> h_sg_triangle_counts(d_sg_triangle_counts.size());
        raft::update_host(h_sg_triangle_counts.data(),
                          d_sg_triangle_counts.data(),
                          d_sg_triangle_counts.size(),
                          handle.get_stream());

        handle.sync_stream();

        ASSERT_TRUE(std::equal(h_mg_aggregate_triangle_counts.begin(),
                               h_mg_aggregate_triangle_counts.end(),
                               h_sg_triangle_counts.begin()));
      }
    }
  }
};

using Tests_MGTriangleCount_File = Tests_MGTriangleCount<cugraph::test::File_Usecase>;
using Tests_MGTriangleCount_Rmat = Tests_MGTriangleCount<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTriangleCount_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTriangleCount_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTriangleCount_Rmat, CheckInt32Int64)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTriangleCount_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_tests,
  Tests_MGTriangleCount_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(TriangleCount_Usecase{0.1}, TriangleCount_Usecase{1.0}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_tests,
                         Tests_MGTriangleCount_Rmat,
                         ::testing::Combine(::testing::Values(TriangleCount_Usecase{0.1},
                                                              TriangleCount_Usecase{1.0}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTriangleCount_Rmat,
  ::testing::Combine(::testing::Values(TriangleCount_Usecase{0.1, false},
                                       TriangleCount_Usecase{1.0, false}),
                     ::testing::Values(cugraph::test::Rmat_Usecase(
                       20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

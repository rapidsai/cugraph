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
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
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

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running TriangleCount on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t>
  void run_current_test(TriangleCount_Usecase const& triangle_count_usecase,
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
        *handle_, input_usecase, false, true, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();

    // 2. generate a vertex subset to compute triangle counts

    std::optional<std::vector<vertex_t>> h_mg_vertices{std::nullopt};
    if (triangle_count_usecase.vertex_subset_ratio < 1.0) {
      std::default_random_engine generator{
        static_cast<long unsigned int>(handle_->get_comms().get_rank()) /* seed */};
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
                                           (*h_mg_vertices).size(), handle_->get_stream())
                                       : std::nullopt;
    if (d_mg_vertices) {
      raft::update_device((*d_mg_vertices).data(),
                          (*h_mg_vertices).data(),
                          (*h_mg_vertices).size(),
                          handle_->get_stream());
    }

    // 3. run MG TriangleCount

    rmm::device_uvector<edge_t> d_mg_triangle_counts(
      d_mg_vertices ? (*d_mg_vertices).size() : mg_graph_view.local_vertex_partition_range_size(),
      handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG TriangleCount");
    }

    cugraph::triangle_count<vertex_t, edge_t, true>(
      *handle_,
      mg_graph_view,
      d_mg_vertices ? std::make_optional<raft::device_span<vertex_t const>>(
                        (*d_mg_vertices).begin(), (*d_mg_vertices).end())
                    : std::nullopt,
      raft::device_span<edge_t>(d_mg_triangle_counts.begin(), d_mg_triangle_counts.end()),
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 4. copmare SG & MG results

    if (triangle_count_usecase.check_correctness) {
      // 4-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_vertices =
        d_mg_vertices ? std::optional<rmm::device_uvector<vertex_t>>{cugraph::test::device_gatherv(
                          *handle_, (*d_mg_vertices).data(), (*d_mg_vertices).size())}
                      : std::nullopt;
      auto d_mg_aggregate_triangle_counts = cugraph::test::device_gatherv(
        *handle_, d_mg_triangle_counts.data(), d_mg_triangle_counts.size());

      if (handle_->get_comms().get_rank() == int{0}) {
        // 4-2. unrenumbr MG results

        if (d_mg_aggregate_vertices) {
          cugraph::unrenumber_int_vertices<vertex_t, false>(
            *handle_,
            (*d_mg_aggregate_vertices).data(),
            (*d_mg_aggregate_vertices).size(),
            d_mg_aggregate_renumber_map_labels.data(),
            std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
          std::tie(d_mg_aggregate_vertices, d_mg_aggregate_triangle_counts) =
            cugraph::test::sort_by_key(
              *handle_, *d_mg_aggregate_vertices, d_mg_aggregate_triangle_counts);
        } else {
          std::tie(std::ignore, d_mg_aggregate_triangle_counts) = cugraph::test::sort_by_key(
            *handle_, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_triangle_counts);
        }

        // 4-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);
        std::tie(sg_graph, std::ignore, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
            *handle_, input_usecase, false, false, false, true);

        auto sg_graph_view = sg_graph.view();

        ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

        // 4-4. run SG TriangleCount

        rmm::device_uvector<edge_t> d_sg_triangle_counts(d_mg_aggregate_vertices
                                                           ? (*d_mg_aggregate_vertices).size()
                                                           : sg_graph_view.number_of_vertices(),
                                                         handle_->get_stream());

        cugraph::triangle_count<vertex_t, edge_t, false>(
          *handle_,
          sg_graph_view,
          d_mg_aggregate_vertices
            ? std::make_optional<raft::device_span<vertex_t const>>(
                (*d_mg_aggregate_vertices).begin(), (*d_mg_aggregate_vertices).end())
            : std::nullopt,
          raft::device_span<edge_t>(d_sg_triangle_counts.begin(), d_sg_triangle_counts.end()),
          false);

        // 4-5. compare

        auto h_mg_aggregate_triangle_counts =
          cugraph::test::to_host(*handle_, d_mg_aggregate_triangle_counts);
        auto h_sg_triangle_counts = cugraph::test::to_host(*handle_, d_sg_triangle_counts);

        ASSERT_TRUE(std::equal(h_mg_aggregate_triangle_counts.begin(),
                               h_mg_aggregate_triangle_counts.end(),
                               h_sg_triangle_counts.begin()));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGTriangleCount<input_usecase_t>::handle_ = nullptr;

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

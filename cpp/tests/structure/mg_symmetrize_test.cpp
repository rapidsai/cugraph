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
#include <utilities/mg_utilities.hpp>
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

#include <random>

struct Symmetrize_Usecase {
  bool reciprocal{false};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGSymmetrize
  : public ::testing::TestWithParam<std::tuple<Symmetrize_Usecase, input_usecase_t>> {
 public:
  Tests_MGSymmetrize() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(Symmetrize_Usecase const& symmetrize_usecase,
                        input_usecase_t const& input_usecase)
  {
    HighResClock hr_clock{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, symmetrize_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 2. run MG symmetrize

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    *d_mg_renumber_map_labels = mg_graph.symmetrize(
      *handle_, std::move(*d_mg_renumber_map_labels), symmetrize_usecase.reciprocal);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG symmetrize took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 3. copmare SG & MG results

    if (symmetrize_usecase.check_correctness) {
      // 3-1. decompress MG results

      auto [d_mg_srcs, d_mg_dsts, d_mg_weights] =
        mg_graph.decompress_to_edgelist(*handle_, d_mg_renumber_map_labels, false);

      // 3-2. aggregate MG results

      auto d_mg_aggregate_srcs =
        cugraph::test::device_gatherv(*handle_, d_mg_srcs.data(), d_mg_srcs.size());
      auto d_mg_aggregate_dsts =
        cugraph::test::device_gatherv(*handle_, d_mg_dsts.data(), d_mg_dsts.size());
      std::optional<rmm::device_uvector<weight_t>> d_mg_aggregate_weights{std::nullopt};
      if (d_mg_weights) {
        d_mg_aggregate_weights =
          cugraph::test::device_gatherv(*handle_, (*d_mg_weights).data(), (*d_mg_weights).size());
      }

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> sg_graph(*handle_);
        std::tie(sg_graph, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
            *handle_, input_usecase, symmetrize_usecase.test_weighted, false);

        // 3-4. run SG symmetrize

        auto d_sg_renumber_map_labels =
          sg_graph.symmetrize(*handle_, std::nullopt, symmetrize_usecase.reciprocal);
        ASSERT_FALSE(d_sg_renumber_map_labels.has_value());

        // 3-5. decompress SG results

        auto [d_sg_srcs, d_sg_dsts, d_sg_weights] =
          sg_graph.decompress_to_edgelist(*handle_, std::nullopt, false);

        // 3-6. compare

        ASSERT_TRUE(mg_graph.number_of_vertices() == sg_graph.number_of_vertices());
        ASSERT_TRUE(mg_graph.number_of_edges() == sg_graph.number_of_edges());

        auto h_mg_aggregate_srcs    = cugraph::test::to_host(*handle_, d_mg_aggregate_srcs);
        auto h_mg_aggregate_dsts    = cugraph::test::to_host(*handle_, d_mg_aggregate_dsts);
        auto h_mg_aggregate_weights = cugraph::test::to_host(*handle_, d_mg_aggregate_weights);

        auto h_sg_srcs    = cugraph::test::to_host(*handle_, d_sg_srcs);
        auto h_sg_dsts    = cugraph::test::to_host(*handle_, d_sg_dsts);
        auto h_sg_weights = cugraph::test::to_host(*handle_, d_sg_weights);

        if (symmetrize_usecase.test_weighted) {
          std::vector<std::tuple<vertex_t, vertex_t, weight_t>> mg_aggregate_edges(
            h_mg_aggregate_srcs.size());
          for (size_t i = 0; i < mg_aggregate_edges.size(); ++i) {
            mg_aggregate_edges[i] = std::make_tuple(
              h_mg_aggregate_srcs[i], h_mg_aggregate_dsts[i], (*h_mg_aggregate_weights)[i]);
          }
          std::vector<std::tuple<vertex_t, vertex_t, weight_t>> sg_edges(h_sg_srcs.size());
          for (size_t i = 0; i < sg_edges.size(); ++i) {
            sg_edges[i] = std::make_tuple(h_sg_srcs[i], h_sg_dsts[i], (*h_sg_weights)[i]);
          }
          std::sort(mg_aggregate_edges.begin(), mg_aggregate_edges.end());
          std::sort(sg_edges.begin(), sg_edges.end());
          ASSERT_TRUE(
            std::equal(mg_aggregate_edges.begin(), mg_aggregate_edges.end(), sg_edges.begin()));
        } else {
          std::vector<std::tuple<vertex_t, vertex_t>> mg_aggregate_edges(
            h_mg_aggregate_srcs.size());
          for (size_t i = 0; i < mg_aggregate_edges.size(); ++i) {
            mg_aggregate_edges[i] = std::make_tuple(h_mg_aggregate_srcs[i], h_mg_aggregate_dsts[i]);
          }
          std::vector<std::tuple<vertex_t, vertex_t>> sg_edges(h_sg_srcs.size());
          for (size_t i = 0; i < sg_edges.size(); ++i) {
            sg_edges[i] = std::make_tuple(h_sg_srcs[i], h_sg_dsts[i]);
          }
          std::sort(mg_aggregate_edges.begin(), mg_aggregate_edges.end());
          std::sort(sg_edges.begin(), sg_edges.end());
          ASSERT_TRUE(
            std::equal(mg_aggregate_edges.begin(), mg_aggregate_edges.end(), sg_edges.begin()));
        }
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGSymmetrize<input_usecase_t>::handle_ = nullptr;

using Tests_MGSymmetrize_File = Tests_MGSymmetrize<cugraph::test::File_Usecase>;
using Tests_MGSymmetrize_Rmat = Tests_MGSymmetrize<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGSymmetrize_File, CheckInt32Int32FloatTransposedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGSymmetrize_File, CheckInt32Int32FloatTransposedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGSymmetrize_Rmat, CheckInt32Int32FloatTransposedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGSymmetrize_Rmat, CheckInt32Int32FloatTransposedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGSymmetrize_Rmat, CheckInt32Int64FloatTransposedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGSymmetrize_Rmat, CheckInt32Int64FloatTransposedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGSymmetrize_Rmat, CheckInt64Int64FloatTransposedFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGSymmetrize_Rmat, CheckInt64Int64FloatTransposedTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGSymmetrize_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Symmetrize_Usecase{false, false},
                      Symmetrize_Usecase{true, false},
                      Symmetrize_Usecase{false, true},
                      Symmetrize_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGSymmetrize_Rmat,
                         ::testing::Combine(
                           // enable correctness checks
                           ::testing::Values(Symmetrize_Usecase{false, false},
                                             Symmetrize_Usecase{true, false},
                                             Symmetrize_Usecase{false, true},
                                             Symmetrize_Usecase{true, true}),
                           ::testing::Values(cugraph::test::Rmat_Usecase(
                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGSymmetrize_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Symmetrize_Usecase{false, false, false},
                      Symmetrize_Usecase{true, false, false},
                      Symmetrize_Usecase{false, true, false},
                      Symmetrize_Usecase{true, true, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

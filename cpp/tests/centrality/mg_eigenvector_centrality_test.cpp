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
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct EigenvectorCentrality_Usecase {
  size_t max_iterations{std::numeric_limits<size_t>::max()};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGEigenvectorCentrality
  : public ::testing::TestWithParam<std::tuple<EigenvectorCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_MGEigenvectorCentrality() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running Eigenvector Centrality on multiple GPUs to that of a single-GPU
  // run
  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(EigenvectorCentrality_Usecase const& eigenvector_usecase,
                        input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, true>(
        *handle_, input_usecase, eigenvector_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    // 2. run MG Eigenvector Centrality

    weight_t constexpr epsilon{1e-6};

    rmm::device_uvector<weight_t> d_mg_centralities(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG eigenvector centrality");
    }

    d_mg_centralities = cugraph::eigenvector_centrality(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::optional<raft::device_span<weight_t const>>{},
      // std::make_optional(raft::device_span<weight_t
      // const>{d_mg_centralities.data(), d_mg_centralities.size()}),
      epsilon,
      eigenvector_usecase.max_iterations,
      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (eigenvector_usecase.check_correctness) {
      // 3-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_centralities =
        cugraph::test::device_gatherv(*handle_, d_mg_centralities.data(), d_mg_centralities.size());

      if (handle_->get_comms().get_rank() == int{0}) {
        // 3-2. Sort MG results by original vertex id
        std::tie(std::ignore, d_mg_aggregate_centralities) = cugraph::test::sort_by_key(
          *handle_, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_centralities);

        // 3-3. create SG graph
        auto [sg_graph, sg_edge_weights, d_sg_renumber_map_labels] =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
            *handle_, input_usecase, eigenvector_usecase.test_weighted, true);

        auto sg_graph_view = sg_graph.view();
        auto sg_edge_weight_view =
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        // 3-4. run SG Eigenvector Centrality
        rmm::device_uvector<weight_t> d_sg_centralities(sg_graph_view.number_of_vertices(),
                                                        handle_->get_stream());

        d_sg_centralities = cugraph::eigenvector_centrality(
          *handle_,
          sg_graph_view,
          sg_edge_weight_view,
          std::optional<raft::device_span<weight_t const>>{},
          // std::make_optional(raft::device_span<weight_t const>{d_sg_centralities.data(),
          // d_sg_centralities.size()}),
          epsilon,
          eigenvector_usecase.max_iterations,
          false);

        std::tie(std::ignore, d_sg_centralities) =
          cugraph::test::sort_by_key(*handle_, *d_sg_renumber_map_labels, d_sg_centralities);

        // 3-5. compare

        auto h_mg_aggregate_centralities =
          cugraph::test::to_host(*handle_, d_mg_aggregate_centralities);
        auto h_sg_centralities = cugraph::test::to_host(*handle_, d_sg_centralities);

        auto max_centrality =
          *std::max_element(h_mg_aggregate_centralities.begin(), h_mg_aggregate_centralities.end());

        // skip comparison for low Eigenvector Centrality vertices (lowly ranked vertices)
        auto threshold_magnitude = max_centrality * epsilon;

        auto nearly_equal = [epsilon, threshold_magnitude](auto lhs, auto rhs) {
          return std::abs(lhs - rhs) < std::max(std::max(lhs, rhs) * epsilon, threshold_magnitude);
        };

        // FIND DIFFERENCES...
        size_t count_differences{0};
        for (size_t i = 0; i < h_mg_aggregate_centralities.size(); ++i) {
          if (nearly_equal(h_mg_aggregate_centralities[i], h_sg_centralities[i])) {
          } else {
            if (count_differences < 10) {
              std::cout << "unequal [" << i << "] " << h_mg_aggregate_centralities[i]
                        << " != " << h_sg_centralities[i] << std::endl;
            }
            ++count_differences;
          }
        }

        ASSERT_EQ(count_differences, size_t{0})
          << "Eigenvector centrality values do not match with the reference "
             "values.";
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGEigenvectorCentrality<input_usecase_t>::handle_ = nullptr;

using Tests_MGEigenvectorCentrality_File =
  Tests_MGEigenvectorCentrality<cugraph::test::File_Usecase>;
using Tests_MGEigenvectorCentrality_Rmat =
  Tests_MGEigenvectorCentrality<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGEigenvectorCentrality_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGEigenvectorCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGEigenvectorCentrality_Rmat, CheckInt32Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGEigenvectorCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGEigenvectorCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EigenvectorCentrality_Usecase{500, false},
                      EigenvectorCentrality_Usecase{500, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGEigenvectorCentrality_Rmat,
                         ::testing::Combine(
                           // enable correctness checks
                           ::testing::Values(EigenvectorCentrality_Usecase{500, false},
                                             EigenvectorCentrality_Usecase{500, true}),
                           ::testing::Values(cugraph::test::Rmat_Usecase(
                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGEigenvectorCentrality_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(EigenvectorCentrality_Usecase{500, false, false},
                      EigenvectorCentrality_Usecase{500, true, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

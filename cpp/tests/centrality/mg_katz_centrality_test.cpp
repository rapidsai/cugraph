/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct KatzCentrality_Usecase {
  bool test_weighted{false};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGKatzCentrality
  : public ::testing::TestWithParam<std::tuple<KatzCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_MGKatzCentrality() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running Katz Centrality on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(KatzCentrality_Usecase const& katz_usecase,
                        input_usecase_t const& input_usecase)
  {
    using edge_type_t = int32_t;

    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, true>(
        *handle_, input_usecase, katz_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (katz_usecase.edge_masking) {
      edge_mask = cugraph::test::generate<decltype(mg_graph_view), bool>::edge_property(
        *handle_, mg_graph_view, 2);
      mg_graph_view.attach_edge_mask((*edge_mask).view());
    }

    // 2. compute max in-degree

    auto max_in_degree = mg_graph_view.compute_max_in_degree(*handle_);

    // 3. run MG Katz Centrality

    result_t const alpha = result_t{1.0} / static_cast<result_t>(max_in_degree + 1);
    result_t constexpr beta{1.0};
    result_t constexpr epsilon{1e-6};

    rmm::device_uvector<result_t> d_mg_katz_centralities(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Katz centrality");
    }

    cugraph::katz_centrality(*handle_,
                             mg_graph_view,
                             mg_edge_weight_view,
                             static_cast<result_t*>(nullptr),
                             d_mg_katz_centralities.data(),
                             alpha,
                             beta,
                             epsilon,
                             std::numeric_limits<size_t>::max(),
                             false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 4. copmare SG & MG results

    if (katz_usecase.check_correctness) {
      // 4-1. aggregate MG results

      rmm::device_uvector<result_t> d_mg_aggregate_katz_centralities(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_katz_centralities) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<result_t const>(d_mg_katz_centralities.data(),
                                            d_mg_katz_centralities.size()));

      cugraph::graph_t<vertex_t, edge_t, true, false> sg_graph(*handle_);
      std::optional<cugraph::edge_property_t<edge_t, weight_t>> sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph_view,
          mg_edge_weight_view,
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // 4-2. run SG Katz Centrality

        auto sg_graph_view = sg_graph.view();
        auto sg_edge_weight_view =
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

        ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph_view.number_of_vertices());

        rmm::device_uvector<result_t> d_sg_katz_centralities(sg_graph_view.number_of_vertices(),
                                                             handle_->get_stream());

        cugraph::katz_centrality(*handle_,
                                 sg_graph_view,
                                 sg_edge_weight_view,
                                 static_cast<result_t*>(nullptr),
                                 d_sg_katz_centralities.data(),
                                 alpha,
                                 beta,
                                 epsilon,
                                 std::numeric_limits<size_t>::max(),  // max_iterations
                                 false);

        // 4-3. compare

        auto h_mg_aggregate_katz_centralities =
          cugraph::test::to_host(*handle_, d_mg_aggregate_katz_centralities);
        auto h_sg_katz_centralities = cugraph::test::to_host(*handle_, d_sg_katz_centralities);

        auto threshold_ratio = 1e-3;
        auto threshold_magnitude =
          1e-6;  // skip comparison for low KatzCentrality verties (lowly ranked vertices)
        auto nearly_equal = [threshold_ratio, threshold_magnitude](auto lhs, auto rhs) {
          return std::abs(lhs - rhs) <
                 std::max(std::max(lhs, rhs) * threshold_ratio, threshold_magnitude);
        };

        ASSERT_TRUE(std::equal(h_mg_aggregate_katz_centralities.begin(),
                               h_mg_aggregate_katz_centralities.end(),
                               h_sg_katz_centralities.begin(),
                               nearly_equal));
      }
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGKatzCentrality<input_usecase_t>::handle_ = nullptr;

using Tests_MGKatzCentrality_File = Tests_MGKatzCentrality<cugraph::test::File_Usecase>;
using Tests_MGKatzCentrality_Rmat = Tests_MGKatzCentrality<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGKatzCentrality_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGKatzCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGKatzCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGKatzCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KatzCentrality_Usecase{false, false},
                      KatzCentrality_Usecase{false, true},
                      KatzCentrality_Usecase{true, false},
                      KatzCentrality_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGKatzCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KatzCentrality_Usecase{false, false},
                      KatzCentrality_Usecase{false, true},
                      KatzCentrality_Usecase{true, false},
                      KatzCentrality_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGKatzCentrality_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(KatzCentrality_Usecase{false, false},
                      KatzCentrality_Usecase{false, true},
                      KatzCentrality_Usecase{true, false},
                      KatzCentrality_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGKatzCentrality_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(KatzCentrality_Usecase{false, false, false},
                      KatzCentrality_Usecase{false, true, false},
                      KatzCentrality_Usecase{true, false, false},
                      KatzCentrality_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
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

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  // Compare the results of running PageRank on multiple GPUs to that of a single-GPU run
  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(PageRank_Usecase const& pagerank_usecase,
                        input_usecase_t const& input_usecase)
  {
    HighResTimer hr_timer{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, true>(
        *handle_, input_usecase, pagerank_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    // 2. generate personalization vertex/value pairs

    std::optional<rmm::device_uvector<vertex_t>> d_mg_personalization_vertices{std::nullopt};
    std::optional<rmm::device_uvector<result_t>> d_mg_personalization_values{std::nullopt};
    if (pagerank_usecase.personalization_ratio > 0.0) {
      raft::random::RngState rng_state(handle_->get_comms().get_rank());

      d_mg_personalization_vertices = cugraph::select_random_vertices(
        *handle_,
        mg_graph_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        rng_state,
        std::max(
          static_cast<size_t>(mg_graph_view.number_of_vertices() *
                              pagerank_usecase.personalization_ratio),
          std::min(
            static_cast<size_t>(mg_graph_view.number_of_vertices()),
            size_t{1})),  // there should be at least one vertex unless the graph is an empty graph
        false,
        false);
      d_mg_personalization_values = rmm::device_uvector<result_t>(
        (*d_mg_personalization_vertices).size(), handle_->get_stream());
      cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                           (*d_mg_personalization_values).data(),
                                           (*d_mg_personalization_values).size(),
                                           result_t{0.0},
                                           result_t{1.0},
                                           rng_state);
    }

    // 3. run MG PageRank

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG PageRank");
    }

    auto [d_mg_pageranks, metadata] = cugraph::pagerank<vertex_t, edge_t, weight_t>(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
      std::nullopt,
      d_mg_personalization_vertices
        ? std::make_optional(std::make_tuple(
            raft::device_span<vertex_t const>{d_mg_personalization_vertices->data(),
                                              d_mg_personalization_vertices->size()},
            raft::device_span<result_t const>{d_mg_personalization_values->data(),
                                              d_mg_personalization_values->size()}))
        : std::nullopt,
      std::optional<raft::device_span<result_t const>>{std::nullopt},
      alpha,
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

    if (pagerank_usecase.check_correctness) {
      // 4-1. aggregate MG results

      std::optional<rmm::device_uvector<vertex_t>> d_mg_aggregate_personalization_vertices{
        std::nullopt};
      std::optional<rmm::device_uvector<result_t>> d_mg_aggregate_personalization_values{
        std::nullopt};
      if (d_mg_personalization_vertices) {
        std::tie(d_mg_aggregate_personalization_vertices, d_mg_aggregate_personalization_values) =
          cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
            *handle_,
            std::make_optional<raft::device_span<vertex_t const>>((*d_mg_renumber_map).data(),
                                                                  (*d_mg_renumber_map).size()),
            mg_graph_view.local_vertex_partition_range(),
            std::optional<raft::device_span<vertex_t const>>{std::nullopt},
            std::make_optional<raft::device_span<vertex_t const>>(
              (*d_mg_personalization_vertices).data(), (*d_mg_personalization_vertices).size()),
            raft::device_span<result_t const>((*d_mg_personalization_values).data(),
                                              (*d_mg_personalization_values).size()));
      }

      rmm::device_uvector<result_t> d_mg_aggregate_pageranks(0, handle_->get_stream());
      std::tie(std::ignore, d_mg_aggregate_pageranks) =
        cugraph::test::mg_vertex_property_values_to_sg_vertex_property_values(
          *handle_,
          std::make_optional<raft::device_span<vertex_t const>>((*d_mg_renumber_map).data(),
                                                                (*d_mg_renumber_map).size()),
          mg_graph_view.local_vertex_partition_range(),
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          std::optional<raft::device_span<vertex_t const>>{std::nullopt},
          raft::device_span<result_t const>(d_mg_pageranks.data(), d_mg_pageranks.size()));

      cugraph::graph_t<vertex_t, edge_t, true, false> sg_graph(*handle_);
      std::optional<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, false>, weight_t>>
        sg_edge_weights{std::nullopt};
      std::tie(sg_graph, sg_edge_weights, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
        *handle_,
        mg_graph_view,
        mg_edge_weight_view,
        std::make_optional<raft::device_span<vertex_t const>>((*d_mg_renumber_map).data(),
                                                              (*d_mg_renumber_map).size()),
        false);

      if (handle_->get_comms().get_rank() == int{0}) {
        // 4-2. run SG PageRank

        auto sg_graph_view = sg_graph.view();
        auto sg_edge_weight_view =
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

        ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

        auto [d_sg_pageranks, sg_metadata] = cugraph::pagerank<vertex_t, edge_t, weight_t>(
          *handle_,
          sg_graph_view,
          sg_edge_weight_view,
          std::nullopt,
          d_mg_aggregate_personalization_vertices
            ? std::make_optional(std::make_tuple(
                raft::device_span<vertex_t const>{d_mg_aggregate_personalization_vertices->data(),
                                                  d_mg_aggregate_personalization_vertices->size()},
                raft::device_span<result_t const>{d_mg_aggregate_personalization_values->data(),
                                                  d_mg_aggregate_personalization_values->size()}))
            : std::nullopt,
          std::optional<raft::device_span<result_t const>>{std::nullopt},
          alpha,
          epsilon,
          std::numeric_limits<size_t>::max(),  // max_iterations
          false);

        // 4-3. compare

        auto h_mg_aggregate_pageranks = cugraph::test::to_host(*handle_, d_mg_aggregate_pageranks);
        auto h_sg_pageranks           = cugraph::test::to_host(*handle_, d_sg_pageranks);

        auto threshold_ratio = 1e-3;
        auto threshold_magnitude =
          1e-6;  // skip comparison for low PageRank verties (lowly ranked vertices)
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

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGPageRank<input_usecase_t>::handle_ = nullptr;

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
    ::testing::Values(cugraph::test::File_Usecase("karate.csv"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_tests,
                         Tests_MGPageRank_Rmat,
                         ::testing::Combine(::testing::Values(PageRank_Usecase{0.0, false},
                                                              PageRank_Usecase{0.5, false},
                                                              PageRank_Usecase{0.0, true},
                                                              PageRank_Usecase{0.5, true}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGPageRank_Rmat,
  ::testing::Combine(
    ::testing::Values(PageRank_Usecase{0.0, false, false},
                      PageRank_Usecase{0.5, false, false},
                      PageRank_Usecase{0.0, true, false},
                      PageRank_Usecase{0.5, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

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
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
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

    auto [mg_graph, mg_edge_weights, d_mg_renumber_map_labels] =
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

    std::optional<std::vector<vertex_t>> h_mg_personalization_vertices{std::nullopt};
    std::optional<std::vector<result_t>> h_mg_personalization_values{std::nullopt};
    if (pagerank_usecase.personalization_ratio > 0.0) {
      std::default_random_engine generator{
        static_cast<long unsigned int>(handle_->get_comms().get_rank()) /* seed */};
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
                                                            handle_->get_stream())
        : std::nullopt;
    auto d_mg_personalization_values =
      h_mg_personalization_values
        ? std::make_optional<rmm::device_uvector<result_t>>((*d_mg_personalization_vertices).size(),
                                                            handle_->get_stream())
        : std::nullopt;
    if (d_mg_personalization_vertices) {
      raft::update_device((*d_mg_personalization_vertices).data(),
                          (*h_mg_personalization_vertices).data(),
                          (*h_mg_personalization_vertices).size(),
                          handle_->get_stream());
      raft::update_device((*d_mg_personalization_values).data(),
                          (*h_mg_personalization_values).data(),
                          (*h_mg_personalization_values).size(),
                          handle_->get_stream());
    }

    // 3. run MG PageRank

    result_t constexpr alpha{0.85};
    result_t constexpr epsilon{1e-6};

    rmm::device_uvector<result_t> d_mg_pageranks(mg_graph_view.local_vertex_partition_range_size(),
                                                 handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG PageRank");
    }

    cugraph::pagerank<vertex_t, edge_t, weight_t>(
      *handle_,
      mg_graph_view,
      mg_edge_weight_view,
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
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 4. copmare SG & MG results

    if (pagerank_usecase.check_correctness) {
      // 4-1. aggregate MG results

      auto d_mg_aggregate_renumber_map_labels = cugraph::test::device_gatherv(
        *handle_, (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size());
      auto d_mg_aggregate_personalization_vertices =
        d_mg_personalization_vertices
          ? std::optional<rmm::device_uvector<vertex_t>>{cugraph::test::device_gatherv(
              *handle_,
              (*d_mg_personalization_vertices).data(),
              (*d_mg_personalization_vertices).size())}
          : std::nullopt;
      auto d_mg_aggregate_personalization_values =
        d_mg_personalization_values
          ? std::optional<rmm::device_uvector<result_t>>{cugraph::test::device_gatherv(
              *handle_,
              (*d_mg_personalization_values).data(),
              (*d_mg_personalization_values).size())}
          : std::nullopt;
      auto d_mg_aggregate_pageranks =
        cugraph::test::device_gatherv(*handle_, d_mg_pageranks.data(), d_mg_pageranks.size());

      if (handle_->get_comms().get_rank() == int{0}) {
        // 4-2. unrenumbr MG results

        if (d_mg_aggregate_personalization_vertices) {
          cugraph::unrenumber_int_vertices<vertex_t, false>(
            *handle_,
            (*d_mg_aggregate_personalization_vertices).data(),
            (*d_mg_aggregate_personalization_vertices).size(),
            d_mg_aggregate_renumber_map_labels.data(),
            std::vector<vertex_t>{mg_graph_view.number_of_vertices()});
          std::tie(d_mg_aggregate_personalization_vertices, d_mg_aggregate_personalization_values) =
            cugraph::test::sort_by_key(*handle_,
                                       *d_mg_aggregate_personalization_vertices,
                                       *d_mg_aggregate_personalization_values);
        }
        std::tie(std::ignore, d_mg_aggregate_pageranks) = cugraph::test::sort_by_key(
          *handle_, d_mg_aggregate_renumber_map_labels, d_mg_aggregate_pageranks);

        // 4-3. create SG graph

        cugraph::graph_t<vertex_t, edge_t, true, false> sg_graph(*handle_);
        std::optional<
          cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, true, false>, weight_t>>
          sg_edge_weights{std::nullopt};
        std::tie(sg_graph, sg_edge_weights, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
            *handle_, input_usecase, pagerank_usecase.test_weighted, false);

        auto sg_graph_view = sg_graph.view();
        auto sg_edge_weight_view =
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt;

        ASSERT_EQ(mg_graph_view.number_of_vertices(), sg_graph_view.number_of_vertices());

        // 4-4. run SG PageRank

        rmm::device_uvector<result_t> d_sg_pageranks(sg_graph_view.number_of_vertices(),
                                                     handle_->get_stream());

        cugraph::pagerank<vertex_t, edge_t, weight_t>(
          *handle_,
          sg_graph_view,
          sg_edge_weight_view,
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

        // 4-5. compare

        auto h_mg_aggregate_pageranks = cugraph::test::to_host(*handle_, d_mg_aggregate_pageranks);
        auto h_sg_pageranks           = cugraph::test::to_host(*handle_, d_sg_pageranks);

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

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
#include <utilities/mg_utilities.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/partition_manager.hpp>

#include <link_prediction/similarity_compare.hpp>

struct Similarity_Usecase {
  bool use_weights{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGSimilarity
  : public ::testing::TestWithParam<std::tuple<Similarity_Usecase, input_usecase_t>> {
 public:
  Tests_MGSimilarity() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename test_functor_t>
  void run_current_test(std::tuple<Similarity_Usecase const&, input_usecase_t const&> param,
                        test_functor_t const& test_functor)
  {
    auto [similarity_usecase, input_usecase] = param;
    HighResClock hr_clock{};

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    auto [mg_graph, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 2. run similarity

    auto mg_graph_view = mg_graph.view();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_clock.start();
    }

    std::optional<raft::device_span<vertex_t const>> first  = std::nullopt;
    std::optional<raft::device_span<vertex_t const>> second = std::nullopt;

#if 0
    auto [result_src, result_dst, result_score] =
      test_functor.run(*handle_, mg_graph_view, first, second, similarity_usecase.use_weights);
#else
    EXPECT_THROW(
      test_functor.run(*handle_, mg_graph_view, first, second, similarity_usecase.use_weights),
      std::exception);
#endif

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "MG " << test_functor.testname << " took " << elapsed_time * 1e-6 << " s.\n";
    }

    // 3. compare SG & MG results

    if (similarity_usecase.check_correctness) {
#if 0
      auto [src, dst, wgt] = cugraph::test::graph_to_host_coo(*handle_, mg_graph_view);

      result_src = cugraph::test::device_gatherv(*handle_, result_src.data(), result_src.size());
      result_dst = cugraph::test::device_gatherv(*handle_, result_dst.data(), result_dst.size());
      result_score =
        cugraph::test::device_gatherv(*handle_, result_score.data(), result_score.size());

      if (result_src.size() > 0) {
        std::vector<vertex_t> h_result_src(result_src.size());
        std::vector<vertex_t> h_result_dst(result_dst.size());
        std::vector<weight_t> h_result_score(result_score.size());

        raft::update_host(
          h_result_src.data(), result_src.data(), result_src.size(), handle_->get_stream());
        raft::update_host(
          h_result_dst.data(), result_dst.data(), result_dst.size(), handle_->get_stream());
        raft::update_host(
          h_result_score.data(), result_score.data(), result_score.size(), handle_->get_stream());

        similarity_compare(mg_graph_view.number_of_vertices(),
                           std::move(src),
                           std::move(dst),
                           std::move(wgt),
                           std::move(h_result_src),
                           std::move(h_result_dst),
                           std::move(h_result_score),
                           test_functor);
      }
#endif
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGSimilarity<input_usecase_t>::handle_ = nullptr;

using Tests_MGSimilarity_File = Tests_MGSimilarity<cugraph::test::File_Usecase>;
using Tests_MGSimilarity_Rmat = Tests_MGSimilarity<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGSimilarity_File, CheckInt32Int32FloatFloatJaccard)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt32Int32FloatFloatJaccard)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt32Int64FloatFloatJaccard)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt64Int64FloatFloatJaccard)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_jaccard_t{});
}

TEST_P(Tests_MGSimilarity_File, CheckInt32Int32FloatSorensen)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt32Int32FloatSorensen)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt32Int64FloatSorensen)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt64Int64FloatSorensen)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_sorensen_t{});
}

TEST_P(Tests_MGSimilarity_File, CheckInt32Int32FloatOverlap)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt32Int32FloatOverlap)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt32Int64FloatOverlap)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

TEST_P(Tests_MGSimilarity_Rmat, CheckInt64Int64FloatOverlap)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()), cugraph::test::test_overlap_t{});
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGSimilarity_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Similarity_Usecase{true, false}, Similarity_Usecase{true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGSimilarity_Rmat,
                         ::testing::Combine(
                           // enable correctness checks
                           ::testing::Values(Similarity_Usecase{true, false},
                                             Similarity_Usecase{true, true}),
                           ::testing::Values(cugraph::test::Rmat_Usecase(
                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGSimilarity_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Similarity_Usecase{false, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

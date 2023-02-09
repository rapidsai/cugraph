/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <link_prediction/similarity_compare.hpp>

struct Similarity_Usecase {
  bool use_weights{false};
  bool check_correctness{true};
  size_t max_seeds{std::numeric_limits<size_t>::max()};
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
    HighResTimer hr_timer{};

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    // 1. create MG graph

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, d_mg_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 2. run similarity

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    rmm::device_uvector<vertex_t> d_start_vertices(
      std::min(
        static_cast<size_t>(mg_graph_view.local_vertex_partition_range_size()),
        similarity_usecase.max_seeds / comm_size +
          (static_cast<size_t>(comm_rank) < similarity_usecase.max_seeds % comm_size ? 1 : 0)),
      handle_->get_stream());
    cugraph::test::populate_vertex_ids(
      *handle_, d_start_vertices, mg_graph_view.local_vertex_partition_range_first());

    auto [d_offsets, two_hop_nbrs] = cugraph::k_hop_nbrs(
      *handle_,
      mg_graph_view,
      raft::device_span<vertex_t const>(d_start_vertices.data(), d_start_vertices.size()),
      2);

    auto h_start_vertices = cugraph::test::to_host(*handle_, d_start_vertices);
    auto h_offsets        = cugraph::test::to_host(*handle_, d_offsets);

    std::vector<vertex_t> h_v1(h_offsets.back());
    for (size_t i = 0; i < h_start_vertices.size(); ++i) {
      std::fill(h_v1.begin() + h_offsets[i], h_v1.begin() + h_offsets[i + 1], h_start_vertices[i]);
    }

    auto d_v1 = cugraph::test::to_device(*handle_, h_v1);
    auto d_v2 = std::move(two_hop_nbrs);

    std::tie(d_v1, d_v2, std::ignore, std::ignore) =
      cugraph::detail::shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                  edge_t,
                                                                                  weight_t,
                                                                                  int32_t>(
        *handle_,
        std::move(d_v1),
        std::move(d_v2),
        std::nullopt,
        std::nullopt,
        mg_graph_view.vertex_partition_range_lasts());

    std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs{
      {d_v1.data(), d_v1.size()}, {d_v2.data(), d_v2.size()}};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG similarity test");
    }

    auto result_score = test_functor.run(
      *handle_, mg_graph_view, mg_edge_weight_view, vertex_pairs, similarity_usecase.use_weights);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. compare SG & MG results

    if (similarity_usecase.check_correctness) {
      auto [src, dst, wgt] =
        cugraph::test::graph_to_host_coo(*handle_, mg_graph_view, mg_edge_weight_view);

      d_v1 = cugraph::test::device_gatherv(*handle_, d_v1.data(), d_v1.size());
      d_v2 = cugraph::test::device_gatherv(*handle_, d_v2.data(), d_v2.size());
      result_score =
        cugraph::test::device_gatherv(*handle_, result_score.data(), result_score.size());

      if (d_v1.size() > 0) {
        auto h_vertex_pair1 = cugraph::test::to_host(*handle_, d_v1);
        auto h_vertex_pair2 = cugraph::test::to_host(*handle_, d_v2);
        auto h_result_score = cugraph::test::to_host(*handle_, result_score);

        similarity_compare(mg_graph_view.number_of_vertices(),
                           std::tie(src, dst, wgt),
                           std::tie(h_vertex_pair1, h_vertex_pair2),
                           h_result_score,
                           test_functor);
      }
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
    // Disable weighted computation testing in 22.10
    //::testing::Values(Similarity_Usecase{true, true, 20}, Similarity_Usecase{false, true, 20}),
    ::testing::Values(Similarity_Usecase{false, true, 20}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGSimilarity_Rmat,
                         ::testing::Combine(
                           // enable correctness checks
                           // Disable weighted computation testing in 22.10
                           //::testing::Values(Similarity_Usecase{true, true, 20},
                           // Similarity_Usecase{false, true, 20}),
                           ::testing::Values(Similarity_Usecase{false, true, 20}),
                           ::testing::Values(cugraph::test::Rmat_Usecase(
                             10, 16, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGSimilarity_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Similarity_Usecase{false, false, 20}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, true, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <cugraph/utilities/high_res_timer.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>

#include <chrono>
#include <random>

#include <gtest/gtest.h>

struct MaximalIndependentSet_Usecase {
  size_t select_count{std::numeric_limits<size_t>::max()};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGMaximalIndependentSet
  : public ::testing::TestWithParam<std::tuple<MaximalIndependentSet_Usecase, input_usecase_t>> {
 public:
  Tests_MGMaximalIndependentSet() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<MaximalIndependentSet_Usecase, input_usecase_t> const& param)
  {
    auto [select_random_vertices_usecase, input_usecase] = param;

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    // Test MIS

    auto d_mis =
      cugraph::compute_mis<vertex_t, edge_t, weight_t, true>(*handle_, mg_graph_view, std::nullopt);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    std::vector<vertex_t> h_mis(d_mis.size());
    raft::update_host(h_mis.data(), d_mis.data(), d_mis.size(), handle_->get_stream());

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    for (int i = 0; i < comm_size; ++i) {
      if (comm_rank == i) {
        if (h_mis.size() <= 50) {
          std::cout << "MIS (rank:" << comm_rank << "): ";
          std::copy(h_mis.begin(), h_mis.end(), std::ostream_iterator<int>(std::cout, " "));
          std::cout << std::endl;
        }
        auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
        auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

        std::for_each(h_mis.begin(), h_mis.end(), [vertex_first, vertex_last](vertex_t v) {
          ASSERT_TRUE((v >= vertex_first) && (v < vertex_last));
        });
      }

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
    }
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGMaximalIndependentSet<input_usecase_t>::handle_ = nullptr;

using Tests_MGMaximalIndependentSet_File =
  Tests_MGMaximalIndependentSet<cugraph::test::File_Usecase>;
using Tests_MGMaximalIndependentSet_Rmat =
  Tests_MGMaximalIndependentSet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_MGMaximalIndependentSet_File,
  ::testing::Combine(::testing::Values(MaximalIndependentSet_Usecase{20, false},
                                       MaximalIndependentSet_Usecase{20, false}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGMaximalIndependentSet_Rmat,
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{50, false},
                      MaximalIndependentSet_Usecase{50, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(6, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGMaximalIndependentSet_Rmat,
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{500, false},
                      MaximalIndependentSet_Usecase{500, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

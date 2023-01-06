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
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct TransposeStorage_Usecase {
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTransposeStorage
  : public ::testing::TestWithParam<std::tuple<TransposeStorage_Usecase, input_usecase_t>> {
 public:
  Tests_MGTransposeStorage() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(TransposeStorage_Usecase const& transpose_storage_usecase,
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
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, transpose_storage_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 2. run MG transpose storage

    auto mg_graph_number_of_vertices = mg_graph.number_of_vertices();
    auto mg_graph_number_of_edges    = mg_graph.number_of_edges();

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transpose storage");
    }

    cugraph::graph_t<vertex_t, edge_t, !store_transposed, true> mg_storage_transposed_graph(
      *handle_);
    std::optional<
      cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, !store_transposed, true>,
                               weight_t>>
      mg_storage_transposed_edge_weights{std::nullopt};
    std::tie(
      mg_storage_transposed_graph, mg_storage_transposed_edge_weights, d_mg_renumber_map_labels) =
      cugraph::transpose_graph_storage(
        *handle_,
        std::move(mg_graph),
        std::move(mg_edge_weights),
        d_mg_renumber_map_labels
          ? std::optional<rmm::device_uvector<vertex_t>>{std::move(*d_mg_renumber_map_labels)}
          : std::nullopt);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 3. copmare SG & MG results

    if (transpose_storage_usecase.check_correctness) {
      // 3-1. decompress MG results

      auto [d_mg_srcs, d_mg_dsts, d_mg_weights] = cugraph::decompress_to_edgelist(
        *handle_,
        mg_storage_transposed_graph.view(),
        mg_storage_transposed_edge_weights
          ? std::make_optional((*mg_storage_transposed_edge_weights).view())
          : std::nullopt,
        d_mg_renumber_map_labels
          ? std::make_optional<raft::device_span<vertex_t const>>(
              (*d_mg_renumber_map_labels).data(), (*d_mg_renumber_map_labels).size())
          : std::nullopt);

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

        cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
        std::optional<
          cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, false>,
                                   weight_t>>
          sg_edge_weights{std::nullopt};
        std::tie(sg_graph, sg_edge_weights, std::ignore) =
          cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, false>(
            *handle_, input_usecase, transpose_storage_usecase.test_weighted, false);

        // 3-4. decompress SG results

        auto [d_sg_srcs, d_sg_dsts, d_sg_weights] = cugraph::decompress_to_edgelist(
          *handle_,
          sg_graph.view(),
          sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt,
          std::optional<raft::device_span<vertex_t const>>{std::nullopt});

        // 3-5. compare

        ASSERT_TRUE(mg_graph_number_of_vertices == sg_graph.number_of_vertices());
        ASSERT_TRUE(mg_graph_number_of_edges == sg_graph.number_of_edges());

        auto h_mg_aggregate_srcs    = cugraph::test::to_host(*handle_, d_mg_aggregate_srcs);
        auto h_mg_aggregate_dsts    = cugraph::test::to_host(*handle_, d_mg_aggregate_dsts);
        auto h_mg_aggregate_weights = cugraph::test::to_host(*handle_, d_mg_aggregate_weights);

        auto h_sg_srcs    = cugraph::test::to_host(*handle_, d_sg_srcs);
        auto h_sg_dsts    = cugraph::test::to_host(*handle_, d_sg_dsts);
        auto h_sg_weights = cugraph::test::to_host(*handle_, d_sg_weights);

        if (transpose_storage_usecase.test_weighted) {
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
std::unique_ptr<raft::handle_t> Tests_MGTransposeStorage<input_usecase_t>::handle_ = nullptr;

using Tests_MGTransposeStorage_File = Tests_MGTransposeStorage<cugraph::test::File_Usecase>;
using Tests_MGTransposeStorage_Rmat = Tests_MGTransposeStorage<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTransposeStorage_File, CheckInt32Int32FloatTransposeStoragedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransposeStorage_File, CheckInt32Int32FloatTransposeStoragedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTransposeStorage_Rmat, CheckInt32Int32FloatTransposeStoragedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransposeStorage_Rmat, CheckInt32Int32FloatTransposeStoragedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransposeStorage_Rmat, CheckInt32Int64FloatTransposeStoragedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransposeStorage_Rmat, CheckInt32Int64FloatTransposeStoragedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransposeStorage_Rmat, CheckInt64Int64FloatTransposeStoragedFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTransposeStorage_Rmat, CheckInt64Int64FloatTransposeStoragedTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTransposeStorage_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(TransposeStorage_Usecase{false}, TransposeStorage_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_MGTransposeStorage_Rmat,
                         ::testing::Combine(
                           // enable correctness checks
                           ::testing::Values(TransposeStorage_Usecase{false},
                                             TransposeStorage_Usecase{true}),
                           ::testing::Values(cugraph::test::Rmat_Usecase(
                             10, 16, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTransposeStorage_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(TransposeStorage_Usecase{false, false},
                      TransposeStorage_Usecase{true, false}),
    ::testing::Values(
      cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false, 0, true))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

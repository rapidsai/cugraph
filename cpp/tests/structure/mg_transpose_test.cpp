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
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <random>

struct Transpose_Usecase {
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGTranspose
  : public ::testing::TestWithParam<std::tuple<Transpose_Usecase, input_usecase_t>> {
 public:
  Tests_MGTranspose() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(Transpose_Usecase const& transpose_usecase,
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
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, store_transposed, true>(
        *handle_, input_usecase, transpose_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 2. create SG graph from MG graph before transposing (for correctness check)

    cugraph::graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(*handle_);
    std::optional<cugraph::edge_property_t<edge_t, weight_t>> sg_edge_weights{std::nullopt};
    if (transpose_usecase.check_correctness) {
      std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, std::ignore) =
        cugraph::test::mg_graph_to_sg_graph(
          *handle_,
          mg_graph.view(),
          mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt,
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>{std::nullopt},
          std::make_optional<raft::device_span<vertex_t const>>((*mg_renumber_map).data(),
                                                                (*mg_renumber_map).size()),
          false);
    }

    // 3. run MG transpose

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.start("MG transpose");
    }

    std::tie(mg_graph, mg_edge_weights, mg_renumber_map) = cugraph::transpose_graph(
      *handle_,
      std::move(mg_graph),
      std::move(mg_edge_weights),
      mg_renumber_map ? std::optional<rmm::device_uvector<vertex_t>>{std::move(*mg_renumber_map)}
                      : std::nullopt);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 4. copmare SG & MG results

    if (transpose_usecase.check_correctness) {
      // 4-1. decompress MG results
      rmm::device_uvector<vertex_t> d_mg_srcs(0, handle_->get_stream());
      rmm::device_uvector<vertex_t> d_mg_dsts(0, handle_->get_stream());
      std::optional<rmm::device_uvector<weight_t>> d_mg_weights{std::nullopt};

      std::tie(d_mg_srcs, d_mg_dsts, d_mg_weights, std::ignore, std::ignore) =
        cugraph::decompress_to_edgelist(
          *handle_,
          mg_graph.view(),
          mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt,
          std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
          std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
          mg_renumber_map ? std::make_optional<raft::device_span<vertex_t const>>(
                              (*mg_renumber_map).data(), (*mg_renumber_map).size())
                          : std::nullopt);

      // 4-2. aggregate MG results

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
        // 4-3. run SG transpose

        std::tie(sg_graph, sg_edge_weights, std::ignore) =
          cugraph::transpose_graph(*handle_,
                                   std::move(sg_graph),
                                   std::move(sg_edge_weights),
                                   std::optional<rmm::device_uvector<vertex_t>>{std::nullopt});

        // 4-4. decompress SG results
        rmm::device_uvector<vertex_t> d_sg_srcs(0, handle_->get_stream());
        rmm::device_uvector<vertex_t> d_sg_dsts(0, handle_->get_stream());
        std::optional<rmm::device_uvector<weight_t>> d_sg_weights{std::nullopt};

        std::tie(d_sg_srcs, d_sg_dsts, d_sg_weights, std::ignore, std::ignore) =
          cugraph::decompress_to_edgelist(
            *handle_,
            sg_graph.view(),
            sg_edge_weights ? std::make_optional((*sg_edge_weights).view()) : std::nullopt,
            std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
            std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
            std::optional<raft::device_span<vertex_t const>>{std::nullopt});

        // 4-5. compare

        ASSERT_TRUE(mg_graph.number_of_vertices() == sg_graph.number_of_vertices());
        ASSERT_TRUE(mg_graph.number_of_edges() == sg_graph.number_of_edges());

        auto h_mg_aggregate_srcs    = cugraph::test::to_host(*handle_, d_mg_aggregate_srcs);
        auto h_mg_aggregate_dsts    = cugraph::test::to_host(*handle_, d_mg_aggregate_dsts);
        auto h_mg_aggregate_weights = cugraph::test::to_host(*handle_, d_mg_aggregate_weights);

        std::vector<vertex_t> h_sg_srcs = cugraph::test::to_host(*handle_, d_sg_srcs);
        std::vector<vertex_t> h_sg_dsts = cugraph::test::to_host(*handle_, d_sg_dsts);
        auto h_sg_weights               = cugraph::test::to_host(*handle_, d_sg_weights);

        if (transpose_usecase.test_weighted) {
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
std::unique_ptr<raft::handle_t> Tests_MGTranspose<input_usecase_t>::handle_ = nullptr;

using Tests_MGTranspose_File = Tests_MGTranspose<cugraph::test::File_Usecase>;
using Tests_MGTranspose_Rmat = Tests_MGTranspose<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGTranspose_File, CheckInt32Int32FloatTransposedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTranspose_File, CheckInt32Int32FloatTransposedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_MGTranspose_Rmat, CheckInt32Int32FloatTransposedFalse)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTranspose_Rmat, CheckInt32Int32FloatTransposedTrue)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTranspose_Rmat, CheckInt64Int64FloatTransposedFalse)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, false>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_MGTranspose_Rmat, CheckInt64Int64FloatTransposedTrue)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float, true>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_MGTranspose_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Transpose_Usecase{false}, Transpose_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_MGTranspose_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Transpose_Usecase{false}, Transpose_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGTranspose_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Transpose_Usecase{false}, Transpose_Usecase{true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGTranspose_Rmat,
  ::testing::Combine(
    // disable correctness checks for large graphs
    ::testing::Values(Transpose_Usecase{false, false}, Transpose_Usecase{true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()

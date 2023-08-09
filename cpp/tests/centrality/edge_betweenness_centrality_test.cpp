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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#include <centrality/betweenness_centrality_reference.hpp>
#include <centrality/betweenness_centrality_validate.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

struct EdgeBetweennessCentrality_Usecase {
  size_t num_seeds{std::numeric_limits<size_t>::max()};
  bool normalized{false};
  bool test_weighted{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_EdgeBetweennessCentrality
  : public ::testing::TestWithParam<
      std::tuple<EdgeBetweennessCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_EdgeBetweennessCentrality() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<EdgeBetweennessCentrality_Usecase, input_usecase_t> const& param)
  {
    constexpr bool renumber           = true;
    constexpr bool do_expensive_check = false;

    auto [betweenness_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, betweenness_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    raft::random::RngState rng_state(0);
    auto d_seeds = cugraph::select_random_vertices(
      handle,
      graph_view,
      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      rng_state,
      betweenness_usecase.num_seeds,
      false,
      true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Edge betweenness centrality");
    }

    auto d_centralities = cugraph::edge_betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      std::make_optional<raft::device_span<vertex_t const>>(
        raft::device_span<vertex_t const>{d_seeds.data(), d_seeds.size()}),
      betweenness_usecase.normalized,
      do_expensive_check);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (betweenness_usecase.check_correctness) {
      // Compute reference edge betweenness result
      auto [h_offsets, h_indices, h_wgt] =
        cugraph::test::graph_to_host_csr(handle, graph_view, edge_weight_view);

      auto h_seeds = cugraph::test::to_host(handle, d_seeds);

      auto h_reference_centralities =
        edge_betweenness_centrality_reference(h_offsets,
                                              h_indices,
                                              h_wgt,
                                              h_seeds,
                                              !graph_view.is_symmetric(),
                                              betweenness_usecase.normalized);

      rmm::device_uvector<vertex_t> d_reference_src_vertex_ids(0, handle.get_stream());
      rmm::device_uvector<vertex_t> d_reference_dst_vertex_ids(0, handle.get_stream());

      std::tie(d_reference_src_vertex_ids, d_reference_dst_vertex_ids, std::ignore) =
        cugraph::test::graph_to_device_coo(handle, graph_view, edge_weight_view);

      auto d_reference_centralities = cugraph::test::to_device(handle, h_reference_centralities);

      auto [d_cugraph_src_vertex_ids, d_cugraph_dst_vertex_ids, d_cugraph_results] =
        cugraph::test::graph_to_device_coo(
          handle, graph_view, std::make_optional(d_centralities.view()));

      cugraph::test::edge_betweenness_centrality_validate(handle,
                                                          d_cugraph_src_vertex_ids,
                                                          d_cugraph_dst_vertex_ids,
                                                          *d_cugraph_results,
                                                          d_reference_src_vertex_ids,
                                                          d_reference_dst_vertex_ids,
                                                          d_reference_centralities);
    }
  }
};

using Tests_EdgeBetweennessCentrality_File =
  Tests_EdgeBetweennessCentrality<cugraph::test::File_Usecase>;
using Tests_EdgeBetweennessCentrality_Rmat =
  Tests_EdgeBetweennessCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_EdgeBetweennessCentrality_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_EdgeBetweennessCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_EdgeBetweennessCentrality_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_EdgeBetweennessCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  run_current_test<int64_t, int64_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_EdgeBetweennessCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EdgeBetweennessCentrality_Usecase{20, false, false, true},
                      EdgeBetweennessCentrality_Usecase{20, false, true, true},
                      EdgeBetweennessCentrality_Usecase{20, true, false, true},
                      EdgeBetweennessCentrality_Usecase{20, true, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_EdgeBetweennessCentrality_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(EdgeBetweennessCentrality_Usecase{50, false, false, true},
                      EdgeBetweennessCentrality_Usecase{50, false, true, true},
                      EdgeBetweennessCentrality_Usecase{50, true, false, true},
                      EdgeBetweennessCentrality_Usecase{50, true, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_EdgeBetweennessCentrality_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(EdgeBetweennessCentrality_Usecase{500, false, false, false},
                      EdgeBetweennessCentrality_Usecase{500, false, true, false},
                      EdgeBetweennessCentrality_Usecase{500, true, false, false},
                      EdgeBetweennessCentrality_Usecase{500, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

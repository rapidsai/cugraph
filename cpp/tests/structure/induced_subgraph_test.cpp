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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#include <structure/induced_subgraph_validate.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_graphs.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

struct InducedSubgraph_Usecase {
  std::vector<size_t> subgraph_sizes{};
  bool test_weighted{false};
  bool check_correctness{false};
};

template <typename input_usecase_t>
class Tests_InducedSubgraph
  : public ::testing::TestWithParam<std::tuple<InducedSubgraph_Usecase, input_usecase_t>> {
 public:
  Tests_InducedSubgraph() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(
    std::tuple<InducedSubgraph_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [induced_subgraph_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResClock hr_clock{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    auto [graph, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, induced_subgraph_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "construct_graph took " << elapsed_time * 1e-6 << " s.\n";
    }

    auto graph_view = graph.view();

    // Construct random subgraph vertex lists
    std::vector<size_t> h_subgraph_offsets(induced_subgraph_usecase.subgraph_sizes.size() + 1, 0);
    std::partial_sum(induced_subgraph_usecase.subgraph_sizes.begin(),
                     induced_subgraph_usecase.subgraph_sizes.end(),
                     h_subgraph_offsets.begin() + 1);

    rmm::device_uvector<vertex_t> all_vertices(graph_view.number_of_vertices(),
                                               handle.get_stream());
    cugraph::detail::sequence_fill(
      handle.get_stream(), all_vertices.data(), all_vertices.size(), vertex_t{0});

    rmm::device_uvector<vertex_t> d_subgraph_vertices(h_subgraph_offsets.back(),
                                                      handle.get_stream());

    for (size_t i = 0; i < induced_subgraph_usecase.subgraph_sizes.size(); ++i) {
      auto start = h_subgraph_offsets[i];
      auto last  = h_subgraph_offsets[i + 1];
      ASSERT_TRUE(last - start <= graph_view.number_of_vertices()) << "Invalid subgraph size.";
      // this is inefficient if last - start << graph_view.number_of_vertices() but this is for
      // the test purpose only and the time & memory cost is only linear to
      // graph_view.number_of_vertices(), so this may not matter.

      auto vertices = cugraph::test::randomly_select(handle, all_vertices, (last - start), true);
      raft::copy(
        d_subgraph_vertices.data() + start, vertices.data(), vertices.size(), handle.get_stream());
    }

    auto d_subgraph_offsets = cugraph::test::to_device(handle, h_subgraph_offsets);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    // FIXME: turn-off do_expensive_check once verified.
    auto [d_subgraph_edgelist_majors,
          d_subgraph_edgelist_minors,
          d_subgraph_edgelist_weights,
          d_subgraph_edge_offsets] =
      cugraph::extract_induced_subgraphs(
        handle,
        graph_view,
        raft::device_span<size_t const>(d_subgraph_offsets.data(), d_subgraph_offsets.size()),
        raft::device_span<vertex_t const>(d_subgraph_vertices.data(), d_subgraph_vertices.size()),
        true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      double elapsed_time{0.0};
      hr_clock.stop(&elapsed_time);
      std::cout << "induced subgraph took " << elapsed_time * 1e-6 << " s.\n";
    }

    if (induced_subgraph_usecase.check_correctness) {
      auto h_subgraph_vertices = cugraph::test::to_host(handle, d_subgraph_vertices);

      auto [h_offsets, h_indices, h_weights] = cugraph::test::graph_to_host_csr(handle, graph_view);

      auto h_cugraph_subgraph_edgelist_majors =
        cugraph::test::to_host(handle, d_subgraph_edgelist_majors);
      auto h_cugraph_subgraph_edgelist_minors =
        cugraph::test::to_host(handle, d_subgraph_edgelist_minors);
      auto h_cugraph_subgraph_edgelist_weights =
        cugraph::test::to_host(handle, d_subgraph_edgelist_weights);
      auto h_cugraph_subgraph_edge_offsets =
        cugraph::test::to_host(handle, d_subgraph_edge_offsets);

      induced_subgraph_validate(h_offsets,
                                h_indices,
                                h_weights,
                                h_subgraph_offsets,
                                h_subgraph_vertices,
                                h_cugraph_subgraph_edgelist_majors,
                                h_cugraph_subgraph_edgelist_minors,
                                h_cugraph_subgraph_edgelist_weights,
                                h_cugraph_subgraph_edge_offsets);
    }
  }
};

using Tests_InducedSubgraph_File = Tests_InducedSubgraph<cugraph::test::File_Usecase>;
using Tests_InducedSubgraph_Rmat = Tests_InducedSubgraph<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations

TEST_P(Tests_InducedSubgraph_File, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_InducedSubgraph_File, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_InducedSubgraph_Rmat, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_InducedSubgraph_Rmat, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  karate_test,
  Tests_InducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{0}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{1}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{10}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{34}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{10, 0, 5}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{9, 3, 10}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{5, 12, 13}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  web_google_test,
  Tests_InducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{250, 130, 15}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{125, 300, 70}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  ljournal_2008_test,
  Tests_InducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{300, 20, 400}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{9130, 1200, 300}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  webbase_1M_test,
  Tests_InducedSubgraph_File,
  ::testing::Combine(
    ::testing::Values(InducedSubgraph_Usecase{std::vector<size_t>{700}, false, true},
                      InducedSubgraph_Usecase{std::vector<size_t>{500}, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

CUGRAPH_TEST_PROGRAM_MAIN()

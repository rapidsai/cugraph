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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#include <structure/induced_subgraph_validate.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<vertex_t>,
           std::vector<vertex_t>,
           std::optional<std::vector<weight_t>>,
           std::vector<size_t>>
extract_induced_subgraph_reference(std::vector<edge_t> const& offsets,
                                   std::vector<vertex_t> const& indices,
                                   std::optional<std::vector<weight_t>> const& weights,
                                   std::vector<size_t> const& subgraph_offsets,
                                   std::vector<vertex_t> const& subgraph_vertices,
                                   size_t num_vertices)
{
  std::vector<vertex_t> edgelist_majors{};
  std::vector<vertex_t> edgelist_minors{};
  auto edgelist_weights = weights ? std::make_optional<std::vector<weight_t>>(0) : std::nullopt;
  std::vector<size_t> subgraph_edge_offsets({0});

  for (size_t i = 0; i < (subgraph_offsets.size() - 1); ++i) {
    std::for_each(subgraph_vertices.data() + subgraph_offsets[i],
                  subgraph_vertices.data() + subgraph_offsets[i + 1],
                  [offsets,
                   indices,
                   weights,
                   subgraph_vertices,
                   subgraph_offsets,
                   &edgelist_majors,
                   &edgelist_minors,
                   &edgelist_weights,
                   i](auto v) {
                    auto first = offsets[v];
                    auto last  = offsets[v + 1];
                    for (auto j = first; j < last; ++j) {
                      if (std::binary_search(subgraph_vertices.data() + subgraph_offsets[i],
                                             subgraph_vertices.data() + subgraph_offsets[i + 1],
                                             indices[j])) {
                        edgelist_majors.push_back(v);
                        edgelist_minors.push_back(indices[j]);
                        if (weights) { (*edgelist_weights).push_back((*weights)[j]); }
                      }
                    }
                  });
    subgraph_edge_offsets.push_back(edgelist_majors.size());
  }

  return std::make_tuple(edgelist_majors, edgelist_minors, edgelist_weights, subgraph_edge_offsets);
}

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
    constexpr bool do_expensive_check{false};

    auto [induced_subgraph_usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, induced_subgraph_usecase.test_weighted, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

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
      hr_timer.start("Induced-subgraph");
    }

    auto [d_subgraph_edgelist_majors,
          d_subgraph_edgelist_minors,
          d_subgraph_edgelist_weights,
          d_subgraph_edge_offsets] =
      cugraph::extract_induced_subgraphs(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<size_t const>(d_subgraph_offsets.data(), d_subgraph_offsets.size()),
        raft::device_span<vertex_t const>(d_subgraph_vertices.data(), d_subgraph_vertices.size()),
        do_expensive_check);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (induced_subgraph_usecase.check_correctness) {
      auto h_subgraph_vertices = cugraph::test::to_host(handle, d_subgraph_vertices);

      auto [h_offsets, h_indices, h_weights] =
        cugraph::test::graph_to_host_csr(handle, graph_view, edge_weight_view);

      auto [h_reference_subgraph_edgelist_majors,
            h_reference_subgraph_edgelist_minors,
            h_reference_subgraph_edgelist_weights,
            h_reference_subgraph_edge_offsets] =
        extract_induced_subgraph_reference(h_offsets,
                                           h_indices,
                                           h_weights,
                                           h_subgraph_offsets,
                                           h_subgraph_vertices,
                                           graph_view.number_of_vertices());

      auto d_reference_subgraph_edgelist_majors =
        cugraph::test::to_device(handle, h_reference_subgraph_edgelist_majors);
      auto d_reference_subgraph_edgelist_minors =
        cugraph::test::to_device(handle, h_reference_subgraph_edgelist_minors);
      auto d_reference_subgraph_edgelist_weights =
        cugraph::test::to_device(handle, h_reference_subgraph_edgelist_weights);
      auto d_reference_subgraph_edge_offsets =
        cugraph::test::to_device(handle, h_reference_subgraph_edge_offsets);

      induced_subgraph_validate(handle,
                                d_subgraph_edgelist_majors,
                                d_subgraph_edgelist_minors,
                                d_subgraph_edgelist_weights,
                                d_subgraph_edge_offsets,
                                d_reference_subgraph_edgelist_majors,
                                d_reference_subgraph_edgelist_minors,
                                d_reference_subgraph_edgelist_weights,
                                d_reference_subgraph_edge_offsets);
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

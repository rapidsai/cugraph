/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

template <bool store_transposed, typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_reference(vertex_t const* p_src_vertices,
                vertex_t const* p_dst_vertices,
                std::optional<weight_t const*> p_edge_weights,
                vertex_t number_of_vertices,
                edge_t number_of_edges)
{
  std::vector<edge_t> offsets(number_of_vertices + 1, edge_t{0});
  std::vector<vertex_t> indices(number_of_edges, vertex_t{0});
  auto weights = p_edge_weights
                   ? std::make_optional<std::vector<weight_t>>(number_of_edges, weight_t{0.0})
                   : std::nullopt;

  for (edge_t i = 0; i < number_of_edges; ++i) {
    auto major = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    offsets[1 + major]++;
  }
  std::partial_sum(offsets.begin() + 1, offsets.end(), offsets.begin() + 1);

  for (edge_t i = 0; i < number_of_edges; ++i) {
    auto major           = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    auto minor           = store_transposed ? p_src_vertices[i] : p_dst_vertices[i];
    auto start           = offsets[major];
    auto degree          = offsets[major + 1] - start;
    auto idx             = indices[start + degree - 1]++;
    indices[start + idx] = minor;
    if (p_edge_weights) { (*weights)[start + idx] = (*p_edge_weights)[i]; }
  }

  return std::make_tuple(std::move(offsets), std::move(indices), std::move(weights));
}

struct Graph_Usecase {
  bool test_weighted{false};
  bool multi_graph{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_Graph : public ::testing::TestWithParam<std::tuple<Graph_Usecase, input_usecase_t>> {
 public:
  Tests_Graph() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(std::tuple<Graph_Usecase const&, input_usecase_t const&> const& param)
  {
    raft::handle_t handle{};
    auto [graph_usecase, input_usecase] = param;

    auto [d_srcs, d_dsts, d_weights, d_vertices, number_of_vertices, is_symmetric] =
      input_usecase
        .template construct_edgelist<vertex_t, edge_t, weight_t, store_transposed, false>(
          handle, graph_usecase.test_weighted);

    edge_t number_of_edges = static_cast<edge_t>(d_srcs.size());

    std::vector<vertex_t> h_srcs(number_of_edges);
    std::vector<vertex_t> h_dsts(number_of_edges);
    auto h_weights =
      d_weights ? std::make_optional<std::vector<weight_t>>(number_of_edges) : std::nullopt;

    raft::update_host(h_srcs.data(), d_srcs.data(), number_of_edges, handle.get_stream());
    raft::update_host(h_dsts.data(), d_dsts.data(), number_of_edges, handle.get_stream());
    if (h_weights) {
      raft::update_host(
        (*h_weights).data(), (*d_weights).data(), number_of_edges, handle.get_stream());
    }
    handle.sync_stream();

    auto [h_reference_offsets, h_reference_indices, h_reference_weights] =
      graph_reference<store_transposed>(
        h_srcs.data(),
        h_dsts.data(),
        h_weights ? std::optional<weight_t const*>{(*h_weights).data()} : std::nullopt,
        number_of_vertices,
        number_of_edges);

    cugraph::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_srcs.data(),
      d_dsts.data(),
      d_weights ? std::optional<weight_t const*>{(*d_weights).data()} : std::nullopt,
      number_of_edges};

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto graph = cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false>(
      handle,
      edgelist,
      cugraph::graph_meta_t<vertex_t, edge_t, false>{
        number_of_vertices,
        cugraph::graph_properties_t{is_symmetric, graph_usecase.multi_graph},
        std::nullopt},
      true);

    auto graph_view = graph.view();

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    ASSERT_EQ(graph_view.number_of_vertices(), number_of_vertices);
    ASSERT_EQ(graph_view.number_of_edges(), number_of_edges);

    if (graph_usecase.check_correctness) {
      std::vector<edge_t> h_cugraph_offsets(graph_view.number_of_vertices() + 1);
      std::vector<vertex_t> h_cugraph_indices(graph_view.number_of_edges());
      auto h_cugraph_weights =
        graph.is_weighted() ? std::optional<std::vector<weight_t>>(graph_view.number_of_edges())
                            : std::nullopt;

      raft::update_host(h_cugraph_offsets.data(),
                        graph_view.local_edge_partition_view().offsets(),
                        graph_view.number_of_vertices() + 1,
                        handle.get_stream());
      raft::update_host(h_cugraph_indices.data(),
                        graph_view.local_edge_partition_view().indices(),
                        graph_view.number_of_edges(),
                        handle.get_stream());
      if (h_cugraph_weights) {
        raft::update_host((*h_cugraph_weights).data(),
                          *(graph_view.local_edge_partition_view().weights()),
                          graph_view.number_of_edges(),
                          handle.get_stream());
      }

      handle.sync_stream();

      ASSERT_TRUE(std::equal(
        h_reference_offsets.begin(), h_reference_offsets.end(), h_cugraph_offsets.begin()))
        << "Graph compressed sparse format offsets do not match with the reference values.";
      ASSERT_EQ(h_reference_weights.has_value(), h_cugraph_weights.has_value());
      if (h_reference_weights) {
        ASSERT_EQ((*h_reference_weights).size(), (*h_cugraph_weights).size());
      }
      for (vertex_t i = 0; i < number_of_vertices; ++i) {
        auto start  = h_reference_offsets[i];
        auto degree = h_reference_offsets[i + 1] - start;
        if (graph_usecase.test_weighted) {
          std::vector<std::tuple<vertex_t, weight_t>> reference_pairs(degree);
          std::vector<std::tuple<vertex_t, weight_t>> cugraph_pairs(degree);
          for (edge_t j = 0; j < degree; ++j) {
            reference_pairs[j] =
              std::make_tuple(h_reference_indices[start + j], (*h_reference_weights)[start + j]);
            cugraph_pairs[j] =
              std::make_tuple(h_cugraph_indices[start + j], (*h_cugraph_weights)[start + j]);
          }
          std::sort(reference_pairs.begin(), reference_pairs.end());
          std::sort(cugraph_pairs.begin(), cugraph_pairs.end());
          ASSERT_TRUE(
            std::equal(reference_pairs.begin(), reference_pairs.end(), cugraph_pairs.begin()))
            << "Graph compressed sparse format indices & weights for vertex " << i
            << " do not match with the reference values.";
        } else {
          std::vector<vertex_t> reference_indices(h_reference_indices.begin() + start,
                                                  h_reference_indices.begin() + (start + degree));
          std::vector<vertex_t> cugraph_indices(h_cugraph_indices.begin() + start,
                                                h_cugraph_indices.begin() + (start + degree));
          std::sort(reference_indices.begin(), reference_indices.end());
          std::sort(cugraph_indices.begin(), cugraph_indices.end());
          ASSERT_TRUE(
            std::equal(reference_indices.begin(), reference_indices.end(), cugraph_indices.begin()))
            << "Graph compressed sparse format indices for vertex " << i
            << " do not match with the reference values.";
        }
      }
    }
  }
};

using Tests_Graph_File = Tests_Graph<cugraph::test::File_Usecase>;
using Tests_Graph_Rmat = Tests_Graph<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Graph_File, CheckStoreTransposedFalse_32_32_float)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedFalse_32_64_float)
{
  run_current_test<int32_t, int64_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedFalse_64_64_float)
{
  run_current_test<int64_t, int64_t, float, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedFalse_32_32_double)
{
  run_current_test<int32_t, int32_t, double, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedFalse_32_64_double)
{
  run_current_test<int32_t, int64_t, double, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedFalse_64_64_double)
{
  run_current_test<int64_t, int64_t, double, false>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedTrue_32_32_float)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedTrue_32_64_float)
{
  run_current_test<int32_t, int64_t, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedTrue_64_64_float)
{
  run_current_test<int64_t, int64_t, float, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedTrue_32_32_double)
{
  run_current_test<int32_t, int32_t, double, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedTrue_32_64_double)
{
  run_current_test<int32_t, int64_t, double, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_File, CheckStoreTransposedTrue_64_64_double)
{
  run_current_test<int64_t, int64_t, double, true>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedFalse_32_32_float)
{
  run_current_test<int32_t, int32_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedFalse_32_64_float)
{
  run_current_test<int32_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedFalse_64_64_float)
{
  run_current_test<int64_t, int64_t, float, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedFalse_32_32_double)
{
  run_current_test<int32_t, int32_t, double, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedFalse_32_64_double)
{
  run_current_test<int32_t, int64_t, double, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedFalse_64_64_double)
{
  run_current_test<int64_t, int64_t, double, false>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedTrue_32_32_float)
{
  run_current_test<int32_t, int32_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedTrue_32_64_float)
{
  run_current_test<int32_t, int64_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedTrue_64_64_float)
{
  run_current_test<int64_t, int64_t, float, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedTrue_32_32_double)
{
  run_current_test<int32_t, int32_t, double, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedTrue_32_64_double)
{
  run_current_test<int32_t, int64_t, double, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Graph_Rmat, CheckStoreTransposedTrue_64_64_double)
{
  run_current_test<int64_t, int64_t, double, true>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Graph_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Graph_Usecase{false}, Graph_Usecase{true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_Graph_Rmat,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(Graph_Usecase{false, true}, Graph_Usecase{true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  file_benchmark_test, /* note that the test filename can be overridden in benchmarking (with
                          --gtest_filter to select only the file_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one File_Usecase that differ only in filename
                          (to avoid running same benchmarks more than once) */
  Tests_Graph_File,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(Graph_Usecase{false, false, false}, Graph_Usecase{true, false, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test,
  Tests_Graph_Rmat,
  ::testing::Combine(
    // disable correctness checks
    ::testing::Values(Graph_Usecase{false, true, false}, Graph_Usecase{true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()

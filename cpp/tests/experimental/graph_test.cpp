/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <utilities/test_utilities.hpp>

#include <experimental/graph.hpp>
#include <experimental/graph_view.hpp>

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
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::vector<weight_t>> graph_reference(
  vertex_t const* p_src_vertices,
  vertex_t const* p_dst_vertices,
  weight_t const* p_edge_weights,
  vertex_t number_of_vertices,
  edge_t number_of_edges)
{
  std::vector<edge_t> offsets(number_of_vertices + 1, edge_t{0});
  std::vector<vertex_t> indices(number_of_edges, vertex_t{0});
  std::vector<weight_t> weights(p_edge_weights != nullptr ? number_of_edges : 0, weight_t{0.0});

  for (size_t i = 0; i < number_of_edges; ++i) {
    auto major = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    offsets[1 + major]++;
  }
  std::partial_sum(offsets.begin() + 1, offsets.end(), offsets.begin() + 1);

  for (size_t i = 0; i < number_of_edges; ++i) {
    auto major           = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    auto minor           = store_transposed ? p_src_vertices[i] : p_dst_vertices[i];
    auto start           = offsets[major];
    auto degree          = offsets[major + 1] - start;
    auto idx             = indices[start + degree - 1]++;
    indices[start + idx] = minor;
    if (p_edge_weights != nullptr) { weights[start + idx] = p_edge_weights[i]; }
  }

  return std::make_tuple(std::move(offsets), std::move(indices), std::move(weights));
}

typedef struct Graph_Usecase_t {
  std::string graph_file_full_path{};
  bool test_weighted{false};

  Graph_Usecase_t(std::string const& graph_file_path, bool test_weighted)
    : test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} Graph_Usecase;

class Tests_Graph : public ::testing::TestWithParam<Graph_Usecase> {
 public:
  Tests_Graph() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(Graph_Usecase const& configuration)
  {
    raft::handle_t handle{};

    rmm::device_uvector<vertex_t> d_rows(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_cols(0, handle.get_stream());
    rmm::device_uvector<weight_t> d_weights(0, handle.get_stream());
    vertex_t number_of_vertices{};
    bool is_symmetric{};
    std::tie(d_rows, d_cols, d_weights, number_of_vertices, is_symmetric) =
      cugraph::test::read_edgelist_from_matrix_market_file<vertex_t, weight_t>(
        handle, configuration.graph_file_full_path, configuration.test_weighted);
    edge_t number_of_edges = static_cast<edge_t>(d_rows.size());

    std::vector<vertex_t> h_rows(number_of_edges);
    std::vector<vertex_t> h_cols(number_of_edges);
    std::vector<weight_t> h_weights(configuration.test_weighted ? number_of_edges : edge_t{0});

    raft::update_host(h_rows.data(), d_rows.data(), number_of_edges, handle.get_stream());
    raft::update_host(h_cols.data(), d_cols.data(), number_of_edges, handle.get_stream());
    if (configuration.test_weighted) {
      raft::update_host(h_weights.data(), d_weights.data(), number_of_edges, handle.get_stream());
    }
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    std::vector<edge_t> h_reference_offsets{};
    std::vector<vertex_t> h_reference_indices{};
    std::vector<weight_t> h_reference_weights{};

    std::tie(h_reference_offsets, h_reference_indices, h_reference_weights) =
      graph_reference<store_transposed>(
        h_rows.data(),
        h_cols.data(),
        configuration.test_weighted ? h_weights.data() : static_cast<weight_t*>(nullptr),
        number_of_vertices,
        number_of_edges);

    cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_rows.data(),
      d_cols.data(),
      configuration.test_weighted ? d_weights.data() : nullptr,
      number_of_edges};

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto graph =
      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle,
        edgelist,
        number_of_vertices,
        cugraph::experimental::graph_properties_t{is_symmetric, false, configuration.test_weighted},
        false,
        true);

    auto graph_view = graph.view();

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    ASSERT_EQ(graph_view.get_number_of_vertices(), number_of_vertices);
    ASSERT_EQ(graph_view.get_number_of_edges(), number_of_edges);

    std::vector<edge_t> h_cugraph_offsets(graph_view.get_number_of_vertices() + 1);
    std::vector<vertex_t> h_cugraph_indices(graph_view.get_number_of_edges());
    std::vector<weight_t> h_cugraph_weights(
      configuration.test_weighted ? graph_view.get_number_of_edges() : 0);

    raft::update_host(h_cugraph_offsets.data(),
                      graph_view.offsets(),
                      graph_view.get_number_of_vertices() + 1,
                      handle.get_stream());
    raft::update_host(h_cugraph_indices.data(),
                      graph_view.indices(),
                      graph_view.get_number_of_edges(),
                      handle.get_stream());
    if (configuration.test_weighted) {
      raft::update_host(h_cugraph_weights.data(),
                        graph_view.weights(),
                        graph_view.get_number_of_edges(),
                        handle.get_stream());
    }

    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    ASSERT_TRUE(
      std::equal(h_reference_offsets.begin(), h_reference_offsets.end(), h_cugraph_offsets.begin()))
      << "Graph compressed sparse format offsets do not match with the reference values.";
    ASSERT_EQ(h_reference_weights.size(), h_cugraph_weights.size());
    for (vertex_t i = 0; i < number_of_vertices; ++i) {
      auto start  = h_reference_offsets[i];
      auto degree = h_reference_offsets[i + 1] - start;
      if (configuration.test_weighted) {
        std::vector<std::tuple<vertex_t, weight_t>> reference_pairs(degree);
        std::vector<std::tuple<vertex_t, weight_t>> cugraph_pairs(degree);
        for (edge_t j = 0; j < degree; ++j) {
          reference_pairs[j] =
            std::make_tuple(h_reference_indices[start + j], h_reference_weights[start + j]);
          cugraph_pairs[j] =
            std::make_tuple(h_cugraph_indices[start + j], h_cugraph_weights[start + j]);
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
};

TEST_P(Tests_Graph, CheckStoreTransposedFalse)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
  run_current_test<int32_t, int64_t, float, false>(GetParam());
  run_current_test<int64_t, int64_t, float, false>(GetParam());
  run_current_test<int32_t, int32_t, double, false>(GetParam());
  run_current_test<int32_t, int64_t, double, false>(GetParam());
  run_current_test<int64_t, int64_t, double, false>(GetParam());
}

TEST_P(Tests_Graph, CheckStoreTransposedTrue)
{
  run_current_test<int32_t, int32_t, float, true>(GetParam());
  run_current_test<int32_t, int64_t, float, true>(GetParam());
  run_current_test<int64_t, int64_t, float, true>(GetParam());
  run_current_test<int32_t, int32_t, double, true>(GetParam());
  run_current_test<int32_t, int64_t, double, true>(GetParam());
  run_current_test<int64_t, int64_t, double, true>(GetParam());
}

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_Graph,
                        ::testing::Values(Graph_Usecase("test/datasets/karate.mtx", false),
                                          Graph_Usecase("test/datasets/karate.mtx", true),
                                          Graph_Usecase("test/datasets/web-Google.mtx", false),
                                          Graph_Usecase("test/datasets/web-Google.mtx", true),
                                          Graph_Usecase("test/datasets/ljournal-2008.mtx", false),
                                          Graph_Usecase("test/datasets/ljournal-2008.mtx", true),
                                          Graph_Usecase("test/datasets/webbase-1M.mtx", false),
                                          Graph_Usecase("test/datasets/webbase-1M.mtx", true)));

CUGRAPH_TEST_PROGRAM_MAIN()

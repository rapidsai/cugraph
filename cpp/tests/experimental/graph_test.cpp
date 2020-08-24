/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <experimental/graph.hpp>
#include <experimental/graph_view.hpp>
#include <utilities/test_utilities.hpp>

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
    auto major   = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    offsets[1 + major]++;
  }
  std::partial_sum(offsets.begin() + 1, offsets.end(), offsets.begin() + 1);

  for (size_t i = 0; i < number_of_edges; ++i) {
    auto major   = store_transposed ? p_dst_vertices[i] : p_src_vertices[i];
    auto minor   = store_transposed ? p_src_vertices[i] : p_dst_vertices[i];
    auto start   = offsets[major];
    auto degree  = offsets[major + 1] - start;
    auto idx     = indices[start + degree - 1]++;
    indices[start + idx] = minor;
    if (p_edge_weights != nullptr) { weights[start + idx] = p_edge_weights[i]; }
  }

  return std::make_tuple(std::move(offsets), std::move(indices), std::move(weights));
}

typedef struct Graph_Usecase_t {
  std::string graph_file_path;
  std::string graph_file_full_path;
  bool test_weighted;

  Graph_Usecase_t(std::string const& graph_file_path, bool test_weighted)
    : graph_file_path(graph_file_path), test_weighted(test_weighted)
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
    MM_typecode mc{};
    vertex_t m{};
    vertex_t k{};
    edge_t nnz{};

    FILE* file = fopen(configuration.graph_file_full_path.c_str(), "r");
    ASSERT_NE(file, nullptr) << "fopen (" << configuration.graph_file_full_path << ") failure.";

    ASSERT_EQ(cugraph::test::mm_properties<int>(file, 1, &mc, &m, &k, &nnz), 0)
      << "could not read Matrix Market file properties\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    std::vector<vertex_t> h_rows(nnz, vertex_t{0});
    std::vector<vertex_t> h_cols(nnz, vertex_t{0});
    std::vector<weight_t> h_weights(nnz, weight_t{0.0});

    ASSERT_EQ((cugraph::test::mm_to_coo<vertex_t, weight_t>(
                file, 1, nnz, h_rows.data(), h_cols.data(), h_weights.data(), nullptr)),
              0)
      << "could not read matrix data\n";
    ASSERT_EQ(fclose(file), 0);

    std::vector<edge_t> h_reference_offsets{};
    std::vector<vertex_t> h_reference_indices{};
    std::vector<weight_t> h_reference_weights{};

    std::tie(h_reference_offsets, h_reference_indices, h_reference_weights) =
      graph_reference<store_transposed>(h_rows.data(),
                                        h_cols.data(),
                                        configuration.test_weighted ? h_weights.data() : nullptr,
                                        m,
                                        nnz);

    raft::handle_t handle{};

    rmm::device_uvector<vertex_t> d_rows(nnz, handle.get_stream());
    rmm::device_uvector<vertex_t> d_cols(nnz, handle.get_stream());
    rmm::device_uvector<weight_t> d_weights(configuration.test_weighted ? nnz : 0,
                                            handle.get_stream());

    raft::update_device(d_rows.data(), h_rows.data(), h_rows.size(), handle.get_stream());
    raft::update_device(d_cols.data(), h_cols.data(), h_cols.size(), handle.get_stream());
    if (configuration.test_weighted) {
      raft::update_device(
        d_weights.data(), h_weights.data(), h_weights.size(), handle.get_stream());
    }

    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    cugraph::experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
      d_rows.data(), d_cols.data(), configuration.test_weighted ? d_weights.data() : nullptr, nnz};

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto graph =
      cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, edgelist, m, mm_is_symmetric(mc), false, configuration.test_weighted, false, true);

    auto graph_view = graph.view();

    ASSERT_EQ(graph_view.get_number_of_vertices(), m);
    ASSERT_EQ(graph_view.get_number_of_edges(), nnz);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

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
    for (vertex_t i = 0; i < m; ++i) {
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

// FIXME: add tests for type combinations
TEST_P(Tests_Graph, CheckInt32Int32FloatFalse)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
}

// FIXME: add tests for type combinations
TEST_P(Tests_Graph, CheckInt32Int32FloatTrue)
{
  run_current_test<int32_t, int32_t, float, true>(GetParam());
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

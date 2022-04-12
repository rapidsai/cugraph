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
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

template <typename vertex_t, typename edge_t>
void degree_reference(edge_t const* offsets,
                      vertex_t const* indices,
                      edge_t* degrees,
                      vertex_t num_vertices,
                      bool major)
{
  if (major) {
    std::adjacent_difference(offsets + 1, offsets + num_vertices + 1, degrees);
  } else {
    std::fill(degrees, degrees + num_vertices, edge_t{0});
    for (vertex_t i = 0; i < num_vertices; ++i) {
      for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
        auto nbr = indices[j];
        ++degrees[nbr];
      }
    }
  }

  return;
}

typedef struct Degree_Usecase_t {
  std::string graph_file_full_path{};

  Degree_Usecase_t(std::string const& graph_file_path)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} Degree_Usecase;

class Tests_Degree : public ::testing::TestWithParam<Degree_Usecase> {
 public:
  Tests_Degree() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(Degree_Usecase const& configuration)
  {
    raft::handle_t handle{};

    cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> graph(handle);
    std::tie(graph, std::ignore) = cugraph::test::
      read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, configuration.graph_file_full_path, false, false);
    auto graph_view = graph.view();

    std::vector<edge_t> h_offsets(graph_view.number_of_vertices() + 1);
    std::vector<vertex_t> h_indices(graph_view.number_of_edges());
    raft::update_host(h_offsets.data(),
                      graph_view.local_edge_partition_view().offsets(),
                      graph_view.number_of_vertices() + 1,
                      handle.get_stream());
    raft::update_host(h_indices.data(),
                      graph_view.local_edge_partition_view().indices(),
                      graph_view.number_of_edges(),
                      handle.get_stream());
    handle.sync_stream();

    std::vector<edge_t> h_reference_in_degrees(graph_view.number_of_vertices());
    std::vector<edge_t> h_reference_out_degrees(graph_view.number_of_vertices());

    degree_reference(h_offsets.data(),
                     h_indices.data(),
                     h_reference_in_degrees.data(),
                     graph_view.number_of_vertices(),
                     store_transposed);

    degree_reference(h_offsets.data(),
                     h_indices.data(),
                     h_reference_out_degrees.data(),
                     graph_view.number_of_vertices(),
                     !store_transposed);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    auto d_in_degrees  = graph_view.compute_in_degrees(handle);
    auto d_out_degrees = graph_view.compute_out_degrees(handle);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<edge_t> h_cugraph_in_degrees(graph_view.number_of_vertices());
    std::vector<edge_t> h_cugraph_out_degrees(graph_view.number_of_vertices());

    raft::update_host(
      h_cugraph_in_degrees.data(), d_in_degrees.data(), d_in_degrees.size(), handle.get_stream());
    raft::update_host(h_cugraph_out_degrees.data(),
                      d_out_degrees.data(),
                      d_out_degrees.size(),
                      handle.get_stream());
    handle.sync_stream();

    ASSERT_TRUE(std::equal(
      h_reference_in_degrees.begin(), h_reference_in_degrees.end(), h_cugraph_in_degrees.begin()))
      << "In-degree values do not match with the reference values.";
    ASSERT_TRUE(std::equal(h_reference_out_degrees.begin(),
                           h_reference_out_degrees.end(),
                           h_cugraph_out_degrees.begin()))
      << "Out-degree values do not match with the reference values.";
  }
};

// FIXME: add tests for type combinations

TEST_P(Tests_Degree, CheckInt32Int32FloatTransposeFalse)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
}

TEST_P(Tests_Degree, CheckInt32Int32FloatTransposeTrue)
{
  run_current_test<int32_t, int32_t, float, true>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(simple_test,
                         Tests_Degree,
                         ::testing::Values(Degree_Usecase("test/datasets/karate.mtx"),
                                           Degree_Usecase("test/datasets/web-Google.mtx"),
                                           Degree_Usecase("test/datasets/ljournal-2008.mtx"),
                                           Degree_Usecase("test/datasets/webbase-1M.mtx")));

CUGRAPH_TEST_PROGRAM_MAIN()

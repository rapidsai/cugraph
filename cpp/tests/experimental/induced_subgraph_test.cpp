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

#include <cugraph/experimental/graph.hpp>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/experimental/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<vertex_t>, std::vector<vertex_t>, std::vector<weight_t>, std::vector<size_t>>
extract_induced_subgraph_reference(edge_t const* offsets,
                                   vertex_t const* indices,
                                   weight_t const* weights,
                                   size_t const* subgraph_offsets,
                                   vertex_t const* subgraph_vertices,
                                   vertex_t num_vertices,
                                   size_t num_subgraphs)
{
  std::vector<vertex_t> edgelist_majors{};
  std::vector<vertex_t> edgelist_minors{};
  std::vector<weight_t> edgelist_weights{};
  std::vector<size_t> subgraph_edge_offsets{0};

  for (size_t i = 0; i < num_subgraphs; ++i) {
    std::for_each(subgraph_vertices + subgraph_offsets[i],
                  subgraph_vertices + subgraph_offsets[i + 1],
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
                      if (std::binary_search(subgraph_vertices + subgraph_offsets[i],
                                             subgraph_vertices + subgraph_offsets[i + 1],
                                             indices[j])) {
                        edgelist_majors.push_back(v);
                        edgelist_minors.push_back(indices[j]);
                        if (weights != nullptr) { edgelist_weights.push_back(weights[j]); }
                      }
                    }
                  });
    subgraph_edge_offsets.push_back(edgelist_majors.size());
  }

  return std::make_tuple(edgelist_majors, edgelist_minors, edgelist_weights, subgraph_edge_offsets);
}

typedef struct InducedSubgraph_Usecase_t {
  std::string graph_file_full_path{};
  std::vector<size_t> subgraph_sizes{};
  bool test_weighted{false};

  InducedSubgraph_Usecase_t(std::string const& graph_file_path,
                            std::vector<size_t> const& subgraph_sizes,
                            bool test_weighted)
    : subgraph_sizes(subgraph_sizes), test_weighted(test_weighted)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} InducedSubgraph_Usecase;

class Tests_InducedSubgraph : public ::testing::TestWithParam<InducedSubgraph_Usecase> {
 public:
  Tests_InducedSubgraph() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
  void run_current_test(InducedSubgraph_Usecase const& configuration)
  {
    raft::handle_t handle{};

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, false> graph(
      handle);
    std::tie(graph, std::ignore) = cugraph::test::
      read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed, false>(
        handle, configuration.graph_file_full_path, configuration.test_weighted, false);
    auto graph_view = graph.view();

    std::vector<edge_t> h_offsets(graph_view.get_number_of_vertices() + 1);
    std::vector<vertex_t> h_indices(graph_view.get_number_of_edges());
    std::vector<weight_t> h_weights{};
    raft::update_host(h_offsets.data(),
                      graph_view.offsets(),
                      graph_view.get_number_of_vertices() + 1,
                      handle.get_stream());
    raft::update_host(h_indices.data(),
                      graph_view.indices(),
                      graph_view.get_number_of_edges(),
                      handle.get_stream());
    if (graph_view.is_weighted()) {
      h_weights.assign(graph_view.get_number_of_edges(), weight_t{0.0});
      raft::update_host(h_weights.data(),
                        graph_view.weights(),
                        graph_view.get_number_of_edges(),
                        handle.get_stream());
    }
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    std::vector<size_t> h_subgraph_offsets(configuration.subgraph_sizes.size() + 1, 0);
    std::partial_sum(configuration.subgraph_sizes.begin(),
                     configuration.subgraph_sizes.end(),
                     h_subgraph_offsets.begin() + 1);
    std::vector<vertex_t> h_subgraph_vertices(
      h_subgraph_offsets.back(), cugraph::experimental::invalid_vertex_id<vertex_t>::value);
    std::default_random_engine generator{};
    std::uniform_int_distribution<vertex_t> distribution{0,
                                                         graph_view.get_number_of_vertices() - 1};

    for (size_t i = 0; i < configuration.subgraph_sizes.size(); ++i) {
      auto start = h_subgraph_offsets[i];
      auto last  = h_subgraph_offsets[i + 1];
      ASSERT_TRUE(last - start <= graph_view.get_number_of_vertices()) << "Invalid subgraph size.";
      // this is inefficient if last - start << graph_view.get_number_of_vertices() but this is for
      // the test puspose only and the time & memory cost is only linear to
      // graph_view.get_number_of_vertices(), so this may not matter.
      std::vector<vertex_t> vertices(graph_view.get_number_of_vertices());
      std::iota(vertices.begin(), vertices.end(), vertex_t{0});
      std::random_shuffle(vertices.begin(), vertices.end());
      std::copy(
        vertices.begin(), vertices.begin() + (last - start), h_subgraph_vertices.begin() + start);
      std::sort(h_subgraph_vertices.begin() + start, h_subgraph_vertices.begin() + last);
    }

    rmm::device_uvector<size_t> d_subgraph_offsets(h_subgraph_offsets.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_subgraph_vertices(h_subgraph_vertices.size(),
                                                      handle.get_stream());
    raft::update_device(d_subgraph_offsets.data(),
                        h_subgraph_offsets.data(),
                        h_subgraph_offsets.size(),
                        handle.get_stream());
    raft::update_device(d_subgraph_vertices.data(),
                        h_subgraph_vertices.data(),
                        h_subgraph_vertices.size(),
                        handle.get_stream());

    std::vector<vertex_t> h_reference_subgraph_edgelist_majors{};
    std::vector<vertex_t> h_reference_subgraph_edgelist_minors{};
    std::vector<weight_t> h_reference_subgraph_edgelist_weights{};
    std::vector<size_t> h_reference_subgraph_edge_offsets{};
    std::tie(h_reference_subgraph_edgelist_majors,
             h_reference_subgraph_edgelist_minors,
             h_reference_subgraph_edgelist_weights,
             h_reference_subgraph_edge_offsets) =
      extract_induced_subgraph_reference(
        h_offsets.data(),
        h_indices.data(),
        h_weights.size() > 0 ? h_weights.data() : static_cast<weight_t*>(nullptr),
        h_subgraph_offsets.data(),
        h_subgraph_vertices.data(),
        graph_view.get_number_of_vertices(),
        configuration.subgraph_sizes.size());

    rmm::device_uvector<vertex_t> d_subgraph_edgelist_majors(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_subgraph_edgelist_minors(0, handle.get_stream());
    rmm::device_uvector<weight_t> d_subgraph_edgelist_weights(0, handle.get_stream());
    rmm::device_uvector<size_t> d_subgraph_edge_offsets(0, handle.get_stream());

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    // FIXME: turn-off do_expensive_check once verified.
    std::tie(d_subgraph_edgelist_majors,
             d_subgraph_edgelist_minors,
             d_subgraph_edgelist_weights,
             d_subgraph_edge_offsets) =
      cugraph::experimental::extract_induced_subgraphs(handle,
                                                       graph_view,
                                                       d_subgraph_offsets.data(),
                                                       d_subgraph_vertices.data(),
                                                       configuration.subgraph_sizes.size(),
                                                       true);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<vertex_t> h_cugraph_subgraph_edgelist_majors(d_subgraph_edgelist_majors.size());
    std::vector<vertex_t> h_cugraph_subgraph_edgelist_minors(d_subgraph_edgelist_minors.size());
    std::vector<weight_t> h_cugraph_subgraph_edgelist_weights(d_subgraph_edgelist_weights.size());
    std::vector<size_t> h_cugraph_subgraph_edge_offsets(d_subgraph_edge_offsets.size());

    raft::update_host(h_cugraph_subgraph_edgelist_majors.data(),
                      d_subgraph_edgelist_majors.data(),
                      d_subgraph_edgelist_majors.size(),
                      handle.get_stream());
    raft::update_host(h_cugraph_subgraph_edgelist_minors.data(),
                      d_subgraph_edgelist_minors.data(),
                      d_subgraph_edgelist_minors.size(),
                      handle.get_stream());
    if (configuration.test_weighted) {
      raft::update_host(h_cugraph_subgraph_edgelist_weights.data(),
                        d_subgraph_edgelist_weights.data(),
                        d_subgraph_edgelist_weights.size(),
                        handle.get_stream());
    }
    raft::update_host(h_cugraph_subgraph_edge_offsets.data(),
                      d_subgraph_edge_offsets.data(),
                      d_subgraph_edge_offsets.size(),
                      handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    ASSERT_TRUE(h_reference_subgraph_edge_offsets.size() == h_cugraph_subgraph_edge_offsets.size())
      << "Returned subgraph edge offset vector has an invalid size.";
    ASSERT_TRUE(std::equal(h_reference_subgraph_edge_offsets.begin(),
                           h_reference_subgraph_edge_offsets.end(),
                           h_cugraph_subgraph_edge_offsets.begin()))
      << "Returned subgraph edge offset values do not match with the reference values.";

    for (size_t i = 0; i < configuration.subgraph_sizes.size(); ++i) {
      auto start = h_reference_subgraph_edge_offsets[i];
      auto last  = h_reference_subgraph_edge_offsets[i + 1];
      if (configuration.test_weighted) {
        std::vector<std::tuple<vertex_t, vertex_t, weight_t>> reference_tuples(last - start);
        std::vector<std::tuple<vertex_t, vertex_t, weight_t>> cugraph_tuples(last - start);
        for (auto j = start; j < last; ++j) {
          reference_tuples[j - start] = std::make_tuple(h_reference_subgraph_edgelist_majors[j],
                                                        h_reference_subgraph_edgelist_minors[j],
                                                        h_reference_subgraph_edgelist_weights[j]);
          cugraph_tuples[j - start]   = std::make_tuple(h_cugraph_subgraph_edgelist_majors[j],
                                                      h_cugraph_subgraph_edgelist_minors[j],
                                                      h_cugraph_subgraph_edgelist_weights[j]);
        }
        ASSERT_TRUE(
          std::equal(reference_tuples.begin(), reference_tuples.end(), cugraph_tuples.begin()))
          << "Extracted subgraph edges do not match with the edges extracted by the reference "
             "implementation.";
      } else {
        std::vector<std::tuple<vertex_t, vertex_t>> reference_tuples(last - start);
        std::vector<std::tuple<vertex_t, vertex_t>> cugraph_tuples(last - start);
        for (auto j = start; j < last; ++j) {
          reference_tuples[j - start] = std::make_tuple(h_reference_subgraph_edgelist_majors[j],
                                                        h_reference_subgraph_edgelist_minors[j]);
          cugraph_tuples[j - start]   = std::make_tuple(h_cugraph_subgraph_edgelist_majors[j],
                                                      h_cugraph_subgraph_edgelist_minors[j]);
        }
        ASSERT_TRUE(
          std::equal(reference_tuples.begin(), reference_tuples.end(), cugraph_tuples.begin()))
          << "Extracted subgraph edges do not match with the edges extracted by the reference "
             "implementation.";
      }
    }
  }
};

// FIXME: add tests for type combinations

TEST_P(Tests_InducedSubgraph, CheckInt32Int32FloatTransposed)
{
  run_current_test<int32_t, int32_t, float, true>(GetParam());
}

TEST_P(Tests_InducedSubgraph, CheckInt32Int32FloatUntransposed)
{
  run_current_test<int32_t, int32_t, float, false>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  simple_test,
  Tests_InducedSubgraph,
  ::testing::Values(
    InducedSubgraph_Usecase("test/datasets/karate.mtx", std::vector<size_t>{0}, false),
    InducedSubgraph_Usecase("test/datasets/karate.mtx", std::vector<size_t>{1}, false),
    InducedSubgraph_Usecase("test/datasets/karate.mtx", std::vector<size_t>{10}, false),
    InducedSubgraph_Usecase("test/datasets/karate.mtx", std::vector<size_t>{34}, false),
    InducedSubgraph_Usecase("test/datasets/karate.mtx", std::vector<size_t>{10, 0, 5}, false),
    InducedSubgraph_Usecase("test/datasets/karate.mtx", std::vector<size_t>{9, 3, 10}, false),
    InducedSubgraph_Usecase("test/datasets/karate.mtx", std::vector<size_t>{5, 12, 13}, true),
    InducedSubgraph_Usecase("test/datasets/web-Google.mtx",
                            std::vector<size_t>{250, 130, 15},
                            false),
    InducedSubgraph_Usecase("test/datasets/web-Google.mtx",
                            std::vector<size_t>{125, 300, 70},
                            true),
    InducedSubgraph_Usecase("test/datasets/ljournal-2008.mtx",
                            std::vector<size_t>{300, 20, 400},
                            false),
    InducedSubgraph_Usecase("test/datasets/ljournal-2008.mtx",
                            std::vector<size_t>{9130, 1200, 300},
                            true),
    InducedSubgraph_Usecase("test/datasets/webbase-1M.mtx", std::vector<size_t>{700}, false),
    InducedSubgraph_Usecase("test/datasets/webbase-1M.mtx", std::vector<size_t>{500}, true)));

CUGRAPH_TEST_PROGRAM_MAIN()

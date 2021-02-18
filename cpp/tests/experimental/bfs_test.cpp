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
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <experimental/graph.hpp>
#include <experimental/graph_view.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <iterator>
#include <limits>
#include <vector>

template <typename vertex_t, typename edge_t>
void bfs_reference(edge_t const* offsets,
                   vertex_t const* indices,
                   vertex_t* distances,
                   vertex_t* predecessors,
                   vertex_t num_vertices,
                   vertex_t source,
                   vertex_t depth_limit = std::numeric_limits<vertex_t>::max())
{
  vertex_t depth{0};

  std::fill(distances, distances + num_vertices, std::numeric_limits<vertex_t>::max());
  std::fill(predecessors, predecessors + num_vertices, cugraph::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = depth;
  std::vector<vertex_t> cur_frontier_rows{source};
  std::vector<vertex_t> new_frontier_rows{};

  while (cur_frontier_rows.size() > 0) {
    for (auto const row : cur_frontier_rows) {
      auto nbr_offset_first = *(offsets + row);
      auto nbr_offset_last  = *(offsets + row + 1);
      for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
        auto nbr = *(indices + nbr_offset);
        if (*(distances + nbr) == std::numeric_limits<vertex_t>::max()) {
          *(distances + nbr)    = depth + 1;
          *(predecessors + nbr) = row;
          new_frontier_rows.push_back(nbr);
        }
      }
    }
    std::swap(cur_frontier_rows, new_frontier_rows);
    new_frontier_rows.clear();
    ++depth;
    if (depth >= depth_limit) { break; }
  }

  return;
}

typedef struct BFS_Usecase_t {
  std::string graph_file_full_path{};
  size_t source{false};

  BFS_Usecase_t(std::string const& graph_file_path, size_t source) : source(source)
  {
    if ((graph_file_path.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path;
    } else {
      graph_file_full_path = graph_file_path;
    }
  };
} BFS_Usecase;

class Tests_BFS : public ::testing::TestWithParam<BFS_Usecase> {
 public:
  Tests_BFS() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(BFS_Usecase const& configuration)
  {
    using weight_t = float;

    raft::handle_t handle{};

    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    std::tie(graph, std::ignore) =
      cugraph::test::read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, false, false>(
        handle, configuration.graph_file_full_path, false, false);
    auto graph_view = graph.view();

    std::vector<edge_t> h_offsets(graph_view.get_number_of_vertices() + 1);
    std::vector<vertex_t> h_indices(graph_view.get_number_of_edges());
    raft::update_host(h_offsets.data(),
                      graph_view.offsets(),
                      graph_view.get_number_of_vertices() + 1,
                      handle.get_stream());
    raft::update_host(h_indices.data(),
                      graph_view.indices(),
                      graph_view.get_number_of_edges(),
                      handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    ASSERT_TRUE(configuration.source >= 0 &&
                configuration.source <= graph_view.get_number_of_vertices())
      << "Starting sources should be >= 0 and"
      << " less than the number of vertices in the graph.";

    std::vector<vertex_t> h_reference_distances(graph_view.get_number_of_vertices());
    std::vector<vertex_t> h_reference_predecessors(graph_view.get_number_of_vertices());

    bfs_reference(h_offsets.data(),
                  h_indices.data(),
                  h_reference_distances.data(),
                  h_reference_predecessors.data(),
                  graph_view.get_number_of_vertices(),
                  static_cast<vertex_t>(configuration.source),
                  std::numeric_limits<vertex_t>::max());

    rmm::device_uvector<vertex_t> d_distances(graph_view.get_number_of_vertices(),
                                              handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.get_number_of_vertices(),
                                                 handle.get_stream());

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    cugraph::experimental::bfs(handle,
                               graph_view,
                               d_distances.begin(),
                               d_predecessors.begin(),
                               static_cast<vertex_t>(configuration.source),
                               false,
                               std::numeric_limits<vertex_t>::max(),
                               false);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<vertex_t> h_cugraph_distances(graph_view.get_number_of_vertices());
    std::vector<vertex_t> h_cugraph_predecessors(graph_view.get_number_of_vertices());

    raft::update_host(
      h_cugraph_distances.data(), d_distances.data(), d_distances.size(), handle.get_stream());
    raft::update_host(h_cugraph_predecessors.data(),
                      d_predecessors.data(),
                      d_predecessors.size(),
                      handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));

    ASSERT_TRUE(std::equal(
      h_reference_distances.begin(), h_reference_distances.end(), h_cugraph_distances.begin()))
      << "distances do not match with the reference values.";

    for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
      auto i = std::distance(h_cugraph_predecessors.begin(), it);
      if (*it == cugraph::invalid_vertex_id<vertex_t>::value) {
        ASSERT_TRUE(h_reference_predecessors[i] == *it)
          << "vertex reachability do not match with the reference.";
      } else {
        ASSERT_TRUE(h_reference_distances[*it] + 1 == h_reference_distances[i])
          << "distance to this vertex != distance to the predecessor vertex + 1.";
        bool found{false};
        for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
          if (h_indices[j] == i) {
            found = true;
            break;
          }
        }
        ASSERT_TRUE(found) << "no edge from the predecessor vertex to this vertex.";
      }
    }
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_BFS, CheckInt32Int32) { run_current_test<int32_t, int32_t>(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_BFS,
                        ::testing::Values(BFS_Usecase("test/datasets/karate.mtx", 0),
                                          BFS_Usecase("test/datasets/polbooks.mtx", 0),
                                          BFS_Usecase("test/datasets/netscience.mtx", 0),
                                          BFS_Usecase("test/datasets/netscience.mtx", 100),
                                          BFS_Usecase("test/datasets/wiki2003.mtx", 1000),
                                          BFS_Usecase("test/datasets/wiki-Talk.mtx", 1000)));

CUGRAPH_TEST_PROGRAM_MAIN()

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

#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <graph.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <iterator>
#include <limits>
#include <vector>

template <typename vertex_t, typename edge_t>
void bfs_reference(edge_t* offsets,
                   vertex_t* indices,
                   vertex_t* distances,
                   vertex_t* predecessors,
                   vertex_t num_vertices,
                   vertex_t source,
                   vertex_t depth_limit = std::numeric_limits<vertex_t>::max())
{
  vertex_t depth{0};

  std::fill(distances, distances + num_vertices, std::numeric_limits<vertex_t>::max());
  std::fill(predecessors,
            predecessors + num_vertices,
            cugraph::experimental::invalid_vertex_id<vertex_t>::value);

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
  std::string graph_file_path_;
  std::string graph_file_full_path_;
  size_t source_;

  BFS_Usecase_t(std::string const& graph_file_path, size_t source)
    : graph_file_path_(graph_file_path), source_(source)
  {
    if ((graph_file_path_.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path_ = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path_;
    } else {
      graph_file_full_path_ = graph_file_path_;
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
    // FIXME: directed is a misnomer.
    bool directed{false};
    auto p_csr_graph =
      cugraph::test::generate_graph_csr_from_mm<vertex_t, edge_t, float /* weight_t, dummy */>(
        directed, configuration.graph_file_full_path_);
    auto csr_graph_view = p_csr_graph->view();
    // FIXME: this shouldn't be necessary
    csr_graph_view.prop.directed = directed;

    ASSERT_TRUE(configuration.source_ >= 0 &&
                configuration.source_ <= csr_graph_view.number_of_vertices)
      << "Starting sources should be >= 0 and"
      << " less than the number of vertices in the graph.";

    std::vector<edge_t> h_offsets(csr_graph_view.number_of_vertices + 1);
    std::vector<vertex_t> h_indices(csr_graph_view.number_of_edges);
    std::vector<vertex_t> h_reference_distances(csr_graph_view.number_of_vertices);
    std::vector<vertex_t> h_reference_predecessors(csr_graph_view.number_of_vertices);

    CUDA_TRY(cudaMemcpy(h_offsets.data(),
                        csr_graph_view.offsets,
                        sizeof(edge_t) * h_offsets.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_indices.data(),
                        csr_graph_view.indices,
                        sizeof(vertex_t) * h_indices.size(),
                        cudaMemcpyDeviceToHost));

    bfs_reference(h_offsets.data(),
                  h_indices.data(),
                  h_reference_distances.data(),
                  h_reference_predecessors.data(),
                  csr_graph_view.number_of_vertices,
                  static_cast<vertex_t>(configuration.source_),
                  std::numeric_limits<vertex_t>::max());

    raft::handle_t handle{};

    rmm::device_uvector<vertex_t> d_distances(csr_graph_view.number_of_vertices,
                                              handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(csr_graph_view.number_of_vertices,
                                                 handle.get_stream());

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    cugraph::experimental::bfs(handle,
                               csr_graph_view,
                               d_distances.begin(),
                               d_predecessors.begin(),
                               static_cast<vertex_t>(configuration.source_),
                               false,
                               std::numeric_limits<vertex_t>::max(),
                               false);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<vertex_t> h_cugraph_distances(csr_graph_view.number_of_vertices);
    std::vector<vertex_t> h_cugraph_predecessors(csr_graph_view.number_of_vertices);

    CUDA_TRY(cudaMemcpy(h_cugraph_distances.data(),
                        d_distances.data(),
                        sizeof(vertex_t) * d_distances.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_cugraph_predecessors.data(),
                        d_predecessors.data(),
                        sizeof(vertex_t) * d_predecessors.size(),
                        cudaMemcpyDeviceToHost));

    ASSERT_TRUE(std::equal(
      h_reference_distances.begin(), h_reference_distances.end(), h_cugraph_distances.begin()))
      << "distances do not match with the reference values.";

    for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
      auto i = std::distance(h_cugraph_predecessors.begin(), it);
      if (*it == cugraph::experimental::invalid_vertex_id<vertex_t>::value) {
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

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}

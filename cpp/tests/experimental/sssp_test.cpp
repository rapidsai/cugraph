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
#include <queue>
#include <tuple>
#include <vector>

// Dijkstra's algorithm
template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_reference(edge_t* offsets,
                    vertex_t* indices,
                    weight_t* weights,
                    weight_t* distances,
                    vertex_t* predecessors,
                    vertex_t num_vertices,
                    vertex_t source,
                    weight_t cutoff = std::numeric_limits<weight_t>::max())
{
  using queue_iterm_t = std::tuple<weight_t, vertex_t>;

  std::fill(distances, distances + num_vertices, std::numeric_limits<weight_t>::max());
  std::fill(predecessors,
            predecessors + num_vertices,
            cugraph::experimental::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = static_cast<weight_t>(0.0);
  std::priority_queue<queue_iterm_t, std::vector<queue_iterm_t>, std::greater<queue_iterm_t>>
    queue{};
  queue.push(std::make_tuple(static_cast<weight_t>(0.0), source));

  while (queue.size() > 0) {
    weight_t distance{};
    vertex_t row{};
    std::tie(distance, row) = queue.top();
    queue.pop();
    if (distance > *(distances + row)) { continue; }
    auto nbr_offsets     = *(offsets + row);
    auto nbr_offset_last = *(offsets + row + 1);
    for (auto nbr_offset = nbr_offsets; nbr_offset != nbr_offset_last; ++nbr_offset) {
      auto nbr          = *(indices + nbr_offset);
      auto new_distance = distance + *(weights + nbr_offset);
      auto threshold    = std::min(*(distances + nbr), cutoff);
      if (new_distance < threshold) {
        *(distances + nbr)    = new_distance;
        *(predecessors + nbr) = row;
        queue.push(std::make_tuple(new_distance, nbr));
      }
    }
  }

  return;
}

typedef struct SSSP_Usecase_t {
  std::string graph_file_path_;
  std::string graph_file_full_path_;
  size_t source_;

  SSSP_Usecase_t(std::string const& graph_file_path, size_t source)
    : graph_file_path_(graph_file_path), source_(source)
  {
    if ((graph_file_path_.length() > 0) && (graph_file_path[0] != '/')) {
      graph_file_full_path_ = cugraph::test::get_rapids_dataset_root_dir() + "/" + graph_file_path_;
    } else {
      graph_file_full_path_ = graph_file_path_;
    }
  };
} SSSP_Usecase;

class Tests_SSSP : public ::testing::TestWithParam<SSSP_Usecase> {
 public:
  Tests_SSSP() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(SSSP_Usecase const& configuration)
  {
    // FIXME: directed is a misnomer.
    bool directed{false};
    auto p_csr_graph =
      cugraph::test::generate_graph_csr_from_mm<vertex_t, edge_t, weight_t>(
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
    std::vector<weight_t> h_weights(csr_graph_view.number_of_edges);
    std::vector<weight_t> h_reference_distances(csr_graph_view.number_of_vertices);
    std::vector<vertex_t> h_reference_predecessors(csr_graph_view.number_of_vertices);

    CUDA_TRY(cudaMemcpy(h_offsets.data(),
                        csr_graph_view.offsets,
                        sizeof(edge_t) * h_offsets.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_indices.data(),
                        csr_graph_view.indices,
                        sizeof(vertex_t) * h_indices.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_weights.data(),
                        csr_graph_view.edge_data,
                        sizeof(weight_t) * h_weights.size(),
                        cudaMemcpyDeviceToHost));

    sssp_reference(h_offsets.data(),
                   h_indices.data(),
                   h_weights.data(),
                   h_reference_distances.data(),
                   h_reference_predecessors.data(),
                   csr_graph_view.number_of_vertices,
                   static_cast<vertex_t>(configuration.source_));

    raft::handle_t handle{};

    rmm::device_uvector<weight_t> d_distances(csr_graph_view.number_of_vertices,
                                              handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(csr_graph_view.number_of_vertices,
                                                 handle.get_stream());

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    cugraph::experimental::sssp(handle,
                                csr_graph_view,
                                d_distances.begin(),
                                d_predecessors.begin(),
                                static_cast<vertex_t>(configuration.source_),
                                std::numeric_limits<weight_t>::max(),
                                false);

    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::vector<weight_t> h_cugraph_distances(csr_graph_view.number_of_vertices);
    std::vector<vertex_t> h_cugraph_predecessors(csr_graph_view.number_of_vertices);

    CUDA_TRY(cudaMemcpy(h_cugraph_distances.data(),
                        d_distances.data(),
                        sizeof(weight_t) * d_distances.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_cugraph_predecessors.data(),
                        d_predecessors.data(),
                        sizeof(vertex_t) * d_predecessors.size(),
                        cudaMemcpyDeviceToHost));

    auto max_weight_element = std::max_element(h_weights.begin(), h_weights.end());
    auto epsilon            = *max_weight_element * static_cast<weight_t>(1e-6);
    auto nearly_equal = [epsilon](auto lhs, auto rhs) { return std::fabs(lhs - rhs) < epsilon; };

    ASSERT_TRUE(std::equal(h_reference_distances.begin(),
                           h_reference_distances.end(),
                           h_cugraph_distances.begin(),
                           nearly_equal))
      << "distances do not match with the reference values.";

    for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
      auto i = std::distance(h_cugraph_predecessors.begin(), it);
      if (*it == cugraph::experimental::invalid_vertex_id<vertex_t>::value) {
        ASSERT_TRUE(h_reference_predecessors[i] == *it)
          << "vertex reachability do not match with the reference.";
      } else {
        auto pred_distance = h_reference_distances[*it];
        bool found{false};
        for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
          if (h_indices[j] == i) {
            if (nearly_equal(pred_distance + h_weights[j], h_reference_distances[i])) {
              found = true;
              break;
            }
          }
        }
        ASSERT_TRUE(found)
          << "no edge from the predecessor vertex to this vertex with the matching weight.";
      }
    }
  }
};

// FIXME: add tests for type combinations
TEST_P(Tests_SSSP, CheckInt32Int32Float) { run_current_test<int32_t, int32_t, float>(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_SSSP,
                        ::testing::Values(SSSP_Usecase("test/datasets/karate.mtx", 0),
                                          SSSP_Usecase("test/datasets/dblp.mtx", 0),
                                          SSSP_Usecase("test/datasets/wiki2003.mtx", 1000)));

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}

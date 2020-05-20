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

#include <queue>
#include <stack>
#include <vector>

#include <algorithms.hpp>

#include "rmm_utils.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include "bfs_ref.h"

// NOTE: This could be common to other files but we might not want the same precision
// depending on the algorithm
#ifndef TEST_EPSILON  // It is currently use for relative error
#define TEST_EPSILON 0.0001
#endif

// NOTE: Defines under which values the difference should  be discarded when
// considering values are close to zero
//  i.e: Do we consider that the difference between 1.3e-9 and 8.e-12 is
// significant
#ifndef TEST_ZERO_THRESHOLD
#define TEST_ZERO_THRESHOLD 1e-10
#endif
// ============================================================================
// C++ Reference Implementation
// ============================================================================
template <typename T, typename precision_t>
bool compare_close(const T &a, const T &b, const precision_t epsilon, precision_t zero_threshold)
{
  return ((zero_threshold > a && zero_threshold > b)) ||
         (a >= b * (1.0 - epsilon)) && (a <= b * (1.0 + epsilon));
}

// ============================================================================
// Test Suite
// ============================================================================
typedef struct BFS_Usecase_t {
  std::string config_;     // Path to graph file
  std::string file_path_;  // Complete path to graph using dataset_root_dir
  int source_;             // Starting point from the traversal
  BFS_Usecase_t(const std::string &config, int source) : config_(config), source_(source)
  {
    const std::string &rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((config_ != "") && (config_[0] != '/')) {
      file_path_ = rapidsDatasetRootDir + "/" + config_;
    } else {
      file_path_ = config_;
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

  // VT                 vertex identifier data type
  // ET                 edge identifier data type
  // WT                 edge weight data type
  // return_sp_counter  should BFS return shortest path countner
  template <typename VT, typename ET, typename WT, bool return_sp_counter>
  void run_current_test(const BFS_Usecase &configuration)
  {
    // Step 1: Construction of the graph based on configuration
    VT number_of_vertices;
    ET number_of_edges;
    CSR_Result_Weighted<VT, WT> csr_result;
    bool directed = false;
    generate_graph_csr_from_mm<VT, ET, WT>(
      csr_result, number_of_vertices, number_of_edges, directed, configuration.file_path_);
    CUDA_CHECK_LAST();
    cugraph::experimental::GraphCSRView<VT, ET, WT> G(csr_result.rowOffsets,
                                                      csr_result.colIndices,
                                                      csr_result.edgeWeights,
                                                      number_of_vertices,
                                                      number_of_edges);
    CUDA_CHECK_LAST();
    G.prop.directed = directed;

    ASSERT_TRUE(configuration.source_ >= 0 && configuration.source_ <= G.number_of_vertices)
      << "Starting sources should be >= 0 and"
      << " less than the number of vertices in the graph";

    VT source = configuration.source_;

    number_of_vertices = G.number_of_vertices;
    number_of_edges    = G.number_of_edges;

    std::vector<VT> indices(number_of_edges);
    std::vector<ET> offsets(number_of_vertices + 1);

    CUDA_TRY(
      cudaMemcpy(indices.data(), G.indices, sizeof(VT) * indices.size(), cudaMemcpyDeviceToHost));
    CUDA_TRY(
      cudaMemcpy(offsets.data(), G.offsets, sizeof(ET) * offsets.size(), cudaMemcpyDeviceToHost));

    std::queue<VT> Q;
    std::stack<VT> S;
    std::vector<VT> ref_bfs_dist(number_of_vertices);
    std::vector<std::vector<VT>> ref_bfs_pred(number_of_vertices);
    std::vector<double> ref_bfs_sigmas(number_of_vertices);

    ref_bfs<VT, ET>(indices.data(),
                    offsets.data(),
                    number_of_vertices,
                    Q,
                    S,
                    ref_bfs_dist,
                    ref_bfs_pred,
                    ref_bfs_sigmas,
                    source);

    // Device data for cugraph_bfs
    rmm::device_vector<VT> d_cugraph_dist(number_of_vertices);
    rmm::device_vector<VT> d_cugraph_pred(number_of_vertices);
    rmm::device_vector<double> d_cugraph_sigmas(number_of_vertices);

    std::vector<VT> cugraph_dist(number_of_vertices);
    std::vector<VT> cugraph_pred(number_of_vertices);
    std::vector<double> cugraph_sigmas(number_of_vertices);

    cugraph::bfs<VT, ET, WT>(G,
                             d_cugraph_dist.data().get(),
                             d_cugraph_pred.data().get(),
                             d_cugraph_sigmas.data().get(),
                             source,
                             G.prop.directed);
    CUDA_TRY(cudaMemcpy(cugraph_dist.data(),
                        d_cugraph_dist.data().get(),
                        sizeof(VT) * d_cugraph_dist.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(cugraph_pred.data(),
                        d_cugraph_pred.data().get(),
                        sizeof(VT) * d_cugraph_pred.size(),
                        cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(cugraph_sigmas.data(),
                        d_cugraph_sigmas.data().get(),
                        sizeof(double) * d_cugraph_sigmas.size(),
                        cudaMemcpyDeviceToHost));

    for (VT i = 0; i < number_of_vertices; ++i) {
      // Check distances: should be an exact match as we use signed int 32-bit
      EXPECT_EQ(cugraph_dist[i], ref_bfs_dist[i])
        << "[MISMATCH] vaid = " << i << ", cugraph = " << cugraph_sigmas[i]
        << " c++ ref = " << ref_bfs_sigmas[i];
      // Check predecessor: We do not enforce the predecessor, we simply verifiy
      // that the predecessor obtained with the GPU implementation is one of the
      // predecessors obtained during the C++ BFS traversal
      VT pred = cugraph_pred[i];  // It could be equal to -1 if the node is never reached
      if (pred == -1) {
        EXPECT_TRUE(ref_bfs_pred[i].empty())
          << "[MISMATCH][PREDECESSOR] vaid = " << i << " cugraph had not predecessor,"
          << "while c++ ref found at least one.";
      } else {
        // This can get expensive to check, we could have simply verified that based
        // on the the distance from the source to the predecessor, but this ensures that there
        // are no misassignations
        auto it = std::find(ref_bfs_pred[i].begin(), ref_bfs_pred[i].end(), pred);
        EXPECT_TRUE(it != ref_bfs_pred[i].end())
          << "[MISMATCH][PREDECESSOR] vaid = " << i << " cugraph = " << cugraph_sigmas[i]
          << " , c++ ref did not consider it as a predecessor.";
      }
      EXPECT_TRUE(
        compare_close(cugraph_sigmas[i], ref_bfs_sigmas[i], TEST_EPSILON, TEST_ZERO_THRESHOLD))
        << "[MISMATCH] vaid = " << i << ", cugraph = " << cugraph_sigmas[i]
        << " c++ ref = " << ref_bfs_sigmas[i];

      if (return_sp_counter) {
        EXPECT_TRUE(
          compare_close(cugraph_sigmas[i], ref_bfs_sigmas[i], TEST_EPSILON, TEST_ZERO_THRESHOLD))
          << "[MISMATCH] vaid = " << i << ", cugraph = " << cugraph_sigmas[i]
          << " c++ ref = " << ref_bfs_sigmas[i];
      }
    }
  }
};

// ============================================================================
// Tests
// ============================================================================
TEST_P(Tests_BFS, CheckFP32_NO_SP_COUNTER) { run_current_test<int, int, float, false>(GetParam()); }

TEST_P(Tests_BFS, CheckFP64_NO_SP_COUNTER)
{
  run_current_test<int, int, double, false>(GetParam());
}

TEST_P(Tests_BFS, CheckFP32_SP_COUNTER) { run_current_test<int, int, float, true>(GetParam()); }

TEST_P(Tests_BFS, CheckFP64_SP_COUNTER) { run_current_test<int, int, double, true>(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_BFS,
                        ::testing::Values(BFS_Usecase("test/datasets/karate.mtx", 0),
                                          BFS_Usecase("test/datasets/polbooks.mtx", 0),
                                          BFS_Usecase("test/datasets/netscience.mtx", 0),
                                          BFS_Usecase("test/datasets/netscience.mtx", 100),
                                          BFS_Usecase("test/datasets/wiki2003.mtx", 1000),
                                          BFS_Usecase("test/datasets/wiki-Talk.mtx", 1000)));

int main(int argc, char **argv)
{
  rmmInitialize(nullptr);
  testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  rmmFinalize();
  return rc;
}

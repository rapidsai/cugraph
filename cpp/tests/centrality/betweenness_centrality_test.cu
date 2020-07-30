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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <traversal/bfs_ref.h>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <graph.hpp>

#include <raft/error.hpp>
#include <raft/handle.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <thrust/device_vector.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <queue>
#include <stack>
#include <utility>

#ifndef TEST_EPSILON
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
template <typename VT, typename ET, typename WT, typename result_t>
void ref_accumulation(result_t *result,
                      VT const number_of_vertices,
                      std::stack<VT> &S,
                      std::vector<std::vector<VT>> &pred,
                      std::vector<double> &sigmas,
                      std::vector<double> &deltas,
                      VT source)
{
  for (VT v = 0; v < number_of_vertices; ++v) { deltas[v] = 0; }
  while (!S.empty()) {
    VT w = S.top();
    S.pop();
    for (VT v : pred[w]) { deltas[v] += (sigmas[v] / sigmas[w]) * (1.0 + deltas[w]); }
    if (w != source) { result[w] += deltas[w]; }
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void ref_endpoints_accumulation(result_t *result,
                                VT const number_of_vertices,
                                std::stack<VT> &S,
                                std::vector<std::vector<VT>> &pred,
                                std::vector<double> &sigmas,
                                std::vector<double> &deltas,
                                VT source)
{
  result[source] += S.size() - 1;
  for (VT v = 0; v < number_of_vertices; ++v) { deltas[v] = 0; }
  while (!S.empty()) {
    VT w = S.top();
    S.pop();
    for (VT v : pred[w]) { deltas[v] += (sigmas[v] / sigmas[w]) * (1.0 + deltas[w]); }
    if (w != source) { result[w] += deltas[w] + 1; }
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void ref_edge_accumulation(result_t *result,
                           VT const number_of_vertices,
                           std::stack<VT> &S,
                           std::vector<std::vector<VT>> &pred,
                           std::vector<double> &sigmas,
                           std::vector<double> &deltas,
                           VT source)
{
  for (VT v = 0; v < number_of_vertices; ++v) { deltas[v] = 0; }
  while (!S.empty()) {
    VT w = S.top();
    S.pop();
    for (VT v : pred[w]) { deltas[v] += (sigmas[v] / sigmas[w]) * (1.0 + deltas[w]); }
    if (w != source) { result[w] += deltas[w]; }
  }
}

// Algorithm 1: Shortest-path vertex betweenness, (Brandes, 2001)
template <typename VT, typename ET, typename WT, typename result_t>
void reference_betweenness_centrality_impl(VT *indices,
                                           ET *offsets,
                                           VT const number_of_vertices,
                                           result_t *result,
                                           bool endpoints,
                                           VT const *sources,
                                           VT const number_of_sources)
{
  std::queue<VT> Q;
  std::stack<VT> S;
  // NOTE: dist is of type VT not WT
  std::vector<VT> dist(number_of_vertices);
  std::vector<std::vector<VT>> pred(number_of_vertices);
  std::vector<double> sigmas(number_of_vertices);
  std::vector<double> deltas(number_of_vertices);

  std::vector<VT> neighbors;

  if (sources) {
    for (VT source_idx = 0; source_idx < number_of_sources; ++source_idx) {
      VT s = sources[source_idx];
      // Step 1: Single-source shortest-paths problem
      //   a. Initialization
      ref_bfs<VT, ET>(indices, offsets, number_of_vertices, Q, S, dist, pred, sigmas, s);
      //  Step 2: Accumulation
      //          Back propagation of dependencies
      if (endpoints) {
        ref_endpoints_accumulation<VT, ET, WT, result_t>(
          result, number_of_vertices, S, pred, sigmas, deltas, s);
      } else {
        ref_accumulation<VT, ET, WT, result_t>(
          result, number_of_vertices, S, pred, sigmas, deltas, s);
      }
    }
  } else {
    for (VT s = 0; s < number_of_vertices; ++s) {
      // Step 1: Single-source shortest-paths problem
      //   a. Initialization
      ref_bfs<VT, ET>(indices, offsets, number_of_vertices, Q, S, dist, pred, sigmas, s);
      //  Step 2: Accumulation
      //          Back propagation of dependencies
      if (endpoints) {
        ref_endpoints_accumulation<VT, ET, WT, result_t>(
          result, number_of_vertices, S, pred, sigmas, deltas, s);
      } else {
        ref_accumulation<VT, ET, WT, result_t>(
          result, number_of_vertices, S, pred, sigmas, deltas, s);
      }
    }
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void reference_rescale(result_t *result,
                       bool directed,
                       bool normalize,
                       bool endpoints,
                       VT const number_of_vertices,
                       VT const number_of_sources)
{
  bool modified                      = false;
  result_t rescale_factor            = static_cast<result_t>(1);
  result_t casted_number_of_sources  = static_cast<result_t>(number_of_sources);
  result_t casted_number_of_vertices = static_cast<result_t>(number_of_vertices);
  if (normalize) {
    if (number_of_vertices > 2) {
      if (endpoints) {
        rescale_factor /= (casted_number_of_vertices * (casted_number_of_vertices - 1));
      } else {
        rescale_factor /= ((casted_number_of_vertices - 1) * (casted_number_of_vertices - 2));
      }
      modified = true;
    }
  } else {
    if (!directed) {
      rescale_factor /= static_cast<result_t>(2);
      modified = true;
    }
  }
  if (modified) {
    if (number_of_sources > 0) {
      rescale_factor *= (casted_number_of_vertices / casted_number_of_sources);
    }
  }
  for (auto idx = 0; idx < number_of_vertices; ++idx) { result[idx] *= rescale_factor; }
}

template <typename VT, typename ET, typename WT, typename result_t>
void reference_betweenness_centrality(cugraph::GraphCSRView<VT, ET, WT> const &graph,
                                      result_t *result,
                                      bool normalize,
                                      bool endpoints,  // This is not yet implemented
                                      VT const number_of_sources,
                                      VT const *sources)
{
  VT number_of_vertices = graph.number_of_vertices;
  ET number_of_edges    = graph.number_of_edges;
  thrust::host_vector<VT> h_indices(number_of_edges);
  thrust::host_vector<ET> h_offsets(number_of_vertices + 1);

  thrust::device_ptr<VT> d_indices((VT *)&graph.indices[0]);
  thrust::device_ptr<ET> d_offsets((ET *)&graph.offsets[0]);

  thrust::copy(d_indices, d_indices + number_of_edges, h_indices.begin());
  thrust::copy(d_offsets, d_offsets + (number_of_vertices + 1), h_offsets.begin());

  cudaDeviceSynchronize();

  reference_betweenness_centrality_impl<VT, ET, WT, result_t>(&h_indices[0],
                                                              &h_offsets[0],
                                                              number_of_vertices,
                                                              result,
                                                              endpoints,
                                                              sources,
                                                              number_of_sources);
  reference_rescale<VT, ET, WT, result_t>(
    result, graph.prop.directed, normalize, endpoints, number_of_vertices, number_of_sources);
}
// Explicit instantiation
/*    FIXME!!!
template void reference_betweenness_centrality<int, int, float, float>(
  cugraph::GraphCSRView<int, int, float> const &,
  float *,
  bool,
  bool,
  const int,
  int const *);
template void reference_betweenness_centrality<int, int, double, double>(
  cugraph::GraphCSRView<int, int, double> const &,
  double *,
  bool,
  bool,
  const int,
  int const *);
*/

// =============================================================================
// Utility functions
// =============================================================================
// Compare while allowing relatie error of epsilon
// zero_threshold indicates when  we should drop comparison for small numbers
template <typename T, typename precision_t>
bool compare_close(const T &a, const T &b, const precision_t epsilon, precision_t zero_threshold)
{
  return ((zero_threshold > a && zero_threshold > b)) ||
         (a >= b * (1.0 - epsilon)) && (a <= b * (1.0 + epsilon));
}

// =============================================================================
// Test Suite
// =============================================================================
// Defines Betweenness Centrality UseCase
// SSSP's test suite code uses type of Graph parameter that could be used
// (MTX / RMAT)
typedef struct BC_Usecase_t {
  std::string config_;     // Path to graph file
  std::string file_path_;  // Complete path to graph using dataset_root_dir
  int number_of_sources_;  // Starting point from the traversal
  BC_Usecase_t(const std::string &config, int number_of_sources)
    : config_(config), number_of_sources_(number_of_sources)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // FIXME: Use platform independent stuff from c++14/17 on compiler update
    const std::string &rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((config_ != "") && (config_[0] != '/')) {
      file_path_ = rapidsDatasetRootDir + "/" + config_;
    } else {
      file_path_ = config_;
    }
  };
} BC_Usecase;

class Tests_BC : public ::testing::TestWithParam<BC_Usecase> {
  raft::handle_t handle;

 public:
  Tests_BC() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}
  // VT         vertex identifier data type
  // ET         edge identifier data type
  // WT         edge weight data type
  // result_t   result data type
  // normalize  should the result be normalized
  // endpoints  should the endpoints be included
  template <typename VT,
            typename ET,
            typename WT,
            typename result_t,
            bool normalize,
            bool endpoints>
  void run_current_test(const BC_Usecase &configuration)
  {
    // Step 1: Construction of the graph based on configuration
    bool is_directed = false;
    auto csr =
      cugraph::test::generate_graph_csr_from_mm<VT, ET, WT>(is_directed, configuration.file_path_);
    cudaDeviceSynchronize();
    cugraph::GraphCSRView<VT, ET, WT> G = csr->view();
    G.prop.directed                     = is_directed;
    CUDA_TRY(cudaGetLastError());
    std::vector<result_t> result(G.number_of_vertices, 0);
    std::vector<result_t> expected(G.number_of_vertices, 0);

    // Step 2: Generation of sources based on configuration
    //         if number_of_sources_ is 0 then sources must be nullptr
    //         Otherwise we only  use the first k values
    ASSERT_TRUE(configuration.number_of_sources_ >= 0 &&
                configuration.number_of_sources_ <= G.number_of_vertices)
      << "Number number of sources should be >= 0 and"
      << " less than the number of vertices in the graph";
    std::vector<VT> sources(configuration.number_of_sources_);
    thrust::sequence(thrust::host, sources.begin(), sources.end(), 0);

    VT *sources_ptr = nullptr;
    if (configuration.number_of_sources_ > 0) { sources_ptr = sources.data(); }

    reference_betweenness_centrality(
      G, expected.data(), normalize, endpoints, configuration.number_of_sources_, sources_ptr);

    sources_ptr = nullptr;
    if (configuration.number_of_sources_ > 0) { sources_ptr = sources.data(); }

    VT total_number_of_sources = configuration.number_of_sources_;
    if (total_number_of_sources == 0) { total_number_of_sources = G.number_of_vertices; }
    thrust::device_vector<result_t> d_result(G.number_of_vertices);
    cugraph::betweenness_centrality(handle,
                                    G,
                                    d_result.data().get(),
                                    normalize,
                                    endpoints,
                                    static_cast<WT *>(nullptr),
                                    configuration.number_of_sources_,
                                    sources_ptr,
                                    total_number_of_sources);
    cudaDeviceSynchronize();
    CUDA_TRY(cudaMemcpy(result.data(),
                        d_result.data().get(),
                        sizeof(result_t) * G.number_of_vertices,
                        cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    for (int i = 0; i < G.number_of_vertices; ++i)
      EXPECT_TRUE(compare_close(result[i], expected[i], TEST_EPSILON, TEST_ZERO_THRESHOLD))
        << "[MISMATCH] vaid = " << i << ", cugraph = " << result[i]
        << " expected = " << expected[i];
  }
};

// ============================================================================
// Tests
// ============================================================================
// Verifiy Un-Normalized results
TEST_P(Tests_BC, CheckFP32_NO_NORMALIZE_NO_ENDPOINTS)
{
  run_current_test<int, int, float, float, false, false>(GetParam());
}

TEST_P(Tests_BC, CheckFP64_NO_NORMALIZE_NO_ENDPOINTS)
{
  run_current_test<int, int, double, double, false, false>(GetParam());
}

TEST_P(Tests_BC, CheckFP32_NO_NORMALIZE_ENDPOINTS)
{
  run_current_test<int, int, float, float, false, true>(GetParam());
}

TEST_P(Tests_BC, CheckFP64_NO_NORMALIZE_ENDPOINTS)
{
  run_current_test<int, int, double, double, false, true>(GetParam());
}

// Verifiy Normalized results
TEST_P(Tests_BC, CheckFP32_NORMALIZE_NO_ENDPOINTS)
{
  run_current_test<int, int, float, float, true, false>(GetParam());
}

TEST_P(Tests_BC, CheckFP64_NORMALIZE_NO_ENDPOINTS)
{
  run_current_test<int, int, double, double, true, false>(GetParam());
}

TEST_P(Tests_BC, CheckFP32_NORMALIZE_ENDPOINTS)
{
  run_current_test<int, int, float, float, true, true>(GetParam());
}

TEST_P(Tests_BC, CheckFP64_NORMALIZE_ENDPOINTS)
{
  run_current_test<int, int, double, double, true, true>(GetParam());
}

// FIXME: There is an InvalidValue on a Memcopy only on tests/datasets/dblp.mtx
INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_BC,
                        ::testing::Values(BC_Usecase("test/datasets/karate.mtx", 0),
                                          BC_Usecase("test/datasets/netscience.mtx", 0),
                                          BC_Usecase("test/datasets/netscience.mtx", 4),
                                          BC_Usecase("test/datasets/wiki2003.mtx", 4),
                                          BC_Usecase("test/datasets/wiki-Talk.mtx", 4)));

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}

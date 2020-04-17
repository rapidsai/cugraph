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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <thrust/device_vector.h>
#include "test_utils.h"
#include <utility>

#include <graph.hpp>
#include <algorithms.hpp>

#include <queue> // C++ Reference Algorithm
#include <stack> // C++ Reference Algorithm

#include <converters/COOtoCSR.cuh> // Loads GraphCSR from .mtx
#include <fstream>

#ifndef TEST_EPSILON
 #define TEST_EPSILON 0.0001
#endif


// =============================================================================
// C++ Reference Implementation
// =============================================================================
template<typename VT, typename ET>
void populate_neighbors(VT *indices, ET *offsets,
                        VT w, std::vector<VT> &neighbors) {
  ET edge_start = offsets[w];
  ET edge_end = offsets[w + 1];
  ET edge_count = edge_end - edge_start;

  neighbors.clear(); // Reset neighbors vector's size
  for (ET edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
    VT dst = indices[edge_start + edge_idx];
    neighbors.push_back(dst);
  }
}

// TODO: This colud be moved to BFS testing on the c++ side
// This implements the BFS from (Brandes, 2001) with shortest path counting
template<typename VT, typename ET, typename WT, typename result_t>
void ref_bfs(VT *indices, ET *offsets, VT const number_of_vertices,
             std::queue<VT> &Q,
             std::stack<VT> &S,
             std::vector<VT> &dist,
             std::vector<std::vector<VT>> &pred,
             std::vector<result_t> &sigmas,
             VT source) {
  std::vector<VT> neighbors;
  for (VT w = 0 ; w < number_of_vertices; ++w) {
    pred[w].clear();
    dist[w] = std::numeric_limits<VT>::max();
    sigmas[w] = 0;
  }
  dist[source] = 0;
  sigmas[source] = 1;
  Q.push(source);
  //   b. Traversal
  while (!Q.empty()) {
    VT v = Q.front();
    Q.pop();
    S.push(v);
    populate_neighbors<VT, ET>(indices, offsets, v, neighbors);
    for (VT w : neighbors) {
      // Path Discovery:
      // Found for the first time?
      if (dist[w] == std::numeric_limits<VT>::max()) {
        dist[w] = dist[v] + 1;
        Q.push(w);
      }
      // Path counting
      // Edge(v, w) on  a shortest path?
      if (dist[w] == dist[v] + 1) {
        sigmas[w] +=  sigmas[v];
        pred[w].push_back(v);
      }
    }
  }
}

template<typename VT, typename ET, typename WT, typename result_t>
void ref_accumulation(result_t *result,
                      VT const number_of_vertices,
                      std::stack<VT> &S,
                      std::vector<std::vector<VT>> &pred,
                      std::vector<result_t> &sigmas,
                      std::vector<result_t> &deltas,
                      VT source) {
  for (VT v = 0; v < number_of_vertices; ++v) {
    deltas[v] = 0;
  }
  while (!S.empty()) {
    VT w = S.top();
    S.pop();
    for (VT v : pred[w]) {
      deltas[v] += (sigmas[v] / sigmas[w]) * (1.0 + deltas[w]);
    }
    if (w != source) {
      result[w] += deltas[w];
    }
  }
}

// Algorithm 1: Shortest-path vertex betweenness, (Brandes, 2001)
template <typename VT, typename ET, typename WT, typename result_t>
void reference_betweenness_centrality_impl(VT *indices, ET *offsets,
                                           VT const number_of_vertices,
                                           result_t *result,
                                           VT const *sources,
                                           VT const number_of_sources) {
  std::queue<VT> Q;
  std::stack<VT> S;
  // NOTE: dist is of type VT not WT
  std::vector<VT> dist(number_of_vertices);
  std::vector<std::vector<VT>> pred(number_of_vertices);
  std::vector<result_t> sigmas(number_of_vertices);
  std::vector<result_t> deltas(number_of_vertices);

  std::vector<VT> neighbors;

  if (sources) {
    for (VT source_idx = 0; source_idx < number_of_sources; ++source_idx) {
      VT s = sources[source_idx];
      // Step 1: Single-source shortest-paths problem
      //   a. Initialization
      ref_bfs<VT, ET, WT, result_t>(indices, offsets, number_of_vertices,
                                    Q, S,
                                    dist, pred, sigmas, s);
      //  Step 2: Accumulation
      //          Back propagation of dependencies
      ref_accumulation<VT, ET, WT, result_t>(result,
                                             number_of_vertices,
                                             S,
                                             pred,
                                             sigmas,
                                             deltas,
                                             s);
    }
  } else {
    for (VT s = 0; s < number_of_vertices; ++s) {
      // Step 1: Single-source shortest-paths problem
      //   a. Initialization
      ref_bfs<VT, ET, WT, result_t>(indices, offsets, number_of_vertices,
                                    Q, S,
                                    dist, pred, sigmas, s);
      //  Step 2: Accumulation
      //          Back propagation of dependencies
      ref_accumulation<VT, ET, WT, result_t>(result,
                                             number_of_vertices,
                                             S,
                                             pred,
                                             sigmas,
                                             deltas,
                                             s);
    }
  }
}

template <typename VT, typename ET, typename WT, typename result_t>
void reference_betweenness_centrality(cugraph::experimental::GraphCSR<VT, ET, WT> const &graph,
                                      result_t *result,
                                      bool normalize,
                                      bool endpoints, // This is not yet implemented
                                      VT const number_of_sources,
                                      VT const *sources) {

  VT number_of_vertices = graph.number_of_vertices;
  ET number_of_edges = graph.number_of_edges;
  thrust::host_vector<VT> h_indices(number_of_edges);
  thrust::host_vector<ET> h_offsets(number_of_vertices + 1);

  thrust::device_ptr<VT>  d_indices((VT *)&graph.indices[0]);
  thrust::device_ptr<ET>  d_offsets((ET *)&graph.offsets[0]);

  thrust::copy(d_indices, d_indices + number_of_edges, h_indices.begin());
  thrust::copy(d_offsets, d_offsets + (number_of_vertices + 1), h_offsets.begin());


  /*
  cudaMemcpyAsync(&h_indices[0], &graph.indices[0],
             sizeof(VT) * h_indices.size(), cudaMemcpyDeviceToHost, nullptr);
  cudaMemcpyAsync(&h_offsets[0], &graph.offsets[0],
             h_offsets.size() * sizeof(ET), cudaMemcpyDeviceToHost, nullptr);
  CUDA_CHECK_LAST();
  */
  cudaDeviceSynchronize();
  reference_betweenness_centrality_impl<VT, ET, WT, result_t>(&h_indices[0],
                                                              &h_offsets[0],
                                                              number_of_vertices,
                                                              result, sources,
                                                              number_of_sources);
  if (normalize && number_of_vertices > 2) {
    result_t factor = static_cast<result_t>(number_of_vertices - 1) * static_cast<result_t>(number_of_vertices - 2);
    for (VT v = 0; v < number_of_vertices; ++v) {
      result[v] /= factor;
    }
  }
}
// Explicit declaration
template void reference_betweenness_centrality<int, int, float, float>(cugraph::experimental::GraphCSR<int, int, float> const&,
                                                                  float *, bool, bool, const int, int const *);
// =============================================================================
// Utility functions
// =============================================================================
/**
 * @brief     Extract betweenness centality values from file
 *
 * This function reads the content of a file containing betweenness values
 * The expected format per line is '<vertex_idx> <betweenness_centrality>'
 *
 * @tparam VT           Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET           Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam result_t     Type of betweenness centrality value: float
 *
 * @param[out] result   Reference to a vector that is resized and filled with betweenness value
 * @param[in] bc_file   Path to the file to extract betweenness from
 *
 */
// FIXME: This is not BC specific, it simply reads '<VT> <result_t>\n' files
template <typename VT, typename result_t>
void extract_bc(std::vector<result_t> &result, std::string bc_file) {
  VT vid = 0; // Not really usefull, nx_bc_file is expected to be sorted
  result_t bc = 0; // Not really usefull, nx_bc_file is expected to be sorted

  result.clear();
  std::ifstream ifs(bc_file);
  ASSERT_TRUE(ifs.is_open());

  while (ifs >> vid >> bc) {
    result.push_back(bc);
  }
  ifs.close();
}

// TODO(xcadet): This could be useful in other testsuite (SSSP, BFS, ...)
template<typename VT, typename ET, typename WT>
void generate_graph_csr(CSR_Result_Weighted<VT, WT> &csr_result, VT &m, VT &nnz, bool &is_directed, std::string matrix_file) {
  FILE* fpin = fopen(matrix_file.c_str(),"r");
  ASSERT_NE(fpin, nullptr) << "fopen (" << matrix_file << ") failure.";

  int k;
  MM_typecode mc;
  ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz),0) << "could not read Matrix Market file properties"<< "\n";
  ASSERT_TRUE(mm_is_matrix(mc));
  ASSERT_TRUE(mm_is_coordinate(mc));
  ASSERT_FALSE(mm_is_complex(mc));
  ASSERT_FALSE(mm_is_skew(mc));
  is_directed = !mm_is_symmetric(mc);

  // Allocate memory on host
  std::vector<int> cooRowInd(nnz), cooColInd(nnz);
  std::vector<float> cooVal(nnz);

  // Read
  ASSERT_EQ( (mm_to_coo<int, float>(fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)) , 0)<< "could not read matrix data"<< "\n";
  ASSERT_EQ(fclose(fpin),0);

  ConvertCOOtoCSR_weighted(&cooRowInd[0], &cooColInd[0], &cooVal[0], nnz, csr_result);
  CUDA_CHECK_LAST();
}

// TODO(xcadet): This may actually operate an exact comparison when b == 0
template <typename T>
bool compare_close(const T &a, const T&b, const double epsilon) {
  return (a >= b * (1.0 - epsilon)) and (a <= b * (1.0 + epsilon));
}


// =============================================================================
// Test Suite
// =============================================================================

// Defines Betweenness Centrality UseCase
// SSSP codes uses type of Graph parameter that could be used
//TODO(xcadet) Use VT for srcs
typedef struct BC_Usecase_t {
  std::string config_;
  std::string file_path_;
  int number_of_sources_;
  BC_Usecase_t(const std::string& config, int number_of_sources)
               : config_(config), number_of_sources_(number_of_sources) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // FIXME: Use platform independent stuff from c++14/17 on compiler update
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((config_ != "") && (config_[0] != '/')) {
      file_path_ = rapidsDatasetRootDir + "/" + config_;
    } else {
      file_path_ = config_;
    }
  };
} BC_Usecase;

/*
struct BetweennessCentralityTest : public ::testing::Test
{
};
*/
class Tests_BC : public ::testing::TestWithParam<BC_Usecase> {
  public:
  Tests_BC() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}
  // TODO(xcadet) Should normalize be part of the configuration?
  template <typename VT, typename ET, typename WT, typename result_t,
            bool normalize, bool endpoints>
  void run_current_test(const BC_Usecase &configuration) {
    // Step 1: Construction of the graph based on configuration
    VT m;
    ET nnz;
    CSR_Result_Weighted<VT, WT> csr_result;
    bool is_directed = false;
    generate_graph_csr<VT, ET, WT>(csr_result, m, nnz, is_directed,
                                   configuration.file_path_);
    cudaDeviceSynchronize();
    cugraph::experimental::GraphCSR<VT, ET, WT> G(csr_result.rowOffsets,
                                                  csr_result.colIndices,
                                                  csr_result.edgeWeights,
                                                  m, nnz);
    G.prop.directed = is_directed;

    CUDA_CHECK_LAST();
    std::vector<result_t> result(G.number_of_vertices, 0);
    std::vector<result_t> expected(G. number_of_vertices, 0);

    // Step 2: Generation of sources based on configuration
    //         if number_of_sources_ is 0 then sources must be nullptr
    //         Otherwise we only  use the first k values
    ASSERT_TRUE(configuration.number_of_sources_ >= 0
           && configuration.number_of_sources_ <= G.number_of_vertices)
           << "Number number of sources should be >= 0 and"
           << " less than the number of vertices in the graph";
    std::vector<VT> sources(configuration.number_of_sources_);
    std::iota(sources.begin(), sources.end(), 0);

    VT *sources_ptr = nullptr;
    if (configuration.number_of_sources_ > 0) {
      sources_ptr = sources.data();
    }

    // TODO(xcadet) reference should also include normalize, endpooint, number_of_sources and sources
    reference_betweenness_centrality(G, expected.data(),
                                     normalize, endpoints,
                                     //weights
                                     configuration.number_of_sources_,
                                     sources_ptr);

    sources_ptr = nullptr;
    if (configuration.number_of_sources_ > 0) {
      sources_ptr = sources.data();
    }

    printf("[DBG] Number of vertices %d\n", G.number_of_vertices);
    thrust::device_vector<result_t>  d_result(G.number_of_vertices);
    cugraph::betweenness_centrality(G, d_result.data().get(),
                                    normalize, endpoints,
                                    static_cast<WT*>(nullptr),
                                    configuration.number_of_sources_,
                                    sources_ptr,
                                    cugraph::cugraph_bc_implem_t::CUGRAPH_DEFAULT);
    cudaDeviceSynchronize();
    std::cout << "[DBG][BC] CUGRAPH IS DONE COMPUTING" << std::endl;
    cudaMemcpy(result.data(), d_result.data().get(),
               sizeof(result_t) * G.number_of_vertices,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0 ; i < G.number_of_vertices ; ++i)
      EXPECT_NEAR(result[i], expected[i], TEST_EPSILON) <<
                  "[MISMATCH] vaid = " << i << ", cugraph = " <<
                  result[i] << " expected = " << expected[i];
    std::cout << "[DBG][BC] Perfect math over " << G.number_of_vertices << std::endl;
  }



};

struct BetweennessCentralityBFSTest : public ::testing::Test
{
};


/*
// BFS: Checking for shortest_path counting correctness
// -----------------------------------------------------------------------------
// TODO(xcadet) Parametrize this part for VT, ET, WT, result_t

TEST_F(BetweennessCentralityBFSTest, CheckReference) {
  // TODO(xcadet) This dataset was manually generated and is not provided
  //std::string matrix_file(get_rapids_dataset_root_dir() + "/" + "email-Eu-core-gen.mtx");
  std::string matrix_file("../../datasets/email-Eu-core-gen.mtx");
  //std::string matrix_file("../../datasets/karate-directed.mtx");
  int m, nnz;
  CSR_Result_Weighted<int, float> csr_result;
  generate_graph_csr<int, int, float>(csr_result, m, nnz, matrix_file);
  cugraph::experimental::GraphCSR<int, int, float> graph(csr_result.rowOffsets,
                                                     csr_result.colIndices,
                                                     csr_result.edgeWeights,
                                                     m, nnz);
  // FIXME: THIS IS CRITICAL:
  graph.prop.directed = true;
  std::vector<float> result(graph.number_of_vertices);

  //int source = 2;
  int source = 12;
  // Ref BC_BFS requires many working values
  int number_of_vertices = graph.number_of_vertices;
  int number_of_edges = graph.number_of_edges;
  //
  std::vector<int> indices(number_of_edges);
  std::vector<int> offsets(number_of_vertices + 1);

  cudaMemcpy(indices.data(), graph.indices,
             sizeof(int) * indices.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(offsets.data(), graph.offsets,
             sizeof(int) * offsets.size(), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::queue<int> Q;
  std::stack<int> S;
  std::vector<int> ref_bfs_dist(number_of_vertices);
  std::vector<std::vector<int>> ref_bfs_pred(number_of_vertices);
  std::vector<float> ref_bfs_sigmas(number_of_vertices);

  ref_bfs<int, int, float, float>(indices.data(), offsets.data(),
                                  number_of_vertices, Q, S,
                                  ref_bfs_dist, ref_bfs_pred,
                                  ref_bfs_sigmas, source);



  // Device data for cugraph_bfs
  thrust::device_vector<int> d_cugraph_dist(number_of_vertices);
  thrust::device_vector<int> d_cugraph_pred(number_of_vertices);
  thrust::device_vector<int> d_cugraph_sigmas(number_of_vertices);

  // This test only checks for sigmas equality
  std::vector<int> cugraph_sigmas(number_of_vertices);

  printf("Is graph directed ? %d\n", graph.prop.directed);
  cugraph::bfs<int, int, float>(graph, d_cugraph_dist.data().get(),
                                d_cugraph_pred.data().get(),
                                d_cugraph_sigmas.data().get(),
                                source, graph.prop.directed);
  cudaMemcpy(cugraph_sigmas.data(), d_cugraph_sigmas.data().get(),
             sizeof(int) * d_cugraph_sigmas.size(), cudaMemcpyDeviceToHost);
  // TODO(xcadet): The implicit cast comes from BFS shortest_path counter being
  // of type VT, while the ref_bfs uses float values
  for (int i = 0 ; i < number_of_vertices ; ++i) {
    EXPECT_TRUE(compare_close((float)cugraph_sigmas[i], ref_bfs_sigmas[i], 0.0001)) <<
                "[MISMATCH] vaid = " << i << ", cugraph = " <<
                cugraph_sigmas[i] << " c++ ref = " << ref_bfs_sigmas[i];
    //std::cout << "Sigmas[" << i << "] = " << cugraph_sigmas[i] << std::endl;
  }
  std::cout << "Graph number_of_vertices " << number_of_vertices << ", number_of_edges " << number_of_edges << std::endl;
  int sum_sigmas_cugraph = thrust::reduce(thrust::host, cugraph_sigmas.begin(), cugraph_sigmas.end(), 0);
  int sum_sigmas_ref = thrust::reduce(thrust::host, ref_bfs_sigmas.begin(), ref_bfs_sigmas.end(), 0);
  std::cout << "Source " << source << ", cugraph: " << sum_sigmas_cugraph << ", ref " << sum_sigmas_ref << std::endl;;
}
*/


// BC
// -----------------------------------------------------------------------------
/*
TEST_F(BetweennessCentralityTest, SimpleGraph)
{
  std::vector<int>  graph_offsets{ { 0, 1, 2, 5, 7, 10, 12, 14 } };
  std::vector<int>  graph_indices{ { 2, 2, 0, 1, 3, 2, 4, 3, 5, 6, 4, 6, 4, 5 } };

  std::vector<float> expected{ {0.0, 0.0, 0.6, 0.6, 0.5333333, 0.0, 0.0 } };

  int num_verts = graph_offsets.size() - 1;
  int num_edges = graph_indices.size();

  thrust::device_vector<int>    d_graph_offsets(graph_offsets);
  thrust::device_vector<int>    d_graph_indices(graph_indices);
  thrust::device_vector<float>  d_result(num_verts);

  std::vector<float>            result(num_verts);

  cugraph::experimental::GraphCSR<int,int,float> G(d_graph_offsets.data().get(),
                                                   d_graph_indices.data().get(),
                                                   nullptr,
                                                   num_verts,
                                                   num_edges);

  cugraph::betweenness_centrality(G, d_result.data().get());

  cudaMemcpy(result.data(), d_result.data().get(), sizeof(float) * num_verts, cudaMemcpyDeviceToHost);

  for (int i = 0 ; i < num_verts ; ++i)
    EXPECT_FLOAT_EQ(result[i], expected[i]);

  // TODO(xcadet) Remove this part, it is for testing the reference
  std::vector<float> ref_result(num_verts);
  reference_betweenness_centrality(G, ref_result.data(), true);
  for (int i = 0 ; i < num_verts ; ++i)
    EXPECT_FLOAT_EQ(ref_result[i], expected[i]);
}
*/
// Verifiy Un-Normalized results
TEST_P(Tests_BC, CheckFP32_NO_NORMALIZE_NO_ENDPOINTS) {
  run_current_test<int, int, float, float, false, false>(GetParam());
}

TEST_P(Tests_BC, CheckFP64_NO_NORMALIZE_NO_ENDPOINTS) {
  run_current_test<int, int, float, float, false, false>(GetParam());
}

// Verifiy Normalized results
TEST_P(Tests_BC, CheckFP32_NORMALIZE_NO_ENPOINTS) {
  run_current_test<int, int, float, float, true, false>(GetParam());
}

TEST_P(Tests_BC, CheckFP64_NORMALIZE_NO_ENPOINTS) {
  run_current_test<int, int, float, float, true, false>(GetParam());
}

// FIXME: There is an InvalidValue on a Memcopy only on tests/datasets/dblp.mtx
INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_BC,
  ::testing::Values(
      BC_Usecase("test/datasets/karate.mtx", 0),
      BC_Usecase("test/datasets/karate.mtx", 4),
      BC_Usecase("test/datasets/karate.mtx", 10),
      BC_Usecase("test/datasets/polbooks.mtx", 0),
      BC_Usecase("test/datasets/polbooks.mtx", 4),
      BC_Usecase("test/datasets/polbooks.mtx", 10),
      BC_Usecase("test/datasets/netscience.mtx", 0),
      BC_Usecase("test/datasets/netscience.mtx", 4),
      BC_Usecase("test/datasets/netscience.mtx", 100),
      BC_Usecase("test/datasets/wiki2003.mtx", 100),
      BC_Usecase("test/datasets/wiki2003.mtx", 1000)
    )
);


int main( int argc, char** argv )
{
  rmmInitialize(nullptr);
  testing::InitGoogleTest(&argc,argv);
  int rc = RUN_ALL_TESTS();
  rmmFinalize();
  return rc;
}
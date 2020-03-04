/*
 * Copyright (c) 2020 NVIDIA CORPORATION.
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

// Author: Xavier Cadet xcadet@nvidia.com
#include <fstream>
#include <gtest/gtest.h>
#include <cugraph.h>

#include "test_utils.h"

//TODO(xcadet) Need remove this
#include <iostream>
#include <fstream>

// We assume here that the betweennees are ordered based on Vertices Indices
template<typename WT>
std::vector<WT> extract_ref_betweenness(std::ifstream &fs_ref) {
  std::vector<WT> vec;
  WT val;
  while (fs_ref >> val) {
    vec.push_back(val);
  }
  return vec;
}


typedef struct BC_Usecase_t {
  std::string graph_file;
  std::string ref_file;
  BC_Usecase_t(const std::string& a, const std::string& b) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      graph_file = rapidsDatasetRootDir + "/" + a;
    } else {
      graph_file = a;
    }
    if ((b != "") && (b[0] != '/')) {
      ref_file = rapidsDatasetRootDir + "/" + b;
    } else {
      ref_file = b;
    }
  }
  BC_Usecase_t& operator=(const BC_Usecase_t& rhs) {
    graph_file = rhs.graph_file;
    ref_file = rhs.ref_file;
    return *this;
  }
} BC_Usecase;

class Tests_BC  : public ::testing::TestWithParam<BC_Usecase> {
public:
  Tests_BC() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  template<typename VT, typename ET, typename WT>
  void run_current_test(const BC_Usecase& param) {
    gdf_column col_src, col_dest, col_weights;
    WT *betweenness_centrality = nullptr;

    VT num_vertices;
    //ET num_edges;

    // Input:
    // --- Verify that graph_file and ref_file can be opened ---
    FILE *fp_in = fopen(param.graph_file.c_str(), "r");
    ASSERT_NE(fp_in, static_cast<FILE*>(nullptr)) << "fopen (" << param.graph_file << ") failure.";

    std::ifstream fs_ref(param.ref_file);
    ASSERT_TRUE(fs_ref.is_open()) << "fopen (" << param.ref_file << ") failure.";

    // --- Set up gdf_columns ---
    // Check and prepare col_src
    col_src.data = nullptr;
    if (std::is_same<VT, int>::value)
      col_src.dtype = GDF_INT32;
    else
      ASSERT_TRUE(0); //  We don't have support for other types yet
    col_src.valid = nullptr;
    col_src.null_count = 0;

    // Check and prepare col_dest
    col_dest.data = nullptr;
    if (std::is_same<VT, int>::value)
      col_dest.dtype = GDF_INT32;
    else
      ASSERT_TRUE(0); //  We don't have support for other types yet
    col_dest.valid = nullptr;
    col_dest.null_count = 0;

    // Check and prepare col_weigts
    col_weights.data = nullptr;
    if (std::is_same<WT, float>::value)
      col_weights.dtype = GDF_FLOAT32;
    else if (std::is_same<WT, double>::value)
      col_weights.dtype = GDF_FLOAT64;
    else
      ASSERT_TRUE(0); //  We don't have support for other types yet
    col_weights.valid = nullptr;
    col_weights.null_count = 0;
    // --- MTX format is expected ---
    VT m = 0;       // m:   Number of rows
    VT n = 0;       // n:   Number of columns
    ET nz = 0;      // nz:  Number of non 0 values
    MM_typecode mc; // mc:  MatrixMarket type

    ASSERT_EQ(mm_properties<VT>(fp_in, 1, &mc, &m, &n, &nz),0) << "could not read Matrix Market file properties"<< "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    // --- Memory allocation ---
    // Allocation on Host
    std::vector<VT> cooRowInd(nz), cooColInd(nz);
    std::vector<WT> cooVal(nz); // TODO(xcadet) need to look more into this aspect
    // --- Filling cooRowInd, cooColInd and cooVal ---
    if (!mm_is_pattern(mc)) {
      ASSERT_EQ((mm_to_coo<VT, WT>(fp_in, 1, nz, &cooRowInd[0], &cooColInd[0],
                                   &cooVal[0], nullptr)),
                 0) << "could not read matrix data"<< "\n";
    } else { // This is an unweighted graph
      ASSERT_EQ((mm_to_coo<VT, WT>(fp_in, 1, nz, &cooRowInd[0], &cooColInd[0],
                                   nullptr, nullptr)),
                 0) << "could not read matrix data"<< "\n";
      std::fill(cooVal.begin(), cooVal.end(), static_cast<WT>(1));
    }
    // --- Read Betweenness from ref_file ---
    std::vector<WT> ref_betweenness_centrality = extract_ref_betweenness<WT>(fs_ref);

    // --- Closing files ---
    ASSERT_EQ(fclose(fp_in), 0);
    fs_ref.close();

    // TODO(xcadet) Update needed for new Graph without gdf_column
    create_gdf_column(cooRowInd, &col_src);
    create_gdf_column(cooColInd, &col_dest);
    create_gdf_column(cooVal, &col_weights);

    num_vertices = m;
    //num_edges = nz;

    // --- Generating the Graph ---
    /*
    GraphCOO(VT const *src_indices_, VT const *dst_indices_, WT const *edge_data_,
           VT number_of_vertices_, ET number_of_edges_):
    */
    cugraph::Graph G;
    cugraph::edge_list_view(&G, &col_src, &col_dest, &col_weights);
    cugraph::add_adj_list(&G);

    // Host Alloc
    std::vector<WT> betweenness_centrality_vec;//(num_vertices, static_cast<WT>(0));

    // Device alloc
    rmm::device_vector<WT> dbetweenness_centrality_vec;

    betweenness_centrality_vec = std::vector<WT>(num_vertices, static_cast<WT>(0));
    dbetweenness_centrality_vec.resize(num_vertices);
    thrust::fill(dbetweenness_centrality_vec.begin(), dbetweenness_centrality_vec.end(), static_cast<WT>(0));
    betweenness_centrality = thrust::raw_pointer_cast(dbetweenness_centrality_vec.data());

    // --- Betweenness Centrality call ---
    cugraph::betweenness_centrality<VT, ET, WT>(&G, betweenness_centrality);
    cudaDeviceSynchronize();

    // MTX may have zero-degree vertices. So reset num_vertices after
    // conversion to CSR
    num_vertices = G.adjList->offsets->size - 1;

    // --- Copy back to host ---
    cudaMemcpy((void*) &betweenness_centrality_vec[0],
              betweenness_centrality,
              sizeof(WT) * num_vertices,
              cudaMemcpyDeviceToHost);
    // --- Actual tests ---
    for (auto idx = 0; idx < num_vertices;  ++idx) {
      ASSERT_EQ(betweenness_centrality_vec[idx], ref_betweenness_centrality[idx])
      << "idx: " << idx << " ref betweenness " << ref_betweenness_centrality[idx]
      << " actual betweenness " << betweenness_centrality_vec[idx];
    }
  }
};

// TODO(xcadet): Adding some small tests for local use, need to updated for last version
INSTANTIATE_TEST_CASE_P(simple_test, Tests_BC,
                        ::testing::Values(BC_Usecase("../../bc_simple_data/data/line3.mtx", "../../bc_simple_data/ref/line3.csv")//,
                                          //BC_Usecase("../../bc_simple_data/data/line4.mtx", "../../bc_simple_data/ref/line4.csv"),
                                          //BC_Usecase("karate.mtx", "../../bc_simple_data/ref/karate.csv")//,
                                          //BC_Usecase("../../bc_simple_data/data/bridge3.mtx", "../../bc_simple_data/ref/bridge3.csv"),
                                          //BC_Usecase("../../bc_simple_data/data/bridge4.mtx", "../../bc_simple_data/ref/bridge4.csv")
                                         )
                       );

TEST_P(Tests_BC, CheckFP32) {
  run_current_test<int, int, float>(GetParam());
}

TEST_P(Tests_BC, CheckFP64) {
  run_current_test<int, int, double>(GetParam());
}

int main(int argc, char **argv) {
  int rc = 0;

  rmmInitialize(nullptr);
  testing::InitGoogleTest(&argc, argv);
  rc = RUN_ALL_TESTS();
  rmmFinalize();

  return rc;
}
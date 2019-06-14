/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <gtest/gtest.h>
#include <cugraph.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <queue>
#include <nvgraph/nvgraph.h>
#include <nvgraph_gdf.h>
#include "test_utils.h"
#include "high_res_clock.h"

typedef enum graph_type { RMAT, MTX } GraphType;

template <typename MaxEType, typename MaxVType, typename DistType>
void ref_bfs(const std::vector<MaxEType>& rowPtr,
             const std::vector<MaxVType>& colInd,
             const MaxVType source_vertex,
             std::vector<DistType>& distances,
             std::vector<MaxVType>& predecessors) {
  size_t n = rowPtr.size() - 1;
  size_t nnz = colInd.size();

  ASSERT_LE(n, std::numeric_limits<MaxVType>::max() - 1);
  ASSERT_LE(nnz, std::numeric_limits<MaxEType>::max());
  ASSERT_EQ(distances.size(), rowPtr.size() - 1);

  std::fill(distances.begin(),
            distances.end(),
            std::numeric_limits<DistType>::max());
  std::fill(predecessors.begin(), predecessors.end(), -1);

  std::queue<MaxVType> q;
  q.push(source_vertex);
  distances[source_vertex] = 0;

  while (!q.empty()) {
    MaxVType u = q.front();
    q.pop();

    for (auto iCol = rowPtr[u]; iCol != rowPtr[u + 1]; ++iCol) {
      MaxVType v = colInd[iCol];
      // undiscovered
      if (distances[v] == std::numeric_limits<DistType>::max()) {
        distances[v] = distances[u] + 1;
        predecessors[v] = u;
        q.push(v);
      }
    }
  }
}

template <typename MaxEType, typename MaxVType, typename DistType>
void ref_sssp(const std::vector<MaxEType>& rowPtr,
              const std::vector<MaxVType>& colInd,
              const std::vector<DistType>& weights,
              const MaxVType source_vertex,
              std::vector<DistType>& distances,
              std::vector<MaxVType>& predecessors) {
  size_t n = rowPtr.size() - 1;
  size_t nnz = colInd.size();

  ASSERT_LE(n, std::numeric_limits<MaxVType>::max() - 1);
  ASSERT_LE(nnz, std::numeric_limits<MaxEType>::max());
  ASSERT_EQ(nnz, weights.size());
  ASSERT_EQ(distances.size(), rowPtr.size() - 1);

  std::fill(distances.begin(),
            distances.end(),
            std::numeric_limits<DistType>::max());
  std::fill(predecessors.begin(), predecessors.end(), -1);

  std::set<MaxVType> curr_frontier;
  curr_frontier.insert(source_vertex);
  distances[source_vertex] = 0;
  MaxVType nf = 1;

  while (nf > 0) {
    std::set<MaxVType> next_frontier;
    for (auto it = curr_frontier.begin(); it != curr_frontier.end(); ++it) {
      MaxVType u = *it;

      for (auto iCol = rowPtr[u]; iCol != rowPtr[u + 1]; ++iCol) {
        MaxVType v = colInd[iCol];
        // relax
        if (distances[u] + weights[iCol] < distances[v]) {
          distances[v] = distances[u] + weights[iCol];
          next_frontier.insert(v);
          predecessors[v] = u;
        }
      }
    }

    curr_frontier = next_frontier;
    nf = curr_frontier.size();
  }
}

// do the perf measurements
// enabled by command line parameter s'--perf'
static int PERF = 0;

// iterations for perf tests
// enabled by command line parameter '--perf-iters"
static int PERF_MULTIPLIER = 5;

typedef struct SSSP_Usecase_t {
  GraphType type_;
  std::string config_;
  std::string file_path_;
  int src_;
  SSSP_Usecase_t(const GraphType& type,
                 const std::string& config,
                 const int src)
      : type_(type), config_(config), src_(src) {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // FIXME: Use platform independent stuff from c++14/17 on compiler update
    if (type_ == MTX) {
      const std::string& rapidsDatasetRootDir = get_rapids_dataset_root_dir();
      if ((config_ != "") && (config_[0] != '/')) {
        file_path_ = rapidsDatasetRootDir + "/" + config_;
      } else {
        file_path_ = config_;
      }
    }
  };
} SSSP_Usecase;

class Tests_SSSP : public ::testing::TestWithParam<SSSP_Usecase> {
 public:
  Tests_SSSP() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {
    if (PERF) {
      for (unsigned int i = 0; i < SSSP_time.size(); ++i) {
        std::cout << SSSP_time[i] / PERF_MULTIPLIER << std::endl;
      }
    }
  }
  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> SSSP_time;

  template <typename DistType, bool DoDist, bool DoPreds>
  void run_current_test(const SSSP_Usecase& param) {
    gdf_column col_src, col_dest, col_weights, col_distances, col_preds;

    size_t v, e;
    // Input
    col_src.data = nullptr;
    col_src.dtype = GDF_INT32;
    col_src.valid = nullptr;
    col_src.null_count = 0;

    col_dest.data = nullptr;
    col_dest.dtype = GDF_INT32;
    col_dest.valid = nullptr;
    col_dest.null_count = 0;

    col_weights.data = nullptr;
    if (std::is_same<DistType, float>::value)
      col_weights.dtype = GDF_FLOAT32;
    else if (std::is_same<DistType, double>::value)
      col_weights.dtype = GDF_FLOAT64;
    else
      ASSERT_TRUE(0);
    col_weights.valid = nullptr;
    col_weights.null_count = 0;

    // Output
    col_distances.data = nullptr;
    col_distances.size = 0;
    col_preds.data = nullptr;
    col_preds.size = 0;

    if (param.type_ == RMAT) {
      ASSERT_EQ(
          gdf_grmat_gen(
              param.config_.c_str(), v, e, &col_src, &col_dest, &col_weights),
          GDF_SUCCESS);
    } else if (param.type_ == MTX) {
      int m, k, nnz;
      MM_typecode mc;

      FILE* fpin = fopen(param.file_path_.c_str(), "r");

      ASSERT_EQ(mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz), 0)
          << "could not read Matrix Market file properties"
          << "\n";
      ASSERT_TRUE(mm_is_matrix(mc));
      ASSERT_TRUE(mm_is_coordinate(mc));
      ASSERT_FALSE(mm_is_complex(mc));
      ASSERT_FALSE(mm_is_skew(mc));

      // Allocate memory on host
      std::vector<int> cooRowInd(nnz), cooColInd(nnz);
      std::vector<DistType> cooVal;

      // Read weights if given
      if (!mm_is_pattern(mc)) {
        cooVal.resize(nnz);
        ASSERT_EQ(
            (mm_to_coo<int, DistType>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)),
            0)
            << "could not read matrix data"
            << "\n";
      } else {
        ASSERT_EQ((mm_to_coo<int, DistType>(
                      fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], NULL, NULL)),
                  0)
            << "could not read matrix data"
            << "\n";
        // Set random weights
        if (std::is_same<DistType, float>::value ||
            std::is_same<DistType, double>::value) {
          cooVal.resize(nnz);
          for (auto i = 0; i < nnz; i++) {
            cooVal[i] = static_cast<DistType>(rand()) /
                static_cast<DistType>(RAND_MAX);
          }
        }
      }

      ASSERT_EQ(fclose(fpin), 0);
      // gdf columns
      create_gdf_column(cooRowInd, &col_src);
      create_gdf_column(cooColInd, &col_dest);
      create_gdf_column(cooVal, &col_weights);
      v = m;
      e = nnz;
    } else {
      ASSERT_TRUE(0);
    }

    // std::cout << "v = " << v << "e = " << e << "\n";
    gdf_graph G;
    ASSERT_EQ(gdf_edge_list_view(&G, &col_src, &col_dest, &col_weights),
              GDF_SUCCESS);

    std::vector<DistType> dist_vec;
    std::vector<int32_t> pred_vec;

    if (DoDist) {
      dist_vec =
          std::vector<DistType>(v, std::numeric_limits<DistType>::max());
      create_gdf_column(dist_vec, &col_distances);
    }

    if (DoPreds) {
      pred_vec = std::vector<int32_t>(v, -1);
      create_gdf_column(pred_vec, &col_preds);
    }

    gdf_error ret;
    HighResClock hr_clock;
    double time_tmp;

    cudaDeviceSynchronize();
    if (PERF) {
      hr_clock.start();
      for (int i = 0; i < PERF_MULTIPLIER; ++i) {
        ret = gdf_sssp(&G, &col_distances, &col_preds, param.src_);
        cudaDeviceSynchronize();
      }
      hr_clock.stop(&time_tmp);
      SSSP_time.push_back(time_tmp);
    } else {
      ret = gdf_sssp(&G, &col_distances, &col_preds, param.src_);
      cudaDeviceSynchronize();
    }

    ASSERT_EQ(ret, GDF_SUCCESS);

    // MTX may have zero-degree vertices. So reset v after conversion to CSR
    v = G.adjList->offsets->size - 1;

    if (DoDist)
      cudaMemcpy((void*)&dist_vec[0],
                 col_distances.data,
                 sizeof(DistType) * v,
                 cudaMemcpyDeviceToHost);

    if (DoPreds)
      cudaMemcpy((void*)&pred_vec[0],
                 col_preds.data,
                 sizeof(int32_t) * v,
                 cudaMemcpyDeviceToHost);

    // Create ref host structures

    std::vector<int32_t> vlist(v + 1), elist(e);
    std::vector<DistType> ref_distances(v), weights(e);
    std::vector<int32_t> ref_predecessors(v);

    cudaMemcpy((void*)&vlist[0],
               G.adjList->offsets->data,
               sizeof(int32_t) * (v + 1),
               cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&elist[0],
               G.adjList->indices->data,
               sizeof(int32_t) * (e),
               cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&weights[0],
               G.adjList->edge_data->data,
               sizeof(DistType) * (e),
               cudaMemcpyDeviceToHost);

    std::unordered_map<uint64_t, DistType> min_edge_map;

    if (DoPreds) {
      for (auto i = 0; i < v; ++i) {
        for (auto e = vlist[i]; e < vlist[i + 1]; ++e) {
          DistType weight = weights[e];
          uint64_t key = (uint64_t)i << 32 | (uint64_t)elist[e];
          if (min_edge_map.find(key) != min_edge_map.end()) {
            min_edge_map[key] == std::min(weight, min_edge_map[key]);
          } else {
            min_edge_map[key] = weight;
          }
        }
      }
    }

    ref_sssp(
        vlist, elist, weights, param.src_, ref_distances, ref_predecessors);

    for (auto i = 0; i < v; ++i) {
      if (DoDist)
        ASSERT_EQ(dist_vec[i], ref_distances[i])
            << "vid: " << i << "ref dist " << ref_distances[i]
            << " actual dist " << dist_vec[i];

      if (DoPreds) {
        if (pred_vec[i] != -1) {
          uint64_t key = (uint64_t)pred_vec[i] << 32 | (uint64_t)i;
          DistType min_edge_weight = min_edge_map.at(key);
          ASSERT_EQ(ref_distances[pred_vec[i]] + min_edge_weight,
                    ref_distances[i])
              << "vid: " << i << "pred " << pred_vec[i] << " ref dist "
              << ref_distances[i] << " observed " << ref_distances[pred_vec[i]]
              << " + " << min_edge_weight << " = "
              << ref_distances[pred_vec[i]] + min_edge_weight << "\n";
        } else {
          ASSERT_EQ(pred_vec[i], ref_predecessors[i])
              << "vid: " << i << "ref pred " << ref_predecessors[i]
              << " actual " << pred_vec[i];
        }
      }
    }
  }
};

std::vector<double> Tests_SSSP::SSSP_time;

TEST_P(Tests_SSSP, CheckFP32_DIST_NO_PREDS) {
  run_current_test<float, true, false>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP32_NO_DIST_PREDS) {
  run_current_test<float, false, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP32_DIST_PREDS) {
  run_current_test<float, true, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_DIST_NO_PREDS) {
  run_current_test<double, true, false>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_NO_DIST_PREDS) {
  run_current_test<double, false, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_DIST_PREDS) {
  run_current_test<double, true, true>(GetParam());
}

// --gtest_filter=*simple_test*

INSTANTIATE_TEST_CASE_P(
    simple_test,
    Tests_SSSP,
    ::testing::Values(
        SSSP_Usecase(RMAT,
                     "grmat --rmat_scale=10 --rmat_edgefactor=16 --device=0  "
                     "--normalized --quiet",
                     0),
        SSSP_Usecase(RMAT,
                     "grmat --rmat_scale=12 --rmat_edgefactor=8 --device=0  "
                     "--normalized --quiet",
                     10),
        SSSP_Usecase(MTX, "test/datasets/dblp.mtx", 100),
        SSSP_Usecase(MTX, "test/datasets/wiki2003.mtx", 100000),
        SSSP_Usecase(MTX, "test/datasets/karate.mtx", 1)
        //,SSSP_Usecase(MTX, "test/datasets/cit-Patents.mtx", 100)
        ));

int main(int argc, char** argv) {
  srand(42);
  ::testing::InitGoogleTest(&argc, argv);

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--perf") == 0) {
      PERF = 1;
    }
    if (strcmp(argv[i], "--perf-iters") == 0) {
      PERF_MULTIPLIER = atoi(argv[i + 1]);
    }
  }
  return RUN_ALL_TESTS();
}

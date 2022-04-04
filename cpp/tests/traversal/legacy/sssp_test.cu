/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited
 *
 */

#include <utilities/base_fixture.hpp>
#include <utilities/high_res_clock.h>
#include <utilities/test_utilities.hpp>

#include <rmm/device_vector.hpp>

#include <converters/COOtoCSR.cuh>
#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <rmm/device_vector.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <utility>

typedef enum graph_type { RMAT, MTX } GraphType;

template <typename MaxEType, typename MaxVType, typename DistType>
void ref_bfs(const std::vector<MaxEType>& rowPtr,
             const std::vector<MaxVType>& colInd,
             const MaxVType source_vertex,
             std::vector<DistType>& distances,
             std::vector<MaxVType>& predecessors)
{
  typename std::vector<MaxEType>::size_type n   = rowPtr.size() - 1;
  typename std::vector<MaxVType>::size_type nnz = colInd.size();

  ASSERT_LE(n, static_cast<decltype(n)>(std::numeric_limits<MaxVType>::max()) - 1);
  ASSERT_LE(nnz, static_cast<decltype(nnz)>(std::numeric_limits<MaxEType>::max()));
  ASSERT_EQ(distances.size(), rowPtr.size() - 1);

  std::fill(distances.begin(), distances.end(), std::numeric_limits<DistType>::max());
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
        distances[v]    = distances[u] + 1;
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
              std::vector<MaxVType>& predecessors)
{
  typename std::vector<MaxEType>::size_type n   = rowPtr.size() - 1;
  typename std::vector<MaxVType>::size_type nnz = colInd.size();

  ASSERT_LE(n, static_cast<decltype(n)>(std::numeric_limits<MaxVType>::max()) - 1);
  ASSERT_LE(nnz, static_cast<decltype(nnz)>(std::numeric_limits<MaxEType>::max()));
  ASSERT_EQ(nnz, weights.size());
  ASSERT_EQ(distances.size(), rowPtr.size() - 1);

  std::fill(distances.begin(), distances.end(), std::numeric_limits<DistType>::max());
  std::fill(predecessors.begin(), predecessors.end(), -1);

  std::set<MaxVType> curr_frontier;
  curr_frontier.insert(source_vertex);
  distances[source_vertex] = 0;
  MaxVType nf              = 1;

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
    nf            = curr_frontier.size();
  }
}

// iterations for perf tests
static int PERF_MULTIPLIER = 5;

typedef struct SSSP_Usecase_t {
  GraphType type_;
  std::string config_;
  std::string file_path_;
  uint64_t src_;
  SSSP_Usecase_t(const GraphType& type, const std::string& config, const int src)
    : type_(type), config_(config), src_(src)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    // FIXME: Use platform independent stuff from c++14/17 on compiler update
    if (type_ == MTX) {
      const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
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
  static void TearDownTestCase()
  {
    if (cugraph::test::g_perf) {
      for (size_t i = 0; i < SSSP_time.size(); ++i) {
        std::cout << SSSP_time[i] / PERF_MULTIPLIER << std::endl;
      }
    }
  }
  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> SSSP_time;

  // MaxVType:        Data type of vertices id (signed int-32)
  // MaxEType:        Data type of edges id (signed int-32)
  // DistType:        Data type of weights (float / double)
  // DoRandomWeights: SSSP Implementation requires weights to operate,
  //                    if true: generate random weights before calling
  //                    else:    relies on self inner code using fake 1.0 weights
  // DoDist:          User can provide weights or not, simulate this behavior
  // DoPreds:         User can provide predecessors or not, simulate this behavior
  template <typename MaxVType,
            typename MaxEType,
            typename DistType,
            bool DoRandomWeights,
            bool DoDist,
            bool DoPreds>
  void run_current_test(const SSSP_Usecase& param)
  {
    // Allocate memory on host (We will resize later on)
    std::vector<MaxVType> cooRowInd;
    std::vector<MaxVType> cooColInd;
    std::vector<DistType> cooVal;

    DistType* distances = nullptr;
    MaxVType* preds     = nullptr;

    MaxVType num_vertices;
    MaxEType num_edges;
    const MaxVType src = param.src_;

    ASSERT_LE(param.src_, static_cast<uint64_t>(std::numeric_limits<MaxVType>::max()));
    // src = static_cast<MaxVType>(param.src_);

    // Input
    ASSERT_TRUE(typeid(MaxVType) == typeid(int));  // We don't have support for other types yet
    ASSERT_TRUE(typeid(MaxEType) == typeid(int));  // We don't have support for other types yet
    ASSERT_TRUE((typeid(DistType) == typeid(float)) || (typeid(DistType) == typeid(double)));
    if (param.type_ == RMAT) {
      // This is size_t due to grmat_gen which should be fixed there
      // FIXME: rmat is disabled
      return;
    } else if (param.type_ == MTX) {
      MaxVType m, k;
      MaxEType nnz;
      MM_typecode mc;

      FILE* fpin = fopen(param.file_path_.c_str(), "r");
      ASSERT_NE(fpin, static_cast<FILE*>(nullptr)) << "fopen (" << param.file_path_ << ") failure.";

      // mm_properties has only one template param which should be fixed there
      ASSERT_EQ(cugraph::test::mm_properties<MaxVType>(fpin, 1, &mc, &m, &k, &nnz), 0)
        << "could not read Matrix Market file properties"
        << "\n";
      ASSERT_TRUE(mm_is_matrix(mc));
      ASSERT_TRUE(mm_is_coordinate(mc));
      ASSERT_FALSE(mm_is_complex(mc));
      ASSERT_FALSE(mm_is_skew(mc));

      // Allocate memory on host
      cooRowInd.resize(nnz);
      cooColInd.resize(nnz);

      // Read weights if given
      if (!mm_is_pattern(mc)) {
        cooVal.resize(nnz);
        ASSERT_EQ((cugraph::test::mm_to_coo(fpin,
                                            1,
                                            nnz,
                                            &cooRowInd[0],
                                            &cooColInd[0],
                                            &cooVal[0],
                                            static_cast<DistType*>(nullptr))),
                  0)
          << "could not read matrix data"
          << "\n";
      } else {
        ASSERT_EQ((cugraph::test::mm_to_coo(fpin,
                                            1,
                                            nnz,
                                            &cooRowInd[0],
                                            &cooColInd[0],
                                            static_cast<DistType*>(nullptr),
                                            static_cast<DistType*>(nullptr))),
                  0)
          << "could not read matrix data"
          << "\n";
        // Set random weights
        if (std::is_same<DistType, float>::value || std::is_same<DistType, double>::value) {
          cooVal.resize(nnz);
          for (auto i = 0; i < nnz; i++) {
            cooVal[i] = static_cast<DistType>(rand()) / static_cast<DistType>(RAND_MAX);
          }
        }
      }

      ASSERT_EQ(fclose(fpin), 0);

      num_vertices = m;
      num_edges    = nnz;
    } else {
      ASSERT_TRUE(0);
    }

    cugraph::legacy::GraphCOOView<MaxVType, MaxEType, DistType> G_coo(
      &cooRowInd[0],
      &cooColInd[0],
      (DoRandomWeights ? &cooVal[0] : nullptr),
      num_vertices,
      num_edges);
    auto G_unique                                                 = cugraph::coo_to_csr(G_coo);
    cugraph::legacy::GraphCSRView<MaxVType, MaxEType, DistType> G = G_unique->view();
    cudaDeviceSynchronize();

    std::vector<DistType> dist_vec;
    std::vector<MaxVType> pred_vec;
    rmm::device_vector<DistType> ddist_vec;
    rmm::device_vector<MaxVType> dpred_vec;

    if (DoDist) {
      dist_vec = std::vector<DistType>(num_vertices, std::numeric_limits<DistType>::max());
      // device alloc
      ddist_vec.resize(num_vertices);
      thrust::fill(ddist_vec.begin(), ddist_vec.end(), std::numeric_limits<DistType>::max());
      distances = thrust::raw_pointer_cast(ddist_vec.data());
    }

    if (DoPreds) {
      pred_vec = std::vector<MaxVType>(num_vertices, -1);
      dpred_vec.resize(num_vertices);
      preds = thrust::raw_pointer_cast(dpred_vec.data());
    }

    HighResClock hr_clock;
    double time_tmp;

    cudaDeviceSynchronize();
    if (cugraph::test::g_perf) {
      hr_clock.start();
      for (auto i = 0; i < PERF_MULTIPLIER; ++i) {
        cugraph::sssp(G, distances, preds, src);
        cudaDeviceSynchronize();
      }
      hr_clock.stop(&time_tmp);
      SSSP_time.push_back(time_tmp);
    } else {
      cugraph::sssp(G, distances, preds, src);
      cudaDeviceSynchronize();
    }

    // MTX may have zero-degree vertices. So reset num_vertices after
    // conversion to CSR
    num_vertices = G.number_of_vertices;

    if (DoDist)
      cudaMemcpy(
        (void*)&dist_vec[0], distances, sizeof(DistType) * num_vertices, cudaMemcpyDeviceToHost);

    if (DoPreds)
      cudaMemcpy(
        (void*)&pred_vec[0], preds, sizeof(MaxVType) * num_vertices, cudaMemcpyDeviceToHost);

    // Create ref host structures
    std::vector<MaxEType> vlist(num_vertices + 1);
    std::vector<MaxVType> elist(num_edges);
    std::vector<DistType> ref_distances(num_vertices), weights(num_edges);
    std::vector<MaxVType> ref_predecessors(num_vertices);

    cudaMemcpy(
      (void*)&vlist[0], G.offsets, sizeof(MaxEType) * (num_vertices + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&elist[0], G.indices, sizeof(MaxVType) * (num_edges), cudaMemcpyDeviceToHost);
    if (G.edge_data != nullptr) {
      cudaMemcpy(
        (void*)&weights[0], G.edge_data, sizeof(DistType) * (num_edges), cudaMemcpyDeviceToHost);
    } else {  // If SSSP is given no weights it uses unit weights by default
      std::fill(weights.begin(), weights.end(), static_cast<DistType>(1));
    }

    std::map<std::pair<MaxVType, MaxVType>, DistType> min_edge_map;

    if (DoPreds) {
      for (auto i = 0; i < num_vertices; ++i) {
        for (auto offset = vlist[i]; offset < vlist[i + 1]; ++offset) {
          DistType weight = weights[offset];
          auto key        = std::make_pair(i, elist[offset]);
          if (min_edge_map.find(key) != min_edge_map.end()) {
            min_edge_map[key] = std::min(weight, min_edge_map[key]);
          } else {
            min_edge_map[key] = weight;
          }
        }
      }
    }

    ref_sssp(vlist, elist, weights, src, ref_distances, ref_predecessors);

    for (auto i = 0; i < num_vertices; ++i) {
      if (DoDist)
        ASSERT_EQ(dist_vec[i], ref_distances[i])
          << "vid: " << i << "ref dist " << ref_distances[i] << " actual dist " << dist_vec[i];

      if (DoPreds) {
        if (pred_vec[i] != -1) {
          auto key                 = std::make_pair(pred_vec[i], i);
          DistType min_edge_weight = min_edge_map.at(key);

          ASSERT_EQ(ref_distances[pred_vec[i]] + min_edge_weight, ref_distances[i])
            << "vid: " << i << "pred " << pred_vec[i] << " ref dist " << ref_distances[i]
            << " observed " << ref_distances[pred_vec[i]] << " + " << min_edge_weight << " = "
            << ref_distances[pred_vec[i]] + min_edge_weight << "\n";
        } else {
          ASSERT_EQ(pred_vec[i], ref_predecessors[i])
            << "vid: " << i << "ref pred " << ref_predecessors[i] << " actual " << pred_vec[i];
        }
      }
    }
  }
};

std::vector<double> Tests_SSSP::SSSP_time;

TEST_P(Tests_SSSP, CheckFP32_NO_RANDOM_DIST_NO_PREDS)
{
  run_current_test<int, int, float, false, true, false>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP32_NO_RANDOM_NO_DIST_PREDS)
{
  run_current_test<int, int, float, false, false, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP32_NO_RANDOM_DIST_PREDS)
{
  run_current_test<int, int, float, false, true, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_NO_RANDOM_DIST_NO_PREDS)
{
  run_current_test<int, int, double, false, true, false>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_NO_RANDOM_NO_DIST_PREDS)
{
  run_current_test<int, int, double, false, false, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_NO_RANDOM_DIST_PREDS)
{
  run_current_test<int, int, double, false, true, true>(GetParam());
}

// FIXME: There might be some tests that are done twice (MTX that are not patterns)
TEST_P(Tests_SSSP, CheckFP32_RANDOM_DIST_NO_PREDS)
{
  run_current_test<int, int, float, true, true, false>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP32_RANDOM_NO_DIST_PREDS)
{
  run_current_test<int, int, float, true, false, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP32_RANDOM_DIST_PREDS)
{
  run_current_test<int, int, float, true, true, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_RANDOM_DIST_NO_PREDS)
{
  run_current_test<int, int, double, true, true, false>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_RANDOM_NO_DIST_PREDS)
{
  run_current_test<int, int, double, true, false, true>(GetParam());
}
TEST_P(Tests_SSSP, CheckFP64_RANDOM_DIST_PREDS)
{
  run_current_test<int, int, double, true, true, true>(GetParam());
}

// --gtest_filter=*simple_test*

INSTANTIATE_TEST_SUITE_P(simple_test,
                         Tests_SSSP,
                         ::testing::Values(SSSP_Usecase(MTX, "test/datasets/dblp.mtx", 100),
                                           SSSP_Usecase(MTX, "test/datasets/wiki2003.mtx", 100000),
                                           SSSP_Usecase(MTX, "test/datasets/karate.mtx", 1)));

CUGRAPH_TEST_PROGRAM_MAIN()

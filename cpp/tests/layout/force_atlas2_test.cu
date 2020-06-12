/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Force_Atlas2 tests
// Author: Hugo Linsenmaier hlinsenmaier@nvidia.com

#include <rmm/thrust_rmm_allocator.h>
#include <algorithms.hpp>
#include <fstream>
#include <graph.hpp>
#include <iostream>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include "cuda_profiler_api.h"
#include "gtest/gtest.h"
#include "high_res_clock.h"
#include "trust_worthiness.h"
#include "utilities/test_utilities.hpp"

// do the perf measurements
// enabled by command line parameter s'--perf'
static int PERF = 0;

// iterations for perf tests
// enabled by command line parameter '--perf-iters"
static int PERF_MULTIPLIER = 5;

typedef struct Force_Atlas2_Usecase_t {
  std::string matrix_file;
  float score;
  Force_Atlas2_Usecase_t(const std::string& a, const float b)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
    score = b;
  }
  Force_Atlas2_Usecase_t& operator=(const Force_Atlas2_Usecase_t& rhs)
  {
    matrix_file = rhs.matrix_file;
    score       = rhs.score;
    return *this;
  }
} Force_Atlas2_Usecase;

class Tests_Force_Atlas2 : public ::testing::TestWithParam<Force_Atlas2_Usecase> {
 public:
  Tests_Force_Atlas2() {}
  static void SetupTestCase() {}
  static void TearDownTestCase()
  {
    if (PERF) {
      for (unsigned int i = 0; i < force_atlas2_time.size(); ++i) {
        std::cout << force_atlas2_time[i] / PERF_MULTIPLIER << std::endl;
      }
    }
  }
  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> force_atlas2_time;

  void compute_rank() {}

  void trustworthiness(float* X, float* Y) { return; }

  template <typename T>
  void run_current_test(const Force_Atlas2_Usecase& param)
  {
    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") +
                          std::string(test_info->name()) + std::string("_") +
                          cugraph::test::getFileName(param.matrix_file) + std::string("_") +
                          ss.str().c_str();

    int m, k, nnz;
    MM_typecode mc;
    HighResClock hr_clock;
    double time_tmp;

    FILE* fpin = fopen(param.matrix_file.c_str(), "r");
    ASSERT_NE(fpin, nullptr) << "fopen (" << param.matrix_file << ") failure.";
    ASSERT_EQ(cugraph::test::mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz), 0)
      << "could not read Matrix Market file properties"
      << "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_FALSE(mm_is_complex(mc));
    ASSERT_FALSE(mm_is_skew(mc));

    // Allocate memory on host
    std::vector<int> cooRowInd(nnz), cooColInd(nnz);
    std::vector<T> cooVal(nnz);
    std::vector<std::vector<int>> adj_matrix(m, std::vector<int>(m));
    std::vector<float> force_atlas2(m * 2);

    // device alloc
    rmm::device_vector<float> force_atlas2_vector(m * 2);
    float* d_force_atlas2 = force_atlas2_vector.data().get();

    // Read
    ASSERT_EQ((cugraph::test::mm_to_coo<int, T>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)),
              0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    // Build Adjacency Matrix
    for (int i = 0; i < nnz; ++i) {
      auto row             = cooRowInd[i];
      auto col             = cooColInd[i];
      adj_matrix[row][col] = 1;
    }

    // Allocate COO on device
    rmm::device_vector<int> srcs_v(nnz);
    rmm::device_vector<int> dests_v(nnz);
    rmm::device_vector<T> weights_v(nnz);

    int* srcs  = srcs_v.data().get();
    int* dests = dests_v.data().get();
    T* weights = weights_v.data().get();

    // FIXME: RAFT error handling mechanism should be used instead
    CUDA_RT_CALL(cudaMemcpy(srcs, &cooRowInd[0], sizeof(int) * nnz, cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy(dests, &cooColInd[0], sizeof(int) * nnz, cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy(weights, &cooVal[0], sizeof(T) * nnz, cudaMemcpyDefault));
    cugraph::experimental::GraphCOOView<int, int, T> G(srcs, dests, weights, m, nnz);

    const int max_iter                    = 500;
    float* x_start                        = nullptr;
    float* y_start                        = nullptr;
    bool outbound_attraction_distribution = false;
    bool lin_log_mode                     = false;
    bool prevent_overlapping              = false;
    const float edge_weight_influence     = 1.0;
    const float jitter_tolerance          = 1.0;
    bool optimize                         = true;
    const float theta                     = 1.0;
    const float scaling_ratio             = 2.0;
    bool strong_gravity_mode              = false;
    const float gravity                   = 1.0;
    bool verbose                          = false;

    if (PERF) {
      hr_clock.start();
      for (int i = 0; i < PERF_MULTIPLIER; ++i) {
        cugraph::force_atlas2<int, int, T>(G,
                                           d_force_atlas2,
                                           max_iter,
                                           x_start,
                                           y_start,
                                           outbound_attraction_distribution,
                                           lin_log_mode,
                                           prevent_overlapping,
                                           edge_weight_influence,
                                           jitter_tolerance,
                                           optimize,
                                           theta,
                                           scaling_ratio,
                                           strong_gravity_mode,
                                           gravity,
                                           verbose);
        cudaDeviceSynchronize();
      }
      hr_clock.stop(&time_tmp);
      force_atlas2_time.push_back(time_tmp);
    } else {
      cudaProfilerStart();
      cugraph::force_atlas2<int, int, T>(G,
                                         d_force_atlas2,
                                         max_iter,
                                         x_start,
                                         y_start,
                                         outbound_attraction_distribution,
                                         lin_log_mode,
                                         prevent_overlapping,
                                         edge_weight_influence,
                                         jitter_tolerance,
                                         optimize,
                                         theta,
                                         scaling_ratio,
                                         strong_gravity_mode,
                                         gravity,
                                         verbose);
      cudaProfilerStop();
      cudaDeviceSynchronize();
    }

    // Copy pos to host
    std::vector<float> h_pos(m * 2);
    CUDA_RT_CALL(
      cudaMemcpy(&h_pos[0], d_force_atlas2, sizeof(float) * m * 2, cudaMemcpyDeviceToHost));

    // Transpose the data
    std::vector<std::vector<double>> C_contiguous_embedding(m, std::vector<double>(2));
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < 2; j++) C_contiguous_embedding[i][j] = h_pos[j * m + i];
    }

    // Test trustworthiness
    double score_bh = trustworthiness_score(adj_matrix, C_contiguous_embedding, m, 2, 5);
    printf("score: %f\n", score_bh);
    ASSERT_GT(score_bh, param.score);
  }
};

std::vector<double> Tests_Force_Atlas2::force_atlas2_time;

TEST_P(Tests_Force_Atlas2, CheckFP32_T) { run_current_test<float>(GetParam()); }

TEST_P(Tests_Force_Atlas2, CheckFP64_T) { run_current_test<double>(GetParam()); }

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_Force_Atlas2,
                        ::testing::Values(Force_Atlas2_Usecase("test/datasets/karate.mtx", 0.73),
                                          Force_Atlas2_Usecase("test/datasets/dolphins.mtx", 0.69),
                                          Force_Atlas2_Usecase("test/datasets/polbooks.mtx", 0.76),
                                          Force_Atlas2_Usecase("test/datasets/netscience.mtx",
                                                               0.80)));

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}

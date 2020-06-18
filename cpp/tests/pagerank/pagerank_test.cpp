/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Pagerank solver tests
// Author: Alex Fender afender@nvidia.com

#include "high_res_clock.h"
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <graph.hpp>

#include <raft/error.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include "cuda_profiler_api.h"

#include "gtest/gtest.h"

#include <cmath>

// do the perf measurements
// enabled by command line parameter s'--perf'
static int PERF = 0;

// iterations for perf tests
// enabled by command line parameter '--perf-iters"
static int PERF_MULTIPLIER = 5;

typedef struct Pagerank_Usecase_t {
  std::string matrix_file;
  std::string result_file;
  Pagerank_Usecase_t(const std::string& a, const std::string& b)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
    if ((b != "") && (b[0] != '/')) {
      result_file = rapidsDatasetRootDir + "/" + b;
    } else {
      result_file = b;
    }
  }
  Pagerank_Usecase_t& operator=(const Pagerank_Usecase_t& rhs)
  {
    matrix_file = rhs.matrix_file;
    result_file = rhs.result_file;
    return *this;
  }
} Pagerank_Usecase;

class Tests_Pagerank : public ::testing::TestWithParam<Pagerank_Usecase> {
 public:
  Tests_Pagerank() {}
  static void SetupTestCase() {}
  static void TearDownTestCase()
  {
    if (PERF) {
      for (unsigned int i = 0; i < pagerank_time.size(); ++i) {
        std::cout << pagerank_time[i] / PERF_MULTIPLIER << std::endl;
      }
    }
  }
  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> pagerank_time;

  template <typename T>
  void run_current_test(const Pagerank_Usecase& param)
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

    float tol = 1E-5f;

    // Default parameters
    /*
    float alpha = 0.85;
    int max_iter = 500;
    bool has_guess = false;
    */

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
    std::vector<T> cooVal(nnz), pagerank(m);

    // device alloc
    rmm::device_uvector<T> pagerank_vector(static_cast<size_t>(m), nullptr);
    T* d_pagerank = pagerank_vector.data();

    // Read
    ASSERT_EQ((cugraph::test::mm_to_coo<int, T>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)),
              0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    //  Pagerank runs on CSC, so feed COOtoCSR the row/col backwards.
    cugraph::experimental::GraphCOOView<int, int, T> G_coo(
      &cooColInd[0], &cooRowInd[0], &cooVal[0], m, nnz);
    auto G_unique = cugraph::coo_to_csr(G_coo);
    cugraph::experimental::GraphCSCView<int, int, T> G(G_unique->view().offsets,
                                                       G_unique->view().indices,
                                                       G_unique->view().edge_data,
                                                       G_unique->view().number_of_vertices,
                                                       G_unique->view().number_of_edges);

    cudaDeviceSynchronize();
    if (PERF) {
      hr_clock.start();
      for (int i = 0; i < PERF_MULTIPLIER; ++i) {
        cugraph::pagerank<int, int, T>(G.handle[0], G, d_pagerank);
        cudaDeviceSynchronize();
      }
      hr_clock.stop(&time_tmp);
      pagerank_time.push_back(time_tmp);
    } else {
      cudaProfilerStart();
      cugraph::pagerank<int, int, T>(G.handle[0], G, d_pagerank);
      cudaProfilerStop();
      cudaDeviceSynchronize();
    }

    // Check vs golden data
    if (param.result_file.length() > 0) {
      std::vector<T> calculated_res(m);

      CUDA_TRY(
        cudaMemcpy(&calculated_res[0], d_pagerank, sizeof(T) * m, cudaMemcpyDeviceToHost));
      std::sort(calculated_res.begin(), calculated_res.end());
      fpin = fopen(param.result_file.c_str(), "rb");
      ASSERT_TRUE(fpin != NULL) << " Cannot read file with reference data: " << param.result_file
                                << std::endl;
      std::vector<T> expected_res(m);
      ASSERT_EQ(cugraph::test::read_binary_vector(fpin, m, expected_res), 0);
      fclose(fpin);
      T err;
      int n_err = 0;
      for (int i = 0; i < m; i++) {
        err = fabs(expected_res[i] - calculated_res[i]);
        if (err > tol * 1.1) {
          n_err++;  // count the number of mismatches
        }
      }
      if (n_err) {
        EXPECT_LE(n_err, 0.001 * m);  // we tolerate 0.1% of values with a litte difference
      }
    }
  }
};

std::vector<double> Tests_Pagerank::pagerank_time;

TEST_P(Tests_Pagerank, CheckFP32_T) { run_current_test<float>(GetParam()); }

TEST_P(Tests_Pagerank, CheckFP64_T) { run_current_test<double>(GetParam()); }

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_CASE_P(
  simple_test,
  Tests_Pagerank,
  ::testing::Values(Pagerank_Usecase("test/datasets/karate.mtx", ""),
                    Pagerank_Usecase("test/datasets/web-Google.mtx",
                                     "test/ref/pagerank/web-Google.pagerank_val_0.85.bin"),
                    Pagerank_Usecase("test/datasets/ljournal-2008.mtx",
                                     "test/ref/pagerank/ljournal-2008.pagerank_val_0.85.bin"),
                    Pagerank_Usecase("test/datasets/webbase-1M.mtx",
                                     "test/ref/pagerank/webbase-1M.pagerank_val_0.85.bin")));

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  auto resource = std::make_unique<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_default_resource(resource.get());
  int rc = RUN_ALL_TESTS();
  return rc;
}

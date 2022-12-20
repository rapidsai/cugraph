/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// connected components tests
// Author: Andrei Schaffer aschaffer@nvidia.com

#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <converters/legacy/COOtoCSR.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <rmm/device_vector.hpp>

#include <cuda_profiler_api.h>

#include <algorithm>
#include <iterator>

namespace {  // un-nammed
struct Usecase {
  explicit Usecase(const std::string& a)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
  }

  const std::string& get_matrix_file(void) const { return matrix_file; }

 private:
  std::string matrix_file;
};

}  // namespace

struct Tests_Weakly_CC : ::testing::TestWithParam<Usecase> {
  Tests_Weakly_CC() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase()
  {
    if (cugraph::test::g_perf) {
      for (unsigned int i = 0; i < weakly_cc_time.size(); ++i) {
        std::cout << weakly_cc_time[i] << std::endl;
      }
    }
  }
  virtual void SetUp() {}
  virtual void TearDown() {}

  static std::vector<double> weakly_cc_time;

  void run_current_test(const Usecase& param)
  {
    const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream ss;
    std::string test_id = std::string(test_info->test_case_name()) + std::string(".") +
                          std::string(test_info->name()) + std::string("_") +
                          cugraph::test::getFileName(param.get_matrix_file()) + std::string("_") +
                          ss.str().c_str();

    int m, k, nnz;  //
    MM_typecode mc;

    HighResTimer hr_timer{};

    FILE* fpin = fopen(param.get_matrix_file().c_str(), "r");
    ASSERT_NE(fpin, nullptr) << "fopen (" << param.get_matrix_file() << ") failure.";

    ASSERT_EQ(cugraph::test::mm_properties<int>(fpin, 1, &mc, &m, &k, &nnz), 0)
      << "could not read Matrix Market file properties"
      << "\n";
    ASSERT_TRUE(mm_is_matrix(mc));
    ASSERT_TRUE(mm_is_coordinate(mc));
    ASSERT_TRUE(mm_is_symmetric(mc));  // weakly cc only works w/ undirected graphs, for now;

#ifdef _DEBUG_WEAK_CC
    std::cout << "matrix nrows: " << m << "\n";
    std::cout << "matrix nnz: " << nnz << "\n";
#endif

    // Allocate memory on host
    std::vector<int> cooRowInd(nnz);
    std::vector<int> cooColInd(nnz);
    std::vector<int> labels(m);  // for G(V, E), m := |V|
    std::vector<int> verts(m);

    // Read: COO Format
    //
    ASSERT_EQ((cugraph::test::mm_to_coo<int, int>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], nullptr, nullptr)),
              0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    cugraph::legacy::GraphCOOView<int, int, float> G_coo(
      &cooRowInd[0], &cooColInd[0], nullptr, m, nnz);
    auto G_unique                                    = cugraph::coo_to_csr(G_coo);
    cugraph::legacy::GraphCSRView<int, int, float> G = G_unique->view();

    rmm::device_vector<int> d_labels(m);

    if (cugraph::test::g_perf) {
      hr_timer.start("WCC");
      cugraph::connected_components<int, int, float>(
        G, cugraph::cugraph_cc_t::CUGRAPH_WEAK, d_labels.data().get());
      cudaDeviceSynchronize();
      auto time_tmp = hr_timer.stop();
      weakly_cc_time.push_back(time_tmp);
    } else {
      cudaProfilerStart();
      cugraph::connected_components<int, int, float>(
        G, cugraph::cugraph_cc_t::CUGRAPH_WEAK, d_labels.data().get());
      cudaProfilerStop();
      cudaDeviceSynchronize();
    }
  }
};

std::vector<double> Tests_Weakly_CC::weakly_cc_time;

TEST_P(Tests_Weakly_CC, Weakly_CC) { run_current_test(GetParam()); }

// --gtest_filter=*simple_test*
INSTANTIATE_TEST_SUITE_P(simple_test,
                         Tests_Weakly_CC,
                         ::testing::Values(Usecase("test/datasets/dolphins.mtx"),
                                           Usecase("test/datasets/coPapersDBLP.mtx"),
                                           Usecase("test/datasets/coPapersCiteseer.mtx"),
                                           Usecase("test/datasets/hollywood.mtx")));

CUGRAPH_TEST_PROGRAM_MAIN()

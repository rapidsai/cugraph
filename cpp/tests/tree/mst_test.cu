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

// Mst solver tests
// Author: Alex Fender afender@nvidia.com

#include <utilities/high_res_clock.h>
#include <utilities/base_fixture.hpp>
#include <utilities/test_utilities.hpp>

#include <algorithms.hpp>
#include <graph.hpp>

#include <raft/error.hpp>
#include <raft/handle.hpp>

#include <cuda_profiler_api.h>

#include <cmath>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include "../src/converters/COOtoCSR.cuh"

typedef struct Mst_Usecase_t {
  std::string matrix_file;
  Mst_Usecase_t(const std::string& a)
  {
    // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
    const std::string& rapidsDatasetRootDir = cugraph::test::get_rapids_dataset_root_dir();
    if ((a != "") && (a[0] != '/')) {
      matrix_file = rapidsDatasetRootDir + "/" + a;
    } else {
      matrix_file = a;
    }
  }
  Mst_Usecase_t& operator=(const Mst_Usecase_t& rhs)
  {
    matrix_file = rhs.matrix_file;
    return *this;
  }
} Mst_Usecase;

class Tests_Mst : public ::testing::TestWithParam<Mst_Usecase> {
 public:
  Tests_Mst() {}
  static void SetupTestCase() {}
  static void TearDownTestCase() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename T>
  void run_current_test(const Mst_Usecase& param)
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
    std::vector<T> cooVal(nnz), mst(m);

    // Read
    ASSERT_EQ((cugraph::test::mm_to_coo<int, T>(
                fpin, 1, nnz, &cooRowInd[0], &cooColInd[0], &cooVal[0], NULL)),
              0)
      << "could not read matrix data"
      << "\n";
    ASSERT_EQ(fclose(fpin), 0);

    raft::handle_t handle;

    std::cout << std::endl;
    cugraph::GraphCOOView<int, int, T> G_coo(&cooRowInd[0], &cooColInd[0], &cooVal[0], m, nnz);
    auto G_unique = cugraph::coo_to_csr(G_coo);
    cugraph::GraphCSRView<int, int, T> G(G_unique->view().offsets,
                                         G_unique->view().indices,
                                         G_unique->view().edge_data,
                                         G_unique->view().number_of_vertices,
                                         G_unique->view().number_of_edges);

    cudaDeviceSynchronize();

    hr_clock.start();
    cudaProfilerStart();
    auto mst_edges = cugraph::minimum_spanning_tree<int, int, T>(handle, G);
    cudaProfilerStop();

    cudaDeviceSynchronize();
    hr_clock.stop(&time_tmp);
    std::cout << "mst_time: " << time_tmp << " us" << std::endl;

    auto expected_mst_weight = thrust::reduce(
      thrust::device_pointer_cast(G_unique->view().edge_data),
      thrust::device_pointer_cast(G_unique->view().edge_data) + G_unique->view().number_of_edges);

    auto calculated_mst_weight = thrust::reduce(
      thrust::device_pointer_cast(mst_edges->view().edge_data),
      thrust::device_pointer_cast(mst_edges->view().edge_data) + mst_edges->view().number_of_edges);

    std::cout << "calculated_mst_weight: " << calculated_mst_weight << std::endl;
    std::cout << "number_of_MST_edges: " << mst_edges->view().number_of_edges << std::endl;

    EXPECT_LE(calculated_mst_weight, expected_mst_weight);
    EXPECT_LE(mst_edges->view().number_of_edges, 2 * m - 2);
  }
};

TEST_P(Tests_Mst, CheckFP32_T) { run_current_test<float>(GetParam()); }

TEST_P(Tests_Mst, CheckFP64_T) { run_current_test<double>(GetParam()); }

INSTANTIATE_TEST_CASE_P(simple_test,
                        Tests_Mst,
                        ::testing::Values(Mst_Usecase("test/datasets/netscience.mtx")));

CUGRAPH_TEST_PROGRAM_MAIN()
